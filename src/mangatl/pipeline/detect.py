"""A1 -- find the lettering on a page.

comic-text-detector exposes three heads for one letterboxed 1024x1024 page:

    blk  (64512, 7)  YOLOv5-style boxes -- cx, cy, w, h, objectness, 2 classes
    seg  (1, H, W)   per-pixel probability that a pixel is a glyph
    det  (2, H, W)   channel 0 is a text-*line* map, channel 1 its background

Each head is used for the thing it is actually good at, and the choice of
which drives region-finding was made by looking at all three on real pages:

* `det[0]` drives it. It marks one filled bar per line of set lettering, at
  whatever scale the line happens to be, which is the grouping problem already
  solved by the network instead of by a dilation kernel that cannot suit a
  page holding both 6px furigana and a 60px shout. Hand-drawn sound effects
  are absent from this head while dialogue and captions are present, so it
  doubles as the sound-effect classifier -- see `RegionKind.SFX`.

* `seg` builds the erase mask. It is pixel-accurate on glyph edges, which is
  what inpainting needs and what a box can never give.

* `blk` is unused. It is trained on dialogue and misses lettering that still
  needs translating -- on the sample page it skips the chapter title bar
  completely -- and the layout model in `pipeline.layout` does the same job
  properly, with a balloon class besides.

Which lines belong to one block is decided by that layout model, not here: a
block is a detection, and lines are assigned to the block that covers them.
Only lines no block claims are grouped by the pairwise rule below, which
remains for the lettering the layout model does not fire on. That rule is
transitive (union-find), so nothing pairwise can bound how large a group
becomes, and `detect.max_block_area_ratio` is what actually bounds it -- but
it is applied only to groups built that way, never to a block the layout
model drew, whose extent is not this module's to second-guess.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

import cv2
import numpy as np

from .. import runtime
from ..config import Config
from ..types import Box, Region, RegionKind
from .layout import Layout, LayoutDetector

runtime.enable_bundled_cuda()

import onnxruntime as ort  # noqa: E402  -- must follow the DLL path fix


class _Letterbox(NamedTuple):
    """The transform that put a page inside the model's square input."""

    scale: float
    pad_x: int
    pad_y: int
    inner_w: int
    inner_h: int
    page_h: int
    page_w: int

    def map_to_page(self, plane: np.ndarray) -> np.ndarray:
        """Undo the letterbox on a model-resolution single-channel map."""
        inner = plane[self.pad_y : self.pad_y + self.inner_h,
                      self.pad_x : self.pad_x + self.inner_w]
        return cv2.resize(
            inner, (self.page_w, self.page_h), interpolation=cv2.INTER_LINEAR
        )


@dataclass
class _Candidate:
    """A region under construction, before reading order is settled."""

    box: Box
    line_height: float
    lines: list[Box]
    bubble: Box | None = None
    # Set only where the glyph mask had nothing to give and the ink was found
    # by contrast instead. Carried rather than recomputed because everything
    # downstream reads it: classification decides colour and whether the
    # region is a watermark from these pixels, and erasure paints over them.
    mask: np.ndarray | None = None


@dataclass
class DetectResult:
    regions: list[Region]
    glyph_mask: np.ndarray                    # page-sized uint8, 0 or 255
    line_boxes: list[Box] = field(default_factory=list)
    block_boxes: list[Box] = field(default_factory=list)
    layout: Layout = field(default_factory=Layout)


def _letterbox(image: np.ndarray, size: int) -> tuple[np.ndarray, _Letterbox]:
    h, w = image.shape[:2]
    scale = min(size / w, size / h)
    inner_w, inner_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(image, (inner_w, inner_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    pad_x, pad_y = (size - inner_w) // 2, (size - inner_h) // 2
    canvas[pad_y : pad_y + inner_h, pad_x : pad_x + inner_w] = resized
    blob = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return blob.transpose(2, 0, 1)[None], _Letterbox(
        scale, pad_x, pad_y, inner_w, inner_h, h, w
    )


class _UnionFind:
    def __init__(self, n: int) -> None:
        self._parent = list(range(n))

    def find(self, a: int) -> int:
        while self._parent[a] != a:
            self._parent[a] = self._parent[self._parent[a]]
            a = self._parent[a]
        return a

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[rb] = ra

    def groups(self) -> dict[int, list[int]]:
        out: dict[int, list[int]] = {}
        for i in range(len(self._parent)):
            out.setdefault(self.find(i), []).append(i)
        return out


def _components(binary: np.ndarray, min_area: int) -> list[Box]:
    """Connected components as page-space boxes, noise dropped.

    No morphological closing is applied first. The line head emits solid bars
    rather than glyph outlines, so there are no breaks along a line to heal --
    and a closing wide enough to be worth applying reaches across the gutter
    between two balloons standing side by side. Measured on the sample page: a
    9x3 close fused a 72px line in the left balloon with a 52px line in the
    right one into a single 128x23 component, which then dragged both balloons
    into one region through union-find. Thresholding alone gives seven
    correctly separated components in that same crop.
    """
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    boxes: list[Box] = []
    for i in range(1, count):
        x, y, w, h, area = stats[i]
        if area < min_area or w < 4 or h < 3:
            continue
        boxes.append(Box(int(x), int(y), int(x + w), int(y + h)))
    return boxes


def _stroke_width(mask: np.ndarray) -> float:
    """Mean stroke thickness of a glyph mask, from its distance transform.

    Set lettering on a page is drawn at the weight of that page's dialogue.
    Sound effects are drawn far heavier. Measuring the two against each other
    is what makes the comparison hold across chapters and page sizes, where an
    absolute thickness in pixels would not.
    """
    binary = (mask > 0).astype(np.uint8)
    if binary.sum() == 0 or binary.all():
        return 0.0
    # Padded, so glyphs touching the crop edge do not read as infinitely thick.
    padded = cv2.copyMakeBorder(binary, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    dist = cv2.distanceTransform(padded, cv2.DIST_L2, 5)[1:-1, 1:-1]
    widths = 2.0 * dist[binary > 0].astype(np.float64)
    mean = float(widths.mean()) if widths.size else 0.0
    return mean if np.isfinite(mean) and mean > 0 else 0.0


def _slant(mask: np.ndarray) -> float:
    """How far a mask's strokes lean off upright, in degrees, area-weighted.

    Sound effects in this genre are brush-drawn on a slant; set lettering
    stands upright, so its components' principal axes sit near 0 or 90
    degrees. Measured over two chapters: captions the detector missed came in
    at 16.7 degrees at worst, while sound effects and artwork ran to 43.
    """
    binary = (mask > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    angles: list[float] = []
    areas: list[int] = []
    for i in range(1, count):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < 30:
            continue
        points = cv2.findNonZero((labels == i).astype(np.uint8))
        if points is None or len(points) < 12:
            continue
        (_, _), (width, height), angle = cv2.minAreaRect(points)
        if width < height:
            angle += 90.0
        angle %= 180.0
        angles.append(min(abs(angle), abs(angle - 90.0), abs(angle - 180.0)))
        areas.append(area)
    if not angles:
        return 0.0
    return float(np.average(angles, weights=areas))


def _lines_from_map(
    line_map: np.ndarray, strong: float, weak: float, min_area: int
) -> list[Box]:
    """Text lines from the line head, by hysteresis over two floors.

    One floor cannot serve both jobs this stage has. A floor low enough to see
    the faintest real balloon (0.248 peak on the sample page) is also low
    enough that neighbouring balloons' bars swell until they touch, and
    union-find then merges the balloons. A floor high enough to keep them
    apart drops the faint balloon entirely.

    So the strong floor decides where lines *are* -- it is what keeps adjacent
    balloons separate -- and the weak floor only contributes components in
    places the strong floor found nothing at all. A weak blob that bridges two
    strong lines intersects them both and is discarded, so lowering the weak
    floor can add regions but can never merge existing ones.
    """
    strong_boxes = _components((line_map > strong).astype(np.uint8), min_area)
    if weak >= strong:
        return strong_boxes
    weak_boxes = _components((line_map > weak).astype(np.uint8), min_area)
    return strong_boxes + [
        b for b in weak_boxes if not any(b.intersects(s) for s in strong_boxes)
    ]


def _lines_from_mask(patch: np.ndarray, off_x: int, off_y: int) -> list[Box]:
    """Split a glyph mask into lettered lines by its row profile.

    Used only for regions recovered from the box head, where there are no line
    bars to group. Rows holding glyphs form runs, one run per line, and each
    run's horizontal extent comes from the glyphs inside it.
    """
    rows = (patch > 0).any(axis=1)
    if not rows.any():
        return []
    edges = np.flatnonzero(np.diff(np.r_[0, rows.astype(np.int8), 0]))
    boxes: list[Box] = []
    for start, end in zip(edges[::2], edges[1::2]):
        band = patch[start:end]
        columns = np.flatnonzero((band > 0).any(axis=0))
        if columns.size == 0:
            continue
        boxes.append(
            Box(
                off_x + int(columns[0]), off_y + int(start),
                off_x + int(columns[-1]) + 1, off_y + int(end),
            )
        )
    return boxes


def _split_at_gutters(line: Box, glyph_mask: np.ndarray, ratio: float) -> list[Box]:
    """Cut a line box wherever it crosses a gap between two text columns.

    The line head marks a horizontal band, and where two blocks of text sit
    side by side that band covers both. Left joined, such a line bridges the
    two columns into one region through union-find, and recognition then walks
    the region row by row -- taking one line from the left column, one from the
    right, and so on. That is how a page came back reading "IT IF THE WOULD
    MAIDEN BE THE THAT IS PERFECT SUPPORTED", which no translator could
    recover from.

    Word spaces and column gutters are separated by scale rather than by
    position. Measured over the region that produced that line: word gaps run
    to 1.0x the line's height, while the narrowest real gutter was 1.375x.
    """
    patch = line.crop(glyph_mask)
    if patch.size == 0:
        return [line]
    columns = (patch > 0).any(axis=0)
    if not columns.any():
        return [line]

    edges = np.flatnonzero(np.diff(np.r_[0, columns.astype(np.int8), 0]))
    starts, ends = edges[::2], edges[1::2]
    if starts.size < 2:
        return [line]

    limit = ratio * line.height
    pieces: list[Box] = []
    run_start, run_end = int(starts[0]), int(ends[0])
    for start, end in zip(starts[1:], ends[1:]):
        if int(start) - run_end > limit:
            pieces.append(Box(line.x1 + run_start, line.y1, line.x1 + run_end, line.y2))
            run_start = int(start)
        run_end = int(end)
    pieces.append(Box(line.x1 + run_start, line.y1, line.x1 + run_end, line.y2))
    return pieces if len(pieces) > 1 else [line]


def _glyph_runs(line: Box, glyph_mask: np.ndarray) -> list[tuple[int, int]]:
    """The horizontal runs of ink inside a line box, in page pixels."""
    patch = line.crop(glyph_mask)
    if patch.size == 0:
        return []
    columns = (patch > 0).any(axis=0)
    if not columns.any():
        return []
    edges = np.flatnonzero(np.diff(np.r_[0, columns.astype(np.int8), 0]))
    return [
        (line.x1 + int(start), line.x1 + int(end))
        for start, end in zip(edges[::2], edges[1::2])
    ]


def _stand_side_by_side(blocks: list[Box], slack: float = 0.15) -> bool:
    """Whether these blocks are columns beside each other, not one on another.

    Only side-by-side blocks say where a bar crossing them should be cut. Two
    boxes over the same column -- the near-duplicates a fine-tune emits, which
    survive suppression when one sits inside the other and their IoU is
    therefore low -- say nothing about a boundary, and cutting on them would
    take a bar apart in the middle of a word. So an overlap of more than a
    sliver anywhere means this rule stands aside and the gap rule decides.
    """
    for index, one in enumerate(blocks):
        for other in blocks[index + 1 :]:
            shared = min(one.x2, other.x2) - max(one.x1, other.x1)
            if shared > slack * min(one.width, other.width):
                return False
    return True


def _split_across_blocks(
    line: Box, blocks: list[Box], glyph_mask: np.ndarray
) -> list[Box]:
    """Cut a line bar where it crosses out of one detected block into another.

    `_split_at_gutters` asks how wide the gap is, and a gap is not always the
    answer. Two blocks of lettering standing side by side in one balloon --
    "WHAT'S GOING ON...?" beside "AREN'T EAGLE EYE CAPTAINS..." on the page
    this was measured on -- were separated by 19px where the same line's word
    spaces ran to 2px and its own height was 19px. No multiple of the line
    height separates 19 from a word space on one page and from a real space on
    the next, and the one tried let a bar spanning both columns through. Both
    balloons' text then landed in one region, was read as one garbled
    paragraph, and -- because the region's box now covered both columns --
    was lettered back across the whole balloon, on top of its neighbour.

    The layout model has already answered the question the gap was standing in
    for: it drew a box round each of those two columns. So a run of ink is
    assigned to the block that covers it, and the bar is cut where that
    changes. Where the model drew only one block over a bar, or none, nothing
    is cut and the gap rule remains the only judge.
    """
    middle = (line.y1 + line.y2) / 2.0
    over = [
        block
        for block in blocks
        # Vertically the line has to sit in the block, horizontally they need
        # only meet: a bar reaching out of one block into the next is exactly
        # the case being cut, so demanding containment would skip it.
        if block.y1 <= middle <= block.y2 and block.x1 < line.x2 and line.x1 < block.x2
    ]
    if len(over) < 2 or not _stand_side_by_side(over):
        return [line]

    runs = _glyph_runs(line, glyph_mask)
    if len(runs) < 2:
        return [line]

    def owner(run: tuple[int, int]) -> Box | None:
        """The block covering most of one run of ink, if one covers it."""
        left, right = run
        width = right - left
        best, best_cover = None, 0.5
        for block in over:
            cover = max(0, min(right, block.x2) - max(left, block.x1)) / max(1, width)
            if cover > best_cover:
                best, best_cover = block, cover
        return best

    pieces: list[Box] = []
    start, end = runs[0]
    held = owner(runs[0])
    for run in runs[1:]:
        claim = owner(run)
        # An unclaimed run joins whatever it follows rather than starting a
        # piece of its own: a stray mark between two columns is punctuation or
        # noise, and either way it is not a block.
        if claim is not None and held is not None and claim != held:
            pieces.append(Box(start, line.y1, end, line.y2))
            start, held = run[0], claim
        elif held is None:
            held = claim
        end = run[1]
    pieces.append(Box(start, line.y1, end, line.y2))
    return pieces if len(pieces) > 1 else [line]


def _split_fused_lines(lines: list[Box], glyph_mask: np.ndarray) -> list[Box]:
    """Break line boxes that hold more than one line of lettering.

    Large display lettering sits close enough together that the line head
    covers several lines in a single band. That band is then far taller than
    the rest of the page's lines, and the height-similarity rule in
    `_should_join` -- which exists to keep a sound effect from swallowing
    dialogue -- refuses to join it to anything. The balloon comes apart:
    "YES, I AM A SEWER RAT!" was detected as "ES", a 99px band, and "RAT",
    translated as three unrelated fragments.

    The split is decided by the glyph mask's own row profile, so a band that
    really is one tall line yields one run and is left alone.
    """
    if not lines:
        return lines
    median = float(np.median([line.height for line in lines]))
    out: list[Box] = []
    for line in lines:
        if line.height > 1.5 * median:
            parts = _lines_from_mask(line.crop(glyph_mask), line.x1, line.y1)
            if len(parts) > 1:
                out.extend(parts)
                continue
        out.append(line)
    return out


def merge_rows(lines: list[Box]) -> list[Box]:
    """Line boxes merged into the rows of lettering they actually make up.

    The line head marks bars, and one row of lettering can arrive as several
    bars side by side: "INCONCEIVABLE!" came back as one bar over
    "INCONCEIV" and another over "BLE!", because the white outline breaks the
    line map where the letters lean apart. They are one line of text, and
    `_chunks` splits a block *between* its lines -- so read as two lines the
    word was cut in half and came back "TNANA" and "BLE!", which is what
    shipped to the page.

    Bars belong to the same row when they overlap vertically by more than
    half the shorter one *and* are about as tall as each other. Overlap
    rather than proximity, because two lines of a balloon sit a few pixels
    apart and must stay separate, while two bars of one row sit on the same
    pixels whatever the gap between them across.

    The height test is what keeps this off lettering drawn on a slant. A
    banner reading "Thud Thud Thud" down a diagonal produces bars that do
    overlap vertically -- they are stepping down the page, not sitting on a
    row -- along with stray fragments of the same effect. Merged, the crop
    handed to the recogniser stopped containing a whole word and the reading
    lost the "Thud" that identified it as a noise, so the banner was
    translated and lettered over. A row of type has one x-height, and two
    bars differing by more than half are not pieces of it.
    """
    if len(lines) < 2:
        return list(lines)
    rows: list[Box] = []
    for line in sorted(lines, key=lambda b: b.y1):
        merged = False
        for index, row in enumerate(rows):
            overlap = min(row.y2, line.y2) - max(row.y1, line.y1)
            shorter = min(row.height, line.height)
            taller = max(row.height, line.height)
            if overlap > 0.5 * shorter and taller <= 2.0 * shorter:
                rows[index] = Box.union([row, line])
                merged = True
                break
        if not merged:
            rows.append(line)
    return sorted(rows, key=lambda b: b.y1)


def _refine_lines(lines: list[Box], glyph_mask: np.ndarray) -> list[Box]:
    """Split line boxes that actually hold several stacked lines.

    The line head fuses lines that sit very close together, which is normal
    for a narrow balloon lettered in short bursts. The result is a box taller
    than it is wide, and a recogniser handed that crop reads nothing at all --
    "STAY WITH US!" came back empty because its three lines arrived as one
    25x54 box. The glyph mask's row profile separates them.
    """
    if not lines:
        return lines
    median = float(np.median([line.height for line in lines]))
    out: list[Box] = []
    for line in lines:
        stacked = line.height > 1.6 * median or line.width < 1.2 * line.height
        if stacked and line.height >= 10:
            parts = _lines_from_mask(line.crop(glyph_mask), line.x1, line.y1)
            if len(parts) > 1:
                out.extend(parts)
                continue
        out.append(line)
    return sorted(out, key=lambda b: (b.y1, b.x1))


def _should_join(a: Box, b: Box, gap_ratio: float) -> bool:
    """Whether two text lines belong to the same block.

    Horizontal overlap is the rule that keeps two balloons standing side by
    side from being stitched together: their lines interleave vertically, so a
    vertical test alone walks from one balloon into the other.
    """
    upper, lower = (a, b) if a.y1 <= b.y1 else (b, a)
    gap = lower.y1 - upper.y2
    tallest = max(a.height, b.height)
    if gap > gap_ratio * tallest:
        return False
    if gap < -0.6 * tallest:  # heavily stacked, likely one bar split by noise
        return False

    overlap = min(a.x2, b.x2) - max(a.x1, b.x1)
    if overlap <= 0:
        return False
    if overlap < 0.35 * min(a.width, b.width):
        return False

    tall, short = max(a.height, b.height), min(a.height, b.height)
    return tall <= 2.5 * short


def _owning_block(lines: list[Box], blocks: list[Box], min_cover: float = 0.6) -> list[int]:
    """For each line, the index of the detected block containing it, or -1.

    The box head is trained on whole balloons, so it knows an extent that no
    amount of line geometry can recover: two balloons standing side by side
    with their lines interleaved vertically and their edges a few pixels
    apart. A line covered by no box gets -1 and is grouped on geometry alone,
    which is what carries lettering the box head does not fire on -- the
    chapter title bar being the case that showed up first.
    """
    owner = [-1] * len(lines)
    for i, line in enumerate(lines):
        if line.area == 0:
            continue
        best, best_cover = -1, 0.0
        for b, box in enumerate(blocks):
            ix1, iy1 = max(line.x1, box.x1), max(line.y1, box.y1)
            ix2, iy2 = min(line.x2, box.x2), min(line.y2, box.y2)
            cover = (max(0, ix2 - ix1) * max(0, iy2 - iy1)) / line.area
            if cover > best_cover:
                best, best_cover = b, cover
        if best_cover >= min_cover:
            owner[i] = best
    return owner


def _group_lines(
    lines: list[Box], blocks: list[Box], page_area: int, cfg: Config
) -> list[list[int]]:
    """Assign lines to the blocks that cover them; group the rest on geometry.

    Two lines the layout model puts in one block are one region whatever the
    gap between them looks like, and two lines it puts in different blocks are
    never joined. That is the whole point of having a block detector: the
    pairwise rule below cannot tell a generously leaded balloon from two
    balloons stacked in a column, and every threshold that keeps one of those
    right gets the other wrong. Measured across two series, lines inside one
    balloon sat at 1.17-1.20 in the first and 1.60 in the second, and the
    value that held both apart also welded two side-by-side balloons.

    The rule still runs for lines no block claims, because the layout model
    does not fire on everything -- credits, the chapter title bar -- and those
    lines have nothing else to group them.
    """
    if not lines:
        return []
    owner = _owning_block(lines, blocks)
    uf = _UnionFind(len(lines))
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            if owner[i] != -1 and owner[j] != -1:
                if owner[i] == owner[j]:
                    uf.union(i, j)
                continue
            if _should_join(lines[i], lines[j], cfg.detect.merge_gap_ratio):
                uf.union(i, j)

    max_area = cfg.detect.max_block_area_ratio * page_area
    blocks: list[list[int]] = []
    for members in uf.groups().values():
        members.sort(key=lambda k: (lines[k].y1, lines[k].x1))
        # A block the layout model drew is left whole. The cap exists to bound
        # transitive growth, and a detection did not grow -- cutting one in
        # half splits a balloon mid-sentence, which is the fault the cap was
        # meant to prevent.
        claimed = {owner[k] for k in members}
        if claimed != {-1} and len(claimed) == 1:
            blocks.append(members)
            continue
        if Box.union([lines[k] for k in members]).area <= max_area:
            blocks.append(members)
            continue
        # Over the cap. Re-grow greedily in reading order and cut when the
        # accumulated block would break it, rather than dropping the group.
        current: list[int] = []
        for k in members:
            trial = current + [k]
            if current and Box.union([lines[m] for m in trial]).area > max_area:
                blocks.append(current)
                current = [k]
            else:
                current = trial
        if current:
            blocks.append(current)
    return blocks


def _drop_nested(
    candidates: list["_Candidate"], share: float = 0.75
) -> list["_Candidate"]:
    """Drop a block of lettering another block already holds.

    Three paths make candidates -- lines grouped into the blocks the layout
    model drew, blocks the line head missed entirely, and what is left in the
    glyph mask afterwards -- and each is kept clear of the others by IoU,
    which is the wrong measure for a box sitting *inside* another. A 66x106
    box inside a 216x215 one scores 0.15 by that measure and survives, so one
    caption on the reported chapter became two regions: the larger read "THIS
    WAS A DAOIST ART:" and the smaller read the middle of the same words as
    "THIS AS A OIS". Both were translated and both were lettered.

    The larger keeps it, because recognition reads a crop of the region's box
    and the smaller's lettering is inside that crop -- nothing is lost by
    dropping it, and what the fuller box read was the better reading of the
    two every time this was looked at.
    """
    kept: list[_Candidate] = []
    for item in sorted(candidates, key=lambda c: -c.box.area):
        if not any(_holds(other.box, item.box, share) for other in kept):
            kept.append(item)
    return kept


def _holds(outer: Box, inner: Box, share: float) -> bool:
    """Whether `inner` sits far enough inside `outer` to be the same lettering."""
    ix1, iy1 = max(outer.x1, inner.x1), max(outer.y1, inner.y1)
    ix2, iy2 = min(outer.x2, inner.x2), min(outer.y2, inner.y2)
    covered = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    return bool(inner.area) and covered >= share * inner.area


# What a block of lettering looks like once its ink is separated from its
# ground: enough of it to be text, not so much that the whole box is ink.
_INK_FLOOR, _INK_CEILING = 0.04, 0.55


def _glyphs_by_contrast(crop: np.ndarray) -> np.ndarray | None:
    """Ink in a crop, told from its own background rather than by the seg head.

    Otsu against the local median, in whichever direction puts less of the
    crop in the foreground -- so it reads white lettering on a coloured plate
    and dark lettering on a light one without being told which it is.

    Returns None when the result does not look like lettering, because this
    runs on boxes the glyph mask has already declined and inventing a region
    on artwork is worse than missing one.
    """
    if crop.size == 0:
        return None
    grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, dark = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    light = cv2.bitwise_not(dark)
    mask = dark if dark.mean() <= light.mean() else light
    ink = float((mask > 0).mean())
    if not _INK_FLOOR <= ink <= _INK_CEILING:
        return None
    # Lettering is several separate marks. One solid blob is a highlight, a
    # speech-balloon tail, or a piece of the drawing.
    count, _, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), connectivity=8
    )
    marks = sum(1 for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= 4)
    return mask if marks >= 2 else None


class TextDetector:
    """Wraps one comic-text-detector session. Build once, reuse per page."""

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.log_severity_level = 3
        self._session = ort.InferenceSession(
            str(cfg.models.detector),
            sess_options=options,
            providers=cfg.models.providers(),
        )
        self._input = self._session.get_inputs()[0].name
        self._outputs = [o.name for o in self._session.get_outputs()]
        self._layout = LayoutDetector(cfg)

    @property
    def provider(self) -> str:
        return self._session.get_providers()[0]

    def detect(self, image: np.ndarray) -> DetectResult:
        cfg = self._cfg.detect
        blob, lb = _letterbox(image, cfg.input_size)
        raw = self._session.run(None, {self._input: blob})
        heads = dict(zip(self._outputs, raw))

        glyph = lb.map_to_page(heads["seg"][0][0])
        lines_map = lb.map_to_page(heads["det"][0][0])

        glyph_mask = ((glyph > cfg.mask_threshold) * 255).astype(np.uint8)

        page_area = lb.page_h * lb.page_w
        # Run before the line bars are cut, not after: the blocks it draws are
        # what decides where two of those cuts fall.
        layout = self._layout.detect(image)
        block_boxes = layout.boxes

        # min_area is quoted for a page-sized region; scale the line-map floor
        # with it but keep it well under, since one line is a slice of a block.
        line_boxes = _lines_from_map(
            lines_map,
            cfg.line_threshold,
            cfg.line_threshold_weak,
            max(12, cfg.min_area // 8),
        )
        # Cut across column gutters before grouping, not after: a line that
        # spans two columns is exactly what welds them into one region, so by
        # the time grouping has run the damage is done.
        line_boxes = [
            piece
            for line in line_boxes
            for piece in _split_at_gutters(line, glyph_mask, cfg.gutter_ratio)
        ]
        # Then across the boundary between two blocks the layout model drew,
        # which catches the columns whose gutter is too narrow for the rule
        # above to call -- 19px on a page here, against word spaces of 2px.
        line_boxes = [
            piece
            for line in line_boxes
            for piece in _split_across_blocks(line, block_boxes, glyph_mask)
        ]
        # Then vertically, for the same reason: a band holding several lines
        # cannot be grouped with anything, because it looks nothing like the
        # lines around it.
        line_boxes = _split_fused_lines(line_boxes, glyph_mask)

        blocks = _group_lines(line_boxes, block_boxes, page_area, self._cfg)

        page_shape = (lb.page_h, lb.page_w)
        candidates: list[_Candidate] = []
        for members in blocks:
            members = sorted(members, key=lambda k: line_boxes[k].y1)
            lines = [line_boxes[k] for k in members]
            box = Box.union(lines).padded(3, within=page_shape)
            if box.area < cfg.min_area:
                continue
            heights = [line.height for line in lines]
            candidates.append(
                _Candidate(
                    box,
                    float(np.median(heights)),
                    merge_rows(_refine_lines(lines, glyph_mask)),
                )
            )

        candidates.extend(
            self._recover_missed(
                block_boxes, candidates, glyph_mask, page_shape, image
            )
        )
        if cfg.recover_unclaimed:
            candidates.extend(
                self._recover_unclaimed(candidates, glyph_mask, page_shape)
            )
        candidates = _drop_nested(candidates)

        ordered = sorted(
            candidates,
            key=self._reading_key(lb.page_h, self._cfg.detect.reading_order),
        )
        # Resolved here rather than while candidates are built, so that
        # regions recovered from the glyph mask alone are given their balloon
        # too -- those are exactly the ones with the least geometry of their
        # own to fall back on.
        for item in ordered:
            if item.bubble is None:
                item.bubble = layout.bubble_for(item.box)

        regions = [
            Region(
                id=f"r{idx:02d}",
                box=item.box,
                mask=(
                    item.mask
                    if item.mask is not None
                    else item.box.crop(glyph_mask) > 0
                ),
                kind=RegionKind.BALLOON,
                source_line_height=item.line_height,
                lines=item.lines,
                bubble=item.bubble,
            )
            for idx, item in enumerate(ordered)
        ]
        return DetectResult(regions, glyph_mask, line_boxes, block_boxes, layout)

    def _recover_missed(
        self,
        block_boxes: list[Box],
        found: list["_Candidate"],
        glyph_mask: np.ndarray,
        page_shape: tuple[int, int],
        image: np.ndarray,
    ) -> list["_Candidate"]:
        """Rebuild regions the line head missed but the box head is sure of.

        The two heads disagree on small balloons: on the sample page the box
        head puts a confident box on the "LADY REIRIN!" balloon in the
        top-right panel while the line head produces nothing there even at a
        0.20 floor. Left alone that balloon ships with English lettering in
        it, which is the worst outcome the pipeline has -- worse than a rough
        translation, because it is visible to any reader.

        A box is only recovered when there really is lettering inside it, so
        a spurious box on empty art cannot invent a region. The glyph mask
        answers that where it can. Where it cannot -- it is trained on
        greyscale manga and finds nothing at all in white lettering on a
        saturated colour plate -- the ink is separated from its own local
        background instead, which is the same fallback a region drawn by hand
        uses. Two character name plates on the reported page were lost this
        way: the layout model was sure of both, and the glyph mask returned
        1% coverage on one and 0% on the other.
        """
        cfg = self._cfg.detect
        recovered: list[_Candidate] = []
        for box in block_boxes:
            if box.area < cfg.min_area:
                continue
            if any(box.iou(existing.box) > 0.15 for existing in found):
                continue
            patch = box.crop(glyph_mask)
            if patch.size == 0:
                continue
            recovered_mask: np.ndarray | None = None
            if (patch > 0).mean() < 0.03:
                patch = _glyphs_by_contrast(box.crop(image))
                if patch is None:
                    continue
                recovered_mask = patch
            ys, xs = np.nonzero(patch)
            tight = Box(
                box.x1 + int(xs.min()), box.y1 + int(ys.min()),
                box.x1 + int(xs.max()) + 1, box.y1 + int(ys.max()) + 1,
            ).padded(3, within=page_shape)
            if tight.area < cfg.min_area:
                continue
            # No line bars here, so the lines are read off the mask itself.
            lines = _lines_from_mask(patch, box.x1, box.y1)
            heights = [line.height for line in lines]
            line_height = float(np.median(heights)) if heights else float(tight.height)
            local = None
            if recovered_mask is not None:
                # Cut to the same box the region will be, so it lines up with
                # `Region.mask`, which is always the region's own crop.
                local = np.zeros((tight.height, tight.width), bool)
                y1, x1 = tight.y1 - box.y1, tight.x1 - box.x1
                piece = recovered_mask[
                    max(0, y1) : max(0, y1) + tight.height,
                    max(0, x1) : max(0, x1) + tight.width,
                ] > 0
                local[: piece.shape[0], : piece.shape[1]] = piece
            recovered.append(
                _Candidate(tight, line_height, lines or [tight], mask=local)
            )
        return recovered

    def _recover_unclaimed(
        self,
        found: list["_Candidate"],
        glyph_mask: np.ndarray,
        page_shape: tuple[int, int],
    ) -> list["_Candidate"]:
        """Lettering neither head found, judged against the page's own text.

        The line head is what separates set lettering from hand-drawn sound
        effects, and on some captions it is simply silent -- measured on one
        page, a caption scored 0.066 while a sound effect on the same page
        scored 0.367, so no threshold on that head can recover the one without
        taking the other. This looks at what is left in the glyph mask instead
        and asks two questions of it, both answered relative to the lettering
        the detector *did* find on this page rather than in absolute units:

        * is it drawn at the weight of this page's dialogue, or far heavier?
        * does it stand upright, or lean like a brush stroke?

        Measured over two chapters, on the 96 largest unclaimed blobs: the
        gates below keep 6 of the 8 genuinely missed captions and 1 of the 88
        sound effects and artwork fragments -- and that one is a block of
        scanlation credits, which is text.

        Nothing recovered here is erased on the strength of this alone.
        Recognition still has to read words out of it; a region that comes
        back unreadable is marked skipped and keeps its pixels. That is what
        makes it safe to be generous here -- the cost of a wrong guess is a
        wasted OCR call, not damaged artwork.
        """
        cfg = self._cfg.detect
        height, width = page_shape

        reference = float(np.median([item.line_height for item in found])) if found else 0.0
        widths = [
            w for w in (_stroke_width(item.box.crop(glyph_mask)) for item in found) if w > 0
        ]
        if not widths or reference <= 0:
            return []
        stroke_reference = float(np.median(widths))

        claimed = np.zeros((height, width), np.uint8)
        for item in found:
            # Claimed generously: a region's own stroke, and glyphs the line
            # head trimmed off its ends, lie outside its box and must not be
            # mistaken for text nobody found.
            pad = int(max(24.0, 1.2 * max(item.line_height, 8.0)))
            box = item.box.padded(pad, within=page_shape)
            claimed[box.y1 : box.y2, box.x1 : box.x2] = 1

        loose = ((glyph_mask > 0) & (claimed == 0)).astype(np.uint8)
        if not loose.any():
            return []
        grouped = cv2.dilate(loose, np.ones((15, 15), np.uint8))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(grouped, connectivity=8)

        min_w = max(40.0, 2.5 * reference)
        min_h = max(18.0, 0.9 * reference)
        recovered: list[_Candidate] = []
        for i in range(1, count):
            x, y, w, h, _ = stats[i]
            if w < min_w or h < min_h:
                continue
            piece = ((labels[y : y + h, x : x + w] == i) & (loose[y : y + h, x : x + w] > 0))
            piece = piece.astype(np.uint8)
            if piece.sum() < max(400, reference * reference * 0.5):
                continue

            stroke = _stroke_width(piece)
            if stroke <= 0 or stroke / stroke_reference > cfg.recover_max_stroke_ratio:
                continue
            if _slant(piece) > cfg.recover_max_slant:
                continue

            ys, xs = np.nonzero(piece)
            box = Box(
                x + int(xs.min()), y + int(ys.min()),
                x + int(xs.max()) + 1, y + int(ys.max()) + 1,
            ).padded(3, within=page_shape)
            lines = _lines_from_mask(piece, x, y)
            heights = [line.height for line in lines]
            measured = float(np.median(heights)) if heights else reference
            # Clamped to the page's own lettering. A caption in a heavy stroke
            # merges into one run, so its "line height" comes back as the whole
            # block -- 204px for three lines of ordinary text. Everything
            # downstream scales off this number, and left alone it made the
            # erase reach across a sixth of the page.
            line_height = float(min(measured, 2.0 * reference)) if reference else measured
            recovered.append(_Candidate(box, line_height, lines or [box]))
        return recovered

    @staticmethod
    def _reading_key(page_h: int, order: str):
        """Top to bottom in bands, then across each band by `order`.

        This is what makes a page's balloons reach the translator in the order
        a reader meets them, which is the whole reason page-level translation
        beats per-balloon translation. The band is a fraction of page height
        rather than a strict row, so two balloons sitting at slightly
        different heights in one panel are still read together.
        """
        band = max(1, page_h // 24)

        def key(item: "_Candidate") -> tuple[int, int]:
            box = item.box
            return (box.y1 // band, -box.x2 if order == "rtl" else box.x1)

        return key

