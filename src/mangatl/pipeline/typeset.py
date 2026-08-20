"""C1 -- letter the Indonesian text back into the page.

Rendering is Skia, not Pillow. Comic lettering is set at small sizes over
artwork, where hinting and subpixel placement are the difference between
lettering that belongs on the page and lettering that sits on top of it.

Four decisions carry most of the quality:

* **Each line is measured against the balloon at its own y, and centred on the
  balloon's own centre at that y.** Balloons are not symmetric columns --
  shout balloons slant, and a partitioned balloon is a wedge -- so a single
  centre for the whole block puts some lines' centres outside the balloon
  entirely. That reported as text set at two thirds the size of the lettering
  it replaced.

* **The block is anchored on the original lettering**, and slid back inside
  the balloon if it would overhang. Anchoring keeps the translation where the
  artist put it; sliding stops a taller block being refused for want of a few
  pixels that exist a little further up.

* **Size is taken from the line pitch of the lettering being replaced.** The
  page has a voice, set by the letterer, and matching its density is what
  keeps a translated page looking like the original rather than like a
  transcript of it.

* **No two blocks are drawn over each other**, whatever the stages before
  decided. Every region is fitted on its own against a shape meant to be its
  alone, and the shapes are worked out by four different routes of which only
  one knows the neighbours exist -- so this is the last place the question can
  be asked, and `_settle_clashes` asks it. A block that would cover another's
  is fitted again in what is left, and if nothing is left it keeps its
  English: one balloon a reader has to skip, rather than two they cannot read.

Whatever is measured here is what gets drawn: `Layout` carries the settled
position and per-line centres, so the width a line was wrapped against is the
width it is drawn at.
"""

from __future__ import annotations

from collections.abc import Callable
from math import ceil
from dataclasses import dataclass

import cv2
import numpy as np
import skia

from ..config import Config
from ..languages import pair_for
from ..types import Box, Region, RegionStatus
from .fallback import Face


@dataclass
class Layout:
    """A settled block of lettering, in page coordinates."""

    lines: list[str]
    centres: list[float]
    size: float
    line_height: float
    first_baseline: float

    @property
    def height(self) -> float:
        return self.line_height * len(self.lines)


def _region_polygon(region: Region) -> np.ndarray:
    if region.polygon is not None and len(region.polygon) >= 3:
        return region.polygon
    # The balloon before the lettering. Falling back to the box the English
    # occupied sets the Indonesian to the size of a sentence it is not, and
    # Indonesian runs 20-30% longer than English, so that box is the one shape
    # on the page guaranteed to be too small.
    b = region.bubble if region.bubble is not None else region.box
    return np.array(
        [[b.x1, b.y1], [b.x2, b.y1], [b.x2, b.y2], [b.x1, b.y2]], dtype=np.int32
    )


def _scaled(
    polygon: np.ndarray, anchor: tuple[float, float], scale: float,
    bounds: tuple[int, int] | None,
) -> np.ndarray:
    """`polygon` blown up about `anchor`, kept on the page.

    Both axes by the same factor, so the shape keeps its proportions. A block
    of free lettering is a column or a banner because the letterer made it
    one, and stretching only the axis that happens to be short would re-wrap
    a four-line stack into two long lines -- the right size in the wrong
    shape, which is not what the page looked like.
    """
    x, y = anchor
    grown = np.empty_like(polygon)
    grown[:, 0] = np.round(x + (polygon[:, 0] - x) * scale)
    grown[:, 1] = np.round(y + (polygon[:, 1] - y) * scale)
    if bounds is not None:
        height, width = bounds
        np.clip(grown[:, 0], 0, width - 1, out=grown[:, 0])
        np.clip(grown[:, 1], 0, height - 1, out=grown[:, 1])
    return grown


def _drawable_area(
    region: Region, padding_ratio: float, blocked: list[Box] | None = None,
    scale: float = 1.0, bounds: tuple[int, int] | None = None,
) -> tuple[np.ndarray, Box]:
    """A mask of where lettering may go, inset from the balloon edge.

    `blocked` is room another region's lettering has already been placed in.
    It is cut out rather than merely avoided, so the fitting that follows
    measures the space that is actually free -- a block wrapped against room
    it does not have would be refused at the end for a width it was told it
    could use.
    """
    # Only the recovered balloon, not unioned with the region's own box.
    # Adding that rectangle looks like a free safety net -- the space the
    # English occupied must surely be usable -- but it was tried and rescued
    # none of the regions that fail to fit, while pushing a rectangle out past
    # the balloon edge on every partitioned shape.
    polygon = _region_polygon(region)
    if scale != 1.0:
        polygon = _scaled(polygon, region.box.center, scale, bounds)
    x, y, w, h = cv2.boundingRect(polygon)
    box = Box(x, y, x + w, y + h)
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [polygon - np.array([x, y])], 1)

    # Capped, because the ratio is calibrated for a balloon-sized shape and a
    # tall one would otherwise lose a fifth of its width to the inset.
    inset = int(round(min(padding_ratio * min(w, h), 12)))
    if inset > 0:
        kernel = np.ones((2 * inset + 1, 2 * inset + 1), np.uint8)
        eroded = cv2.erode(mask, kernel)
        if eroded.any():
            mask = eroded

    for taken in blocked or ():
        x1, y1 = max(0, taken.x1 - x), max(0, taken.y1 - y)
        x2, y2 = min(w, taken.x2 - x), min(h, taken.y2 - y)
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 0

    # Keep one piece. Insetting a spiky balloon can pinch it apart at its
    # narrow points, and lettering must not be split across two islands.
    # The piece holding this region's own lettering, not simply the biggest:
    # once a neighbour's block has been cut out of a shared balloon, the
    # bigger of the two remaining pieces is as often the neighbour's side.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count > 2:
        centre_x, centre_y = region.box.center
        row = int(np.clip(round(centre_y) - y, 0, h - 1))
        column = int(np.clip(round(centre_x) - x, 0, w - 1))
        keep = int(labels[row, column])
        if keep == 0:
            # The centre landed on room already taken, or on a pinch. The
            # piece covering most of this region's own lettering is still the
            # right one; the biggest piece is only right by coincidence.
            inside = labels[
                int(np.clip(region.box.y1 - y, 0, h)) : int(np.clip(region.box.y2 - y, 0, h)),
                int(np.clip(region.box.x1 - x, 0, w)) : int(np.clip(region.box.x2 - x, 0, w)),
            ]
            seen, sizes = np.unique(inside[inside > 0], return_counts=True)
            keep = (
                int(seen[sizes.argmax()])
                if seen.size
                else 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            )
        mask = (labels == keep).astype(np.uint8)
    return mask, box


class _Area:
    """Where the balloon is, row by row.

    A row is kept as the *runs* of free pixels across it rather than as its
    first and last, because the two stop being the same thing as soon as
    something is cut out of the middle -- a neighbour's block, once that has
    been placed. Read as first-to-last, such a row reports the room either
    side of the hole plus the hole itself, and a line measured against it is
    drawn straight through the neighbour's lettering. That is the fault this
    is here to make impossible; on an uncut balloon there is one run per row
    and the two readings are identical.
    """

    def __init__(
        self, mask: np.ndarray, box: Box, anchor_x: float,
        blocked: list[Box] | None = None,
    ) -> None:
        self._box = box
        self._anchor = anchor_x
        self._blocked = list(blocked or ())
        rows: list[list[tuple[int, int]]] = []
        for row in mask:
            edges = np.flatnonzero(np.diff(np.r_[0, (row > 0).astype(np.int8), 0]))
            rows.append(
                [
                    (box.x1 + int(start), box.x1 + int(end))
                    for start, end in zip(edges[::2], edges[1::2])
                ]
            )
        self._rows = rows

    @property
    def box(self) -> Box:
        return self._box

    @property
    def span(self) -> tuple[float, float] | None:
        """First and last page rows that can hold lettering."""
        usable = [i for i, row in enumerate(self._rows) if row]
        if not usable:
            return None
        return (self._box.y1 + usable[0], self._box.y1 + usable[-1] + 1)

    @property
    def widest(self) -> float:
        """The most room any single run has, for estimating a line count."""
        return max(
            (float(right - left) for row in self._rows for left, right in row),
            default=0.0,
        )

    def line_at(self, y: float, ink: float = 0.0) -> tuple[float, float]:
        """(centre, width) available to a line centred on `y`, `ink` tall.

        The centre is the balloon's own centre at that row rather than a fixed
        one for the block. A single centre has to be inside every row the
        block touches, and on a slanted or wedge-shaped balloon it is not:
        the row then reports zero usable width and the whole layout is thrown
        out, however much room the balloon actually has.

        Where a row holds several runs, the one over the lettering being
        replaced is the block's own -- so the block stays in the column the
        artist put it in instead of stepping across the hole from row to row.

        The balloon is read at that one row, which is what the sizes on this
        page were tuned against. Room already given to another region is not:
        that is measured over the whole height of the line's glyphs, because a
        line whose middle clears a neighbour by two pixels still has an
        ascender and a descender that do not, and "nearly not overlapping" is
        not a thing a reader can see the difference of.
        """
        index = int(round(y)) - self._box.y1
        if index < 0 or index >= len(self._rows):
            return (0.0, 0.0)
        runs = self._rows[index]
        if not runs:
            return (0.0, 0.0)
        if len(runs) > 1:
            # Nearest to the anchor, which is zero for the run containing it.
            runs = [
                min(
                    runs,
                    key=lambda run: max(
                        run[0] - self._anchor, self._anchor - run[1], 0.0
                    ),
                )
            ]
        left, right = float(runs[0][0]), float(runs[0][1])
        if self._blocked and ink > 0:
            top, bottom = y - ink / 2.0, y + ink / 2.0
            for taken in self._blocked:
                if taken.y2 <= top or taken.y1 >= bottom:
                    continue
                left, right = self._clear_of(left, right, taken)
                if right <= left:
                    return (0.0, 0.0)
        return ((left + right) / 2.0, right - left)

    def _clear_of(self, left: float, right: float, taken: Box) -> tuple[float, float]:
        """What is left of a run once `taken` is out of it."""
        if taken.x2 <= left or taken.x1 >= right:
            return (left, right)
        if taken.x1 <= left and taken.x2 >= right:
            return (left, left)
        if taken.x1 <= left:
            return (float(taken.x2), right)
        if taken.x2 >= right:
            return (left, float(taken.x1))
        # It falls inside the run and cuts it in two. The side the block's own
        # lettering is on is the side it keeps; with the anchor in the gap,
        # the wider side, which is where it has the better chance of fitting.
        if self._anchor <= taken.x1:
            return (left, float(taken.x1))
        if self._anchor >= taken.x2:
            return (float(taken.x2), right)
        return (
            (left, float(taken.x1))
            if taken.x1 - left >= right - taken.x2
            else (float(taken.x2), right)
        )


def _breakable(word: str) -> bool:
    """Whether a word may be split across lines at all.

    Short words never are -- breaking "NONA" into "NO-" and "NA" costs a line
    and reads worse than the smaller size that would have avoided it. Nor is
    anything without letters: a trailing "--" is the letterer's dash, and
    hyphenating it produces "TERUS-" above a lone "-".
    """
    return sum(1 for c in word if c.isalpha()) >= 6


def _longest_prefix(word: str, font: Face, available: float) -> int:
    """How much of a word fits, leaving room for the hyphen."""
    for cut in range(len(word) - 1, 0, -1):
        if font.measureText(word[:cut] + "-") <= available:
            return cut
    return 0


def _split_tokens(text: str) -> list[str]:
    """Words to wrap on, splitting where a hyphen already is.

    "REIRIN-SAMA!!" is one token to `str.split`, so a balloon too narrow for
    it whole was set small enough to fit it on one line -- where the original
    lettering simply broke it at the hyphen it already contains. Breaking
    there costs nothing and adds no punctuation.
    """
    tokens: list[str] = []
    for word in text.split():
        while True:
            cut = word.find("-", 1)
            if cut < 0 or cut >= len(word) - 1 or word[cut + 1] == "-":
                break
            tokens.append(word[: cut + 1])
            word = word[cut + 1 :]
        tokens.append(word)
    return tokens


def _ink_height(font: Face) -> float:
    """Ascender to descender, the band a line's glyphs actually cover."""
    metrics = font.getMetrics()
    return -metrics.fAscent + metrics.fDescent


def _wrap(
    words: list[str], font: skia.Font, area: _Area,
    top: float, line_height: float, hyphenate: bool,
) -> tuple[list[str], list[float]] | None:
    """Greedy wrap, each line measured and centred where it will be drawn."""
    lines: list[str] = []
    centres: list[float] = []
    current = ""
    centre = 0.0
    index = 0
    ink = _ink_height(font)
    while index < len(words):
        y = top + line_height * (len(lines) + 0.5)
        centre, available = area.line_at(y, ink)
        if available <= 1.0:
            return None

        word = words[index]
        joiner = "" if current.endswith("-") else " "
        candidate = f"{current}{joiner}{word}" if current else word
        if font.measureText(candidate) <= available:
            current = candidate
            index += 1
            continue

        if current:
            lines.append(current)
            centres.append(centre)
            current = ""
            continue

        if not hyphenate or not _breakable(word):
            return None
        cut = _longest_prefix(word, font, available)
        if cut < 2 or len(word) - cut < 2:
            return None
        lines.append(word[:cut] + "-")
        centres.append(centre)
        words[index] = word[cut:]
    if current:
        lines.append(current)
        centres.append(centre)
    return (lines, centres) if lines else None


# Below this, lettering stops being small and starts being unreadable at any
# print size. Reached only when the balloon holds nothing larger.
_LAST_RESORT_SIZE = 5.0

# How much more room free lettering may take, tried in order and stopped at
# the first step that holds the source's own size. A target 20-30% longer than
# its source needs about 1.3x the area, which is 1.15 on a side; the rest of
# the ladder is for the short captions, where one long word can want half as
# much room again as the whole English line took. It stops at twice, because
# past that the block has stopped being where the letterer put it.
_MORE_ROOM = (1.2, 1.45, 1.7, 2.0)

# Within a couple of percent of the source's size is the same size. Demanding
# it exactly would spend every step of the ladder chasing a rounding error.
_CLOSE_ENOUGH = 0.98


def _covered(boxes: list[Box], other: Box) -> int:
    """How many pixels of `other` a list of boxes covers.

    Summed per box rather than unioned, which is exact for the lines of one
    block -- they are stacked, not overlapping -- and only ever over-counts,
    so a clash cannot be missed by it.
    """
    total = 0
    for box in boxes:
        width = min(box.x2, other.x2) - max(box.x1, other.x1)
        height = min(box.y2, other.y2) - max(box.y1, other.y1)
        total += max(0, width) * max(0, height)
    return total


def _worst_clash(
    drawn: dict[str, list[Box]], regions: dict[str, Region]
) -> tuple[str, str] | None:
    """The (gives way, stays) pair whose lettering collides most, or None.

    Which of the two moves is settled by asking who is in whose way, not by
    who is larger or came first: the block reaching further into the *source
    lettering* of its neighbour is the one in the wrong place, because that
    lettering is what the neighbour is replacing and where it therefore has to
    be drawn. Where neither is reaching into the other -- two blocks meeting
    in empty balloon between them -- the smaller gives way, having the better
    chance of finding room somewhere else.

    A block someone has placed by hand is asked to move only if the other one
    has been too. Someone looking at the page and dragging a block there knows
    something this does not, and the automatic block beside it is the one with
    no opinion about where it sits.
    """
    worst: tuple[int, str, str] | None = None
    ids = sorted(drawn)
    for index, one in enumerate(ids):
        for other in ids[index + 1 :]:
            clash = sum(_covered(drawn[other], box) for box in drawn[one])
            if clash <= 0:
                continue
            set_by_hand = regions[one].is_hand_set, regions[other].is_hand_set
            if set_by_hand[0] != set_by_hand[1]:
                mover, stays = (other, one) if set_by_hand[0] else (one, other)
            else:
                into_other = _covered(drawn[one], regions[other].box)
                into_one = _covered(drawn[other], regions[one].box)
                if into_other == into_one:
                    mover, stays = (
                        (one, other)
                        if regions[one].box.area <= regions[other].box.area
                        else (other, one)
                    )
                else:
                    mover, stays = (
                        (one, other) if into_other > into_one else (other, one)
                    )
            if worst is None or clash > worst[0]:
                worst = (clash, mover, stays)
    return (worst[1], worst[2]) if worst is not None else None


def _says_the_same(region: Region) -> bool:
    """Whether a translation differs from its source only in how it is typed.

    Case and spacing do not count, because neither is a translation: "ABYDOS"
    to "Abydos" is the same word set differently, and the artist already set
    it better than this can.
    """
    def bare(text: str) -> str:
        return "".join(text.split()).casefold()

    source = region.text_en_fixed or region.text_en
    return bool(source) and bare(source) == bare(region.text_id)


def _largest_that_fits(
    attempt: "Callable[[float], Layout | None]", floor: float, upper: float
) -> "Layout | None":
    """The biggest size in [floor, upper] that fits, by bisection.

    Stepping down from the top in fixed decrements looks simpler and has a
    fault that shows the moment sizes are set by hand: where the search starts
    decides which sizes it ever tries. Two starting points a hair apart walk
    different grids, so asking a balloon for *larger* type could hand back
    something a point smaller than before -- 37.3 became 36.5 on a page here,
    which reads as the button doing the opposite of what it says.

    Bisection has no grid. It also costs about four attempts where the walk
    cost a dozen, and each attempt is a full wrap of the text.
    """
    settled = attempt(upper)
    if settled is not None:
        return settled
    best: Layout | None = None
    low, high = floor, upper
    for _ in range(14):
        if high - low < 0.25:
            break
        middle = (low + high) / 2.0
        found = attempt(middle)
        if found is not None:
            best, low = found, middle
        else:
            high = middle
    return best


class Typesetter:
    """Draws translated text into cleaned pages."""

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        self._source_is_cjk = pair_for(cfg.translate.pair).source.is_cjk
        self._typeface = skia.Typeface.MakeFromFile(str(cfg.typeset.font))
        if self._typeface is None:
            raise RuntimeError(f"could not load font {cfg.typeset.font}")
        # Optional, and the regular face stands in where it is missing: a
        # bolded line set in the regular face is a line that did not get
        # emphasis, where refusing to load leaves the whole page unlettered.
        self._bold_typeface = (
            skia.Typeface.MakeFromFile(str(cfg.typeset.font_bold))
            or self._typeface
        )
        metrics = skia.Font(self._typeface, 100.0).getMetrics()
        # Ascender-to-descender extent per point of size, measured from the
        # face itself rather than assumed, since it is what converts the
        # source lettering's line pitch into a font size.
        self._span_per_size = (-metrics.fAscent + metrics.fDescent) / 100.0

    def _font(self, size: float, bold: bool = False) -> Face:
        """The lettering face at one size, able to draw what it lacks itself.

        A `Face` rather than a `skia.Font`: the comic face has no glyph for
        the decorative marks Japanese lettering uses, and Skia draws a missing
        glyph as nothing at all without saying so. See `pipeline.fallback`.
        """
        return Face(self._bold_typeface if bold else self._typeface, size)

    def _source_size(self, region: Region) -> float:
        """The size that matches the lettering being replaced.

        Taken from the line *pitch* -- the region's height over its line count
        -- rather than from the detector's line-bar heights. Those bars mark
        the x-height core of a line and measured 10.5px on a balloon whose
        lettering was plainly around 23px, because a faint extra bar dragged
        their median down. Height over line count does not have that failure
        mode, and agreed with the drawn lettering on every region checked.
        """
        spacing = self._cfg.typeset.line_spacing
        lines = len(region.lines)
        if lines <= 0:
            # Nothing to divide the box by. Treating it as a single line would
            # read a four-line block's full height as one glyph and ask for
            # display type, so the detector's bar height is used instead --
            # less accurate, but wrong in a bounded way.
            if region.source_line_height > 0:
                return region.source_line_height / self._span_per_size
            return region.box.height / self._span_per_size
        if lines == 1:
            # One line, so the box is the glyph extent with no leading in it.
            return region.box.height / self._span_per_size
        pitch = region.box.height / lines
        return pitch / (self._span_per_size * spacing)

    def layout(
        self, region: Region, blocked: list[Box] | None = None,
        bounds: tuple[int, int] | None = None,
    ) -> Layout | None:
        """Fit a region's translation, or None if it cannot be fitted.

        `blocked` is room already given to another region's lettering, and is
        taken out of the shape before anything is measured against it.
        `bounds` is the page's (height, width), which only free lettering
        needs: it is the one edge such a region has.
        """
        cfg = self._cfg.typeset
        text = region.text_id.strip()
        if not text:
            return None
        if cfg.uppercase:
            text = text.upper()

        target = self._source_size(region) * max(0.2, region.size_scale)
        found = self._fit(region, text, blocked, 1.0, bounds)
        if not region.is_unbounded or self._source_is_cjk:
            return found

        # Lettering laid straight onto artwork has no balloon to overflow, and
        # the box the source occupied is not an edge -- it is only where the
        # words happened to land. Read as one, a translation 20-30% longer
        # than its source has nowhere to go but down in size: "HUH? I'M DOING
        # WHAT?" was drawn at 24px and came back at 13, small enough that the
        # page had to be zoomed in to read one of its own captions.
        #
        # So an unbounded region is given more room instead, in the smallest
        # step that lets it keep the size the letterer set. Growing rather
        # than shrinking is what a letterer does with a caption over art, and
        # the room is not free -- it is artwork -- so it is taken a step at a
        # time and only while the size is still short of the source's.
        for scale in _MORE_ROOM:
            if found is not None and found.size >= target * _CLOSE_ENOUGH:
                break
            wider = self._fit(region, text, blocked, scale, bounds)
            if wider is not None and (found is None or wider.size > found.size):
                found = wider
        return found

    def _fit(
        self, region: Region, text: str, blocked: list[Box] | None,
        scale: float, bounds: tuple[int, int] | None,
    ) -> Layout | None:
        """Fit `text` into this region's shape, blown up by `scale`."""
        cfg = self._cfg.typeset
        mask, box = _drawable_area(region, cfg.padding_ratio, blocked, scale, bounds)
        if not mask.any():
            return None
        anchor_x, anchor_y = region.box.center
        area = _Area(mask, box, anchor_x + region.offset_x, blocked)
        anchor_y += region.offset_y

        # The hand-set scale moves the whole search window rather than the
        # answer, so asking for larger type asks the fitter to *try* larger
        # type. If the balloon will not hold it the fitter still shrinks back
        # to what fits, which is the point: the one thing an enlargement must
        # not do is put lettering outside the shape it belongs in.
        target = self._source_size(region) * max(0.2, region.size_scale)
        # `max_font_size` guards the automatic path against a runaway on a
        # region whose measured line height is nonsense. It is a global number
        # and it is not the right authority over someone looking at the page
        # and asking for bigger type, so asking lifts it -- on this chapter
        # three of nine balloons were already sitting on the cap, and pressing
        # + on any of them did nothing at all. The balloon still binds: that
        # is the constraint that is actually about this region.
        ceiling = cfg.max_font_size * max(1.0, region.size_scale)
        if self._source_is_cjk:
            # `max_growth_over_source` keeps a page's voice by holding the
            # replacement near the size of the lettering it replaces. That
            # comparison needs the two to be set the same way round, and here
            # they are not: Japanese and Chinese balloons are usually set in
            # vertical columns, so the line head marks each *character* as a
            # line -- ten of them in one 196px balloon -- and the pitch it
            # implies is a third of the type actually drawn. Every balloon on
            # a page came out at about 14pt where the source was 40.
            #
            # There is no line pitch to match, so the balloon decides instead:
            # aim for the largest type allowed and let the fitter shrink to
            # what the shape holds, which is what a letterer does by eye.
            upper = ceiling
        elif region.is_unbounded and len(region.lines) > 1:
            # No cap but the source's own size. `max_font_size` is a guard
            # against a nonsense line-height measurement, and free lettering
            # is where display type is normal: the chapter title measured 56px
            # of drawn glyph and "I'M... ALIVE." 52px, and both were held to
            # 44 and set smaller than the lettering they replaced.
            #
            # Only where the measurement is a *pitch*, though -- a box divided
            # by the lines in it. With one line there is nothing to divide by
            # and `_source_size` is reading the box itself, which is right
            # only if the box really holds one line. The line head fails on
            # exactly the lettering this branch is about: an eight-line column
            # of ordinary dialogue came back as one line 443px tall, and
            # uncapped it was set at 90px sprawling across three panels.
            # Below the cap that reading is merely optimistic; above it, it is
            # the largest thing on the page. Of 71 free regions measuring over
            # the cap across three chapters, the 57 with one line were junk,
            # sound effects, or that same failure.
            upper = target * cfg.max_growth_over_source
        else:
            upper = min(ceiling, target * cfg.max_growth_over_source)
        # The floor follows the source too. A balloon lettered small was
        # always going to need small type, and refusing to go below a global
        # minimum leaves it in the source language instead -- which is worse
        # than small.
        floor = min(cfg.min_font_size, max(_LAST_RESORT_SIZE, target * 0.55))
        upper = max(upper, floor)

        # `min_font_size` is an absolute number of pixels, and pages are not
        # a fixed size: 8px is a comfortable floor on a 2560px scan and a
        # third of a name plate on a 770px one. So it is a preference, not a
        # rule -- if nothing fits above it, the search runs again down to the
        # readability limit rather than giving up and leaving the source
        # lettering on the page. Two regions of the reported page failed here,
        # both of which fit perfectly well at 6px.
        found = self._search(region, area, anchor_y, text, floor, upper)
        if found is None and floor > _LAST_RESORT_SIZE:
            found = self._search(
                region, area, anchor_y, text, _LAST_RESORT_SIZE, floor
            )
        return found

    def _search(
        self, region: Region, area: "_Area", anchor_y: float, text: str,
        floor: float, upper: float,
    ) -> Layout | None:
        """The largest size in [floor, upper] the balloon will hold."""
        cfg = self._cfg.typeset
        tokens = _split_tokens(text)
        # Shrink before breaking words. Hyphenation can make almost anything
        # "fit" at almost any size, so offering it during the size search
        # stops the search at the first, largest size and returns a balloon
        # full of broken words.
        for hyphenate in (False, True):
            if hyphenate and not cfg.hyphenate:
                break

            def attempt(size: float, hyphenate: bool = hyphenate) -> Layout | None:
                return self._try_size(
                    tokens,
                    self._font(size, region.bold),
                    area,
                    anchor_y,
                    self._span_per_size * size * cfg.line_spacing,
                    size,
                    hyphenate,
                    region.offset_x,
                )

            found = _largest_that_fits(attempt, floor, upper)
            if found is not None:
                return found
        return None

    def _try_size(
        self, tokens: list[str], font: Face, area: _Area, anchor_y: float,
        line_height: float, size: float, hyphenate: bool, shift: float = 0.0,
    ) -> Layout | None:
        """Wrap at one size, settling the block's position as it goes.

        Wrapping and placing depend on each other -- where a block sits
        decides how wide each of its lines may be, and how many lines it has
        decides where it sits -- so this walks between them. Two things about
        that walk are worth stating, because getting either wrong rejected
        sizes the balloon could hold and left the lettering far smaller than
        the space around it:

        * **Every candidate is verified at the position it was wrapped for.**
          Lines wrapped near the middle of an ellipse, where it is widest, and
          then checked higher up, where it is not, fail a test they should
          never have been given. One balloon measured its lines against 82px
          of room after wrapping them to 87px.

        * **The walk does not have to converge.** In a balloon whose width
          changes quickly with height the line count oscillates -- 5, 4, 5, 4
          -- and neither is wrong; they are two different ways to set the same
          text. Taking the first that verifies is enough, and demanding a
          fixed point threw both away.
        """
        span = area.span
        if span is None:
            return None
        first, last = span

        def settle(n: int) -> float:
            """Where an n-line block sits: on the anchor, slid to stay inside.

            Anchoring on the original lettering keeps the translation where
            the artist put it, but a block taller than the source it replaces
            can reach past the balloon. Sliding it back is right where
            refusing it is not -- the room exists, it is simply not centred on
            the same point.
            """
            top = anchor_y - (n * line_height) / 2.0
            if top + n * line_height > last:
                top = last - n * line_height
            return max(top, first)

        # The fewest lines this text could possibly take: everything packed
        # at the widest row the shape has. A real wrap will need at least
        # this many and usually a few more, so it is where the search starts.
        #
        # It is computed rather than wrapped for, because there is nowhere
        # good to wrap a trial block. Starting at the anchor gives a
        # multi-line block only the lower half of the shape and the wrap runs
        # out of rows; starting at the top of an ellipse gives it a row eight
        # pixels wide and nothing fits at all. Both refused sizes the balloon
        # holds comfortably -- one balloon shipped 6pt text into a shape with
        # room for 9pt.
        widest = area.widest
        if widest <= 1.0:
            return None
        count = max(1, ceil(font.measureText(" ".join(tokens)) / widest))

        # Each candidate line count places the block, wraps it there, and is
        # judged there. Counts are tried from the estimate upward: a block
        # that will not fit usually needs one or two more lines than the
        # generous guess, since settling it centres it where the balloon is
        # narrower.
        #
        # The walk between wrapping and placing does not have to reach a fixed
        # point. In a balloon whose width changes quickly with height the
        # count oscillates -- 5, 4, 5, 4 -- and neither is wrong; they are two
        # ways to set the same text. Demanding convergence threw both away.
        for extra in range(10):
            top = settle(count + extra)
            wrapped = _wrap(list(tokens), font, area, top, line_height, hyphenate)
            if wrapped is None:
                continue
            lines, _ = wrapped
            settled = self._verify(lines, font, area, top, line_height, size, shift)
            if settled is not None:
                return settled
        return None

    def _verify(
        self, lines: list[str], font: Face, area: _Area, top: float,
        line_height: float, size: float, shift: float,
    ) -> Layout | None:
        """Whether `lines` really fit, laid out from `top`."""
        span = area.span
        if span is None or top + len(lines) * line_height > span[1] + 0.5:
            return None

        centres: list[float] = []
        ink = _ink_height(font)
        for index, line in enumerate(lines):
            y = top + line_height * (index + 0.5)
            centre, available = area.line_at(y, ink)
            width = font.measureText(line)
            if available <= 1.0 or width > available + 0.5:
                return None
            # Sideways movement is clamped to the room the balloon has at
            # this row rather than refused, so dragging a block towards an
            # edge stops against it instead of the whole layout failing and
            # the line vanishing mid-drag.
            if shift:
                slack = max(0.0, (available - width) / 2.0)
                centre += max(-slack, min(slack, shift))
            centres.append(centre)

        metrics = font.getMetrics()
        ascent = -metrics.fAscent
        leading = line_height - (ascent + metrics.fDescent)
        return Layout(lines, centres, size, line_height, top + ascent + leading / 2.0)

    def plan(
        self, regions: list[Region], bounds: tuple[int, int] | None = None
    ) -> dict[str, Layout]:
        """Fit every translated region, before anything is erased.

        Planning is separated from drawing so the pipeline can know which
        regions it can letter *before* it rubs the originals out. A region
        whose translation will not fit is marked here and skipped by erasure,
        so it keeps its English rather than becoming an empty balloon -- which
        is what was happening: on one reported page six labels were erased and
        left blank because the fit was only discovered afterwards.
        """
        layouts: dict[str, Layout] = {}
        for region in regions:
            if region.status is not RegionStatus.TRANSLATED:
                continue
            if _says_the_same(region):
                # Nothing was gained by translating it, so nothing is gained
                # by drawing it: erasing the artist's lettering and setting
                # the same word back in the comic face can only make the page
                # worse. The ID card on a character's uniform reading "ABYDOS"
                # came back as "Abydos" and was lettered over a drawn prop.
                region.status = RegionStatus.SKIPPED_SAME
                region.note = "the translation is the source; left as drawn"
                continue
            found = self.layout(region, bounds=bounds)
            if found is None:
                region.fail(
                    RegionStatus.FAILED_FIT,
                    f"{region.text_id!r} does not fit its balloon",
                )
                continue
            layouts[region.id] = found
        self._settle_clashes(regions, layouts, bounds)
        return layouts

    def lines_drawn(self, region: Region, layout: Layout) -> list[Box]:
        """The page pixels a settled block's lines will actually cover."""
        font = self._font(layout.size, region.bold)
        metrics = font.getMetrics()
        boxes: list[Box] = []
        for index, (line, centre) in enumerate(zip(layout.lines, layout.centres)):
            baseline = layout.first_baseline + index * layout.line_height
            half = font.measureText(line) / 2.0
            boxes.append(
                Box(
                    int(centre - half), int(baseline + metrics.fAscent),
                    int(ceil(centre + half)), int(ceil(baseline + metrics.fDescent)),
                )
            )
        return boxes

    def _settle_clashes(
        self, regions: list[Region], layouts: dict[str, Layout],
        bounds: tuple[int, int] | None = None,
    ) -> None:
        """Make sure no two blocks of lettering are drawn over each other.

        Each region is fitted on its own, against a shape that is supposed to
        be its alone. Four different things arrive at that shape and only one
        of them -- the flood fill partitioned between the regions in a shared
        balloon -- knows about the neighbours; `classify._separate_polygons`
        carves the other three apart afterwards, and it carves by which
        region's *lettering* is nearer, which cannot separate two blocks whose
        boxes overlap at all. That leaves a gap, and what comes through it is
        the worst thing this program can put on a page: two translations
        written across each other, both unreadable, where leaving one balloon
        in English would have cost the reader one balloon.

        So the guarantee is made here instead, where every path has already
        converged on a settled block and the question is no longer which shape
        a region may use but which pixels it is about to cover. Whoever is
        reaching further into the other's lettering gives way and is fitted
        again in what is left; if nothing is left, it keeps its English, which
        is an omission rather than damage.
        """
        drawn = {
            region.id: self.lines_drawn(region, layouts[region.id])
            for region in regions
            if region.id in layouts
        }
        by_id = {region.id: region for region in regions}
        blocked: dict[str, list[Box]] = {}

        def give_up(mover: str, note: str) -> None:
            del layouts[mover], drawn[mover]
            by_id[mover].fail(RegionStatus.FAILED_FIT, note)

        # Every pass either places one block clear of everything it has been
        # told to avoid or gives up on it, and a block never has to avoid less
        # than it already did -- so it walks in one direction. It does not
        # walk straight, though: settling one pair can put a block into a
        # third, so the bound is the page's own size rather than a small
        # fixed number.
        for _ in range(4 * len(drawn) + 4):
            clash = _worst_clash(drawn, by_id)
            if clash is None:
                return
            mover, rival = clash
            blocked.setdefault(mover, []).extend(drawn[rival])
            region = by_id[mover]
            found = self.layout(region, blocked[mover], bounds)
            boxes = self.lines_drawn(region, found) if found is not None else []
            if found is None or any(
                _covered(boxes, taken) for taken in blocked[mover]
            ):
                # Either there was no room, or there was and the fitter could
                # not use it. Both mean the same thing for this region and
                # trying again would mean the same thing again.
                give_up(mover, f"no room left beside {rival}; left as drawn")
                continue
            layouts[mover], drawn[mover] = found, boxes

        # Whatever is still touching after all that gives up its lettering.
        # The promise this makes is the one the page is judged on, and it has
        # to hold on the pages where re-fitting could not settle it either.
        while (last := _worst_clash(drawn, by_id)) is not None:
            give_up(last[0], f"still over {last[1]}'s lettering; left as drawn")

    def draw_page(
        self, clean: np.ndarray, regions: list[Region], layouts: dict[str, Layout]
    ) -> np.ndarray:
        """Draw the planned lettering onto the cleaned page."""
        height, width = clean.shape[:2]
        surface = skia.Surface(width, height)
        canvas = surface.getCanvas()

        rgba = cv2.cvtColor(clean, cv2.COLOR_BGR2RGBA)
        canvas.drawImage(
            skia.Image.fromarray(rgba, colorType=skia.kRGBA_8888_ColorType), 0, 0
        )

        for region in regions:
            layout = layouts.get(region.id)
            if layout is None or region.status is not RegionStatus.TRANSLATED:
                continue
            self._draw(canvas, region, layout)
            region.status = RegionStatus.TYPESET

        # Read back in an explicitly named order. A Surface takes whatever
        # layout the platform prefers -- BGRA on Windows -- so an unqualified
        # toarray() returns BGRA there while returning RGBA elsewhere.
        # Converting the result as if it were always RGBA swaps red and blue,
        # which turned whole reported pages solid blue.
        image = surface.makeImageSnapshot().toarray(
            colorType=skia.ColorType.kBGRA_8888_ColorType
        )
        return np.ascontiguousarray(image[:, :, :3])

    def render_page(self, clean: np.ndarray, regions: list[Region]) -> np.ndarray:
        """Plan and draw in one call, for callers that erase nothing."""
        return self.draw_page(clean, regions, self.plan(regions, clean.shape[:2]))

    def _draw(self, canvas: skia.Canvas, region: Region, layout: Layout) -> None:
        blue, green, red = region.text_color
        fill = skia.Paint(
            AntiAlias=True, Color=skia.ColorSetARGB(255, red, green, blue)
        )

        # Lettering drawn straight onto artwork gets its stroke back. The
        # letterer put one there so dark text stays readable over dark art;
        # reproducing the ink without it leaves the balloon-less captions
        # unreadable, which is exactly what they looked like.
        stroke: skia.Paint | None = None
        if region.outline_color is not None:
            ob, og, orr = region.outline_color
            width = layout.size * self._cfg.typeset.outline_width_ratio
            stroke = skia.Paint(
                AntiAlias=True,
                Color=skia.ColorSetARGB(255, orr, og, ob),
                Style=skia.Paint.kStroke_Style,
                StrokeWidth=width,
                StrokeJoin=skia.Paint.kRound_Join,
                StrokeCap=skia.Paint.kRound_Cap,
            )

        font = self._font(layout.size, region.bold)
        for index, (line, centre) in enumerate(zip(layout.lines, layout.centres)):
            baseline = layout.first_baseline + index * layout.line_height
            x = centre - font.measureText(line) / 2.0
            # Stroke first: Skia centres a stroke on the outline, so drawing
            # the fill afterwards keeps the letterforms their proper weight
            # and the halo entirely outside them.
            if stroke is not None:
                font.draw(canvas, line, x, baseline, stroke)
            font.draw(canvas, line, x, baseline, fill)
