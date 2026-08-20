"""A2 -- sample each region's colours, and recover the space it may use.

Three things are decided here, all of them from pixels rather than from the
network:

* **Ink and ground.** The replacement lettering is drawn in the colour the
  original lettering was drawn in, and erasing has to leave behind the colour
  that surrounded it. Sampling both ways round matters: the chapter title bar
  on the sample page is white-on-black, and lettering it in black would be as
  wrong as leaving it in English.

* **Whether the ground is flat.** A region sitting on a uniform fill can be
  erased by painting that fill back, which is exact. A region sitting on art
  needs inpainting, which is a guess. Choosing per region rather than per page
  keeps the guess off the balloons, which are most of the page.

* **How much room the translation actually has.** The detected box is only as
  big as the English was. Indonesian runs 20-30% longer, so typesetting into
  that box means shrinking the lettering on nearly every balloon. Flood
  filling the balloon interior recovers the room the artist left, and the
  difference is the difference between lettering that matches the page and
  lettering that is visibly too small.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..config import Config
from ..types import Box, Region, RegionKind, RegionStatus


def _median_color(pixels: np.ndarray) -> tuple[int, int, int]:
    if pixels.size == 0:
        return (255, 255, 255)
    med = np.median(pixels.reshape(-1, 3), axis=0)
    return (int(med[0]), int(med[1]), int(med[2]))


def _ink_color(
    pixels: np.ndarray, ground: tuple[int, int, int]
) -> tuple[int, int, int]:
    """The colour the lettering was drawn in, read off its stroke pixels.

    Taken from the tail of the distribution rather than its middle. Comic
    lettering at page resolution is two or three pixels wide, so even a
    correct glyph mask is half anti-aliased edge, and the median of it lands
    between ink and paper -- (112,112,112) mid-grey where the ink is actually
    near-black. The tail to take is the one facing away from the ground, which
    is how the same code reads black-on-white dialogue and the white-on-black
    chapter title without being told which it is looking at.
    """
    if pixels.size == 0:
        return (0, 0, 0)
    flat = pixels.reshape(-1, 3)
    ground_luma = float(np.mean(ground))
    darker = float(np.median(flat.mean(axis=1))) < ground_luma
    percentile = 15.0 if darker else 85.0
    value = np.percentile(flat, percentile, axis=0)
    return (int(value[0]), int(value[1]), int(value[2]))


def _ink_and_ground(
    mask: np.ndarray, inner: int, outer: int
) -> tuple[np.ndarray, np.ndarray]:
    """Stroke cores to sample ink from, and a band of ground outside them.

    Ink is sampled from an *eroded* mask, never a dilated one. Comic lettering
    is thin, so a mask grown even one step is mostly the paper between the
    strokes: sampling that reported (216,216,216) ink on (254,254,254) ground
    for most balloons on the sample page -- near-white letters on white -- and
    would have had the typesetter draw white on white. Erosion is skipped when
    it would eat the strokes outright, which is what happens on the smallest
    lettering.
    """
    kernel = np.ones((3, 3), np.uint8)
    core = cv2.erode(mask, kernel)
    if core.sum() < max(8, mask.sum() * 0.15):
        core = mask
    body = cv2.dilate(mask, np.ones((inner, inner), np.uint8))
    around = cv2.dilate(mask, np.ones((outer, outer), np.uint8))
    return core.astype(bool), (around.astype(bool) & ~body.astype(bool))


def _outline_colour(
    region: Region, near: tuple[int, int, int], min_contrast: float
) -> tuple[int, int, int] | None:
    """The stroke to draw around lettering laid straight onto artwork.

    Whether a block sits on artwork at all has already been decided, by
    measuring how uniform the ring just outside its glyphs is: uniform means a
    balloon, varied means artwork. That leaves only what colour the stroke is,
    and the ring answers that too -- it is the stroke, because the stroke is
    what lies between the glyphs and the page.

    This used to ask a further question, and asked it with the wrong
    instrument: it compared the ring against the *median* of the artwork
    around the block, and drew a stroke only where the two were far apart.
    Over dark art they are, and those captions came out right. Over screentone
    they are not -- a median of mostly-white paper with dots through it reads
    about 220, against a stroke at 245 -- so the test said "no stroke" for
    lettering that plainly had one. Measured over one chapter it missed 23 of
    30 blocks, including every caption on a toned panel, and each of those was
    then drawn in its own dark ink straight onto the tone with nothing between.

    There is no version of that comparison that works, because a median cannot
    see texture. Nor is one needed: a block over artwork needs a stroke
    whatever the original had, since erasure replaces what was behind the
    glyphs with reconstructed artwork and the replacement lettering has to
    stay legible against it.

    Only one thing disqualifies the ring: being too close to the ink to
    separate it from anything. Light lettering on dark art carries a dark
    stroke and dark lettering a light one, and both come back from the ring
    correctly, so this is insurance rather than a case seen in practice.
    """
    if abs(float(np.mean(near)) - float(np.mean(region.text_color))) < min_contrast:
        return None
    return near


def _is_overlay(crop: np.ndarray, glyphs: np.ndarray, colour_fraction: float) -> bool:
    """Whether a region is a site watermark rather than part of the artwork.

    Scan watermarks are the one thing on a *greyscale* page that is reliably
    in colour, which is the whole basis of this test -- so it must not run on
    a page that is in colour to begin with.

    How much of the page is coloured decides that, not how saturated the
    median pixel is. The median was measuring the paper: a full-colour comic
    is still mostly white background, so it came out at 10 and the page was
    judged greyscale. Every piece of coloured lettering on it was then thrown
    away as a watermark -- on the reported page the blue title and all four
    name plates, none of which was translated because none survived to be.

    The fraction separates the two cleanly. Measured over both: greyscale
    scans run 0.000 to 0.004, and colour pages 0.33 and up.
    """
    if colour_fraction > _COLOUR_PAGE:  # a colour page: the test says nothing
        return False
    if not glyphs.any():
        return False
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    return float(hsv[:, :, 1][glyphs].mean()) > 60.0


# How much of a page must be saturated before it counts as a colour page.
# Two orders of magnitude clear of both sides of the measurement above.
_COLOUR_PAGE = 0.04
_COLOURED_PIXEL = 40


def _page_colour_fraction(image: np.ndarray) -> float:
    """How much of a page carries real colour rather than ink and paper."""
    saturation = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 1]
    return float((saturation > _COLOURED_PIXEL).mean())


def _interior_mask(
    image: np.ndarray, region: Region, bg: tuple[int, int, int], tolerance: int
) -> tuple[np.ndarray, Box] | None:
    """The balloon interior around a region, as a mask in a local window.

    Built by thresholding on distance from the ground colour rather than by
    flooding the photograph directly, so that an *opening* can be applied
    before the component is taken. That opening is what makes this usable: a
    balloon outline is a drawn line, and drawn lines have thin places, so a
    plain flood leaks through one into the panel and the whole recovery has to
    be thrown away. Opening severs any bridge narrower than the kernel while
    leaving the interior itself untouched. On the sample page it takes
    recovery from 5 balloons out of 15 to nearly all of them.
    """
    h, w = image.shape[:2]
    box = region.box
    if region.bubble is not None:
        # The balloon's own extent, which is not a guess. The margin matters:
        # a window laid exactly on the balloon has the interior touching all
        # four sides by construction, and the leak test below would then throw
        # away every balloon on the page.
        margin = max(8, int(0.06 * max(region.bubble.width, region.bubble.height)))
        window = region.bubble.padded(margin, within=(h, w))
    else:
        # Generous, because the window has to hold the *balloon*, and a balloon
        # is not shaped like the text in it. "SILENCE." on the sample page is a
        # 115x21 box inside an oval nearly 300px tall: a window scaled to the
        # box clipped the oval, the interior then reached the window edge, and
        # a perfectly good balloon was thrown out as a leak.
        window = box.padded(int(max(box.width, box.height) * 2.0) + 24, within=(h, w))
    crop = window.crop(image)
    if crop.size == 0:
        return None

    ground = np.array(bg, dtype=np.int16)
    distance = np.abs(crop.astype(np.int16) - ground).max(axis=2)
    candidate = (distance <= tolerance).astype(np.uint8)

    # The lettering is inside the balloon even though it is not the ground
    # colour, so add it back before the component is taken.
    off_x, off_y = box.x1 - window.x1, box.y1 - window.y1
    if region.mask is not None:
        # Clipped to the window, which is not always big enough to hold the
        # whole mask: both the window and the region's box are clamped to the
        # page, so a region against an edge has a box a pixel or two wider
        # than the window around it, and the unclipped assignment raises
        # rather than drawing -- taking the whole chapter run down with it.
        mh, mw = region.mask.shape
        y1, x1 = max(0, off_y), max(0, off_x)
        y2 = min(candidate.shape[0], off_y + mh)
        x2 = min(candidate.shape[1], off_x + mw)
        if y2 > y1 and x2 > x1:
            candidate[y1:y2, x1:x2] |= region.mask.astype(np.uint8)[
                y1 - off_y : y2 - off_y, x1 - off_x : x2 - off_x
            ]

    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    count, labels = cv2.connectedComponents(candidate, connectivity=4)
    if count < 2:
        return None
    seed_y = min(candidate.shape[0] - 1, off_y + box.height // 2)
    seed_x = min(candidate.shape[1] - 1, off_x + box.width // 2)
    label = labels[seed_y, seed_x]
    if label == 0:
        # The centre landed on a glyph stroke that the opening removed; take
        # the component covering most of the region instead.
        patch = labels[off_y : off_y + box.height, off_x : off_x + box.width]
        values, counts = np.unique(patch[patch > 0], return_counts=True)
        if values.size == 0:
            return None
        label = int(values[counts.argmax()])

    interior = _fill_holes((labels == label).astype(np.uint8))
    # Reaching a window edge means the fill escaped the balloon -- unless that
    # edge is the page edge, where the window was clamped and there was no
    # more page to escape into.
    if (
        (window.y1 > 0 and interior[0, :].any())
        or (window.y2 < h and interior[-1, :].any())
        or (window.x1 > 0 and interior[:, 0].any())
        or (window.x2 < w and interior[:, -1].any())
    ):
        return None
    return interior, window


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """Solidify a mask, so lettering inside it does not read as absence.

    Only the region's own glyphs are added back when the interior is built, so
    every *other* region's lettering punches holes in it. Left there, those
    holes drag the measured overlap between two balloons down to whatever
    their text density happens to be: on the sample page one direction of the
    same pair measured 0.283 and the other 0.356, which puts any threshold in
    between inside the spread of real values rather than outside it. Filling
    makes the measurement about geometry instead of about text density.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    solid = np.zeros_like(mask)
    cv2.drawContours(solid, contours, -1, 1, thickness=cv2.FILLED)
    return solid


def _distance_to(box: Box, window: Box, shape: tuple[int, int]) -> np.ndarray:
    """Distance from every pixel of a window to a box, in window pixels."""
    h, w = shape
    x1, y1 = max(0, box.x1 - window.x1), max(0, box.y1 - window.y1)
    x2, y2 = min(w, box.x2 - window.x1), min(h, box.y2 - window.y1)
    if x2 <= x1 or y2 <= y1:
        return np.full(shape, 1e6, np.float32)
    blocker = np.ones(shape, np.uint8)
    blocker[y1:y2, x1:x2] = 0
    return cv2.distanceTransform(blocker, cv2.DIST_L2, 3)


def _split_between(
    interior: np.ndarray, window: Box, own: Box, others: list[Box]
) -> np.ndarray:
    """Keep only the part of a shared interior that belongs to `own`.

    Balloons drawn overlapping share one white region, so flood filling from
    either one returns the same shape -- on the sample page three separate
    pairs do this. Handing both regions the whole shape would let the
    typesetter draw two different translations over each other. Splitting it
    by which text box is nearer gives each region the half a reader would
    already read it as owning.
    """
    shape = interior.shape
    own_distance = _distance_to(own, window, shape)
    keep = np.ones(shape, bool)
    for box in others:
        keep &= own_distance <= _distance_to(box, window, shape)
    return (interior.astype(bool) & keep).astype(np.uint8)


def _largest_part_at(mask: np.ndarray, at: tuple[int, int]) -> np.ndarray | None:
    """The connected component covering a point, or the biggest one near it."""
    count, labels = cv2.connectedComponents(mask, connectivity=4)
    if count < 2:
        return None
    y, x = at
    y = int(np.clip(y, 0, mask.shape[0] - 1))
    x = int(np.clip(x, 0, mask.shape[1] - 1))
    label = int(labels[y, x])
    if label == 0:
        values, counts = np.unique(labels[labels > 0], return_counts=True)
        if values.size == 0:
            return None
        label = int(values[counts.argmax()])
    return (labels == label).astype(np.uint8)


def _shape_within_bubble(
    image: np.ndarray, region: Region, bg: tuple[int, int, int], tolerance: int
) -> np.ndarray | None:
    """The balloon's own outline, taken inside the box the detector drew.

    The strict recovery in `_interior_mask` throws a fill away if it reaches
    the edge of its window or comes back too ragged to be a balloon, because
    without knowing where the balloon is, a fill that escaped one is
    indistinguishable from a fill that found one. Inside a detected balloon
    that reasoning does not apply: an escape has nowhere to go, so the tests
    can be dropped and the fill kept.

    This is what the rectangle was standing in for. A rectangle is the right
    *part* of the page but the wrong shape, and lettering set to it runs into
    the corners a round balloon does not have -- on one page a line of it was
    drawn straight over the artwork past the balloon's edge.
    """
    if region.bubble is None:
        return None
    height, width = image.shape[:2]
    window = region.bubble
    crop = window.crop(image)
    if crop.size == 0:
        return None

    ground = np.array(bg, dtype=np.int16)
    distance = np.abs(crop.astype(np.int16) - ground).max(axis=2)
    candidate = (distance <= tolerance).astype(np.uint8)

    off_x, off_y = region.box.x1 - window.x1, region.box.y1 - window.y1
    if region.mask is not None:
        mh, mw = region.mask.shape
        y1, x1 = max(0, off_y), max(0, off_x)
        y2, x2 = min(candidate.shape[0], off_y + mh), min(candidate.shape[1], off_x + mw)
        if y2 > y1 and x2 > x1:
            candidate[y1:y2, x1:x2] |= region.mask.astype(np.uint8)[
                y1 - off_y : y2 - off_y, x1 - off_x : x2 - off_x
            ]

    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    centre = (
        region.box.y1 + region.box.height // 2 - window.y1,
        region.box.x1 + region.box.width // 2 - window.x1,
    )
    part = _largest_part_at(candidate, centre)
    if part is None:
        return None
    interior = _fill_holes(part)

    contours, _ = cv2.findContours(
        interior, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    # It still has to hold the lettering it was recovered for. Anything
    # smaller is a highlight inside the balloon, not the balloon.
    if cv2.contourArea(contour) < region.box.area:
        return None
    polygon = contour.reshape(-1, 2).astype(np.int32)
    polygon[:, 0] = np.clip(polygon[:, 0] + window.x1, 0, width)
    polygon[:, 1] = np.clip(polygon[:, 1] + window.y1, 0, height)
    return polygon


def _bubble_polygon(bubble: Box) -> np.ndarray:
    return np.array(
        [
            [bubble.x1, bubble.y1],
            [bubble.x2, bubble.y1],
            [bubble.x2, bubble.y2],
            [bubble.x1, bubble.y2],
        ],
        dtype=np.int32,
    )


def _grow_into_ground(
    image: np.ndarray, region: Region, bg: tuple[int, int, int],
    tolerance: int, limit: float = 2.2,
) -> np.ndarray | None:
    """Expand a region's box outward while the ground stays uniform.

    The fallback for regions where no balloon could be recovered -- a caption
    laid on a flat bar, or a balloon whose outline defeated the flood. Without
    it such a region may only use the box the *English* occupied, and since
    Indonesian runs 20-30% longer the translation frequently cannot be fitted
    at any size: "Festival Keterampilan Mendoakan" had to go into the 134x12
    box that held a much shorter English caption.

    Each side advances only while the strip being added is almost entirely
    ground, so growth stops at the artwork rather than covering it.
    """
    height, width = image.shape[:2]
    ground = np.array(bg, dtype=np.int16)
    box = region.box
    max_w, max_h = box.width * limit, box.height * limit
    if region.bubble is not None:
        # A multiple of the *text* is the wrong bound when the balloon holding
        # it is known: a short line in a large balloon stops growing well
        # inside it, and the translation is then fitted to a fraction of the
        # room that is actually there.
        max_w, max_h = max(max_w, region.bubble.width), max(max_h, region.bubble.height)
    x1, y1, x2, y2 = box.x1, box.y1, box.x2, box.y2
    step = 2

    def clear(strip: np.ndarray) -> bool:
        if strip.size == 0:
            return False
        distance = np.abs(strip.astype(np.int16) - ground).max(axis=2)
        return float((distance <= tolerance).mean()) >= 0.92

    growing = True
    while growing:
        growing = False
        if x1 - step >= 0 and (x2 - x1) < max_w and clear(image[y1:y2, x1 - step : x1]):
            x1 -= step
            growing = True
        if x2 + step <= width and (x2 - x1) < max_w and clear(image[y1:y2, x2 : x2 + step]):
            x2 += step
            growing = True
        if y1 - step >= 0 and (y2 - y1) < max_h and clear(image[y1 - step : y1, x1:x2]):
            y1 -= step
            growing = True
        if y2 + step <= height and (y2 - y1) < max_h and clear(image[y2 : y2 + step, x1:x2]):
            y2 += step
            growing = True

    if region.bubble is not None:
        # Ground continues past a balloon drawn open at one edge, or through a
        # tail into the balloon next door. The detection says where to stop.
        bubble = region.bubble
        x1, y1 = max(x1, bubble.x1), max(y1, bubble.y1)
        x2, y2 = min(x2, bubble.x2), min(y2, bubble.y2)
    if x2 - x1 <= box.width and y2 - y1 <= box.height:
        return None
    return np.array(
        [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32
    )


def _polygon_of(
    interior: np.ndarray, window: Box, region: Region, page_area: int
) -> np.ndarray | None:
    """Validate a recovered interior and return it as a page-space polygon."""
    contours, _ = cv2.findContours(interior, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    if area < region.box.area * 0.9:
        return None  # smaller than the text it must hold: not a balloon

    # A balloon is a closed, roughly convex shape. A fill that has escaped one
    # is not: the chapter title bar on the sample page ran out along the black
    # gutters between panels and came back covering 1170x759 of the page at
    # solidity 0.24, where every correctly recovered balloon measured 0.77 or
    # better. Nothing else separated that case -- its area was smaller than
    # two of the genuine balloons.
    hull = cv2.contourArea(cv2.convexHull(contour))
    if hull > 0 and area / hull < 0.55:
        return None
    if area > page_area * 0.20:
        return None

    polygon = contour.reshape(-1, 2).astype(np.int32)
    polygon[:, 0] += window.x1
    polygon[:, 1] += window.y1
    return polygon


def classify(image: np.ndarray, regions: list[Region], cfg: Config) -> list[Region]:
    """Fill in colours, kind and polygon. Returns the regions worth keeping."""
    colour_fraction = _page_colour_fraction(image)
    page_area = image.shape[0] * image.shape[1]

    kept: list[Region] = []
    for region in regions:
        crop = region.box.crop(image)
        if crop.size == 0 or region.mask is None:
            continue
        mask = region.mask.astype(np.uint8)
        body, ring = _ink_and_ground(mask, inner=3, outer=11)

        if _is_overlay(crop, body, colour_fraction):
            region.fail(RegionStatus.SKIPPED_EMPTY, "colour overlay, likely a watermark")
            continue

        region.bg_color = _median_color(crop[ring]) if ring.any() else (255, 255, 255)
        region.text_color = _ink_color(crop[body], region.bg_color)

        surround = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)[ring] if ring.any() else None
        flat = surround is not None and float(surround.std()) <= cfg.classify.flat_bg_std
        region.kind = RegionKind.BALLOON if flat else RegionKind.FREE

        # Only lettering over artwork carries a stroke. Inside a balloon the
        # ring outside the glyphs *is* the background, and testing there would
        # only find noise: balloons measured 0.7 apart at the 90th percentile.
        if region.kind is RegionKind.FREE:
            region.outline_color = _outline_colour(
                region, region.bg_color, cfg.classify.min_outline_contrast
            )
        kept.append(region)

    _recover_balloons(image, kept, page_area, cfg)
    _separate_polygons(kept)
    return kept


def _recover_balloons(
    image: np.ndarray, regions: list[Region], page_area: int, cfg: Config
) -> None:
    """Give every balloon region the polygon of the space it may letter into."""
    tolerance = int(cfg.classify.flat_bg_std)
    found: dict[str, tuple[np.ndarray, Box]] = {}
    for region in regions:
        if region.kind is not RegionKind.BALLOON and region.bubble is None:
            continue
        result = _interior_mask(image, region, region.bg_color, tolerance)
        if result is not None:
            found[region.id] = result

    for region in regions:
        if region.id not in found:
            region.polygon = _fallback_shape(image, region, tolerance)
            continue
        interior, window = found[region.id]
        rivals = [
            other.box
            for other in regions
            if other.id != region.id and _covers(interior, window, other.box)
        ]
        if rivals:
            interior = _split_between(interior, window, region.box, rivals)
        centre = (
            region.box.y1 + region.box.height // 2 - window.y1,
            region.box.x1 + region.box.width // 2 - window.x1,
        )
        part = _largest_part_at(interior, centre)
        polygon = (
            _polygon_of(part, window, region, page_area) if part is not None else None
        )
        # A recovery that was rejected leaves the region with only the box the
        # English occupied, which is usually too small for the Indonesian. Fall
        # back to growing into whatever flat ground surrounds it, and past that
        # to the balloon itself -- a rectangle the size of the balloon is a
        # coarse shape, but it is the right part of the page, which the box the
        # English sat in very often is not.
        region.polygon = (
            polygon if polygon is not None else _fallback_shape(image, region, tolerance)
        )


def _fallback_shape(
    image: np.ndarray, region: Region, tolerance: int
) -> np.ndarray | None:
    """Where a region may letter into when the strict recovery found nothing.

    In order of how much each knows about the page: the balloon's own outline
    taken inside the detected box, then whatever flat ground surrounds the
    lettering, then the detected box itself. The last is a poor shape and a
    good place -- still better than the box the English occupied, which is the
    one shape guaranteed to be too small for the Indonesian.
    """
    shape = _shape_within_bubble(image, region, region.bg_color, tolerance)
    if shape is not None:
        return shape
    grown = _grow_into_ground(image, region, region.bg_color, tolerance)
    if grown is not None:
        return grown
    return _bubble_polygon(region.bubble) if region.bubble is not None else None


def _separate_polygons(regions: list[Region]) -> None:
    """Stop two regions being told they may letter into the same pixels.

    The flood-fill path partitions a shared balloon interior between the
    regions in it, but that is one of four ways a polygon is arrived at, and
    the other three -- growing into flat ground, the balloon outline taken
    inside a detected box, and the detected box itself -- each hand back a
    shape that knows nothing about its neighbours. Two adjacent balloons then
    both receive room the other is also using, and the typesetter, which
    fits each region on its own, draws two translations across each other.
    That is what the reported page shows: two blocks of Indonesian in one
    balloon, one written over the other.

    Contested pixels go to whichever region's lettering is nearer, which is
    how a reader assigns them too.
    """
    owners = [r for r in regions if r.polygon is not None]
    for index, region in enumerate(owners):
        carve(
            region,
            [
                owners[other].box
                for other in range(len(owners))
                if other != index and _polygons_meet(region, owners[other])
            ],
        )


def carve(region: Region, rivals: list[Box]) -> bool:
    """Cut a region's drawable shape back off its neighbours' lettering.

    Public because a region selected by hand arrives after the page has been
    classified, and it has to be carved against what is already there or it
    takes the whole balloon and letters across the block beside it.
    """
    if region.polygon is None or not rivals:
        return False
    window = Box(
        min(int(region.polygon[:, 0].min()), region.box.x1),
        min(int(region.polygon[:, 1].min()), region.box.y1),
        max(int(region.polygon[:, 0].max()), region.box.x2) + 1,
        max(int(region.polygon[:, 1].max()), region.box.y2) + 1,
    )
    mask = np.zeros((window.height, window.width), np.uint8)
    cv2.fillPoly(mask, [region.polygon - np.array([window.x1, window.y1])], 1)
    kept = _split_between(mask, window, region.box, rivals)
    part = _largest_part_at(
        kept,
        (
            region.box.y1 + region.box.height // 2 - window.y1,
            region.box.x1 + region.box.width // 2 - window.x1,
        ),
    )
    if part is None or not part.any():
        return False
    contours, _ = cv2.findContours(part, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False
    contour = max(contours, key=cv2.contourArea)
    # A sliver is worse than the shape it came from: the region still has to
    # hold its own lettering, and cutting it below that guarantees a failure
    # to fit where an overlap was only a risk.
    if cv2.contourArea(contour) < region.box.area * 0.75:
        return False
    polygon = contour.reshape(-1, 2).astype(np.int32)
    polygon[:, 0] += window.x1
    polygon[:, 1] += window.y1
    region.polygon = polygon
    return True


def _polygons_meet(one: Region, other: Region) -> bool:
    """Whether two regions' drawable shapes could touch at all."""
    if one.polygon is None or other.polygon is None:
        return False
    a = Box(
        int(one.polygon[:, 0].min()), int(one.polygon[:, 1].min()),
        int(one.polygon[:, 0].max()) + 1, int(one.polygon[:, 1].max()) + 1,
    )
    b = Box(
        int(other.polygon[:, 0].min()), int(other.polygon[:, 1].min()),
        int(other.polygon[:, 0].max()) + 1, int(other.polygon[:, 1].max()) + 1,
    )
    return a.intersects(b)


def _covers(interior: np.ndarray, window: Box, box: Box) -> bool:
    """Whether another region sits inside this interior.

    Measured over the whole box rather than at its centre point. Only the
    region's own glyphs were added back into its interior, so a rival's
    lettering is a field of holes in it, and a centre-pixel test lands on a
    stroke and reports no overlap often enough to miss the case entirely.
    """
    h, w = interior.shape
    x1, y1 = max(0, box.x1 - window.x1), max(0, box.y1 - window.y1)
    x2, y2 = min(w, box.x2 - window.x1), min(h, box.y2 - window.y1)
    if x2 <= x1 or y2 <= y1:
        return False
    return float(interior[y1:y2, x1:x2].mean()) > 0.30
