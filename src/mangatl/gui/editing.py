"""Corrections applied to a page that has already been translated.

Everything here works on one `PageResult` in memory and re-letters it. Nothing
touches the file on disk until the page is saved, and nothing changes how the
next page is translated -- these are corrections to *this* page, made while
looking at it.

The one thing worth knowing about re-lettering is that it starts from
`page.clean`, not from what is currently on screen. The cleaned page is kept
for exactly this reason: lettering drawn onto it is never drawn *over*, so a
line can be retyped, resized and retyped again without each pass laying ink on
top of the last.

Two operations change `clean` itself, and both are meant to: erasing by hand,
and translating a region the run never erased. The second is the subtle one.
The pipeline erases exactly what it goes on to letter, so a balloon it never
saw -- one selected by hand afterwards -- still holds the artist's words, and
setting a translation over them puts two sentences in one balloon. `_erased`
tracks which regions have had theirs taken out, seeded from the regions the
run actually lettered.

Keeping a balloon in its original language does *not* change `clean`, on
purpose. The pixels are put back at drawing time instead, so the switch works
both ways: undoing it would otherwise mean erasing the balloon a second time,
and a second erase over rebuilt artwork is not the same picture as the first.

`Typesetter.plan` only considers regions marked TRANSLATED, because in the
pipeline it runs once, between translation and erasure. Re-lettering has to
put the already-drawn regions back into that state first, which is what
`_ready_to_letter` does.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..runner import PageResult, _imwrite
from ..types import Box, Page, Region, RegionKind, RegionStatus
from .worker import RunnerHost


@dataclass(frozen=True)
class _Original:
    """What a region was before anyone touched it, for Reset."""

    text_id: str
    size_scale: float
    bold: bool
    status: RegionStatus
    offset_x: float = 0.0
    offset_y: float = 0.0
    keep_original: bool = False


# How many corrections can be taken back. Every entry holds the settings of
# every region, which is kilobytes, plus -- only for the two operations that
# repaint -- the patch of cleaned page they covered. Storing whole pages
# instead would be 27 MB an entry on a 2560x3595 scan.
_UNDO_DEPTH = 40


@dataclass
class _Step:
    """One undoable correction: what changed, and what it was before."""

    label: str
    regions: dict[str, _Original]
    patch: tuple[Box, np.ndarray] | None = None
    # Regions this step brought into existence, which taking it back has to
    # remove. Distinct from `regions`, which is what it *changed*: a drawn
    # region is both, and restoring its settings without removing it leaves
    # an empty region behind on the page.
    created: tuple[str, ...] = ()
    # Regions whose lettering this step rubbed out. `patch` puts the pixels
    # back; this puts back the fact that they are there, so the next attempt
    # rubs them out again instead of lettering over them.
    erased: tuple[str, ...] = ()


def _runs_off(
    ink: np.ndarray, window: Box, shape: tuple[int, int], margin: int
) -> bool:
    """Whether a block of ink comes near a side of its window and carries on.

    Near, not touching. The next line of a balloon sits a leading away from
    this one, so ink that stops a few pixels short of the edge still has more
    beyond it -- measured on a page here, a window cut off eight pixels above
    the last line of a balloon, the block did not reach the edge, and the
    search stopped one line short with "THAT?!" left in English underneath.
    The margin is that leading.

    A side that is also the edge of the page does not count: there is nothing
    beyond it to have missed, and treating it as unfinished would grow the
    window until it covered everything.
    """
    ys, xs = np.nonzero(ink)
    if ys.size == 0:
        return False
    height, width = shape
    return bool(
        (xs.min() <= margin and window.x1 > 0)
        or (ys.min() <= margin and window.y1 > 0)
        or (xs.max() >= window.width - 1 - margin and window.x2 < width)
        or (ys.max() >= window.height - 1 - margin and window.y2 < height)
    )


def _snapshot(region: Region) -> _Original:
    return _Original(
        region.text_id, region.size_scale, region.bold, region.status,
        region.offset_x, region.offset_y, region.keep_original,
    )


def _restore(region: Region, before: _Original) -> None:
    region.text_id = before.text_id
    region.size_scale = before.size_scale
    region.bold = before.bold
    region.status = before.status
    region.offset_x = before.offset_x
    region.offset_y = before.offset_y
    region.keep_original = before.keep_original
    region.note = ""


class PageEditor:
    """Holds the page being corrected and re-letters it on demand."""

    def __init__(self, host: RunnerHost) -> None:
        self._host = host
        self.result: PageResult | None = None
        self._before: dict[str, _Original] = {}
        self._undo: list[_Step] = []
        self._added: list[str] = []
        # Regions whose own lettering is gone from the cleaned page. The
        # pipeline erases what it translated and nothing else, so a region it
        # skipped -- or one drawn here after the fact -- still has the
        # artist's words underneath, and lettering over them is what that
        # looks like on the page.
        self._erased: set[str] = set()
        self._dirty = False

    # -- the page -------------------------------------------------------

    def open(self, result: PageResult | None) -> None:
        self.result = result
        self._dirty = False
        self._before = {}
        self._undo = []
        self._added = []
        self._erased = set()
        if result is None:
            return
        for region in result.page.regions:
            self._before[region.id] = _snapshot(region)
            # What the run erased is exactly what it went on to letter.
            if region.status is RegionStatus.TYPESET:
                self._erased.add(region.id)

    @property
    def page(self) -> Page | None:
        return self.result.page if self.result else None

    @property
    def editable(self) -> bool:
        """Whether there is a lettered page here to correct."""
        page = self.page
        return page is not None and page.clean is not None

    @property
    def dirty(self) -> bool:
        return self._dirty

    def region(self, region_id: str) -> Region | None:
        page = self.page
        if page is None or not region_id:
            return None
        return next((r for r in page.regions if r.id == region_id), None)

    # -- undo -----------------------------------------------------------

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def undo_label(self) -> str:
        return self._undo[-1].label if self._undo else ""

    def _record(
        self,
        label: str,
        ids: list[str],
        patch: tuple[Box, np.ndarray] | None = None,
        created: tuple[str, ...] = (),
        erased: tuple[str, ...] = (),
    ) -> None:
        """Remember what the named regions were, before they are changed."""
        page = self.page
        if page is None:
            return
        kept = {
            region.id: _snapshot(region)
            for region in page.regions
            if region.id in ids
        }
        self._undo.append(_Step(label, kept, patch, created, erased))
        del self._undo[: max(0, len(self._undo) - _UNDO_DEPTH)]

    def _patch_of(self, box: Box) -> tuple[Box, np.ndarray] | None:
        """The pixels an operation on `box` could disturb, kept for undo.

        Wider than the box itself, because rebuilding artwork does not stop
        at the hole: LaMa is shown `inpaint.context_ratio` of the box around
        it so it has something to continue, and blends its answer back across
        that whole window. A patch cut to the box would restore the middle of
        an undone erase and leave a rebuilt ring around it.
        """
        page = self.page
        if page is None or page.clean is None:
            return None
        reach = int(
            max(box.width, box.height) * self._host.cfg.inpaint.context_ratio
        ) + 12
        wide = box.padded(reach, within=page.clean.shape[:2])
        return wide, wide.crop(page.clean).copy()

    def undo(self) -> bool:
        """Take back the last correction. Returns whether anything changed."""
        page = self.page
        if page is None or not self._undo:
            return False
        step = self._undo.pop()

        if step.patch is not None and page.clean is not None:
            box, pixels = step.patch
            page.clean[box.y1 : box.y2, box.x1 : box.x2] = pixels

        for region_id, before in step.regions.items():
            region = self.region(region_id)
            if region is not None:
                _restore(region, before)
        for region_id in step.erased:
            self._erased.discard(region_id)
        if step.created:
            page.regions = [
                region for region in page.regions if region.id not in step.created
            ]
            for region_id in step.created:
                self._before.pop(region_id, None)
                if region_id in self._added:
                    self._added.remove(region_id)
        self._dirty = bool(self._undo)
        return True

    # -- corrections ----------------------------------------------------

    def set_text(self, region_id: str, text: str) -> bool:
        region = self.region(region_id)
        if region is None or region.text_id == text:
            return False
        self._record("retype", [region_id])
        region.text_id = text
        # Retyping a line the pipeline gave up on is the whole point of being
        # able to retype it, so a hand-written translation clears whatever the
        # failure was and puts the region back in the queue to be lettered.
        region.status = (
            RegionStatus.TRANSLATED if text else RegionStatus.SKIPPED_EMPTY
        )
        region.note = ""
        return self._touch()

    def set_scale(self, region_id: str, scale: float) -> bool:
        region = self.region(region_id)
        if region is None or abs(region.size_scale - scale) < 1e-6:
            return False
        self._record("resize", [region_id])
        region.size_scale = scale
        return self._touch()

    def nudge(self, region_id: str, dx: float, dy: float, merge: bool) -> bool:
        """Move a block within its balloon, in page pixels.

        `merge` folds this into the last move rather than stacking a new one,
        so a drag across the page is one thing to take back rather than forty.
        """
        region = self.region(region_id)
        if region is None or (abs(dx) < 0.01 and abs(dy) < 0.01):
            return False
        if not (merge and self._undo and self._undo[-1].label == "move"):
            self._record("move", [region_id])
        region.offset_x += dx
        region.offset_y += dy
        return self._touch()

    def set_bold(self, region_id: str, bold: bool) -> bool:
        region = self.region(region_id)
        if region is None or region.bold == bold:
            return False
        self._record("bold", [region_id])
        region.bold = bold
        return self._touch()

    def set_keep_original(self, region_id: str, keep: bool) -> bool:
        """Leave this balloon reading as it was drawn, or letter it again.

        Nothing is repainted and nothing is thrown away. `page.clean` is left
        erased and the translation is left on the region; the original pixels
        go back over the hole at drawing time instead -- see `reletter`. That
        is what makes this a switch rather than a decision: putting the
        lettering back costs one redraw, where restoring pixels into `clean`
        would mean erasing the balloon a second time to undo it, and a second
        erase over rebuilt artwork is not the same picture as the first.
        """
        region = self.region(region_id)
        if region is None or region.keep_original == keep:
            return False
        self._record("keep original" if keep else "translate again", [region_id])
        region.keep_original = keep
        region.note = "left in the original language" if keep else ""
        return self._touch()

    def record_translate(self, region_id: str) -> bool:
        """Note that a region is about to be read and translated again.

        Recorded before the worker starts rather than after it finishes, so
        that taking it back restores what the region said beforehand -- by the
        time the reply lands, that is gone. The pixels go in the same step,
        because the lettering under this region is about to be rubbed out and
        one press should be one thing to take back.
        """
        region = self.region(region_id)
        if region is None:
            return False
        self._record(
            "translate",
            [region_id],
            patch=self._patch_of(region.box),
            erased=(region_id,),
        )
        # Asking for a translation is asking for it to be lettered, so a
        # balloon being kept in the original stops being kept. Inside the same
        # undo step, so taking it back puts both halves of that decision back.
        region.keep_original = False
        return self._touch()

    def reset(self, region_id: str) -> bool:
        region = self.region(region_id)
        original = self._before.get(region_id) if region else None
        if region is None or original is None:
            return False
        self._record("reset", [region_id])
        _restore(region, original)
        return self._touch()

    def erase(self, points: list[tuple[float, float]], radius: int) -> bool:
        """Paint out whatever a stroke covered, on the cleaned page."""
        page = self.page
        if page is None or page.clean is None or not points:
            return False
        height, width = page.clean.shape[:2]
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        box = Box(
            int(min(xs)), int(min(ys)), int(max(xs)) + 1, int(max(ys)) + 1
        ).padded(radius + 8, within=(height, width))
        if box.width <= 0 or box.height <= 0:
            return False

        self._record("erase", [], self._patch_of(box))
        mask = np.zeros((box.height, box.width), np.uint8)
        local = [(int(x) - box.x1, int(y) - box.y1) for x, y in points]
        # Drawn as a joined-up line, not as a scatter of dots: a quick drag
        # reports points several pixels apart and stamping circles at each
        # leaves the gaps between them unerased.
        for start, end in zip(local, local[1:]):
            cv2.line(mask, start, end, 1, radius * 2)
        for point in local:
            cv2.circle(mask, point, radius, 1, -1)

        self._host.runner().painter.paint_out(page.clean, mask, box)
        return self._touch()

    # -- a region drawn by hand -----------------------------------------

    def _ink_in(self, box: Box) -> np.ndarray | None:
        """The lettering inside a box someone drew round it, or None.

        Three sources, in order of how much each knows. The segmentation head
        is the best of them where it fired -- but a box drawn by hand is drawn
        precisely where it did *not*, since a balloon the detector found needs
        no box drawn round it. So it is asked first and disbelieved when it
        comes back near-empty, which is the same 3% floor the detector's own
        recovery uses, and then the ink is separated from its own local ground
        instead. Believing an empty answer is what made this refuse the very
        balloons it exists for, with "nothing that reads as lettering".
        """
        from ..pipeline.detect import _glyphs_by_contrast

        page = self.page
        if page is None or page.original is None:
            return None
        crop = box.crop(page.original)
        if crop.size == 0:
            return None

        if page.glyph_mask is not None:
            found = (box.crop(page.glyph_mask) > 0).astype(np.uint8) * 255
            if float((found > 0).mean()) >= 0.03:
                return self._glyph_like(found)

        separated = _glyphs_by_contrast(crop)
        if separated is not None:
            return self._glyph_like(separated)

        # Neither would say. Whatever is darker than the local ground is a
        # coarse answer and still enough to letter against, and refusing here
        # leaves someone looking at a balloon they can plainly see with no way
        # to translate it.
        grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        rough = (grey < max(90, int(grey.mean()) - 40)).astype(np.uint8) * 255
        return self._glyph_like(rough) if rough.any() else None

    @staticmethod
    def _glyph_like(ink: np.ndarray) -> np.ndarray:
        """Drop the marks that are the wrong size and shape to be lettering.

        Telling ink from its ground says nothing about what the ink *is*, and
        round a balloon there is plenty that is not writing: the balloon's own
        outline, the tail, the artwork it overlaps, a character's hair. Left
        in, all of it joins the block that is being closed up, and the block
        then runs off whatever window it is looked at in -- the search widened
        five times on a page here and came back holding a third of it.

        A page's own lettering is the measure. Glyphs are many marks of much
        the same size, so they carry the median between them, and what the
        median rejects is the outline running the height of the window and the
        solid blob that is somebody's hair. Nothing here is absolute: it is
        all relative to the marks in this one window, which is what lets it
        hold from a 6px caption to a 60px shout.
        """
        binary = (ink > 0).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
        marks = [
            (i, int(stats[i, cv2.CC_STAT_HEIGHT]))
            for i in range(1, count)
            if stats[i, cv2.CC_STAT_AREA] >= 6
        ]
        if len(marks) < 4:
            # Too few to have a median worth trusting, and too few to be the
            # case this is guarding against.
            return ink
        mark = float(np.median([height for _, height in marks]))
        kept = [
            i
            for i, height in marks
            if 0.3 * mark <= height <= 3.0 * mark
            and stats[i, cv2.CC_STAT_WIDTH] <= 12.0 * mark
            and stats[i, cv2.CC_STAT_AREA] <= 6.0 * mark * mark
        ]
        if not kept:
            return ink
        return np.where(np.isin(labels, kept), ink, 0)

    def _under_stroke(
        self,
        ink: np.ndarray,
        window: Box,
        points: list[tuple[float, float]],
        radius: int,
    ) -> np.ndarray | None:
        """The block of lettering a stroke was drawn over, out of `ink`.

        Two steps, and the second is what makes a stroke worth drawing at all.
        The marks the stroke actually crosses are only the letters under it --
        set the translation to those and a swipe across "GOING ON" would
        replace two words in the middle of a sentence. So the block is closed
        up first -- see `_blocks` -- and every mark sharing a blob with a
        touched one comes along. Swiping anywhere over a balloon
        therefore takes the balloon, and swiping over one of two columns takes
        that column and not its neighbour.
        """
        band = np.zeros(ink.shape, np.uint8)
        local = [
            (int(round(x)) - window.x1, int(round(y)) - window.y1) for x, y in points
        ]
        for start, end in zip(local, local[1:]):
            cv2.line(band, start, end, 1, radius * 2)
        for point in local:
            cv2.circle(band, point, radius, 1, -1)

        binary = (ink > 0).astype(np.uint8)
        blobs, labels = self._blocks(binary)
        touched = np.unique(labels[(band > 0) & (binary > 0)])
        touched = touched[touched > 0]
        if touched.size == 0:
            # The stroke ran between the marks rather than across them -- over
            # the leading, or through a balloon's empty half. Whichever block
            # it came nearest is still the one meant.
            nearest = self._nearest_block(labels, blobs, band)
            if nearest is None:
                return None
            touched = np.array([nearest])
        return np.where(np.isin(labels, touched), ink, 0)

    @staticmethod
    def _nearest_block(
        labels: np.ndarray, blobs: int, band: np.ndarray
    ) -> int | None:
        """The closed block nearest the stroke, when it crossed none of them."""
        if blobs < 2 or not band.any():
            return None
        reach = cv2.distanceTransform(1 - (band > 0).astype(np.uint8), cv2.DIST_L2, 3)
        best, closest = None, None
        for label in range(1, blobs):
            here = labels == label
            if not here.any():
                continue
            distance = float(reach[here].min())
            if closest is None or distance < closest:
                best, closest = label, distance
        return best

    @staticmethod
    def _leading(ink: np.ndarray) -> int:
        """About how far the next line of the same block would sit.

        Taken from the marks, so it holds from a 6px caption to a 60px shout.
        """
        count, _, stats, _ = cv2.connectedComponentsWithStats(
            (ink > 0).astype(np.uint8), 8
        )
        marks = [
            int(stats[i, cv2.CC_STAT_HEIGHT])
            for i in range(1, count)
            if stats[i, cv2.CC_STAT_AREA] >= 6
        ]
        return max(12, int(round(1.5 * float(np.median(marks))))) if marks else 12

    def _blocks(self, binary: np.ndarray) -> tuple[int, np.ndarray]:
        """Label the ink by block: marks close enough together to be one.

        Down the page the gap to close is the leading and across it only the
        word spaces, so the two reaches are not the same size: reaching as far
        sideways as vertically joins the balloon's own outline back on, and
        measured here the words were 10px apart and the outline 25px away.

        Both scale with the marks themselves rather than being fixed, because
        a reach that closes six lines on a 770px page leaves them as six blocks
        on a 2560px scan -- and a swipe would then take one line of a balloon.
        """
        count, _, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
        marks = [
            int(stats[i, cv2.CC_STAT_HEIGHT])
            for i in range(1, count)
            if stats[i, cv2.CC_STAT_AREA] >= 6
        ]
        mark = float(np.median(marks)) if marks else 10.0
        down = max(9, int(round(1.4 * mark)))
        across = max(5, int(round(0.7 * mark)))
        joined = cv2.dilate(binary, np.ones((down, across), np.uint8))
        return cv2.connectedComponents(joined, connectivity=8)

    def erase_lettering(self, region_id: str) -> bool:
        """Rub out one region's own words, so a translation can go over them.

        The run erases what it went on to letter and nothing else, which is
        right for a page it translated in one pass and wrong the moment a
        region is added or revived afterwards: a balloon the detector missed
        still holds the artist's lettering, and setting the Indonesian on top
        of it produces two sentences in one balloon -- which is what drawing a
        box used to do.

        The page's glyph mask is what the erase stage follows, and it has
        nothing to say here for the same reason the detector missed the
        balloon. So the region's own ink -- found from its local ground when
        the box was drawn -- is added into a copy of that mask, and the
        ordinary erase runs against it: flat fill inside a balloon, a rebuild
        over artwork, bounded by the balloon either way.
        """
        page = self.page
        region = self.region(region_id)
        if page is None or region is None or page.clean is None:
            return False
        if region.status is not RegionStatus.TRANSLATED or region.mask is None:
            return False
        if region_id in self._erased:
            return False

        glyphs = (
            page.glyph_mask.copy()
            if page.glyph_mask is not None
            else np.zeros(page.clean.shape[:2], np.uint8)
        )
        box, own = region.box, region.mask.astype(np.uint8) * 255
        if own.shape == (box.height, box.width):
            glyphs[box.y1 : box.y2, box.x1 : box.x2] |= own
        # Based on the cleaned page rather than the original, so everything
        # erased before this stays erased.
        page.clean = self._host.runner().painter.erase(page.clean, [region], glyphs)
        self._erased.add(region_id)
        return self._touch()

    def region_under(
        self, points: list[tuple[float, float]], radius: int = 9
    ) -> Region | None:
        """A region a stroke was drawn over, if there is one.

        Drawing over lettering that was found already is an easy thing to do
        -- the outlines only show while a tool is active -- and making a
        second region of it would read and letter the same words twice. The
        stroke has to actually cross the region rather than merely pass its
        box, so a swipe through the gap between two balloons finds neither and
        goes on to make a region of what it did land on.
        """
        page = self.page
        if page is None or not points:
            return None
        best, covered = None, 0
        for region in page.regions:
            hits = sum(
                1
                for x, y in points
                if region.box.x1 - radius <= x <= region.box.x2 + radius
                and region.box.y1 - radius <= y <= region.box.y2 + radius
            )
            # Half the stroke, so brushing a neighbour on the way past is not
            # mistaken for pointing at it.
            if hits > max(covered, len(points) // 2):
                best, covered = region, hits
        return best

    def new_region(
        self, points: list[tuple[float, float]], radius: int = 9
    ) -> Region | None:
        """Prepare lettering the detector never found, from a stroke over it.

        The stroke is a seed, not an outline. Someone drags across the words
        they mean; the ink it lands on is found, the marks it touched bring
        the rest of their block with them, and the region is the block. A
        rectangle was tried first and is the wrong shape for the job -- drawn
        tight it cuts a line off, drawn loose it takes in the balloon edge and
        whatever the balloon overlaps, and neither is what anyone meant by
        pointing at some words.

        Only what is cheap and local happens here -- the glyphs, the colours,
        the balloon. Reading and translating are a second or twenty and belong
        on a worker thread, so they are the caller's to run.
        """
        page = self.page
        if page is None or page.original is None or not points:
            return None
        from ..pipeline.classify import carve, classify
        from ..pipeline.detect import _lines_from_mask

        shape = page.original.shape[:2]
        height, width = shape
        reach = max(4, int(radius))
        seed = Box(
            int(min(p[0] for p in points)), int(min(p[1] for p in points)),
            int(max(p[0] for p in points)) + 1, int(max(p[1] for p in points)) + 1,
        )

        # The window is looked at well beyond the stroke, because the block it
        # lands on reaches past it -- a swipe across part of one line still
        # means the block -- and how far past cannot be known before the ink
        # is found. So it grows until the block stops running off the edge:
        # cut short, a swipe over the middle of a balloon returned the middle
        # of the sentence.
        around = max(48, reach * 6)
        found: tuple[np.ndarray, Box] | None = None
        for _ in range(5):
            drawn = seed.padded(around, within=shape)
            ink = self._ink_in(drawn)
            if ink is None:
                return None
            ink = self._under_stroke(ink, drawn, points, reach)
            if ink is None or not ink.any():
                return None
            found = (ink, drawn)
            if not _runs_off(ink, drawn, shape, self._leading(ink)):
                break
            around *= 2
        if found is None:
            return None
        ink, drawn = found
        ys, xs = np.nonzero(ink)
        if ys.size == 0:
            return None
        tight = Box(
            drawn.x1 + int(xs.min()), drawn.y1 + int(ys.min()),
            drawn.x1 + int(xs.max()) + 1, drawn.y1 + int(ys.max()) + 1,
        ).padded(3, within=(height, width))
        # Copied across by page coordinates rather than by shape, because the
        # padding can push the tightened box outside the drawn one on one side
        # and be clamped to the page on the other. Lining the two up by their
        # top-left corners would then slide the mask off its own glyphs.
        patch = np.zeros((tight.height, tight.width), np.uint8)
        sy1, sx1 = max(0, tight.y1 - drawn.y1), max(0, tight.x1 - drawn.x1)
        sy2 = min(ink.shape[0], tight.y2 - drawn.y1)
        sx2 = min(ink.shape[1], tight.x2 - drawn.x1)
        piece = ink[sy1:sy2, sx1:sx2]
        dy, dx = drawn.y1 + sy1 - tight.y1, drawn.x1 + sx1 - tight.x1
        patch[dy : dy + piece.shape[0], dx : dx + piece.shape[1]] = piece

        lines = _lines_from_mask(patch, tight.x1, tight.y1)
        heights = [line.height for line in lines]
        region = Region(
            id=self._next_id(),
            box=tight,
            kind=RegionKind.BALLOON,
            mask=patch > 0,
            lines=lines or [tight],
            source_line_height=float(np.median(heights)) if heights else tight.height,
        )
        kept = classify(page.original, [region], self._host.cfg)
        if not kept:
            return None
        # Classified on its own, so the shape it was given knows nothing about
        # what is already on the page. Two blocks in one balloon share an
        # interior, and a flood fill from either returns the whole of it --
        # so without this the selected block takes the balloon and letters
        # across its neighbour.
        carve(region, [other.box for other in page.regions])
        page.regions.append(region)
        self._before[region.id] = _Original("", 1.0, False, RegionStatus.DETECTED)
        self._added.append(region.id)
        self._record("add region", [region.id], created=(region.id,))
        self._touch()
        return region

    def _next_id(self) -> str:
        page = self.page
        taken = {r.id for r in page.regions} if page else set()
        for index in range(100):
            candidate = f"h{index:02d}"
            if candidate not in taken:
                return candidate
        return f"h{len(taken)}"

    # -- output ---------------------------------------------------------

    def _touch(self) -> bool:
        self._dirty = True
        return True

    def _ready_to_letter(self) -> None:
        """Put already-drawn regions back into the state `plan` looks at."""
        page = self.page
        if page is None:
            return
        redraw = {
            RegionStatus.TYPESET,
            RegionStatus.FAILED_FIT,
            RegionStatus.TRANSLATED,
            RegionStatus.KEPT_ORIGINAL,
        }
        for region in page.regions:
            if region.keep_original:
                # Held out of the fitting entirely rather than fitted and not
                # drawn: it has no room to compete for, and letting it take
                # part would push its neighbours aside for lettering that is
                # never going to appear.
                region.status = RegionStatus.KEPT_ORIGINAL
            elif region.status in redraw:
                region.status = (
                    RegionStatus.TRANSLATED
                    if region.text_id.strip()
                    else RegionStatus.SKIPPED_EMPTY
                )

    def _ground(self) -> np.ndarray | None:
        """The cleaned page with the kept-original balloons drawn back in.

        A copy, and only when there is something to put back: this runs on
        every keystroke that settles, and copying a 2560x3595 page costs 27 MB
        each time for the ordinary case where nothing is being kept.
        """
        page = self.page
        if page is None or page.clean is None:
            return None
        kept = [region for region in page.regions if region.keep_original]
        if not kept or page.original is None:
            return page.clean
        ground = page.clean.copy()
        for region in kept:
            box = region.box.padded(6, within=ground.shape[:2])
            ground[box.y1 : box.y2, box.x1 : box.x2] = page.original[
                box.y1 : box.y2, box.x1 : box.x2
            ]
        return ground

    def reletter(self) -> np.ndarray | None:
        """Redraw the whole page from the cleaned copy. Returns the result."""
        page = self.page
        if page is None or page.clean is None:
            return None
        typesetter = self._host.runner().typesetter
        self._ready_to_letter()
        layouts = typesetter.plan(page.regions, page.shape)
        page.rendered = typesetter.draw_page(self._ground(), page.regions, layouts)
        return page.rendered

    def save(self, out_dir: Path) -> Path | None:
        page = self.page
        if page is None or page.rendered is None:
            return None
        out_dir.mkdir(parents=True, exist_ok=True)
        cfg = self._host.cfg.output
        target = out_dir / f"{page.name}.{cfg.format}"
        params = (
            [cv2.IMWRITE_JPEG_QUALITY, cfg.jpeg_quality]
            if cfg.format == "jpg"
            else [cv2.IMWRITE_PNG_COMPRESSION, 6]
        )
        _imwrite(target, page.rendered, params)
        self._dirty = False
        return target
