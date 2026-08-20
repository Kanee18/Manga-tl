"""Data model shared by every pipeline stage.

A run carries one `Page` through the stages in `mangatl.pipeline`, and each
stage writes only the fields it owns:

    detect    -> Region.box, Region.polygon, Region.mask
    ocr       -> Region.text_en, Region.ocr_confidence
    classify  -> Region.kind, Region.bg_color, Region.text_color
    translate -> Region.text_id
    inpaint   -> Page.clean
    typeset   -> Page.rendered

`Region.status` records how far a region got. Inpainting runs *after*
translation and erases only regions that reached TRANSLATED, so a region that
fails anywhere keeps its original pixels. That ordering is deliberate: a
balloon re-lettered with the English the artist already wrote reads as a bug,
and an erased balloon with nothing drawn back reads as a worse one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import NamedTuple

import numpy as np


class RegionKind(str, Enum):
    """What sort of lettering a region holds."""

    BALLOON = "balloon"  # inside a speech balloon, on a flat fill
    FREE = "free"        # narration or caption laid straight over art
    SFX = "sfx"          # sound-effect lettering, drawn as part of the art


class RegionStatus(str, Enum):
    """How far a region got. Only TYPESET regions are drawn over."""

    DETECTED = "detected"
    READ = "read"
    TRANSLATED = "translated"
    TYPESET = "typeset"
    SKIPPED_SFX = "skipped_sfx"
    SKIPPED_EMPTY = "skipped_empty"
    SKIPPED_SAME = "skipped_same"
    # Asked for by hand in the window: this balloon is to read as it was
    # drawn. Not a failure and not a miss -- a choice, which is why it is its
    # own status rather than being folded into one of the two above.
    KEPT_ORIGINAL = "kept_original"
    FAILED_OCR = "failed_ocr"
    FAILED_TRANSLATE = "failed_translate"
    FAILED_FIT = "failed_fit"

    @property
    def is_failure(self) -> bool:
        return self.value.startswith("failed_")

    @property
    def is_rendered(self) -> bool:
        return self is RegionStatus.TYPESET


class Box(NamedTuple):
    """An axis-aligned box in page pixels, x2/y2 exclusive."""

    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def padded(self, pad: int, within: tuple[int, int] | None = None) -> "Box":
        """Grow by `pad` on every side, clamped to a (height, width) shape."""
        x1, y1 = self.x1 - pad, self.y1 - pad
        x2, y2 = self.x2 + pad, self.y2 + pad
        if within is not None:
            h, w = within
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
        return Box(int(x1), int(y1), int(x2), int(y2))

    def crop(self, image: np.ndarray) -> np.ndarray:
        return image[self.y1 : self.y2, self.x1 : self.x2]

    def iou(self, other: "Box") -> float:
        ix1, iy1 = max(self.x1, other.x1), max(self.y1, other.y1)
        ix2, iy2 = min(self.x2, other.x2), min(self.y2, other.y2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def intersects(self, other: "Box") -> bool:
        return (
            self.x1 < other.x2
            and other.x1 < self.x2
            and self.y1 < other.y2
            and other.y1 < self.y2
        )

    def contains(self, other: "Box", tolerance: int = 0) -> bool:
        return (
            self.x1 - tolerance <= other.x1
            and self.y1 - tolerance <= other.y1
            and self.x2 + tolerance >= other.x2
            and self.y2 + tolerance >= other.y2
        )

    @staticmethod
    def union(boxes: "list[Box]") -> "Box":
        return Box(
            min(b.x1 for b in boxes),
            min(b.y1 for b in boxes),
            max(b.x2 for b in boxes),
            max(b.y2 for b in boxes),
        )


@dataclass
class Region:
    """One block of lettering on a page."""

    id: str
    box: Box
    kind: RegionKind = RegionKind.BALLOON
    status: RegionStatus = RegionStatus.DETECTED

    # Page-sized boolean mask of the glyph pixels, cropped to `box`.
    mask: np.ndarray | None = None
    # Convex-ish outline of the balloon interior this region sits in, in page
    # pixels. None when no balloon was recovered, in which case `box` is used.
    polygon: np.ndarray | None = None
    # The balloon this lettering sits in, as the layout model detected it, or
    # None for lettering laid over artwork. Distinct from `polygon`, which is
    # the interior actually recovered from the pixels: this is the extent the
    # recovery is seeded and bounded by, and what the typesetter falls back to
    # when no interior could be recovered at all. A box the size of the
    # balloon is a far better guess at where a translation may go than a box
    # the size of the English that used to be in it.
    bubble: Box | None = None
    # The individual lettered lines making up this region, top to bottom, in
    # page pixels. Recognition runs per line against these, which is what
    # keeps words in the order they were drawn in.
    lines: list[Box] = field(default_factory=list)

    text_en: str = ""
    # What the translator judged the letterer actually wrote, after repairing
    # OCR damage. Kept separate from `text_en`, which stays the raw reading so
    # the echo and language guards always compare against what was really on
    # the page.
    text_en_fixed: str = ""
    text_id: str = ""
    # The last candidate a guard turned down. Kept so a corrective retry can
    # name what was wrong with it instead of repeating the same request, and
    # so a rejection is visible in the GUI rather than silent.
    rejected: str = ""
    ocr_confidence: float = 0.0
    ocr_lines: list[str] = field(default_factory=list)

    # BGR, sampled from the page. bg is what inpainting should leave behind,
    # text is what the replacement lettering is drawn in.
    bg_color: tuple[int, int, int] = (255, 255, 255)
    text_color: tuple[int, int, int] = (0, 0, 0)
    # BGR of the stroke the letterer put around the glyphs, or None when there
    # is none. Lettering laid straight onto artwork carries one so it stays
    # readable; dropping it leaves dark text on dark art. See
    # `classify._outline_colour`.
    outline_color: tuple[int, int, int] | None = None

    # Median stroke height of the source lettering, in pixels. The replacement
    # is capped near this so a translated balloon keeps the page's voice.
    source_line_height: float = 0.0

    # Set by hand in the window, not by any stage. A translation the
    # pipeline fitted correctly can still be the wrong size or weight for the
    # page it is on -- a shout set at speaking size reads as a mumble -- and
    # judging that needs eyes on the artwork. Both feed the typesetter as
    # adjustments to what it worked out, never as replacements for it: the
    # fitting still has the last word, so enlarging text cannot push it out
    # of its balloon.
    size_scale: float = 1.0
    bold: bool = False
    # Where the block sits relative to where the fitter put it, in page
    # pixels. The fitter anchors on the lettering it replaces, which is right
    # nearly always and wrong when the original sat off-centre in its balloon
    # or the balloon has a tail the text should keep clear of.
    offset_x: float = 0.0
    offset_y: float = 0.0
    # Whether this balloon is to be left reading as the artist drew it. Set in
    # the window, and reversible: the translation is kept, and the original
    # pixels are put back at drawing time rather than by rubbing out what was
    # lettered -- so turning it off letters the same translation again with
    # nothing to redo. A page can be part translated and part not, which is
    # what someone reading along in the source language actually wants.
    keep_original: bool = False

    note: str = ""

    @property
    def is_translatable(self) -> bool:
        return self.kind is not RegionKind.SFX and bool(self.text_en.strip())

    @property
    def is_unbounded(self) -> bool:
        """Whether anything on the page holds this region's lettering in.

        Both witnesses have to agree, because either alone is wrong about a
        real case. `kind` is FREE when the pixels ringing the glyphs are not
        flat, and a balloon whose interior is hatched or screentoned reads
        FREE while very much having an edge. `bubble` is None when the layout
        model boxed no balloon there, and it misses the open, tail-less
        balloons drawn straight onto white. Lettering with neither a flat
        ground around it nor a detected balloon over it is laid on artwork,
        and artwork has no edges.

        Two stages care. The typesetter, because such a region may spread out
        rather than shrink; and the translator, because lettering drawn onto
        artwork at several times the page's dialogue size is usually a noise
        the artist drew rather than a line anybody says.
        """
        return self.kind is RegionKind.FREE and self.bubble is None

    @property
    def line_pitch(self) -> float:
        """How tall one line of this region's lettering is, in page pixels.

        A raw measurement, with no font metrics in it, so it is only good for
        comparing one region on a page against another -- which is all the
        translator needs, and it saves that stage a dependency on the face
        the lettering will eventually be set in.
        """
        return self.box.height / max(1, len(self.lines))

    @property
    def is_moved(self) -> bool:
        return abs(self.offset_x) > 0.5 or abs(self.offset_y) > 0.5

    @property
    def is_hand_set(self) -> bool:
        """Whether anyone has overridden how this region is set."""
        return (
            self.bold
            or self.keep_original
            or abs(self.size_scale - 1.0) > 1e-6
            or self.is_moved
        )

    def fail(self, status: RegionStatus, note: str = "") -> None:
        self.status = status
        if note:
            self.note = note


@dataclass
class Page:
    """One source image and everything derived from it."""

    path: Path
    index: int = 0
    original: np.ndarray | None = None   # BGR, as loaded
    clean: np.ndarray | None = None      # BGR, text erased
    rendered: np.ndarray | None = None   # BGR, translated lettering drawn
    # Kept past detection so the window can work on the page afterwards: a
    # region drawn by hand needs the same glyph mask every detected one got,
    # and re-running the detector to recover it would cost a second and give
    # a different answer for the rest of the page.
    glyph_mask: np.ndarray | None = None
    regions: list[Region] = field(default_factory=list)
    error: str = ""

    @property
    def name(self) -> str:
        return self.path.stem

    @property
    def shape(self) -> tuple[int, int]:
        """(height, width) of the source image."""
        if self.original is None:
            raise ValueError(f"page {self.name} has not been loaded")
        return self.original.shape[:2]

    @property
    def translatable(self) -> list[Region]:
        return [r for r in self.regions if r.is_translatable]

    def stats(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for r in self.regions:
            counts[r.status.value] = counts.get(r.status.value, 0) + 1
        return counts
