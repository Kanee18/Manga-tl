"""A4 -- erase the original lettering.

This stage runs *after* translation, and erases only regions that reached
`RegionStatus.TRANSLATED`. Ordering it this way removes a whole class of
failure rather than handling it: a region that could not be read or could not
be translated is never cleared in the first place, so it keeps its original
pixels and the page ships with one balloon still in English instead of one
balloon blank. A blank balloon is the worse of the two -- it reads as damage,
where untranslated text reads as an omission.

Two ways of erasing, chosen per region by what is behind the glyphs:

* **Flat fill**, where classification found uniform ground. Painting the
  sampled colour back is not an approximation of the right answer, it *is* the
  right answer, and it costs nothing. Most of a manga page is balloons, so
  most regions take this path.
* **LaMa**, where there is artwork behind the lettering and the pixels have to
  be invented. Reserved for the minority of regions that need it, because an
  invented patch is always a guess and guesses are best kept off the balloons.
"""

from __future__ import annotations

import cv2
import numpy as np

from .. import runtime
from ..config import Config
from ..types import Box, Page, Region, RegionKind, RegionStatus

runtime.enable_bundled_cuda()

import onnxruntime as ort  # noqa: E402  -- must follow the DLL path fix


def _depth(mask: np.ndarray) -> float:
    """How far the deepest erased pixel lies from one that was kept.

    This, rather than the hole's area, is what an inpainting model is up
    against: it fills from the rim inward, so a long thin hole following the
    strokes of a letter is easy however many pixels it covers, and a compact
    rectangle of the same area is not.
    """
    if not mask.any():
        return 0.0
    padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    return float(cv2.distanceTransform(padded, cv2.DIST_L2, 5).max())


class Inpainter:
    """Erases lettering. The LaMa session is built on first use."""

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        self._session: ort.InferenceSession | None = None
        self._fixed_size: int | None = None

    def _lama(self) -> ort.InferenceSession:
        if self._session is None:
            options = ort.SessionOptions()
            options.log_severity_level = 3
            self._session = ort.InferenceSession(
                str(self._cfg.models.inpaint),
                sess_options=options,
                providers=self._cfg.models.providers(),
            )
        return self._session

    def _input_size(self) -> int:
        """The square the graph actually takes, whatever the config asks for.

        Both LaMa exports here are built at a fixed 512, so `inpaint.lama_size`
        was a knob that could only be turned to the value it already had --
        anything else raised INVALID_ARGUMENT from onnxruntime at the first
        region of the first page. The graph is asked instead, and the setting
        is used only where the export leaves the axis free.
        """
        if self._fixed_size is None:
            shape = self._lama().get_inputs()[0].shape
            side = shape[2] if len(shape) > 2 else None
            self._fixed_size = int(side) if isinstance(side, int) else 0
        return self._fixed_size or self._cfg.inpaint.lama_size

    def erase(
        self, image: np.ndarray, regions: list[Region], glyph_mask: np.ndarray
    ) -> np.ndarray:
        """A copy of the page with translated regions' lettering removed."""
        clean = image.copy()
        method = self._cfg.inpaint.method
        shape = image.shape[:2]
        for region in regions:
            if region.status is not RegionStatus.TRANSLATED or region.mask is None:
                continue
            flat = region.kind is RegionKind.BALLOON
            if method == "fill" or (method == "auto" and flat):
                mask, box = self._mask_for(region, glyph_mask, shape)
                mask = self._within_balloon(mask, box, region, shape)
                if mask.any():
                    self._fill(clean, region, mask, box)
            else:
                mask, box = self._block_mask(region, glyph_mask, shape)
                mask = self._within_balloon(mask, box, region, shape)
                if mask.any():
                    self._inpaint(clean, mask, box)
        return clean

    def _within_balloon(
        self, mask: np.ndarray, box: Box, region: Region, shape: tuple[int, int]
    ) -> np.ndarray:
        """Keep the erase inside the balloon the lettering sits in.

        Everything that decides how far to reach -- the component selection,
        the allowance for a stroke, the block rectangle handed to LaMa --
        reaches outward from the lettering and has no idea where the balloon
        ends. Reaching past it is what messy cleanup looks like: the balloon
        outline erased along one edge, or artwork just outside it smeared by a
        model asked to invent a rectangle bigger than the hole.

        The bound is the detected balloon with the dilation allowed back, so
        lettering set hard against the edge is still taken whole.
        """
        if region.bubble is None:
            return mask
        limit = region.bubble.padded(max(1, self._cfg.inpaint.dilate), within=shape)
        keep = np.zeros_like(mask)
        x1, y1 = max(0, limit.x1 - box.x1), max(0, limit.y1 - box.y1)
        x2, y2 = min(box.width, limit.x2 - box.x1), min(box.height, limit.y2 - box.y1)
        if x2 > x1 and y2 > y1:
            keep[y1:y2, x1:x2] = 1
        return mask * keep

    def paint_out(
        self, canvas: np.ndarray, mask: np.ndarray, box: Box
    ) -> None:
        """Erase an area chosen by hand rather than found by the detector.

        The stages above erase what a region says is lettering. This erases
        whatever someone painted over, which is the other half of the job: the
        segmentation head misses fragments, and a balloon with a stray stroke
        left in it looks worse than one never touched.

        The choice between filling and rebuilding is made the same way the
        automatic path makes it, by looking at what surrounds the area. Over a
        balloon the ground is uniform and painting it back *is* the right
        answer; over artwork it has to be invented.
        """
        if not mask.any():
            return
        ring = cv2.dilate(mask, np.ones((13, 13), np.uint8)) - mask
        patch = box.crop(canvas)
        if patch.size == 0:
            return
        around = patch[ring > 0]
        flat = (
            around.size > 0
            and float(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)[ring > 0].std())
            <= self._cfg.classify.flat_bg_std
        )
        if flat and self._cfg.inpaint.method != "lama":
            ground = np.median(around.reshape(-1, 3), axis=0)
            alpha = self._alpha(mask)
            blended = patch.astype(np.float32) * (1 - alpha) + ground * alpha
            patch[:, :] = np.clip(blended, 0, 255).astype(np.uint8)
        else:
            self._inpaint(canvas, mask, box)

    def _block_mask(
        self, region: Region, glyph_mask: np.ndarray, shape: tuple[int, int]
    ) -> tuple[np.ndarray, Box]:
        """The whole text block, as the area to rebuild.

        Inpainting follows the glyph mask everywhere else, because it removes
        the least. Over artwork it cannot: the segmentation head under-detects
        decorative and outlined lettering on a dark ground, and inpainting only
        the strokes it did find erases some letters and leaves the rest --
        "Maiden Shu Keigetsu" came back as "Maiden Keiye", and the translation
        was then drawn on top of the remains.

        Covering the block removes the lettering whatever the mask missed. It
        asks the model to invent more of the picture, which is the right trade
        only where there is a picture to invent; balloons keep the exact-glyph
        path, and they are most of a page.
        """
        grow = max(1, self._cfg.inpaint.dilate)

        # Cover the glyphs as the mask actually found them, not just the
        # detector's box: that box is grown from line bands marking the
        # x-height, so caps and descenders stand outside it.
        #
        # Reach further here than for a balloon. Over artwork the line head
        # often finds only a fragment of a block -- one page had a three-line
        # caption detected as "IS ER" -- and the letters it missed are
        # separate components that a tight selection leaves standing right
        # next to the replacement. The reach is scaled to the type, so it
        # gathers the rest of the same lettering without straying into
        # whatever else is on the page.
        # Bounded by the region itself and by an absolute cap. The reach is
        # scaled to the type so it gathers the rest of the same lettering, but
        # a region that reports an implausible line height would otherwise
        # gather half the page: one recovered caption erased 604,000 pixels of
        # artwork that way.
        limit = max(8.0, 0.35 * min(region.box.width, region.box.height))
        reach = int(min(max(6.0, 1.2 * max(region.source_line_height, 6.0)), limit, 60.0))
        glyphs, glyph_box = self._mask_for(region, glyph_mask, shape, reach)
        covered = region.box
        if glyphs.any():
            ys, xs = np.nonzero(glyphs)
            covered = Box(
                min(covered.x1, glyph_box.x1 + int(xs.min())),
                min(covered.y1, glyph_box.y1 + int(ys.min())),
                max(covered.x2, glyph_box.x1 + int(xs.max()) + 1),
                max(covered.y2, glyph_box.y1 + int(ys.max()) + 1),
            )

        # Lettering with a stroke needs the stroke taken too. It lies outside
        # the glyphs and outside the mask, and left behind it reads as pale
        # blobs scattered around the replacement -- which is what the erased
        # captions looked like. Strokes scale with the type, so the allowance
        # does too.
        halo = 0
        if region.outline_color is not None:
            halo = int(min(max(3.0, 0.45 * max(region.source_line_height, 6.0)),
                           limit, 40.0))

        inner = covered.padded(3 + halo, within=shape)
        box = inner.padded(grow + 12, within=shape)
        mask = np.zeros((box.height, box.width), np.uint8)
        x1, y1 = inner.x1 - box.x1, inner.y1 - box.y1
        mask[y1 : y1 + inner.height, x1 : x1 + inner.width] = 1
        mask = cv2.dilate(mask, np.ones((grow, grow), np.uint8))

        # And only if the model can actually fill the hole it makes. A
        # rectangle asks LaMa to invent every pixel inside it from the rim
        # inward, so what decides whether that works is how far in the
        # deepest pixel lies. Past about 150px it stops inventing and returns
        # a smooth grey field: the reported page had a lattice window behind
        # a caption replaced by nothing at all.
        #
        # Measured over 78 blocks of free lettering in one chapter. Above
        # 150px deep -- five blocks -- the rectangle destroyed the panel every
        # time and following the glyphs was plainly better. Below it the
        # rectangle was usually the better of the two, which is why this is a
        # limit and not a change of default: it removes lettering the
        # segmentation head only partly found, and that is worth a little
        # invented artwork when the amount is small.
        if _depth(mask) > self._cfg.inpaint.max_hole_depth:
            return glyphs, glyph_box
        return mask, box

    # -- masking --------------------------------------------------------

    def _mask_for(
        self,
        region: Region,
        glyph_mask: np.ndarray,
        shape: tuple[int, int],
        reach: int = 6,
    ) -> tuple[np.ndarray, Box]:
        """Whole glyphs to erase for a region, and the box holding them.

        Taken as complete connected components of the page's glyph mask rather
        than as the region's own cropped copy. The region box is grown from the
        detector's line bars, and those mark the x-height core of a line, not
        its full extent -- so a `y` descender or the stem of a capital falls
        outside the box, survives the erase, and leaves a legible ghost of the
        original lettering. Selecting by component keeps each glyph whole
        while still ignoring lettering that belongs to some other region.
        """
        grow = max(1, self._cfg.inpaint.dilate)
        box = region.box.padded(grow + 12 + reach, within=shape)
        patch = (box.crop(glyph_mask) > 0).astype(np.uint8)
        if not patch.any():
            return patch, box

        count, labels = cv2.connectedComponents(patch, connectivity=8)
        if count < 2:
            return patch, box

        # Selected against a grown box: punctuation the line head did not
        # cover sits just outside the region -- the closing bracket of a
        # chapter title survived an erase that took every other glyph on its
        # line. `reach` widens this for lettering over artwork, where the
        # detector often finds only a fragment of a block and the letters it
        # missed are left standing beside the replacement.
        inner = region.box.padded(reach, within=shape)
        x1, y1 = inner.x1 - box.x1, inner.y1 - box.y1
        x2, y2 = x1 + inner.width, y1 + inner.height
        core = labels[max(0, y1) : max(0, y2), max(0, x1) : max(0, x2)]
        wanted = np.unique(core[core > 0])
        if wanted.size == 0:
            return patch, box

        mask = np.isin(labels, wanted).astype(np.uint8)
        return cv2.dilate(mask, np.ones((grow, grow), np.uint8)), box

    @staticmethod
    def _alpha(mask: np.ndarray) -> np.ndarray:
        """Coverage that is total over the glyphs and soft just outside them.

        Blurring the mask directly leaves its own boundary at about half
        opacity, so the outermost -- and darkest -- pixels of a letter show
        through at 50% and the erased balloon still reads. The blur is taken
        over a mask grown one step further and then floored at the original,
        which keeps full coverage where the glyphs are and confines the
        feathering to the anti-aliased rim beyond them.
        """
        solid = mask.astype(np.float32)
        skirt = cv2.dilate(mask, np.ones((3, 3), np.uint8)).astype(np.float32)
        feathered = cv2.GaussianBlur(skirt, (5, 5), 0)
        return np.clip(np.maximum(solid, feathered), 0.0, 1.0)[:, :, None]

    def _fill(
        self, canvas: np.ndarray, region: Region, mask: np.ndarray, box: Box
    ) -> None:
        """Paint the sampled ground colour back over the glyphs."""
        patch = box.crop(canvas)
        ground = np.empty_like(patch)
        ground[:, :] = region.bg_color
        alpha = self._alpha(mask)
        blended = patch.astype(np.float32) * (1 - alpha) + ground.astype(np.float32) * alpha
        patch[:, :] = np.clip(blended, 0, 255).astype(np.uint8)

    # -- artwork behind the glyphs --------------------------------------

    def _inpaint(self, canvas: np.ndarray, mask: np.ndarray, box: Box) -> None:
        """Rebuild the artwork behind a region with LaMa."""
        size = self._input_size()
        h, w = canvas.shape[:2]

        # LaMa needs to see around the hole to continue what it finds there.
        pad = int(max(box.width, box.height) * self._cfg.inpaint.context_ratio) + 8
        window = box.padded(pad, within=(h, w))
        full_mask = np.zeros((window.height, window.width), np.uint8)
        off_x, off_y = box.x1 - window.x1, box.y1 - window.y1
        full_mask[off_y : off_y + box.height, off_x : off_x + box.width] = mask

        crop = window.crop(canvas)
        if crop.size == 0 or not full_mask.any():
            return

        small = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
        small_mask = cv2.resize(
            full_mask, (size, size), interpolation=cv2.INTER_NEAREST
        )
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        inputs = {
            "image": rgb.transpose(2, 0, 1)[None],
            "mask": small_mask.astype(np.float32)[None, None],
        }
        out = self._lama().run(None, inputs)[0][0]

        # The export returns 0-255 in some builds and 0-1 in others.
        painted = out.transpose(1, 2, 0)
        if painted.max() <= 1.5:
            painted = painted * 255.0
        painted = cv2.cvtColor(
            np.clip(painted, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR
        )
        painted = cv2.resize(
            painted, (window.width, window.height), interpolation=cv2.INTER_CUBIC
        )

        # Only the hole is replaced. Everything the model repainted outside the
        # mask is its reconstruction of pixels that were never damaged, and
        # keeping it would soften artwork for no reason.
        alpha = self._alpha(full_mask)
        target = window.crop(canvas)
        blended = target.astype(np.float32) * (1 - alpha) + painted.astype(np.float32) * alpha
        target[:, :] = np.clip(blended, 0, 255).astype(np.uint8)
