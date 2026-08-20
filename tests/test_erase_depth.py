"""Erasing lettering must not take the drawing with it.

Lettering laid over artwork is erased as a *block* rather than glyph by
glyph, because the segmentation head only partly finds outlined and
decorative lettering and erasing the strokes it did find leaves half the
letters standing beside the replacement.

A block is a rectangle, and a rectangle asks the inpainting model to invent
every pixel inside it working inward from the rim. That is cheap for a
caption and impossible for a panel: a lattice window behind an eight-line
caption came back as a smooth grey field with nothing in it at all. Four more
blocks in the same chapter were erased that way; they do not all reach a
finished page, because a block classed as a sound effect is never erased at
all, but the rectangle over them was the same unfillable hole.

So the block is used only while the hole it makes is one the model can fill.
What decides that is not the hole's area but its *depth* -- how far the
deepest erased pixel lies from one that was kept -- because a long thin hole
following the strokes of a letter is easy however many pixels it covers.
"""

from __future__ import annotations

import numpy as np
import pytest

from mangatl.config import default_config
from mangatl.pipeline.inpaint import Inpainter, _depth
from mangatl.types import Box, Region, RegionKind, RegionStatus

SHAPE = (1000, 900)


def _config(limit: float = 150.0):
    cfg = default_config().model_copy(deep=True)
    cfg.inpaint.max_hole_depth = limit
    return cfg


def _lettering(box: Box, stroke: int = 4) -> np.ndarray:
    """A page glyph mask holding a few lettered rows inside `box`."""
    mask = np.zeros(SHAPE, np.uint8)
    for y in range(box.y1 + 6, box.y2 - 6, 18):
        mask[y : y + stroke, box.x1 + 6 : box.x2 - 6] = 1
    return mask


def _region(box: Box, line_height: float = 12.0) -> Region:
    region = Region(
        id="r00",
        box=box,
        kind=RegionKind.FREE,
        status=RegionStatus.TRANSLATED,
        source_line_height=line_height,
        lines=[Box(box.x1, y, box.x2, y + 12) for y in range(box.y1, box.y2 - 12, 18)],
    )
    region.mask = np.ones((box.height, box.width), bool)
    return region


# -- what depth means --------------------------------------------------------


def test_depth_of_a_solid_square_is_half_its_side():
    mask = np.zeros((120, 120), np.uint8)
    mask[10:110, 10:110] = 1
    assert _depth(mask) == pytest.approx(50, abs=2)


def test_a_long_thin_hole_is_shallow_however_much_it_covers():
    """Which is the whole reason depth is the measure and area is not."""
    thin = np.zeros((200, 400), np.uint8)
    thin[90:110, 5:395] = 1          # 20 x 390
    square = np.zeros((200, 400), np.uint8)
    square[40:120, 40:120] = 1       # 80 x 80
    assert thin.sum() > square.sum()
    assert _depth(thin) < _depth(square)


def test_an_empty_mask_has_no_depth():
    assert _depth(np.zeros((10, 10), np.uint8)) == 0.0


# -- the cap -----------------------------------------------------------------


def test_a_caption_is_still_erased_as_a_block():
    """The block is what removes lettering the mask only partly found.

    It is worth a little invented artwork when the amount is small, so a
    region the model can fill must keep it.
    """
    painter = Inpainter(_config())
    box = Box(300, 300, 460, 380)
    mask, _ = painter._block_mask(_region(box), _lettering(box), SHAPE)
    assert mask.sum() > _lettering(box).sum()
    assert _depth(mask) <= 150.0


# A block wide and tall enough that a rectangle over it is deeper than the
# model can fill, as the reported 224x443 caption was once its stroke
# allowance was added.
BIG = Box(200, 200, 580, 700)


def test_a_panel_sized_block_follows_the_glyphs_instead():
    """The reported page: a caption over a lattice window.

    Erased as a rectangle the window was replaced by nothing at all.
    """
    painter = Inpainter(_config())
    mask, _ = painter._block_mask(_region(BIG), _lettering(BIG), SHAPE)
    assert _depth(mask) <= 150.0, "the hole must be one the model can fill"
    assert mask.sum() < BIG.area / 2, "a rectangle over the box is far larger"


def test_the_limit_is_what_decides_and_it_can_be_moved():
    """Raised past the hole, the same region keeps its rectangle."""
    glyphs = _lettering(BIG)
    tight, _ = Inpainter(_config(150.0))._block_mask(_region(BIG), glyphs, SHAPE)
    loose, _ = Inpainter(_config(10_000.0))._block_mask(_region(BIG), glyphs, SHAPE)
    assert loose.sum() > tight.sum()
    assert _depth(loose) > 150.0


def test_a_balloon_never_reaches_this_at_all():
    """Flat ground is painted back, not invented, so nothing is at stake."""
    cfg = _config()
    painter = Inpainter(cfg)
    box = Box(200, 200, 424, 643)
    region = _region(box)
    region.kind = RegionKind.BALLOON
    page = np.full((*SHAPE, 3), 255, np.uint8)
    clean = painter.erase(page, [region], _lettering(box))
    assert clean.shape == page.shape


# -- the model's input size --------------------------------------------------


def test_the_graph_decides_its_own_input_size():
    """`lama_size` could only ever be set to the value it already had.

    Both exports here are built at a fixed 512, and asking for anything else
    raised INVALID_ARGUMENT from onnxruntime at the first region of the first
    page. The graph is asked instead.
    """
    cfg = _config()
    cfg.inpaint.lama_size = 4096
    painter = Inpainter(cfg)
    assert painter._input_size() == 512


def test_a_free_axis_falls_back_to_the_setting():
    cfg = _config()
    cfg.inpaint.lama_size = 384
    painter = Inpainter(cfg)
    painter._fixed_size = 0        # as a dynamic axis reports
    assert painter._input_size() == 384
