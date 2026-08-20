"""Blocks and balloons come from a detector now, not from thresholds.

The reported faults -- untranslated balloons, text set to the wrong shape,
awkward translations -- all trace to the same thing: which lines belonged
together, and where the balloon around them was, used to be inferred. Lines
were joined by the gap between them, so a generously leaded balloon came apart
mid-sentence and two balloons side by side welded into one; the balloon itself
was recovered by flooding outward from the lettering, and where that failed
the typesetter fell back to the box the English had occupied.

`pipeline.layout` detects both. These tests hold the consequences in place.
"""

from __future__ import annotations

import numpy as np

from mangatl.config import default_config
from mangatl.pipeline.detect import _group_lines
from mangatl.pipeline.inpaint import Inpainter
from mangatl.pipeline.layout import Layout, TextBlock
from mangatl.types import Box, Region, RegionKind


def _line(y1: int, y2: int, x1: int = 10, x2: int = 90) -> Box:
    return Box(x1, y1, x2, y2)


# -- which lines belong together --------------------------------------------


def test_lines_in_one_block_join_however_far_apart_they_sit():
    """The gap rule cannot tell loose leading from two stacked balloons.

    Measured across two series, lines inside one balloon sat at 1.17-1.20 line
    heights apart in the first and 1.60 in the second, and the setting that
    held the second together welded two side-by-side balloons in the first.
    """
    lines = [_line(0, 20), _line(200, 220)]  # ten line-heights apart
    groups = _group_lines(lines, [Box(0, 0, 100, 240)], 10**6, default_config())
    assert groups == [[0, 1]]


def test_lines_in_different_blocks_never_join():
    lines = [_line(0, 20), _line(24, 44)]  # touching, by geometry one block
    blocks = [Box(0, 0, 100, 22), Box(0, 23, 100, 46)]
    groups = _group_lines(lines, blocks, 10**6, default_config())
    assert sorted(groups) == [[0], [1]]


def test_lines_no_block_claims_are_still_grouped_on_geometry():
    """Credits and title bars are lettering the layout model does not fire on."""
    lines = [_line(0, 20), _line(24, 44)]
    groups = _group_lines(lines, [], 10**6, default_config())
    assert groups == [[0, 1]]


def test_a_detected_block_is_not_cut_up_by_the_area_cap():
    """The cap bounds transitive growth, and a detection did not grow.

    Applied to one, it splits a balloon mid-sentence -- which is the fault the
    cap was there to prevent in the first place.
    """
    lines = [_line(0, 20), _line(30, 50), _line(60, 80)]
    tiny_page_area = 100  # any real block is "over the cap" against this
    groups = _group_lines(lines, [Box(0, 0, 100, 80)], tiny_page_area, default_config())
    assert groups == [[0, 1, 2]]


# -- which balloon the lettering sits in -------------------------------------


def _layout(*bubbles: Box) -> Layout:
    return Layout(blocks=[], bubbles=list(bubbles))


def test_the_balloon_covering_the_lettering_is_the_one_chosen():
    text = Box(40, 40, 60, 60)
    right = Box(20, 20, 80, 80)
    assert _layout(Box(200, 200, 300, 300), right).bubble_for(text) == right


def test_lettering_over_artwork_has_no_balloon():
    assert _layout(Box(200, 200, 300, 300)).bubble_for(Box(0, 0, 20, 20)) is None


def test_the_tighter_of_two_balloons_around_one_line_wins():
    """A shout drawn inside a larger balloon must not take the outer one."""
    text = Box(45, 45, 55, 55)
    inner, outer = Box(40, 40, 60, 60), Box(0, 0, 200, 200)
    assert _layout(outer, inner).bubble_for(text) == inner


# -- erasure stays inside the balloon ----------------------------------------


def test_the_erase_mask_is_cut_off_at_the_balloon_edge():
    """Every reach in the erase stage grows outward and knows no edge.

    Past the balloon that means its outline erased along one side, or artwork
    outside it handed to a model told to invent a rectangle.
    """
    region = Region(id="r00", box=Box(10, 10, 30, 30), kind=RegionKind.FREE)
    region.bubble = Box(0, 0, 40, 40)
    mask = np.ones((80, 80), np.uint8)
    kept = Inpainter(default_config())._within_balloon(
        mask, Box(0, 0, 80, 80), region, (80, 80)
    )
    assert kept[:40, :40].all(), "the balloon's own interior must still be erased"
    assert not kept[60:, 60:].any(), "nothing outside the balloon may be touched"


def test_a_region_with_no_balloon_is_left_to_the_old_bounds():
    region = Region(id="r00", box=Box(10, 10, 30, 30), kind=RegionKind.FREE)
    mask = np.ones((80, 80), np.uint8)
    kept = Inpainter(default_config())._within_balloon(
        mask, Box(0, 0, 80, 80), region, (80, 80)
    )
    assert kept.all()
