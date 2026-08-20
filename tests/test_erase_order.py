"""The pipeline must never erase lettering it cannot replace.

This was the worst of the reported defects: balloons came back empty. Erasure
ran before fitting, so a region was rubbed out and only afterwards found to be
unfittable -- leaving a blank balloon, which reads as damage. Leaving the
English in place reads as an omission, which is the better failure.
"""

from __future__ import annotations

import numpy as np
import pytest

from mangatl.config import default_config
from mangatl.pipeline.inpaint import Inpainter
from mangatl.pipeline.typeset import Typesetter
from mangatl.types import Box, Region, RegionKind, RegionStatus


@pytest.fixture(scope="module")
def cfg():
    return default_config()


def _page_with_text() -> tuple[np.ndarray, np.ndarray]:
    """A white page with a black bar of 'lettering', and its glyph mask."""
    page = np.full((200, 200, 3), 255, np.uint8)
    page[90:110, 60:140] = 0
    mask = np.zeros((200, 200), np.uint8)
    mask[90:110, 60:140] = 255
    return page, mask


def _region(text: str) -> Region:
    region = Region(
        id="r00",
        box=Box(58, 88, 142, 112),
        lines=[Box(60, 90, 140, 110)],
        kind=RegionKind.BALLOON,
        status=RegionStatus.TRANSLATED,
        text_id=text,
        bg_color=(255, 255, 255),
        text_color=(0, 0, 0),
        source_line_height=20.0,
    )
    region.mask = np.zeros((region.box.height, region.box.width), bool)
    region.mask[2:22, 2:82] = True
    return region


def test_a_region_that_cannot_be_fitted_keeps_its_pixels(cfg):
    page, glyph_mask = _page_with_text()
    region = _region("Kalimat panjang sekali yang tidak mungkin muat di sini sama sekali")

    layouts = Typesetter(cfg).plan([region])
    assert region.status is RegionStatus.FAILED_FIT
    assert region.id not in layouts

    clean = Inpainter(cfg).erase(page, [region], glyph_mask)
    assert np.array_equal(clean, page), "an unfittable region was erased anyway"


def test_a_region_that_fits_is_erased_and_lettered(cfg):
    page, glyph_mask = _page_with_text()
    region = _region("Diam.")

    typesetter = Typesetter(cfg)
    layouts = typesetter.plan([region])
    assert region.status is RegionStatus.TRANSLATED
    assert region.id in layouts

    clean = Inpainter(cfg).erase(page, [region], glyph_mask)
    assert not np.array_equal(clean, page), "a fittable region was not erased"
    # The original bar is gone: the balloon interior is back to paper.
    assert clean[95:105, 70:130].min() > 200

    rendered = typesetter.draw_page(clean, [region], layouts)
    assert region.status is RegionStatus.TYPESET
    # Something was drawn where the lettering belongs.
    assert rendered[85:115, 55:145].min() < 128


def test_planning_does_not_letter_anything_by_itself(cfg):
    """plan() must not mark regions typeset -- erasure happens in between."""
    region = _region("Diam.")
    Typesetter(cfg).plan([region])
    assert region.status is RegionStatus.TRANSLATED
