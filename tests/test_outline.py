"""Lettering laid straight onto artwork carries a stroke.

The letterer puts one there so dark text stays readable over dark art. Taken
for the background it caused three faults at once: the replacement was drawn
in the glyph's own dark ink over dark artwork and could not be read, erasing
painted the stroke's pale colour back over the art in blobs, and the stroke
itself was never reproduced.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from mangatl.config import default_config
from mangatl.pipeline.classify import classify
from mangatl.pipeline.inpaint import Inpainter
from mangatl.pipeline.typeset import Typesetter
from mangatl.types import Box, Region, RegionKind, RegionStatus


@pytest.fixture(scope="module")
def cfg():
    return default_config()


def _outlined_page() -> tuple[np.ndarray, np.ndarray]:
    """Dark, busy artwork with black lettering in a thin white stroke on it.

    The stroke is three pixels, as a letterer would draw it. A thick one would
    fill the whole ring the classifier samples, which would read as a flat
    background -- true of the fixture, not of any real page.
    """
    rng = np.random.default_rng(0)
    page = (rng.integers(20, 70, (300, 300, 3))).astype(np.uint8)  # busy dark art
    mask = np.zeros((300, 300), np.uint8)
    for top in (100, 140):
        cv2.rectangle(page, (83, top + 3), (217, top + 19), (255, 255, 255), -1)
        cv2.rectangle(page, (86, top + 6), (214, top + 16), (25, 25, 25), -1)
        mask[top + 6 : top + 16, 86:214] = 255
    return page, mask


def _plain_balloon() -> tuple[np.ndarray, np.ndarray]:
    """Black lettering on a flat white balloon: no stroke anywhere."""
    page = np.full((300, 300, 3), 255, np.uint8)
    mask = np.zeros((300, 300), np.uint8)
    for top in (100, 140):
        cv2.rectangle(page, (86, top + 6), (214, top + 16), (25, 25, 25), -1)
        mask[top + 6 : top + 16, 86:214] = 255
    return page, mask


def _region() -> Region:
    return Region(
        id="r00",
        box=Box(84, 104, 216, 158),
        lines=[Box(86, 106, 214, 116), Box(86, 146, 214, 156)],
        source_line_height=10.0,
    )


def test_a_stroke_over_artwork_is_found(cfg):
    page, mask = _outlined_page()
    region = _region()
    region.mask = region.box.crop(mask) > 0
    kept = classify(page, [region], cfg)
    assert kept
    assert region.kind is RegionKind.FREE
    assert region.outline_color is not None
    assert np.mean(region.outline_color) > 180, "the stroke is the light colour"
    assert np.mean(region.text_color) < 80, "the ink is still the dark glyph"


def _toned_page() -> tuple[np.ndarray, np.ndarray]:
    """The case that was being missed: a stroke over light screentone.

    Screentone is dots of ink on white paper. Its *median* is near-white, and
    so is the stroke, so the two look identical to any comparison that takes a
    median -- which is why lettering like this was going out with no stroke at
    all, drawn in its own dark ink straight onto the tone.
    """
    page = np.full((300, 300, 3), 255, np.uint8)
    page[::3, ::3] = 40          # 40% screentone
    page[1::3, 2::3] = 40
    mask = np.zeros((300, 300), np.uint8)
    for top in (100, 140):
        cv2.rectangle(page, (83, top + 3), (217, top + 19), (255, 255, 255), -1)
        cv2.rectangle(page, (86, top + 6), (214, top + 16), (25, 25, 25), -1)
        mask[top + 6 : top + 16, 86:214] = 255
    return page, mask


def test_a_stroke_over_screentone_is_found(cfg):
    page, mask = _toned_page()
    region = _region()
    region.mask = region.box.crop(mask) > 0
    classify(page, [region], cfg)
    assert region.kind is RegionKind.FREE
    assert region.outline_color is not None, "a toned panel is artwork, not a balloon"
    assert np.mean(region.outline_color) > 180


def test_a_stroke_no_darker_than_the_ink_is_not_worth_drawing(cfg):
    """It would be a halo the same colour as the letter it surrounds."""
    from mangatl.pipeline.classify import _outline_colour

    region = _region()
    region.text_color = (30, 30, 30)
    assert _outline_colour(region, (35, 35, 35), cfg.classify.min_outline_contrast) is None
    assert _outline_colour(region, (250, 250, 250), cfg.classify.min_outline_contrast) is not None


def test_balloon_lettering_is_not_given_a_stroke(cfg):
    """Inside a balloon the ring outside the glyphs *is* the background."""
    page, mask = _plain_balloon()
    region = _region()
    region.mask = region.box.crop(mask) > 0
    classify(page, [region], cfg)
    assert region.kind is RegionKind.BALLOON
    assert region.outline_color is None


def test_the_stroke_is_drawn_back(cfg):
    """Without it the replacement is dark ink on dark art and unreadable."""
    page = np.full((200, 400, 3), 40, np.uint8)
    region = Region(
        id="r00",
        box=Box(40, 70, 360, 130),
        lines=[Box(40, 70, 360, 100), Box(40, 100, 360, 130)],
        source_line_height=26.0,
        status=RegionStatus.TRANSLATED,
        text_id="Sudah terlambat",
        text_color=(20, 20, 20),
        outline_color=(255, 255, 255),
    )
    typesetter = Typesetter(cfg)
    drawn = typesetter.render_page(page, [region])
    assert region.status is RegionStatus.TYPESET

    changed = np.any(drawn != page, axis=2)
    assert changed.any(), "nothing was drawn"
    light = (cv2.cvtColor(drawn, cv2.COLOR_BGR2GRAY) > 200) & changed
    dark = (cv2.cvtColor(drawn, cv2.COLOR_BGR2GRAY) < 60) & changed
    assert light.sum() > 0, "the stroke is missing"
    assert dark.sum() > 0, "the ink is missing"
    # A halo surrounds the ink rather than replacing it.
    assert light.sum() > dark.sum() * 0.3


def test_without_an_outline_nothing_extra_is_drawn(cfg):
    page = np.full((200, 400, 3), 255, np.uint8)
    region = Region(
        id="r00",
        box=Box(40, 70, 360, 130),
        lines=[Box(40, 70, 360, 100), Box(40, 100, 360, 130)],
        source_line_height=26.0,
        status=RegionStatus.TRANSLATED,
        text_id="Diam",
        text_color=(20, 20, 20),
    )
    drawn = Typesetter(cfg).render_page(page, [region])
    changed = np.any(drawn != page, axis=2)
    gray = cv2.cvtColor(drawn, cv2.COLOR_BGR2GRAY)
    # Everything that changed went darker: no halo was painted.
    assert not ((gray > 250) & changed).any()


def test_erasing_takes_the_stroke_with_the_glyphs(cfg):
    """Left behind, the stroke reads as pale blobs around the replacement."""
    page, mask = _outlined_page()
    region = _region()
    region.mask = region.box.crop(mask) > 0
    classify(page, [region], cfg)
    region.status = RegionStatus.TRANSLATED
    region.text_id = "X"

    clean = Inpainter(cfg).erase(page, [region], mask)
    gray = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY)
    # The white stroke ran from y=100..122 and y=140..162 across x=80..220.
    stroke_rows = gray[100:162, 80:220]
    assert (stroke_rows > 200).mean() < 0.02, "the stroke survived the erase"
