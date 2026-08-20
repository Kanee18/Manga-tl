"""Recovering lettering that neither detector head found.

The line head is what separates set lettering from hand-drawn sound effects,
and on some captions it is silent -- measured on one page, a caption scored
0.066 on it while a sound effect on the same page scored 0.367. No threshold
on that head can take the one without the other, so what is left in the glyph
mask is judged on its own shape instead, relative to the lettering the
detector *did* find on the same page.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from mangatl.config import default_config
from mangatl.pipeline import detect as D
from mangatl.types import Box


@pytest.fixture
def detector():
    """A detector without its model: only the pure geometry is exercised."""
    obj = D.TextDetector.__new__(D.TextDetector)
    obj._cfg = default_config()
    return obj


def _bar(canvas, x, y, w, h, angle=0.0):
    patch = np.zeros_like(canvas)
    cv2.rectangle(patch, (x, y), (x + w, y + h), 255, -1)
    if angle:
        centre = (x + w / 2, y + h / 2)
        m = cv2.getRotationMatrix2D(centre, angle, 1.0)
        patch = cv2.warpAffine(patch, m, (canvas.shape[1], canvas.shape[0]))
    canvas |= patch


def test_stroke_width_grows_with_the_stroke():
    thin = np.zeros((80, 200), np.uint8)
    _bar(thin, 20, 30, 160, 4)
    thick = np.zeros((80, 200), np.uint8)
    _bar(thick, 20, 20, 160, 24)
    assert D._stroke_width(thick) > 3 * D._stroke_width(thin)


def test_stroke_width_ignores_an_empty_or_solid_mask():
    assert D._stroke_width(np.zeros((20, 20), np.uint8)) == 0.0
    assert D._stroke_width(np.full((20, 20), 255, np.uint8)) == 0.0


def test_slant_is_near_zero_for_upright_strokes():
    upright = np.zeros((200, 200), np.uint8)
    for x in (20, 70, 120):
        _bar(upright, x, 40, 12, 90)
    assert D._slant(upright) < 5.0


def test_slant_follows_a_leaning_stroke():
    leaning = np.zeros((240, 240), np.uint8)
    for x in (40, 110):
        _bar(leaning, x, 60, 14, 90, angle=30.0)
    assert D._slant(leaning) > 20.0


def _page_with(detector, blobs) -> tuple[list[D._Candidate], np.ndarray]:
    """A page holding one already-found line plus the given loose blobs."""
    mask = np.zeros((900, 900), np.uint8)
    # Lettering the detector already found: thin, upright, 6px strokes.
    found_box = Box(60, 60, 400, 96)
    for x in range(70, 380, 18):
        _bar(mask, x, 66, 6, 24)
    found = [D._Candidate(found_box, 24.0, [found_box])]
    for draw in blobs:
        draw(mask)
    return found, mask


def test_a_caption_at_the_pages_own_weight_is_recovered(detector):
    def caption(mask):
        # Same 6px stroke, upright, far from anything already claimed.
        for row in (500, 560):
            for x in range(200, 520, 18):
                _bar(mask, x, row, 6, 26)

    found, mask = _page_with(detector, [caption])
    got = detector._recover_unclaimed(found, mask, mask.shape)
    # Rows further apart than the grouping reach come back separately; both
    # are lettering, and grouping into blocks happens downstream.
    assert got, "lettering at the page's own weight was not recovered"
    assert all(item.box.y1 > 400 for item in got)
    covered = Box.union([item.box for item in got])
    assert covered.x1 <= 210 and covered.x2 >= 500


def test_a_heavy_sound_effect_is_left_alone(detector):
    def sfx(mask):
        for x in range(220, 520, 56):
            _bar(mask, x, 500, 46, 150)   # far thicker than the page's text

    found, mask = _page_with(detector, [sfx])
    assert detector._recover_unclaimed(found, mask, mask.shape) == []


def test_a_leaning_sound_effect_is_left_alone(detector):
    def sfx(mask):
        for x in range(220, 520, 30):
            _bar(mask, x, 480, 8, 150, angle=32.0)  # right weight, wrong slant

    found, mask = _page_with(detector, [sfx])
    assert detector._recover_unclaimed(found, mask, mask.shape) == []


def test_lettering_beside_a_found_region_is_not_recovered_twice(detector):
    """A region's own stroke and trimmed glyphs lie outside its box."""
    def fringe(mask):
        for x in range(70, 380, 18):
            _bar(mask, x, 100, 6, 22)  # just below the found line

    found, mask = _page_with(detector, [fringe])
    assert detector._recover_unclaimed(found, mask, mask.shape) == []


def test_a_recovered_blocks_height_is_not_taken_for_a_line_height(detector):
    """Everything downstream scales off line height.

    A caption in a heavy stroke merges into one run, so its measured "line
    height" is the whole block. Left unclamped it made the erase reach across
    a sixth of the page.
    """
    def solid(mask):
        cv2.rectangle(mask, (200, 480), (560, 700), 255, -1)
        cv2.rectangle(mask, (210, 490), (550, 690), 0, -1)

    found, mask = _page_with(detector, [solid])
    got = detector._recover_unclaimed(found, mask, mask.shape)
    for item in got:
        assert item.line_height <= 2.0 * 24.0 + 0.01


def test_recovery_can_be_switched_off(detector):
    def caption(mask):
        for row in (500, 560):
            for x in range(200, 520, 18):
                _bar(mask, x, row, 6, 26)

    found, mask = _page_with(detector, [caption])
    assert detector._recover_unclaimed(found, mask, mask.shape)
    detector._cfg = default_config().model_copy(deep=True)
    detector._cfg.detect.recover_max_stroke_ratio = 0.01
    assert detector._recover_unclaimed(found, mask, mask.shape) == []
