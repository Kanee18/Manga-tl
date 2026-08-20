"""Four faults reported after the first real Japanese chapter.

Three of them put source text on the finished page; the fourth put the tool's
lettering over a drawn prop.
"""

from __future__ import annotations

import numpy as np
import pytest

from mangatl.config import default_config
from mangatl.pipeline import detect as D
from mangatl.pipeline.typeset import Typesetter, _drawable_area, _Area, _says_the_same
from mangatl.types import Box, Region, RegionStatus


def _latin_config():
    cfg = default_config().model_copy(deep=True)
    cfg.translate.pair = "en-id"
    return cfg


@pytest.fixture(scope="module")
def typesetter():
    return Typesetter(_latin_config())


def _tall_balloon(text: str) -> Region:
    """A narrow upright ellipse, as a vertical Japanese balloon becomes."""
    centre_x, centre_y = 300, 300
    angles = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    polygon = np.stack(
        [centre_x + 58 * np.cos(angles), centre_y + 80 * np.sin(angles)], axis=1
    ).astype(np.int32)
    return Region(
        id="r00",
        box=Box(centre_x - 28, centre_y - 58, centre_x + 28, centre_y + 58),
        polygon=polygon,
        lines=[Box(centre_x - 28, centre_y - 58 + i * 23, centre_x + 28,
                   centre_y - 58 + i * 23 + 20) for i in range(5)],
        source_line_height=20.0,
        text_id=text,
        status=RegionStatus.TRANSLATED,
    )


# -- lettering fills the space it has ----------------------------------------


def test_a_balloon_with_room_gets_type_that_uses_it(typesetter):
    """Reported as text far too small for the balloon around it.

    The line count was guessed by wrapping a trial block where a *one-line*
    block would sit, which leaves a multi-line block only the lower half of
    the shape. The wrap ran out of rows, the size was refused, and the search
    walked down until something fit in the half it was given.
    """
    region = _tall_balloon("Benar, kalau tidak mengusulkan cara yang realistis")
    layout = typesetter.layout(region)
    assert layout is not None

    mask, box = _drawable_area(region, _latin_config().typeset.padding_ratio)
    area = _Area(mask, box, region.box.center[0])
    first, last = area.span
    used = len(layout.lines) * layout.line_height
    assert used > (last - first) * 0.4, (
        f"used {used:.0f}px of {last - first}px available at size {layout.size:.1f}"
    )


def test_every_line_is_judged_where_it_is_drawn(typesetter):
    """Lines were wrapped at one position and verified at another.

    In an ellipse that means a line wrapped to the widest row is checked
    against a narrower one and fails a test it should never have been given.
    """
    region = _tall_balloon("Benar, kalau tidak mengusulkan cara yang realistis")
    layout = typesetter.layout(region)
    assert layout is not None

    mask, box = _drawable_area(region, _latin_config().typeset.padding_ratio)
    area = _Area(mask, box, region.box.center[0])
    font = typesetter._font(layout.size)
    for index, line in enumerate(layout.lines):
        y = layout.first_baseline + index * layout.line_height
        _, available = area.line_at(y - layout.line_height * 0.3)
        assert font.measureText(line) <= available + 1.0, f"{line!r} overflows"


def test_an_oscillating_wrap_still_produces_a_layout(typesetter):
    """Where width changes quickly with height the line count flips 5, 4, 5, 4.

    Neither is wrong -- they are two ways to set the same text -- and
    demanding a fixed point threw both away.
    """
    for text in (
        "Benar, kalau tidak mengusulkan cara yang realistis",
        "K-Kalian semua... dalam batasan yang kita bisa...",
        "Memang hebat sekali senior Shiroko yang terhormat",
    ):
        assert typesetter.layout(_tall_balloon(text)) is not None, text


# -- a translation that is its own source is not drawn -----------------------


@pytest.mark.parametrize(
    "source,target",
    [("ABYDOS", "Abydos"), ("SERANG BANK", "Serang Bank"), ("...", "...")],
)
def test_a_translation_that_says_the_same_is_left_as_drawn(source, target):
    """The ID card on a uniform read ABYDOS and was lettered over.

    Erasing the artist's lettering to set the same word in the comic face can
    only make the page worse.
    """
    region = Region(id="r00", box=Box(0, 0, 100, 40), text_en=source, text_id=target)
    assert _says_the_same(region)


def test_a_real_translation_is_still_drawn():
    region = Region(id="r00", box=Box(0, 0, 100, 40), text_en="逃げろ", text_id="Lari!")
    assert not _says_the_same(region)


def test_the_repaired_reading_is_what_is_compared():
    """OCR damage must not make an unchanged line look like a translation."""
    region = Region(
        id="r00", box=Box(0, 0, 100, 40),
        text_en="ABYD0S", text_en_fixed="ABYDOS", text_id="Abydos",
    )
    assert _says_the_same(region)


def test_a_page_skips_rather_than_letters_it(typesetter):
    region = Region(
        id="r00", box=Box(0, 0, 100, 40),
        lines=[Box(0, 0, 100, 40)], source_line_height=20.0,
        text_en="ABYDOS", text_id="Abydos", status=RegionStatus.TRANSLATED,
    )
    assert typesetter.plan([region]) == {}
    assert region.status is RegionStatus.SKIPPED_SAME


# -- lettering the glyph mask cannot see -------------------------------------


def _plate(fg, bg) -> np.ndarray:
    """A name plate: three marks in `fg` on a flat `bg`."""
    crop = np.full((26, 60, 3), bg, np.uint8)
    for x in (8, 25, 42):
        crop[8:18, x : x + 9] = fg
    return crop


def test_light_lettering_on_a_colour_plate_is_found():
    """comic-text-detector is trained on greyscale and returns nothing here.

    Two character name plates were lost this way, and neither was translated
    because neither survived to be a region.
    """
    mask = D._glyphs_by_contrast(_plate((255, 255, 255), (200, 90, 40)))
    assert mask is not None and (mask > 0).any()


def test_dark_lettering_on_a_light_plate_is_found():
    assert D._glyphs_by_contrast(_plate((20, 20, 20), (250, 250, 250))) is not None


def test_flat_artwork_is_not_turned_into_a_region():
    """This runs on boxes the glyph mask already declined.

    Inventing a region on artwork is worse than missing one.
    """
    assert D._glyphs_by_contrast(np.full((40, 60, 3), 128, np.uint8)) is None


def test_one_solid_blob_is_not_lettering():
    """A highlight, a balloon tail, or a piece of the drawing."""
    crop = np.full((40, 60, 3), 250, np.uint8)
    crop[10:30, 10:50] = 20
    assert D._glyphs_by_contrast(crop) is None


def test_a_crop_that_is_nearly_all_ink_is_not_lettering():
    crop = np.full((40, 60, 3), 20, np.uint8)
    crop[0:2, 0:2] = 250
    assert D._glyphs_by_contrast(crop) is None
