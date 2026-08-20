"""Three faults found on the first real Japanese page put through the tool.

All three shipped a page with source text still on it, and none of them was
visible on the English material the pipeline was built against.
"""

from __future__ import annotations

import numpy as np
import pytest

from mangatl.config import default_config
from mangatl.languages import pair_for
from mangatl.pipeline import classify as C
from mangatl.pipeline.translate import looks_untranslated
from mangatl.types import Box, Region, RegionStatus

JA_ID = pair_for("ja-id")
EN_ID = pair_for("en-id")


# -- a name that is the same in both languages is not an echo ----------------


@pytest.mark.parametrize(
    "source,target",
    [
        ("ABYDOS", "ABYDOS"),          # a school name, latin in the original
        ("SERANG BANK", "Serang Bank"),  # differs only in capitalisation
        ("ABIDOS HIGH SCHOOL", "ABIDOS HIGH SCHOOL"),
    ],
)
def test_a_latin_name_returned_unchanged_is_accepted(source, target):
    """Both of these were rejected and shipped untranslated.

    The echo test exists to catch a model that gave up and parroted its
    input. A line with no Japanese in it cannot have been parroted in that
    sense -- it is a name written the same way in both languages, and
    returning it unchanged is the right answer.
    """
    assert not looks_untranslated(source, target, JA_ID)


def test_japanese_returned_unchanged_is_still_caught():
    assert looks_untranslated("その通りだよ", "その通りだよ", JA_ID)
    assert looks_untranslated("さすがシロコ先輩", "さすがシロコ先輩", JA_ID)


def test_a_half_translated_line_is_still_caught():
    assert looks_untranslated("もう限界だ", "Aku sudah 限界", JA_ID)


def test_the_english_source_echo_test_is_unchanged():
    """Three words or more of echoed English is still a failure."""
    assert looks_untranslated("SHE IS BARELY CONSCIOUS.", "SHE IS BARELY CONSCIOUS.", EN_ID)
    assert not looks_untranslated("REIRIN", "REIRIN", EN_ID)


# -- coloured lettering on a colour page is lettering, not a watermark -------


def _page(colour_fraction: float) -> np.ndarray:
    """A mostly-white page with `colour_fraction` of it in colour."""
    page = np.full((400, 400, 3), 255, np.uint8)
    rows = int(400 * colour_fraction)
    if rows:
        page[:rows, :] = (200, 90, 40)   # a saturated blue-ish band
    return page


def _coloured_text() -> tuple[np.ndarray, np.ndarray]:
    crop = np.full((40, 200, 3), 255, np.uint8)
    crop[10:30, 10:190] = (200, 90, 40)      # blue lettering
    glyphs = np.zeros((40, 200), bool)
    glyphs[10:30, 10:190] = True
    return crop, glyphs


def test_a_colour_page_is_recognised_by_how_much_colour_is_on_it():
    """The median was measuring the paper.

    A full-colour comic is still mostly white background, so its median
    saturation came out at 10 and the page was judged greyscale. Every piece
    of coloured lettering was then dropped as a watermark -- on the reported
    page that was the title and all four name plates, and none of them was
    translated because none of them survived to be.
    """
    assert C._page_colour_fraction(_page(0.30)) > C._COLOUR_PAGE
    assert C._page_colour_fraction(_page(0.0)) < C._COLOUR_PAGE


def test_coloured_lettering_on_a_colour_page_is_kept():
    crop, glyphs = _coloured_text()
    assert not C._is_overlay(crop, glyphs, C._page_colour_fraction(_page(0.30)))


def test_a_colour_watermark_on_a_greyscale_page_is_still_dropped():
    """Which is the case the test was written for, and still works."""
    crop, glyphs = _coloured_text()
    assert C._is_overlay(crop, glyphs, C._page_colour_fraction(_page(0.0)))


def test_the_title_of_the_reported_page_survives_classification():
    """The blue title measured 216 saturation and was thrown away."""
    page = _page(0.30)
    page[300:330, 40:280] = (200, 90, 40)
    region = Region(id="r00", box=Box(40, 300, 280, 330))
    region.mask = np.ones((30, 240), bool)
    assert C.classify(page, [region], default_config()), "the title was dropped"


# -- the lettering face cannot draw every character it is given --------------


@pytest.mark.parametrize("symbol", ["☆", "★", "♪", "♡", "→", "、", "。"])
def test_a_symbol_the_comic_face_lacks_is_still_drawn(symbol):
    """Skia draws a missing glyph as `.notdef`, which in this face is nothing.

    The translator kept the ☆ in the Indonesian exactly as it should, and the
    typesetter drew a hole where it went.
    """
    from mangatl.pipeline.fallback import Face
    import skia

    cfg = default_config()
    face = Face(skia.Typeface.MakeFromFile(str(cfg.typeset.font)), 40.0)
    runs = face.runs(f"AB{symbol}CD")
    assert len(runs) == 3, f"{symbol} was not split into its own run"
    piece, font = runs[1]
    assert piece == symbol
    assert font.unicharToGlyph(ord(symbol)), f"no face found that can draw {symbol}"


def test_plain_lettering_stays_in_one_run():
    """Splitting text the comic face can draw would cost its kerning."""
    from mangatl.pipeline.fallback import Face
    import skia

    cfg = default_config()
    face = Face(skia.Typeface.MakeFromFile(str(cfg.typeset.font)), 40.0)
    assert len(face.runs("LEPASKAN AKU!")) == 1


def test_a_line_with_a_symbol_measures_wider_than_one_without():
    """The width was being taken from `.notdef`, which is why the hole fitted."""
    from mangatl.pipeline.fallback import Face
    import skia

    cfg = default_config()
    face = Face(skia.Typeface.MakeFromFile(str(cfg.typeset.font)), 40.0)
    assert face.measureText("SHIROKO☆") > face.measureText("SHIROKO")


def test_a_symbol_puts_ink_on_the_page():
    from mangatl.pipeline.typeset import Typesetter
    import cv2

    cfg = default_config().model_copy(deep=True)
    cfg.typeset.uppercase = True
    typesetter = Typesetter(cfg)

    def ink(text: str) -> int:
        page = np.full((150, 620, 3), 255, np.uint8)
        region = Region(
            id="r00", box=Box(30, 40, 590, 110),
            lines=[Box(30, 40, 590, 110)], source_line_height=40.0,
            text_id=text, status=RegionStatus.TRANSLATED,
            polygon=np.array([[20, 25], [600, 25], [600, 125], [20, 125]], np.int32),
        )
        drawn = typesetter.render_page(page, [region])
        return int((cv2.cvtColor(drawn, cv2.COLOR_BGR2GRAY) < 128).sum())

    assert ink("HEBAT☆") > ink("HEBAT")
