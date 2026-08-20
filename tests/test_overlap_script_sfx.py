"""Three faults that each put the wrong thing on a finished page.

One drew two translations across each other, one translated artwork that was
never in the source language, and one lettered over sound effects the artist
had drawn by hand.
"""

from __future__ import annotations

import numpy as np
import pytest

from mangatl.config import default_config
from mangatl.languages import ENGLISH, INDONESIAN, is_sound_effect, pair_for
from mangatl.pipeline import classify as C
from mangatl.pipeline import ocr as O
from mangatl.pipeline.layout import BUBBLE, TEXT_FREE, TEXT_IN_BUBBLE, _merge_overlapping
from mangatl.pipeline.translate import _reply_schema, _unpack
from mangatl.types import Box, Region, RegionKind, RegionStatus


# -- one piece of lettering is one block -------------------------------------


def test_the_two_text_classes_suppress_each_other():
    """They answer the same question and differ only on the balloon.

    Suppressed one class at a time, a shout both were unsure about arrived as
    an identical pair of boxes, became two regions, was read twice and was
    lettered twice, one on top of the other.
    """
    same = Box(100, 100, 200, 150)
    kept = _merge_overlapping([same, same], [0.77, 0.57], 0.5)
    assert len(kept) == 1
    assert kept[0] == 0, "the more confident box should be the one kept"


def test_a_balloon_may_still_overlap_the_lettering_inside_it():
    """Bubbles are suppressed separately, because that overlap is the point."""
    assert BUBBLE != TEXT_IN_BUBBLE != TEXT_FREE


# -- two regions never letter into the same pixels ---------------------------


def _region(rid: str, box: Box, polygon: Box) -> Region:
    return Region(
        id=rid,
        box=box,
        polygon=np.array(
            [
                [polygon.x1, polygon.y1], [polygon.x2, polygon.y1],
                [polygon.x2, polygon.y2], [polygon.x1, polygon.y2],
            ],
            np.int32,
        ),
    )


def _extent(region: Region) -> Box:
    poly = region.polygon
    return Box(
        int(poly[:, 0].min()), int(poly[:, 1].min()),
        int(poly[:, 0].max()), int(poly[:, 1].max()),
    )


def test_two_regions_sharing_room_are_split_between_them():
    """The flood-fill path partitions; the three fallbacks did not.

    Two adjacent balloons each received room the other was also using, and
    the typesetter -- which fits each region alone -- drew both.
    """
    left = _region("r00", Box(100, 100, 200, 160), Box(80, 80, 320, 180))
    right = _region("r01", Box(260, 100, 360, 160), Box(140, 80, 380, 180))
    C._separate_polygons([left, right])
    assert not _extent(left).intersects(_extent(right)), "still sharing room"


def test_a_region_keeps_enough_room_for_its_own_lettering():
    """A sliver is worse than the overlap: it guarantees a failure to fit."""
    big = _region("r00", Box(100, 100, 300, 200), Box(90, 90, 320, 220))
    tiny = _region("r01", Box(150, 120, 170, 140), Box(90, 90, 320, 220))
    C._separate_polygons([big, tiny])
    for region in (big, tiny):
        extent = _extent(region)
        assert extent.width >= region.box.width * 0.7
        assert extent.height >= region.box.height * 0.7


def test_regions_that_do_not_meet_are_left_alone():
    far = _region("r00", Box(0, 0, 50, 50), Box(0, 0, 60, 60))
    away = _region("r01", Box(500, 500, 550, 550), Box(490, 490, 560, 560))
    before = far.polygon.copy()
    C._separate_polygons([far, away])
    assert np.array_equal(far.polygon, before)


# -- the recogniser reads three scripts; only one of them is the source ------


@pytest.fixture
def english_reader():
    cfg = default_config().model_copy(deep=True)
    cfg.translate.pair = "en-id"
    reader = O.TextReader.__new__(O.TextReader)
    reader._cfg = cfg
    reader._source = pair_for("en-id").source
    reader._spaced = True
    return reader


def test_japanese_read_on_an_english_page_is_dropped(english_reader):
    """それでも was read on an English page and lettered back as "Tidak...".

    The recogniser is multilingual and is told nothing in advance, so it
    returns kana on damaged English lettering and on Chinese banners drawn
    into the artwork. Neither is the script being translated.
    """
    assert english_reader._in_source_script("それでも") == ""
    assert english_reader._in_source_script("乞巧節") == ""


def test_a_stray_character_is_repaired_rather_than_losing_the_line(english_reader):
    """One bad character is damage worth mending, not a reason to lose a balloon."""
    assert english_reader._in_source_script("EXPLAIN氏.") == "EXPLAIN."


def test_english_is_left_exactly_as_read(english_reader):
    assert english_reader._in_source_script("AT THIS RATE--") == "AT THIS RATE--"


def test_a_japanese_source_keeps_its_latin_names():
    """ABYDOS and KANT are names in the original, not misreadings."""
    cfg = default_config().model_copy(deep=True)
    cfg.translate.pair = "ja-id"
    reader = O.TextReader.__new__(O.TextReader)
    reader._cfg = cfg
    reader._source = pair_for("ja-id").source
    reader._spaced = False
    assert reader._in_source_script("ABYDOS") == "ABYDOS"
    assert reader._in_source_script("さすがシロコ") == "さすがシロコ"


# -- sound effects are drawn, not written ------------------------------------


def test_the_reply_carries_a_sound_effect_flag():
    """Whether a line is a drawn noise is a question about the words.

    No measurement of the box answers it: over four pages the real sound
    effects measured *smaller* than the real captions, so the size rule this
    once had could never have worked.
    """
    assert _reply_schema(["r01"])["properties"]["r01"]["required"] == ["en", "id", "sfx"]


def test_an_entry_reports_whether_it_is_a_sound_effect():
    assert _unpack({"en": "CRACK", "id": "CRACK", "sfx": True}) == ("CRACK", "CRACK", True)
    assert _unpack({"en": "SILENCE.", "id": "Diam.", "sfx": False})[2] is False


def test_a_reply_without_the_flag_is_not_a_sound_effect():
    """A smaller model that drops the structure says nothing either way."""
    assert _unpack("Diam.") == ("Diam.", "", False)
    assert _unpack({"en": "SILENCE.", "id": "Diam."})[2] is False


def test_a_marked_sound_effect_keeps_its_own_pixels():
    from mangatl.pipeline.translate import OllamaTranslator

    cfg = default_config().model_copy(deep=True)
    cfg.translate.pair = "en-id"
    translator = OllamaTranslator.__new__(OllamaTranslator)
    translator._cfg = cfg
    translator._t = cfg.translate
    translator._pair = pair_for("en-id")

    region = Region(id="r00", box=Box(0, 0, 80, 40), text_en="CRACK")
    assert translator._accept(region, "CRACK", "CRACK", None, sfx=True)
    assert region.status is RegionStatus.SKIPPED_SFX
    assert not region.text_id, "a sound effect must not be lettered"


def test_sound_effects_are_translated_when_asked_for():
    from mangatl.pipeline.translate import OllamaTranslator

    cfg = default_config().model_copy(deep=True)
    cfg.translate.pair = "en-id"
    cfg.translate.translate_sfx = True
    translator = OllamaTranslator.__new__(OllamaTranslator)
    translator._cfg = cfg
    translator._t = cfg.translate
    translator._pair = pair_for("en-id")

    region = Region(id="r00", box=Box(0, 0, 80, 40), text_en="CRACK")
    assert translator._accept(region, "Retak", "CRACK", None, sfx=True)
    assert region.status is RegionStatus.TRANSLATED


def test_a_noise_is_recognised_without_asking_the_model():
    """The flag is the mechanism; the list is the floor under it.

    A 2B model gets the loud noises right and the quiet ones wrong, and
    "SQUEEZE" over a clasped hand came back as speech -- so the drawing under
    it was erased and the word set again in the comic face.
    """
    assert is_sound_effect("SQUEEZE", ENGLISH)
    assert is_sound_effect("THUD", ENGLISH)
    assert is_sound_effect("CRACK...", ENGLISH)


def test_a_stretched_noise_is_the_same_noise():
    """Letterers draw a noise out to fill the space it is drawn in."""
    assert is_sound_effect("SQUEEEEEZE!!", ENGLISH)   # a doubled letter, stretched
    assert is_sound_effect("THUUUD", ENGLISH)         # a single letter, stretched
    assert is_sound_effect("THUD THUD THUD", ENGLISH)
    assert is_sound_effect("BA-DUMP", ENGLISH)


def test_a_line_that_only_contains_a_noise_is_still_speech():
    """Every word has to be one, or a sentence about squeezing is lost."""
    assert not is_sound_effect("Squeeze it harder!", ENGLISH)
    assert not is_sound_effect("THE ROOM WENT SILENT WITH A CRACK", ENGLISH)


def test_an_interjection_is_something_a_character_says():
    """A one-word reply is speech, however short, and must be translated."""
    for line in ("UGH", "HUH?", "TSK!", "AHEM.", "SILENCE.", "STOP!", "MOTHER..."):
        assert not is_sound_effect(line, ENGLISH), line


def test_a_language_with_no_list_claims_nothing():
    assert not is_sound_effect("SQUEEZE", INDONESIAN)


def test_the_noises_are_settled_before_the_model_is_asked():
    """A line nobody asks about cannot be answered wrongly."""
    from mangatl.pipeline.translate import OllamaTranslator

    cfg = default_config().model_copy(deep=True)
    cfg.translate.pair = "en-id"
    translator = OllamaTranslator.__new__(OllamaTranslator)
    translator._cfg = cfg
    translator._t = cfg.translate
    translator._pair = pair_for("en-id")

    noise = Region(id="r00", box=Box(0, 0, 80, 40), text_en="SQUEEZE",
                   status=RegionStatus.READ)
    speech = Region(id="r01", box=Box(0, 0, 80, 40), text_en="STOP IT!",
                    status=RegionStatus.READ)
    translator._mark_sound_effects([noise, speech])
    assert noise.status is RegionStatus.SKIPPED_SFX
    assert not noise.text_id, "a sound effect must not be lettered"
    assert speech.status is RegionStatus.READ, "speech still goes to the model"


def test_asking_for_sound_effects_translates_them_after_all():
    from mangatl.pipeline.translate import OllamaTranslator

    cfg = default_config().model_copy(deep=True)
    cfg.translate.pair = "en-id"
    cfg.translate.translate_sfx = True
    translator = OllamaTranslator.__new__(OllamaTranslator)
    translator._cfg = cfg
    translator._t = cfg.translate
    translator._pair = pair_for("en-id")

    noise = Region(id="r00", box=Box(0, 0, 80, 40), text_en="SQUEEZE",
                   status=RegionStatus.READ)
    translator._mark_sound_effects([noise])
    assert noise.status is RegionStatus.READ


def test_a_noise_ocr_damaged_is_caught_on_the_repaired_reading():
    """Before the request the list only sees what OCR returned.

    Hand-drawn lettering comes back as "5QUEEZE" often enough to matter, and
    repairing the reading is the first thing the model is asked to do -- so
    the list looks again at what it repaired it to.
    """
    from mangatl.pipeline.translate import OllamaTranslator

    cfg = default_config().model_copy(deep=True)
    cfg.translate.pair = "en-id"
    translator = OllamaTranslator.__new__(OllamaTranslator)
    translator._cfg = cfg
    translator._t = cfg.translate
    translator._pair = pair_for("en-id")

    region = Region(id="r00", box=Box(0, 0, 80, 40), text_en="5QUEEZE")
    assert translator._accept(region, "Remas", "SQUEEZE", None, sfx=False)
    assert region.status is RegionStatus.SKIPPED_SFX
    assert not region.text_id


def test_a_region_against_the_page_edge_does_not_take_the_run_down():
    """Both the window and the box are clamped to the page, separately.

    A region against an edge therefore has a box a pixel or two wider than the
    window recovered around it, and the unclipped assignment this used to make
    raised rather than drawing -- ending the chapter run 20 pages in.
    """
    image = np.full((60, 100, 3), 255, np.uint8)
    # Lettering reaching outside the balloon it was matched to, against the
    # left edge of the page: the window is the balloon plus a margin, clamped,
    # and the box starts two pixels to the left of it.
    box = Box(0, 0, 60, 40)
    region = Region(
        id="r00",
        box=box,
        mask=np.ones((box.height, box.width), bool),
        bubble=Box(10, 0, 40, 50),
    )
    C._interior_mask(image, region, (255, 255, 255), 18)
