"""Two faults about lettering the artist drew rather than set.

Both are the same misunderstanding seen from two sides. A balloon is a
container: it decides how much room its lettering has and it guarantees that
what is inside it is speech. Lettering laid straight onto artwork has neither
property, and treating its box as a balloon got both of them wrong.

Read as a boundary, the box the source occupied left no room for a
translation that runs 20-30% longer, so the type shrank instead: "HUH? I'M
DOING WHAT?" was drawn at 24px and lettered back at 13, small enough that the
page had to be zoomed in to read one of its own captions.

Read as a balloon, "GUUUURRRGLE" -- drawn diagonally across a whole panel,
stomach-shaped -- was speech, so the panel was erased and "GUUUURRAGIE DAN"
set across it in the comic face, taking the character underneath with it.
"""

from __future__ import annotations

import numpy as np
import pytest

from mangatl.config import default_config
from mangatl.languages import ENGLISH, JAPANESE, is_stretched_noise, pair_for
from mangatl.pipeline.translate import OllamaTranslator
from mangatl.pipeline.typeset import Typesetter, _scaled
from mangatl.types import Box, Region, RegionKind, RegionStatus


def _latin_config():
    """The English-source config, whatever this machine is set to.

    `default_config()` reads settings.local.yaml, so a developer who has
    switched the app to Japanese would otherwise change what these measure.
    """
    cfg = default_config().model_copy(deep=True)
    cfg.translate.pair = "en-id"
    return cfg


@pytest.fixture(scope="module")
def typesetter():
    return Typesetter(_latin_config())


PAGE = (2106, 1500)


def _drawn_on_art(text: str, box: Box = Box(600, 400, 700, 500)) -> Region:
    """Four short lines laid over artwork, as "HUH? / I'M / DOING / WHAT?" is."""
    pitch = box.height // 4
    return Region(
        id="r00",
        box=box,
        kind=RegionKind.FREE,
        lines=[
            Box(box.x1, box.y1 + i * pitch, box.x2, box.y1 + i * pitch + pitch - 3)
            for i in range(4)
        ],
        text_id=text,
        status=RegionStatus.TRANSLATED,
    )


def _in_a_balloon(text: str, box: Box = Box(600, 400, 700, 500)) -> Region:
    region = _drawn_on_art(text, box)
    region.kind = RegionKind.BALLOON
    region.bubble = box
    return region


# -- what counts as having no edges ------------------------------------------


def test_lettering_over_artwork_has_nothing_holding_it_in():
    assert _drawn_on_art("Hah?").is_unbounded


def test_a_detected_balloon_holds_its_lettering_in():
    assert not _in_a_balloon("Hah?").is_unbounded


def test_a_flat_ground_holds_its_lettering_in():
    """Both witnesses have to agree before a region may spread out.

    A balloon drawn open at one edge is often missed by the layout model, and
    "Flail" over a bare white panel comes back with no balloon anywhere near
    it. The flat ring around the glyphs is what says there is an edge there.
    """
    region = _drawn_on_art("Hah?")
    region.kind = RegionKind.BALLOON
    assert region.bubble is None
    assert not region.is_unbounded


# -- free lettering keeps the size the letterer set --------------------------


def test_free_lettering_takes_more_room_rather_than_smaller_type(typesetter):
    """The reported fault: a 24px caption came back at 13px.

    Nothing was wrong with the fitting. The translation genuinely did not fit
    the box the English had occupied -- and that box was never an edge.
    """
    region = _drawn_on_art("Hah? Apa yang kulakukan?")
    settled = typesetter.layout(region, bounds=PAGE)
    assert settled is not None
    assert settled.size >= typesetter._source_size(region) * 0.98


def test_the_same_words_in_a_balloon_still_shrink_to_fit(typesetter):
    """A balloon is an edge, and the one thing fitting must never do is cross it."""
    held_region = _in_a_balloon("Hah? Apa yang kulakukan?")
    free = typesetter.layout(_drawn_on_art("Hah? Apa yang kulakukan?"), bounds=PAGE)
    held = typesetter.layout(held_region, bounds=PAGE)
    assert held.size < typesetter._source_size(held_region) * 0.98
    assert free.size > held.size


def test_room_is_taken_only_when_it_is_needed(typesetter):
    """Growth is not free -- the room it takes is artwork.

    A translation that already fits where the source sat must be laid out
    exactly as it would have been, so the ladder is never climbed at all.
    """
    short = _drawn_on_art("Ya!")
    grown = typesetter.layout(short, bounds=PAGE)
    ungrown = typesetter._fit(short, "YA!", None, 1.0, PAGE)
    assert grown is not None and ungrown is not None
    assert grown.size == pytest.approx(ungrown.size)
    assert grown.centres == pytest.approx(ungrown.centres)


def test_a_grown_shape_keeps_its_proportions():
    """A column stays a column.

    Stretching only the axis that happens to be short would re-wrap a
    four-line stack into two long lines: the right size in the wrong shape,
    which is not what the page looked like.
    """
    box = np.array([[100, 200], [300, 200], [300, 600], [100, 600]], np.int32)
    grown = _scaled(box, (200.0, 400.0), 1.5, None)
    width = grown[:, 0].max() - grown[:, 0].min()
    height = grown[:, 1].max() - grown[:, 1].min()
    assert width == pytest.approx(200 * 1.5)
    assert height == pytest.approx(400 * 1.5)


def test_growth_stops_at_the_edge_of_the_page():
    box = np.array([[10, 10], [110, 10], [110, 110], [10, 110]], np.int32)
    grown = _scaled(box, (60.0, 60.0), 4.0, (2106, 1500))
    assert grown[:, 0].min() >= 0 and grown[:, 1].min() >= 0
    assert grown[:, 0].max() < 1500 and grown[:, 1].max() < 2106


def test_free_lettering_is_not_held_to_the_global_size_cap(typesetter):
    """`max_font_size` guards against a nonsense line-height measurement.

    Free lettering is where that measurement is at its best -- no balloon
    around it, so the box is drawn tight to the glyphs -- and where display
    type is normal. Held to 44px, a caption drawn at 90 came back at half of
    it.
    """
    cap = _latin_config().typeset.max_font_size
    big = Box(300, 300, 900, 660)
    region = _drawn_on_art("Ya!", big)
    assert typesetter._source_size(region) > cap
    settled = typesetter.layout(region, bounds=PAGE)
    assert settled is not None and settled.size > cap


def test_lettering_in_a_balloon_is_still_held_to_it(typesetter):
    cap = _latin_config().typeset.max_font_size
    settled = typesetter.layout(_in_a_balloon("Ya!", Box(300, 300, 900, 660)),
                                bounds=PAGE)
    assert settled is not None and settled.size <= cap + 0.5


def test_a_block_reported_as_one_line_is_still_held_to_the_cap(typesetter):
    """Because with one line there is no pitch -- only the box.

    The line head fails on exactly the lettering the lifted cap is about: an
    eight-line column of ordinary dialogue came back as one line 443px tall.
    Uncapped it was set at 90px and sprawled across three panels; below the
    cap the same reading is merely optimistic.
    """
    cap = _latin_config().typeset.max_font_size
    region = _drawn_on_art("Oh sayang, aku ingin tidur sangat larut.",
                           Box(300, 300, 524, 743))
    region.lines = [region.box]
    assert typesetter._source_size(region) > cap
    settled = typesetter.layout(region, bounds=PAGE)
    assert settled is not None and settled.size <= cap + 0.5


def test_the_shape_still_has_the_last_word(typesetter):
    """Growth stops; the fitter does not get to letter off the page.

    "GUUUURRRGLE" measures 236px of drawn glyph. Nothing on the page holds it
    in, so it is allowed to ask for that -- and eleven characters at 236px is
    2000px wide, which the panel does not have, so it still comes back
    smaller. Whatever this returns has fitted somewhere.
    """
    region = _drawn_on_art("Guuuurragie dan", Box(510, 1007, 819, 1294))
    region.lines = [region.box]
    settled = typesetter.layout(region, bounds=PAGE)
    assert settled is not None
    assert settled.size < typesetter._source_size(region)


# -- noises drawn into the artwork -------------------------------------------


def test_a_stretched_spelling_is_how_a_letterer_draws_a_noise():
    assert is_stretched_noise("GUUUURRRGLE", ENGLISH)
    assert is_stretched_noise("EEEE", ENGLISH)
    assert is_stretched_noise("WOOOW", ENGLISH)


def test_it_survives_the_damage_the_word_does_not():
    """Which is the whole point of testing the spelling rather than the word.

    Hand lettering is what OCR is worst at, and the noise list can only
    recognise a word it is given: "GUUUURRRGLE" arrived as "GUUUURRAGIE AND",
    which is not gurgle and is not on any list.
    """
    assert is_stretched_noise("GUUUURRAGIE AND", ENGLISH)


def test_a_sentence_with_a_stretched_word_is_somebody_shouting():
    """One stray word is OCR; three is a line of dialogue."""
    assert not is_stretched_noise("I AM SOOOO TIRED", ENGLISH)
    assert not is_stretched_noise("NOOOO! YOU CAN'T DO THAT!", ENGLISH)


def test_the_letterers_dash_is_punctuation_and_not_a_stretch():
    assert not is_stretched_noise("well---", ENGLISH)
    assert not is_stretched_noise("but--", ENGLISH)


def test_ordinary_words_are_left_alone():
    for line in ("Nope", "MAYBE THREE...", "HOW COULD THIS BE?!", "..."):
        assert not is_stretched_noise(line, ENGLISH), line


def test_a_script_without_letters_is_not_asked():
    """A Japanese effect is drawn large and out of column, not spelled long."""
    assert not is_stretched_noise("ドドド", JAPANESE)


class _Offline(OllamaTranslator):
    """The guard, without a server behind it."""

    def __init__(self) -> None:
        cfg = _latin_config()
        self._cfg, self._t = cfg, cfg.translate
        self._pair = pair_for(cfg.translate.pair)


def test_a_stretched_noise_over_artwork_is_left_as_drawn():
    """The reported fault. Erasing it took the character underneath with it."""
    region = _drawn_on_art("")
    region.text_en = "GUUUURRAGIE AND"
    region.status = RegionStatus.READ
    _Offline()._mark_sound_effects([region])
    assert region.status is RegionStatus.SKIPPED_SFX


def test_the_same_spelling_in_a_balloon_is_dialogue():
    """A balloon is what tells a drawn noise from somebody shouting.

    Across three chapters every reading with a stretched run that was inside
    a balloon turned out to be speech, and every one on bare artwork turned
    out to be a noise.
    """
    region = _in_a_balloon("")
    region.text_en = "AAHHH!!"
    region.status = RegionStatus.READ
    _Offline()._mark_sound_effects([region])
    assert region.status is RegionStatus.READ


def test_size_is_not_what_decides():
    """Measured, and it cannot: display lettering is half speech.

    "Nope", "MAYBE THREE..." and "HOW COULD THIS BE?!" are all set three to
    five times the size of the dialogue around them, all laid on artwork, and
    all things a character says. A threshold that caught the noises at that
    size left those in English.
    """
    guard = _Offline()
    for line in ("Nope", "MAYBE THREE...", "HOW COULD THIS BE?!"):
        huge = _drawn_on_art("", Box(300, 300, 900, 660))
        huge.text_en = line
        huge.status = RegionStatus.READ
        guard._mark_sound_effects([huge])
        assert huge.status is RegionStatus.READ, line


def test_asking_for_sound_effects_still_overrides_it():
    region = _drawn_on_art("")
    region.text_en = "GUUUURRAGIE AND"
    region.status = RegionStatus.READ
    guard = _Offline()
    guard._t.translate_sfx = True
    guard._mark_sound_effects([region])
    assert region.status is RegionStatus.READ
