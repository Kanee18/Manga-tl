"""Language pairs, and the four things each one changes.

The pipeline was built English-to-Indonesian and several of its rules quietly
assumed it. What these hold down is the places where an assumption that was
invisible while there was one pair becomes a bug the moment there are five.
"""

from __future__ import annotations

import pytest

from mangatl.languages import (
    CHINESE, ENGLISH, INDONESIAN, JAPANESE, PAIRS, DEFAULT_PAIR, pair_for,
)
from mangatl.pipeline import ocr
from mangatl.pipeline.translate import (
    leftover_source, looks_untranslated, system_prompt,
)

EN_ID = pair_for("en-id")
JA_ID = pair_for("ja-id")
JA_EN = pair_for("ja-en")
ZH_EN = pair_for("zh-en")


# -- the registry ------------------------------------------------------------


def test_every_offered_pair_builds_a_prompt():
    for pair in PAIRS:
        prompt = system_prompt(pair)
        assert pair.source.name in prompt
        assert pair.target.name in prompt
        assert '"en"' in prompt and '"id"' in prompt, "the JSON contract went missing"


def test_an_unknown_pair_falls_back_rather_than_refusing_to_start(self=None):
    """A typo in a settings file should not stop the app opening."""
    assert pair_for("kl-ng").code == DEFAULT_PAIR
    assert pair_for("").code == DEFAULT_PAIR


def test_a_cjk_source_is_told_about_vertical_lettering():
    assert "vertical" in system_prompt(JA_ID).lower()
    assert "vertical" not in system_prompt(EN_ID).lower()


def test_each_source_is_told_about_its_own_kind_of_ocr_damage():
    """Latin damage is lookalike letters; Japanese is lookalike kanji."""
    assert "kanji" in system_prompt(JA_EN)
    assert "kanji" not in system_prompt(EN_ID)
    assert "Z and D" in system_prompt(EN_ID)


# -- the untranslated guard --------------------------------------------------


def test_a_latin_answer_holding_kana_is_untranslated():
    assert leftover_source("Aku sudah 限界です", JA_ID)
    assert looks_untranslated("もう限界だ", "Aku sudah 限界です", JA_ID)


def test_a_clean_translation_out_of_japanese_passes():
    assert not leftover_source("Aku sudah mencapai batas", JA_ID)
    assert not looks_untranslated("もう限界だ", "Aku sudah mencapai batas", JA_ID)


def test_translating_into_english_does_not_reject_english():
    """The English-marker list is the guard for an English *source*.

    Left in place for Japanese-to-English it would reject every correct
    answer, since a correct answer is entirely English function words.
    """
    assert not leftover_source("What are you saying?", JA_EN)
    assert not looks_untranslated("何を言っている", "What are you saying?", JA_EN)


def test_the_english_marker_guard_still_works_where_it_belongs():
    assert leftover_source("Istana Selatan Terttama House of Sn", EN_ID)
    assert looks_untranslated("SHE IS GONE.", "SHE IS GONE.", EN_ID)


def test_a_han_character_left_in_an_english_answer_is_caught():
    assert leftover_source("This 身体 is mine.", ZH_EN)


def test_an_echoed_cjk_line_is_caught_even_though_it_is_one_token():
    """Counted in characters, since 私は猫である splits into one word."""
    assert looks_untranslated("私は猫である", "私は猫である", JA_ID)


# -- joining recognised lines ------------------------------------------------


def test_japanese_lines_join_without_a_space():
    """Japanese is written without spaces; inserting one breaks a word."""
    assert ocr._join(["おはよう", "ございます"], spaced=False) == "おはようございます"


def test_latin_lines_still_join_with_a_space_and_heal_hyphens():
    assert ocr._join(["AT THIS", "RATE--"], spaced=True) == "AT THIS RATE--"
    assert ocr._join(["CONVUL-", "SIONS."], spaced=True) == "CONVULSIONS."


# -- what counts as readable -------------------------------------------------


@pytest.mark.parametrize("text", ["おはよう", "你好吗", "限界", "ｶﾀｶﾅ"])
def test_cjk_is_not_thrown_away_as_punctuation(text):
    """`^[^0-9A-Za-z]+$` is every CJK line ever recognised.

    Left as it was, a Japanese page came back entirely "nothing but
    punctuation" and not one balloon was translated.
    """
    assert not ocr._JUNK.match(text)


@pytest.mark.parametrize("text", ["...", "！？", "--", "「」"])
def test_punctuation_is_still_thrown_away(text):
    assert ocr._JUNK.match(text)


def test_the_loose_letter_test_does_not_run_on_an_unspaced_script():
    """It counts one-character tokens, and CJK is one token however long."""
    assert not ocr._is_gibberish("私 は 猫 で", spaced=False)
    assert ocr._is_gibberish("T R D N V", spaced=True)
