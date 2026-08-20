"""Tests for the behaviour that broke at least once while this was built.

Each of these is a defect that was found by looking at a rendered page rather
than by reasoning about the code, so each is worth a test that fails cheaply
instead of a re-run of the whole pipeline.
"""

from __future__ import annotations

import numpy as np
import pytest

from mangatl.config import default_config
from mangatl.languages import pair_for
from mangatl.pipeline import detect, ocr, translate

# The pair these guards were written for and tuned against.
EN_ID = pair_for("en-id")
from mangatl.types import Box, Region, RegionStatus


# -- geometry ------------------------------------------------------------


def test_box_padding_is_clamped_to_the_page():
    box = Box(4, 4, 20, 20).padded(10, within=(30, 30))
    assert box == Box(0, 0, 30, 30)


def test_union_covers_every_member():
    boxes = [Box(10, 10, 20, 20), Box(40, 5, 50, 30)]
    assert Box.union(boxes) == Box(10, 5, 50, 30)


# -- line grouping -------------------------------------------------------


def _lines(*specs: tuple[int, int, int, int]) -> list[Box]:
    return [Box(*s) for s in specs]


def test_lines_of_one_balloon_are_joined():
    a, b = Box(100, 100, 180, 112), Box(104, 120, 176, 132)
    assert detect._should_join(a, b, 1.5)


def test_balloons_side_by_side_are_not_joined():
    """The case that produced one region spanning two balloons.

    Their lines interleave vertically and are a few pixels apart
    horizontally, so only the overlap rule separates them.
    """
    left = Box(335, 652, 408, 665)
    right = Box(411, 642, 463, 655)
    assert not detect._should_join(left, right, 1.5)


def test_grouping_is_bounded_by_area_even_when_every_pair_joins():
    """Union-find is transitive, so no pairwise rule can bound a block."""
    cfg = default_config().model_copy(deep=True)
    cfg.detect.max_block_area_ratio = 0.02
    stack = _lines(*[(0, y, 100, y + 10) for y in range(0, 400, 14)])
    page_area = 1000 * 1000
    # Every neighbouring pair passes the join rule, so the whole stack is one
    # group before the cap is applied: 100x410 against a 20000px limit.
    assert Box.union(stack).area > page_area * 0.02
    blocks = detect._group_lines(stack, [], page_area, cfg)
    assert len(blocks) > 1
    for members in blocks:
        assert Box.union([stack[i] for i in members]).area <= page_area * 0.02


def test_weak_threshold_cannot_merge_two_strong_lines():
    """Hysteresis: the weak floor may add regions, never join found ones."""
    plane = np.zeros((60, 200), np.float32)
    plane[20:30, 10:90] = 0.9      # strong line, left
    plane[20:30, 110:190] = 0.9    # strong line, right
    plane[20:30, 90:110] = 0.15    # faint bridge between them
    boxes = detect._lines_from_map(plane, strong=0.30, weak=0.10, min_area=20)
    assert len(boxes) == 2
    assert all(box.width < 100 for box in boxes)


def test_weak_threshold_still_recovers_a_faint_line():
    plane = np.zeros((60, 200), np.float32)
    plane[10:20, 10:90] = 0.9
    plane[40:50, 10:90] = 0.18     # only visible below the strong floor
    boxes = detect._lines_from_map(plane, strong=0.30, weak=0.10, min_area=20)
    assert len(boxes) == 2


def test_a_line_spanning_two_columns_is_cut_at_the_gutter():
    """Side-by-side text must not be welded into one region.

    A line box covering both columns bridges them through union-find, and
    recognition then reads across the pair -- one line from the left, one from
    the right -- producing "IT IF THE WOULD MAIDEN BE" out of two perfectly
    ordinary sentences.
    """
    mask = np.zeros((20, 200), np.uint8)
    mask[4:16, 10:80] = 255     # left column
    mask[4:16, 120:190] = 255   # right column, 40px gutter
    pieces = detect._split_at_gutters(Box(0, 0, 200, 20), mask, 1.2)
    assert len(pieces) == 2
    assert pieces[0].x2 < pieces[1].x1


def test_word_spaces_are_not_mistaken_for_a_gutter():
    mask = np.zeros((20, 200), np.uint8)
    mask[4:16, 10:60] = 255
    mask[4:16, 68:120] = 255    # an 8px space in a 12px-tall line
    assert detect._split_at_gutters(Box(0, 0, 200, 20), mask, 1.2) == [
        Box(0, 0, 200, 20)
    ]


def _two_columns() -> np.ndarray:
    """One line bar covering two columns of lettering, 19px apart.

    Measured off the page this was reported from: the two columns of one
    balloon were separated by 19px where the same bar's word spaces ran to
    2px and its own height was 19px, so no multiple of the line height tells
    the gutter from a space.
    """
    mask = np.zeros((19, 300), np.uint8)
    mask[3:16, 41:92] = 255     # left column, first word
    mask[3:16, 94:135] = 255    # left column, second word -- a 2px space
    mask[3:16, 154:280] = 255   # right column, over a 19px gutter
    return mask


def test_two_columns_too_close_for_the_gap_rule_are_cut_by_the_blocks():
    """The layout model has already drawn a box round each column.

    Left to the gap rule the bar goes through whole, both columns land in one
    region, and the region -- now as wide as the balloon -- is lettered back
    across it, on top of its neighbour.
    """
    line = Box(0, 0, 300, 19)
    blocks = [Box(30, -20, 140, 60), Box(150, -20, 290, 60)]
    assert detect._split_at_gutters(line, _two_columns(), 1.2) == [line]
    pieces = detect._split_across_blocks(line, blocks, _two_columns())
    assert len(pieces) == 2
    assert pieces[0].x2 <= 135 and pieces[1].x1 >= 154


def test_a_bar_the_model_drew_one_block_over_is_left_whole():
    """One block over the bar says nothing about where to cut it."""
    line = Box(0, 0, 300, 19)
    assert detect._split_across_blocks(
        line, [Box(30, -20, 290, 60)], _two_columns()
    ) == [line]


def test_a_bar_no_block_covers_is_left_whole():
    line = Box(0, 0, 300, 19)
    assert detect._split_across_blocks(line, [], _two_columns()) == [line]


def test_blocks_beside_a_bar_rather_than_over_it_do_not_cut_it():
    """Two blocks on the row above are not this bar's columns."""
    line = Box(0, 0, 300, 19)
    blocks = [Box(30, -80, 140, -20), Box(150, -80, 290, -20)]
    assert detect._split_across_blocks(line, blocks, _two_columns()) == [line]


def test_a_band_holding_several_lines_is_split_before_grouping():
    """Large lettering is banded together by the line head.

    The band is then far taller than the page's other lines, and the
    height-similarity rule refuses to group it with them -- so a balloon comes
    apart. "YES, I AM A SEWER RAT!" was detected as "ES", a 99px band, and
    "RAT", and translated as three unrelated fragments.
    """
    mask = np.zeros((200, 200), np.uint8)
    for top in (10, 50, 90):          # three lines inside one tall band
        mask[top : top + 24, 20:180] = 255
    mask[150:162, 20:180] = 255       # an ordinary line elsewhere
    lines = [Box(20, 10, 180, 114), Box(20, 150, 180, 162)]
    out = detect._split_fused_lines(lines, mask)
    assert len(out) == 4
    assert all(box.height < 40 for box in out)


def test_a_genuinely_tall_line_is_left_whole():
    """One big line yields one run in the row profile, so nothing is split."""
    mask = np.zeros((200, 200), np.uint8)
    mask[10:70, 20:180] = 255         # a single 60px line of display type
    mask[150:162, 20:180] = 255
    lines = [Box(20, 10, 180, 70), Box(20, 150, 180, 162)]
    assert detect._split_fused_lines(lines, mask) == lines


def test_stacked_lines_in_a_narrow_box_are_split():
    """A box taller than it is wide holds several lines, and reads as none."""
    mask = np.zeros((60, 30), np.uint8)
    for top in (2, 22, 42):
        mask[top : top + 10, 4:26] = 255
    lines = detect._refine_lines([Box(0, 0, 30, 60)], mask)
    assert len(lines) == 3
    assert [line.height for line in lines] == [10, 10, 10]


# -- recognition ---------------------------------------------------------


def test_hyphenated_break_is_healed():
    assert ocr._join(["A MOMENT AGO, SHE HAD A FIT OF CONVUL-", "SIONS."]) == (
        "A MOMENT AGO, SHE HAD A FIT OF CONVULSIONS."
    )


def test_letterers_dash_is_not_a_hyphen():
    assert ocr._join(["AT THIS RATE--", "SO WHAT"]) == "AT THIS RATE-- SO WHAT"


@pytest.mark.parametrize(
    "text", ["T R D N V", "0 INOY G WOM", "I 5 W", "L E T R"]
)
def test_scattered_letters_are_not_sent_to_the_translator(text):
    assert ocr._is_gibberish(text)


@pytest.mark.parametrize(
    "text",
    [
        "A MOMENT AGO, SHE HAD A FIT OF CONVULSIONS.",
        "IF YOU CAN GRANT MY WISH",
        "AT THIS RATE--",
        # Two of three tokens are single letters, and every one of them is a
        # word. Counting them as noise cost a balloon the middle of its own
        # sentence: "YES, I AM A SEWER RAT!" lost its "I AM A".
        "I AM A",
        "A I O",
    ],
)
def test_real_dialogue_with_short_words_is_kept(text):
    assert not ocr._is_gibberish(text)


def test_a_word_broken_across_lines_is_healed():
    """Balloons are narrow, so letterers break words constantly.

    The recogniser returns the letterer's own line breaks, and a word carried
    over one of them has to be put back together before the translator sees
    it -- asked to translate "CONVUL" and "SIONS." it invents Indonesian for
    both.
    """
    assert ocr._join(["I AM HAVING CONVUL-", "SIONS."]) == "I AM HAVING CONVULSIONS."


def test_the_letterers_dash_is_not_a_broken_word():
    """"AT THIS RATE--" ends a line on purpose and joins nothing."""
    assert ocr._join(["AT THIS RATE--", "WE WILL LOSE."]) == "AT THIS RATE-- WE WILL LOSE."


# -- translation guards --------------------------------------------------


@pytest.mark.parametrize(
    "source,target",
    [
        ("SHE IS BARELY CONSCIOUS.", "SHE IS BARELY CONSCIOUS."),
        ("WE SUMMONED THE DOCTOR.", "The doctor has not arrived"),
        ("ANYTHING", ""),
        # Half-translated: one English function word is one too many. This
        # exact line shipped to a page because the guard demanded two.
        ("Southern Territory House of Sn", "Istana Selatan Terttama House of Sn"),
        ("Maiden Ran Houshun", "Perawan Ran Houshun and her retinue"),
    ],
)
def test_english_coming_back_is_rejected(source, target):
    assert translate.looks_untranslated(source, target, EN_ID)


@pytest.mark.parametrize(
    "target",
    [
        # An untranslated rank is a wording problem, not a failed translation.
        # Rejecting these made the model mangle the word rather than fix it:
        # "...menjadi Empress berikutnya" came back as "MENJADI EMPERIUT".
        "Jika Perawan yang didukung menjadi Empress berikutnya,",
        "Rumah Wilayah Selatan dari Sn",
        "Yang Mulia Crown Princess sudah tiba.",
    ],
)
def test_an_untranslated_rank_is_not_treated_as_untranslated_text(target):
    assert not translate.looks_untranslated("ANY SOURCE TEXT HERE", target, EN_ID)


def test_the_leftover_words_are_named_for_the_retry():
    stray = translate.leftover_source("Istana Selatan Terttama House of Sn", EN_ID)
    assert stray == {"of"}


@pytest.mark.parametrize(
    "source,target",
    [
        ("LADY REIRIN!", "Nona Reirin!"),
        ("SILENCE.", "Diam."),
        ("AT THIS RATE--", "Kalau begini terus--"),
        ("REIRIN", "REIRIN"),  # a bare name may legitimately survive
    ],
)
def test_indonesian_is_accepted(source, target):
    assert not translate.looks_untranslated(source, target, EN_ID)


def test_reply_is_read_through_fencing_and_prose():
    raw = 'Sure!\n```json\n{"r01": {"en": "SILENCE.", "id": "Diam."}}\n```'
    table = translate._parse_object(raw)
    assert translate._unpack(table["r01"]) == ("Diam.", "SILENCE.", False)


def test_a_plain_string_reply_is_still_usable():
    # A bare string says nothing about sound effects, so it is not one.
    assert translate._unpack("Diam.") == ("Diam.", "", False)


# -- settings ------------------------------------------------------------


def test_overrides_replace_only_what_they_name():
    """A settings override must not wipe the rest of its own section."""
    from mangatl.config import _merge

    base = {
        "translate": {"model": "a", "temperature": 0.2, "num_ctx": 8192},
        "detect": {"reading_order": "rtl", "min_area": 260},
    }
    merged = _merge(base, {"translate": {"model": "b"}})
    assert merged["translate"] == {"model": "b", "temperature": 0.2, "num_ctx": 8192}
    assert merged["detect"] == base["detect"]


def test_saved_settings_are_read_back(tmp_path, monkeypatch):
    from mangatl import config as config_module

    monkeypatch.setattr(config_module, "OVERRIDES_PATH", tmp_path / "settings.local.yaml")
    config_module.save_overrides(
        {"detect": {"reading_order": "ltr"}, "translate": {"model": "qwen3:4b-instruct"}}
    )
    cfg = config_module.load_config()
    assert cfg.detect.reading_order == "ltr"
    assert cfg.translate.model == "qwen3:4b-instruct"
    # Untouched values still come from config.yaml, with its measurements.
    assert cfg.detect.line_threshold == 0.30
    assert cfg.detect.min_area == 260


# -- ordering ------------------------------------------------------------


def test_pages_sort_numerically_not_lexically(tmp_path):
    from mangatl.runner import find_pages

    for name in ["page_10.png", "page_2.png", "page_1.png"]:
        (tmp_path / name).write_bytes(b"")
    assert [p.stem for p in find_pages(tmp_path)] == ["page_1", "page_2", "page_10"]


def test_reading_order_is_right_to_left_within_a_band():
    key = detect.TextDetector._reading_key(page_h=1600, order="rtl")
    left = detect._Candidate(Box(100, 500, 200, 560), 12.0, [])
    right = detect._Candidate(Box(800, 505, 900, 565), 12.0, [])
    assert sorted([left, right], key=key) == [right, left]


def test_reading_order_can_be_left_to_right():
    key = detect.TextDetector._reading_key(page_h=1600, order="ltr")
    left = detect._Candidate(Box(100, 500, 200, 560), 12.0, [])
    right = detect._Candidate(Box(800, 505, 900, 565), 12.0, [])
    assert sorted([right, left], key=key) == [left, right]


def test_two_boxes_over_one_column_do_not_cut_it():
    """A fine-tune emits near-duplicate boxes, and a nested pair survives NMS.

    Neither says where a boundary is, and cutting on them would take a bar
    apart in the middle of a word.
    """
    line = Box(0, 0, 300, 19)
    nested = [Box(30, -20, 290, 60), Box(90, -10, 240, 50)]
    assert detect._split_across_blocks(line, nested, _two_columns()) == [line]


def _candidate(box: Box) -> detect._Candidate:
    return detect._Candidate(box, float(box.height), [box])


def test_a_block_inside_another_block_is_dropped():
    """IoU is the wrong measure for a box sitting inside another.

    A 66x106 box inside a 216x215 one scores 0.15, survives every check the
    three candidate paths make against each other, and the same caption is
    read, translated and lettered twice.
    """
    outer = _candidate(Box(1955, 2003, 2171, 2218))
    inner = _candidate(Box(2037, 2030, 2103, 2136))
    kept = detect._drop_nested([inner, outer])
    assert [c.box for c in kept] == [outer.box]


def test_blocks_that_merely_touch_are_both_kept():
    """Two balloons whose boxes clip each other are two balloons."""
    left = _candidate(Box(0, 0, 100, 100))
    right = _candidate(Box(90, 0, 190, 100))
    kept = detect._drop_nested([left, right])
    assert len(kept) == 2
