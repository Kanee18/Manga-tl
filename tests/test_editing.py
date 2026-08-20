"""Corrections made by hand on a page that has already been translated.

The pipeline gets most of a page right and never all of it, so the window can
retype a line, resize it, weight it, paint out what the erase missed, and draw
a region the detector never found. What these hold down is the part that is
easy to get subtly wrong: re-lettering starts from the *cleaned* page, and a
correction has to survive being applied twice.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from mangatl.config import default_config
from mangatl.types import Box, Page, Region, RegionStatus

pytest.importorskip("PyQt6")

from mangatl.gui.editing import PageEditor  # noqa: E402
from mangatl.gui.worker import RunnerHost  # noqa: E402
from mangatl.runner import PageResult  # noqa: E402


@pytest.fixture(scope="module")
def host():
    return RunnerHost(default_config())


def _page(tmp_path) -> PageResult:
    """A white page with one balloon-sized region already lettered."""
    image = np.full((400, 400, 3), 255, np.uint8)
    page = Page(path=tmp_path / "001.png")
    page.original = image.copy()
    page.clean = image.copy()
    page.rendered = image.copy()
    region = Region(
        id="r00",
        box=Box(120, 150, 280, 250),
        lines=[Box(120, 150, 280, 200), Box(120, 200, 280, 250)],
        source_line_height=24.0,
        text_en="LET ME GO!",
        text_id="Lepaskan aku!",
        status=RegionStatus.TYPESET,
        polygon=np.array([[100, 130], [300, 130], [300, 270], [100, 270]], np.int32),
    )
    page.regions.append(region)
    return PageResult(page=page)


@pytest.fixture
def editor(host, tmp_path):
    session = PageEditor(host)
    session.open(_page(tmp_path))
    return session


def _only(editor) -> Region:
    return editor.page.regions[0]


# -- re-lettering ------------------------------------------------------------


def test_a_page_can_be_relettered_twice_without_stacking_ink(editor):
    """`Typesetter.plan` only looks at TRANSLATED regions.

    A region that has been drawn is TYPESET, so a second pass would letter
    nothing at all unless the drawn ones are put back first -- and re-lettering
    starts from the cleaned page, never from what is already on screen, so the
    two passes have to come out identical rather than doubled.
    """
    first = editor.reletter()
    second = editor.reletter()
    assert first is not None and second is not None
    assert np.array_equal(first, second)
    assert _only(editor).status is RegionStatus.TYPESET


def test_lettering_is_drawn_over_the_cleaned_page_not_the_last_render(editor):
    editor.reletter()
    inked = int((editor.page.rendered < 128).sum())
    editor.set_text("r00", "Lepaskan aku sekarang juga!")
    editor.reletter()
    editor.set_text("r00", "Lepaskan aku!")
    editor.reletter()
    assert int((editor.page.rendered < 128).sum()) == inked


# -- retyping ----------------------------------------------------------------


def test_retyping_a_line_the_pipeline_gave_up_on_puts_it_back_in_the_queue(editor):
    """Being able to fix a failure is the whole reason for being able to type."""
    region = _only(editor)
    region.fail(RegionStatus.FAILED_TRANSLATE, "still English")
    assert editor.set_text("r00", "Diterjemahkan dengan tangan")
    assert region.status is RegionStatus.TRANSLATED
    assert not region.note


def test_emptying_a_line_letters_nothing_rather_than_an_empty_string(editor):
    assert editor.set_text("r00", "")
    assert _only(editor).status is RegionStatus.SKIPPED_EMPTY
    editor.reletter()
    assert (editor.page.rendered == 255).all(), "an empty line drew something"


def test_an_unchanged_edit_is_not_a_change(editor):
    assert not editor.set_text("r00", _only(editor).text_id)
    assert not editor.set_scale("r00", 1.0)
    assert not editor.set_bold("r00", False)


# -- size and weight ---------------------------------------------------------


def test_reset_puts_everything_back(editor):
    editor.set_text("r00", "Sesuatu yang lain")
    editor.set_scale("r00", 1.6)
    editor.set_bold("r00", True)
    assert editor.reset("r00")
    region = _only(editor)
    assert region.text_id == "Lepaskan aku!"
    assert region.size_scale == 1.0
    assert not region.bold
    assert region.status is RegionStatus.TYPESET


def test_a_hand_set_region_says_so(editor):
    assert not _only(editor).is_hand_set
    editor.set_scale("r00", 1.2)
    assert _only(editor).is_hand_set


# -- erasing and dropping ----------------------------------------------------


def test_erasing_only_touches_what_the_stroke_covered(editor):
    editor.page.clean[:] = 40           # a dark page, so a fill is visible
    editor.page.clean[0:20, 0:20] = 40
    assert editor.erase([(200.0, 200.0), (210.0, 205.0)], 8)
    assert editor.page.clean[0, 0].tolist() == [40, 40, 40], "erased away from the stroke"


# -- regions selected by hand ------------------------------------------------


def test_a_stroke_over_nothing_is_not_made_into_a_region(editor):
    """Blank paper has no lettering in it, whatever was drawn across it."""
    assert editor.new_region([(10.0, 10.0), (80.0, 40.0)]) is None
    assert len(editor.page.regions) == 1


def test_a_stroke_over_lettering_becomes_a_region_with_its_own_id(editor):
    editor.page.original[60:90, 40:200] = 0     # ink to find
    region = editor.new_region([(60.0, 75.0), (180.0, 75.0)])
    assert region is not None
    assert region.id == "h00" and region.id != "r00"
    assert region.mask is not None and region.mask.any()
    assert editor.region("h00") is region


def test_a_bubble_the_glyph_mask_missed_can_still_be_selected(editor):
    """The one case the box tool exists for, and the one it used to refuse.

    A stroke is drawn precisely where the detector found nothing -- that is
    why someone is drawing it -- so the segmentation head is usually empty
    there too. Believing it made the tool answer "nothing that reads as
    lettering" for the very balloons it is for. The ink is separated from
    its own local ground instead.
    """
    page = editor.page
    page.glyph_mask = np.zeros(page.original.shape[:2], np.uint8)  # found nothing
    page.original[60:90, 40:200] = 0
    region = editor.new_region([(60.0, 75.0), (180.0, 75.0)])
    assert region is not None and region.mask.any()


def test_the_region_is_the_lettering_not_the_stroke(editor):
    """A stroke is a seed, not an outline.

    Swiping across part of a line still means the block: set to the marks it
    actually crossed, a swipe over "GOING ON" would replace two words in the
    middle of a sentence.
    """
    editor.page.original[60:90, 40:200] = 0
    region = editor.new_region([(70.0, 75.0), (100.0, 75.0)])   # a third of it
    assert region is not None
    assert region.box.x1 <= 45 and region.box.x2 >= 195, (
        f"only what the stroke crossed: {tuple(region.box)}"
    )
    assert region.mask.shape == (region.box.height, region.box.width)


def test_selecting_text_does_not_translate_it(editor):
    """Which words the stroke landed on is worth seeing before it runs."""
    editor.page.original[60:90, 40:200] = 0
    region = editor.new_region([(60.0, 75.0), (180.0, 75.0)])
    assert region is not None
    assert not region.text_id
    assert region.status is not RegionStatus.TRANSLATED


# -- a balloon kept in the language it was drawn in ---------------------------


def test_keeping_the_original_puts_the_drawn_pixels_back(editor):
    page = editor.page
    page.original[160:240, 130:270] = 17       # the artist's lettering ...
    page.clean[160:240, 130:270] = 255         # ... which the erase removed
    assert editor.set_keep_original("r00", True)
    editor.reletter()
    assert (page.rendered[160:240, 130:270] == 17).all()
    assert _only(editor).status is RegionStatus.KEPT_ORIGINAL


def test_keeping_the_original_keeps_the_translation_too(editor):
    """It is a switch, not a decision: pressing it again costs one redraw."""
    editor.set_keep_original("r00", True)
    editor.reletter()
    assert _only(editor).text_id == "Lepaskan aku!"

    assert editor.set_keep_original("r00", False)
    lettered = editor.reletter()
    assert _only(editor).status is RegionStatus.TYPESET
    assert not np.array_equal(lettered, np.full_like(lettered, 255))


def test_the_cleaned_page_is_never_touched_by_keeping_the_original(editor):
    """Restoring pixels into `clean` would mean erasing twice to undo it."""
    page = editor.page
    page.original[160:240, 130:270] = 17
    before = page.clean.copy()
    editor.set_keep_original("r00", True)
    editor.reletter()
    assert np.array_equal(page.clean, before)


def test_a_kept_balloon_is_held_out_of_the_fitting(editor):
    """It has no room to compete for, and would push its neighbours aside."""
    editor.set_keep_original("r00", True)
    editor._ready_to_letter()
    assert _only(editor).status is RegionStatus.KEPT_ORIGINAL


def test_keeping_the_original_can_be_taken_back(editor):
    editor.set_keep_original("r00", True)
    assert editor.undo()
    assert not _only(editor).keep_original


def test_asking_to_translate_stops_keeping_the_original(editor):
    """Asking for a translation is asking for it to be lettered."""
    editor.set_keep_original("r00", True)
    assert editor.record_translate("r00")
    assert not _only(editor).keep_original
    assert editor.undo()
    assert _only(editor).keep_original, "both halves of the decision come back"


# -- saving ------------------------------------------------------------------


def test_saving_clears_the_unsaved_mark(editor, tmp_path):
    editor.set_text("r00", "Diubah")
    editor.reletter()
    assert editor.dirty
    written = editor.save(tmp_path / "out")
    assert written is not None and written.exists()
    assert not editor.dirty


def test_drawing_over_a_region_that_is_already_there_finds_it(editor):
    """Outlines only show while a tool is active, so this is easy to do.

    Making a second region of it would read and letter the same words twice.
    """
    found = editor.region_under([(140.0, 200.0), (200.0, 200.0), (260.0, 200.0)])
    assert found is not None and found.id == "r00"


def test_a_stroke_brushing_past_a_region_is_not_pointing_at_it(editor):
    """Half the stroke has to be on it, or a swipe past takes the neighbour."""
    passing = [(float(x), 200.0) for x in range(20, 140, 10)] + [(270.0, 200.0)]
    assert editor.region_under(passing) is None


# -- a region the run never erased -------------------------------------------


def _hand_drawn(editor):
    """A balloon the detector missed: ink still on the cleaned page."""
    page = editor.page
    page.original[60:90, 40:200] = 0
    page.clean[60:90, 40:200] = 0      # the run never erased what it never saw
    region = editor.new_region([(60.0, 75.0), (180.0, 75.0)])
    assert region is not None
    region.text_id = "Halo dunia"
    region.status = RegionStatus.TRANSLATED
    return region


def test_a_hand_selected_region_has_its_own_lettering_rubbed_out(editor):
    """Otherwise the translation is set straight on top of the artist's words.

    The run erases what it goes on to letter and nothing else, so a balloon it
    never saw still holds the English -- and drawing over it produced two
    sentences in one balloon.
    """
    region = _hand_drawn(editor)
    assert editor.erase_lettering(region.id)
    assert editor.page.clean[60:90, 40:200].mean() > 200, "the English is still there"


def test_a_region_the_run_already_erased_is_not_erased_again(editor):
    """A second rebuild over already-rebuilt artwork is not the same picture."""
    _only(editor).status = RegionStatus.TRANSLATED
    assert not editor.erase_lettering("r00")


def test_taking_back_a_translate_puts_the_lettering_back(editor):
    region = _hand_drawn(editor)
    editor.record_translate(region.id)
    editor.erase_lettering(region.id)
    assert editor.page.clean[60:90, 40:200].mean() > 200

    assert editor.undo()
    assert editor.page.clean[60:90, 40:200].mean() < 60, "the pixels stayed erased"
    region.status = RegionStatus.TRANSLATED
    assert editor.erase_lettering(region.id), "and it can be rubbed out again"


def test_a_stroke_does_not_take_the_ink_beside_the_words(editor):
    """A stroke crosses a balloon's own outline, and its neighbour's words.

    Both are dark, both are within a line height of the lettering, and
    neither is what was pointed at.
    """
    page = editor.page
    page.original[60:90, 120:200] = 0     # the words
    page.original[40:120, 20:26] = 0      # a balloon edge running past them
    region = editor.new_region([(130.0, 75.0), (190.0, 75.0)])
    assert region is not None
    assert region.box.x1 > 60, f"the edge came along: {tuple(region.box)}"


@pytest.mark.parametrize("scale", [1, 3])
def test_a_block_of_lines_is_taken_whole_whatever_its_scale(editor, scale):
    """Down the page the gap to close is the leading, which scales with the type.

    A reach that closes six lines on a 770px page leaves them as six blocks on
    a 2560px scan, and a swipe would then take one line of a balloon.
    """
    page = editor.page
    page.original = np.full((300 * scale, 300 * scale, 3), 255, np.uint8)
    page.clean = page.original.copy()
    for index in range(4):                       # four lines, generously led
        top = (40 + index * 20) * scale
        page.original[top : top + 10 * scale, 60 * scale : 200 * scale] = 0

    region = editor.new_region(
        [(80.0 * scale, 45.0 * scale), (150.0 * scale, 45.0 * scale)],  # line one
        radius=3 * scale,
    )
    assert region is not None
    assert region.box.y2 >= 108 * scale, (
        f"only the line the stroke crossed: {tuple(region.box)}"
    )


def test_a_selection_never_claims_room_a_neighbour_letters_into(editor):
    """It is classified on its own, so its shape knows nothing about the page.

    Two blocks in one balloon share an interior and a flood fill from either
    returns the whole of it, so the selection has to be carved against what is
    already there -- `classify.carve`, the same cut the pipeline makes between
    the regions it found itself. Without it the selection takes the balloon
    and letters across its neighbour.
    """
    editor.page.original[60:90, 40:140] = 0
    region = editor.new_region([(60.0, 75.0), (120.0, 75.0)])
    assert region is not None and region.polygon is not None
    neighbour = _only(editor).box.center
    assert cv2.pointPolygonTest(region.polygon, neighbour, False) < 0, (
        "the selection claimed the room r00 letters into"
    )


def test_a_block_stopping_a_line_short_of_the_window_is_not_finished():
    """The next line sits a leading away, so ink that stops short has more.

    Measured on a page here: a window cut off eight pixels above the last line
    of a balloon, the block did not touch the edge, and the search stopped one
    line short with "THAT?!" left in English underneath.
    """
    from mangatl.gui.editing import _runs_off

    window = Box(100, 100, 300, 300)
    page = (1000, 1000)

    near = np.zeros((200, 200), np.uint8)
    near[20:180, 20:180] = 1       # 20px clear of the bottom, a leading of 28
    assert _runs_off(near, window, page, margin=28)

    clear = np.zeros((200, 200), np.uint8)
    clear[50:120, 50:120] = 1      # 80px clear, nothing plausibly beyond
    assert not _runs_off(clear, window, page, margin=28)


def test_a_block_against_the_edge_of_the_page_is_finished():
    """There is nothing beyond it to have missed."""
    from mangatl.gui.editing import _runs_off

    ink = np.zeros((200, 200), np.uint8)
    ink[0:180, 0:180] = 1
    assert not _runs_off(ink, Box(0, 0, 200, 200), (200, 200), margin=28)
