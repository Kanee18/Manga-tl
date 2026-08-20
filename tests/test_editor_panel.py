"""The panel's own behaviour, as distinct from what it edits.

The one that matters is the caret. Re-lettering happens *while* someone is
typing, on a pause, and the panel is refilled from the region when it does --
so refilling the text box put the caret back at the start of the line and the
next keystroke landed in front of everything already typed.
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("PyQt6")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from mangatl.gui.editor import EditorPanel  # noqa: E402
from mangatl.gui.preview import ERASE, NUDGE, PAN, SELECT  # noqa: E402
from mangatl.types import Box, Region, RegionStatus  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def panel(app):
    return EditorPanel()


def _region(text: str = "Lepaskan aku!") -> Region:
    return Region(
        id="r00",
        box=Box(0, 0, 100, 40),
        text_en="LET ME GO!",
        text_id=text,
        status=RegionStatus.TYPESET,
    )


# -- the caret ---------------------------------------------------------------


def test_refreshing_mid_edit_leaves_the_caret_where_it_was(panel):
    region = _region()
    panel.show_region(region)
    panel._text.setPlainText("Lepaskan aku sekarang")
    caret = len("Lepaskan aku sekarang")
    cursor = panel._text.textCursor()
    cursor.setPosition(caret)
    panel._text.setTextCursor(cursor)

    # What re-lettering does: the region now holds what was typed, and the
    # panel is refreshed from it.
    region.text_id = "Lepaskan aku sekarang"
    panel.show_region(region, keep_text=True)

    assert panel._text.textCursor().position() == caret
    assert panel._text.toPlainText() == "Lepaskan aku sekarang"


def test_refreshing_still_updates_everything_around_the_text(panel):
    region = _region()
    panel.show_region(region)
    region.size_scale = 1.4
    region.bold = True
    panel.show_region(region, keep_text=True)
    assert panel._size_label.text() == "140%"
    assert panel._bold.isChecked()


def test_choosing_another_line_does_replace_the_text(panel):
    panel.show_region(_region("Satu"))
    panel.show_region(_region("Dua"))
    assert panel._text.toPlainText() == "Dua"


def test_an_unchanged_refresh_does_not_touch_the_box(panel):
    """setPlainText resets the caret even when the text is identical."""
    region = _region()
    panel.show_region(region)
    cursor = panel._text.textCursor()
    cursor.setPosition(5)
    panel._text.setTextCursor(cursor)
    panel.show_region(region)
    assert panel._text.textCursor().position() == 5


# -- tools -------------------------------------------------------------------


def test_the_tools_are_exclusive(panel):
    assert panel._pan.isChecked()
    panel._erase.setChecked(True)
    assert not panel._pan.isChecked()
    assert sum(
        button.isChecked()
        for button in (panel._pan, panel._nudge_tool, panel._select, panel._erase)
    ) == 1


def test_each_tool_reports_its_own_name(panel, app):
    seen: list[str] = []
    panel.tool_changed.connect(seen.append)
    for button, expected in (
        (panel._nudge_tool, NUDGE), (panel._select, SELECT), (panel._erase, ERASE),
        (panel._pan, PAN),
    ):
        button.setChecked(True)
    assert seen == [NUDGE, SELECT, ERASE, PAN]


def test_the_brush_only_matters_while_erasing(panel):
    panel._select.setChecked(True)
    assert not panel._brush.isEnabled()
    panel._erase.setChecked(True)
    assert panel._brush.isEnabled()


# -- keeping the original ----------------------------------------------------


def test_keeping_the_original_latches_both_ways(panel, app):
    """It changes the page rather than the view of it, so it stays where put.

    Peeking at what is underneath is the O key, held; this is the balloon
    someone has decided to leave in the language it was drawn in.
    """
    panel.show_region(_region())
    seen: list[bool] = []
    panel.keep_original_toggled.connect(seen.append)
    panel._original.setChecked(True)
    panel._original.setChecked(False)
    assert seen == [True, False]


def test_the_button_follows_the_region_it_is_showing(panel, app):
    """Selecting a kept balloon must not read as someone pressing the button."""
    seen: list[bool] = []
    panel.keep_original_toggled.connect(seen.append)
    kept = _region()
    kept.keep_original = True
    panel.show_region(kept)
    assert panel._original.isChecked()
    assert seen == [], "filling the panel is not an edit"


def test_translate_is_a_second_press(panel, app):
    """Drawing a box marks where to look; translating it is the next step."""
    asked: list[int] = []
    panel.translate_requested.connect(lambda: asked.append(1))
    panel.show_region(_region())
    assert panel._translate.isEnabled()
    panel._translate.click()
    assert asked == [1]


def test_translate_is_shut_while_one_is_running(panel, app):
    panel.show_region(_region())
    panel.set_translating(True)
    assert not panel._translate.isEnabled()
    # Selecting another line while it runs must not hand the button back.
    panel.show_region(_region())
    assert not panel._translate.isEnabled()
    panel.set_translating(False)
    assert panel._translate.isEnabled()


def test_nothing_selected_disables_the_line_controls(panel):
    panel.show_region(None)
    for widget in (panel._text, panel._bold, panel._original, panel._centre):
        assert not widget.isEnabled()


# -- undo --------------------------------------------------------------------


def test_the_undo_button_names_what_it_will_take_back(panel):
    panel.set_undo(True, "retype")
    assert panel._undo.isEnabled()
    assert "retype" in panel._undo.text()
    panel.set_undo(False)
    assert not panel._undo.isEnabled()
