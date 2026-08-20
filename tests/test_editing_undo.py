"""Taking corrections back, and moving lettering inside its balloon.

Undo is snapshot-based rather than a log of inverses: every step remembers
what the regions it touched *were*, and the two operations that repaint the
cleaned page also keep the patch they covered. Storing whole pages instead
would be 27 MB a step on a 2560x3595 scan.
"""

from __future__ import annotations

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


@pytest.fixture
def editor(host, tmp_path):
    image = np.full((400, 400, 3), 255, np.uint8)
    page = Page(path=tmp_path / "001.png")
    page.original = image.copy()
    page.clean = image.copy()
    page.rendered = image.copy()
    page.regions.append(
        Region(
            id="r00",
            box=Box(120, 150, 280, 250),
            lines=[Box(120, 150, 280, 200), Box(120, 200, 280, 250)],
            source_line_height=24.0,
            text_en="LET ME GO!",
            text_id="Lepaskan aku!",
            status=RegionStatus.TYPESET,
            polygon=np.array([[100, 130], [300, 130], [300, 270], [100, 270]], np.int32),
        )
    )
    session = PageEditor(host)
    session.open(PageResult(page=page))
    return session


def _only(editor) -> Region:
    return editor.page.regions[0]


# -- undo --------------------------------------------------------------------


def test_a_fresh_page_has_nothing_to_undo(editor):
    assert not editor.can_undo
    assert not editor.undo()


def test_undo_takes_back_a_retype(editor):
    editor.set_text("r00", "Sesuatu yang lain")
    assert editor.can_undo and editor.undo_label == "retype"
    assert editor.undo()
    assert _only(editor).text_id == "Lepaskan aku!"
    assert not editor.can_undo


def test_undo_walks_back_one_step_at_a_time(editor):
    editor.set_text("r00", "Satu")
    editor.set_scale("r00", 1.5)
    editor.set_bold("r00", True)

    editor.undo()
    assert not _only(editor).bold and _only(editor).size_scale == 1.5
    editor.undo()
    assert _only(editor).size_scale == 1.0 and _only(editor).text_id == "Satu"
    editor.undo()
    assert _only(editor).text_id == "Lepaskan aku!"


def test_undo_restores_the_pixels_an_erase_painted_over(editor):
    # A dark mark on light ground: erasing one uniform colour onto the same
    # uniform colour is a no-op, and would pass this test without erasing.
    editor.page.clean[190:225, 190:225] = 20
    before = editor.page.clean.copy()
    editor.erase([(200.0, 200.0), (215.0, 210.0)], 12)
    assert not np.array_equal(editor.page.clean, before), "the erase did nothing"
    assert editor.undo()
    assert np.array_equal(editor.page.clean, before)


def test_undo_puts_back_a_balloon_that_was_kept_in_the_original(editor):
    editor.page.original[160:240, 130:270] = 17
    editor.set_keep_original("r00", True)
    assert editor.undo()
    assert not _only(editor).keep_original
    assert _only(editor).text_id == "Lepaskan aku!"
    assert _only(editor).status is RegionStatus.TYPESET


def test_undoing_a_selected_region_removes_it_rather_than_emptying_it(editor):
    editor.page.original[60:90, 40:200] = 0
    region = editor.new_region([(60.0, 75.0), (180.0, 75.0)])
    assert region is not None and len(editor.page.regions) == 2
    assert editor.undo()
    assert len(editor.page.regions) == 1
    assert editor.region(region.id) is None


def test_the_page_is_clean_again_once_everything_is_undone(editor):
    editor.set_text("r00", "Berubah")
    assert editor.dirty
    editor.undo()
    assert not editor.dirty


def test_the_stack_does_not_grow_without_bound(editor):
    from mangatl.gui.editing import _UNDO_DEPTH

    for index in range(_UNDO_DEPTH + 15):
        editor.set_text("r00", f"Baris {index}")
    assert len(editor._undo) == _UNDO_DEPTH


# -- moving text -------------------------------------------------------------


def test_moving_shifts_where_the_block_is_drawn(host, editor):
    typesetter = host.runner().typesetter
    region = _only(editor)
    region.status = RegionStatus.TRANSLATED
    settled = typesetter.layout(region)

    editor.nudge("r00", 0.0, 25.0, False)
    region.status = RegionStatus.TRANSLATED
    moved = typesetter.layout(region)
    assert moved.first_baseline > settled.first_baseline


def test_a_drag_is_one_thing_to_take_back(editor):
    """Forty undo steps for one drag across the page is not an undo history."""
    editor.nudge("r00", 3.0, 0.0, False)
    for _ in range(20):
        editor.nudge("r00", 3.0, 0.0, True)
    assert len(editor._undo) == 1
    editor.undo()
    assert _only(editor).offset_x == 0.0


def test_recentring_puts_a_moved_block_back(editor):
    editor.nudge("r00", 12.0, -8.0, False)
    region = _only(editor)
    assert region.is_moved
    editor.nudge("r00", -region.offset_x, -region.offset_y, False)
    assert not _only(editor).is_moved


def test_a_move_of_nothing_is_not_a_step(editor):
    assert not editor.nudge("r00", 0.0, 0.0, False)
    assert not editor.can_undo


def test_reset_puts_position_back_too(editor):
    editor.nudge("r00", 20.0, 20.0, False)
    editor.reset("r00")
    assert not _only(editor).is_moved


def test_undo_restores_artwork_a_rebuild_reached_outside_the_stroke(editor, host):
    """LaMa blends its answer across a window wider than the hole.

    A patch cut to the stroke would restore the middle of an undone erase and
    leave a rebuilt ring around it.
    """
    rng = np.random.default_rng(0)
    editor.page.clean[:] = rng.integers(0, 255, editor.page.clean.shape, dtype=np.uint8)
    host.cfg.inpaint.method = "lama"
    before = editor.page.clean.copy()
    try:
        assert editor.erase([(200.0, 200.0), (214.0, 208.0)], 10)
        assert not np.array_equal(editor.page.clean, before)
        assert editor.undo()
    finally:
        host.cfg.inpaint.method = "auto"
    assert np.array_equal(editor.page.clean, before)
