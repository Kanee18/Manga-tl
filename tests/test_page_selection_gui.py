"""Unticking a page must actually keep it out of a chapter run.

The persistence is covered in test_selection.py; what matters here is the
wiring -- that the list of pages handed to the worker is the ticked ones, and
that Translate Page still works on a page the user has ticked off, since
choosing a page and pressing that button is an explicit instruction.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import Qt  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from mangatl import selection  # noqa: E402
from mangatl.config import default_config  # noqa: E402
from mangatl.gui import main_window as mw  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def chapter(tmp_path, monkeypatch):
    monkeypatch.setattr(selection, "SELECTION_PATH", tmp_path / "sel.json")
    folder = tmp_path / "ch01"
    folder.mkdir()
    import cv2

    for index in range(1, 6):
        page = np.full((60, 40, 3), 255, np.uint8)
        cv2.imwrite(str(folder / f"{index:03d}.png"), page)
    return folder


class _StubWorker:
    """Captures what a run would have been given, and does nothing."""

    last: "_StubWorker | None" = None

    def __init__(self, host, pages, out_dir):
        self.pages = list(pages)
        _StubWorker.last = self
        for name in ("progress", "page_done", "finished_run", "failed", "finished"):
            setattr(self, name, _Signal())

    def start(self):
        pass

    def stop(self):
        pass

    def wait(self, _ms=0):
        return True


class _Signal:
    def connect(self, _slot):
        pass


@pytest.fixture
def window(app, chapter, monkeypatch):
    monkeypatch.setattr(mw, "TranslateWorker", _StubWorker)
    win = mw.MainWindow(default_config())
    win.load_chapter(chapter)
    app.processEvents()
    _StubWorker.last = None
    return win


def _untick(window, rows):
    for row in rows:
        window._page_list.item(row).setCheckState(Qt.CheckState.Unchecked)


def test_all_pages_are_ticked_to_begin_with(window):
    assert len(window._selected_pages()) == 5


def test_a_chapter_run_only_receives_ticked_pages(window):
    _untick(window, [1, 3])
    window.translate_chapter()
    assert _StubWorker.last is not None
    names = [p.name for p in _StubWorker.last.pages]
    assert names == ["001.png", "003.png", "005.png"]


def test_translate_page_still_works_on_an_unticked_page(window):
    """Selecting a page and pressing the button is an explicit instruction."""
    _untick(window, [2])
    window._page_list.setCurrentRow(2)
    window.translate_page()
    assert _StubWorker.last is not None
    assert [p.name for p in _StubWorker.last.pages] == ["003.png"]


def test_a_run_with_nothing_ticked_does_not_start(window, monkeypatch):
    warned = []
    monkeypatch.setattr(
        mw.QMessageBox, "information", lambda *a, **k: warned.append(a)
    )
    window._set_all_checked(False)
    window.translate_chapter()
    assert _StubWorker.last is None
    assert warned, "the user should be told why nothing happened"


def test_the_ticks_come_back_when_the_chapter_is_reopened(window, chapter):
    _untick(window, [0, 4])
    window.load_chapter(chapter)
    unticked = [
        window._page_list.item(row).text()
        for row in range(window._page_list.count())
        if window._page_list.item(row).checkState() != Qt.CheckState.Checked
    ]
    assert unticked == ["001.png", "005.png"]
