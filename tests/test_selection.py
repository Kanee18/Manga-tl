"""Remembering which pages of a chapter to leave alone.

Only the skipped pages are stored. A chapter nobody has touched has no entry,
and a page dropped into the folder later is translated by default rather than
silently passed over because it was absent when the list was written.
"""

from __future__ import annotations

import json

import pytest

from mangatl import selection


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.setattr(selection, "SELECTION_PATH", tmp_path / "page_selection.json")


def test_a_chapter_nobody_has_touched_skips_nothing(tmp_path):
    assert selection.load_skipped(tmp_path / "ch01") == set()


def test_skips_survive_a_reload(tmp_path):
    chapter = tmp_path / "ch01"
    chapter.mkdir()
    selection.save_skipped(chapter, {"003.jpg", "004.jpg"})
    assert selection.load_skipped(chapter) == {"003.jpg", "004.jpg"}


def test_two_chapters_are_remembered_apart(tmp_path):
    first, second = tmp_path / "ch01", tmp_path / "ch02"
    first.mkdir()
    second.mkdir()
    selection.save_skipped(first, {"001.jpg"})
    selection.save_skipped(second, {"009.jpg"})
    assert selection.load_skipped(first) == {"001.jpg"}
    assert selection.load_skipped(second) == {"009.jpg"}


def test_ticking_everything_again_removes_the_record(tmp_path):
    chapter = tmp_path / "ch01"
    chapter.mkdir()
    selection.save_skipped(chapter, {"003.jpg"})
    selection.save_skipped(chapter, set())
    assert selection.load_skipped(chapter) == set()
    assert not selection.SELECTION_PATH.exists()


def test_a_page_added_later_is_not_skipped(tmp_path):
    """Storing the skipped pages, not the chosen ones, is what buys this."""
    chapter = tmp_path / "ch01"
    chapter.mkdir()
    selection.save_skipped(chapter, {"003.jpg"})
    assert "037.jpg" not in selection.load_skipped(chapter)


def test_a_corrupt_file_is_ignored_rather_than_fatal(tmp_path):
    chapter = tmp_path / "ch01"
    chapter.mkdir()
    selection.SELECTION_PATH.write_text("{ not json", encoding="utf-8")
    assert selection.load_skipped(chapter) == set()


def test_the_file_is_readable_by_a_person(tmp_path):
    chapter = tmp_path / "ch01"
    chapter.mkdir()
    selection.save_skipped(chapter, {"b.jpg", "a.jpg"})
    data = json.loads(selection.SELECTION_PATH.read_text(encoding="utf-8"))
    assert list(data.values())[0] == ["a.jpg", "b.jpg"]  # sorted, not set order
