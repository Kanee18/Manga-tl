"""Which pages of a chapter the user has told the app to leave alone.

Only the *exceptions* are stored -- the pages ticked off -- so a chapter that
has never been touched needs no entry, and pages added to a folder later are
translated by default rather than silently skipped.

Kept next to settings.local.yaml rather than inside the chapter folder: the
chapter folder is the user's own material, and a tool has no business leaving
bookkeeping files in it.
"""

from __future__ import annotations

import json
from pathlib import Path

from .config import PROJECT_ROOT

SELECTION_PATH = PROJECT_ROOT / "page_selection.json"


def _key(chapter: Path) -> str:
    try:
        return str(chapter.resolve())
    except OSError:
        return str(chapter)


def _read() -> dict[str, list[str]]:
    if not SELECTION_PATH.exists():
        return {}
    try:
        with SELECTION_PATH.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {
        str(k): [str(n) for n in v]
        for k, v in data.items()
        if isinstance(v, list)
    }


def load_skipped(chapter: Path) -> set[str]:
    """File names in `chapter` that a chapter run should pass over."""
    return set(_read().get(_key(chapter), []))


def save_skipped(chapter: Path, skipped: set[str]) -> None:
    """Record the skipped pages, dropping the entry when there are none."""
    data = _read()
    key = _key(chapter)
    if skipped:
        data[key] = sorted(skipped)
    else:
        data.pop(key, None)

    try:
        if data:
            with SELECTION_PATH.open("w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, ensure_ascii=False)
        elif SELECTION_PATH.exists():
            SELECTION_PATH.unlink()
    except OSError:
        # Losing the tick state is a nuisance, not a reason to fail a run.
        pass
