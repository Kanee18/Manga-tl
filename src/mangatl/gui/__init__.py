"""The desktop application."""

from __future__ import annotations

import sys
from pathlib import Path


def _claim_taskbar_identity() -> None:
    """Tell Windows this process is MangaTL and not the interpreter running it.

    Without it the taskbar groups the window under python.exe and shows
    Python's icon there, whatever icon the window itself carries -- so setting
    the logo would appear to do nothing in the one place a logo is most seen.
    Must run before any window exists.
    """
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("MangaTL")
    except Exception:  # noqa: BLE001 -- cosmetic, never worth failing to start
        pass


def main(argv: list[str] | None = None) -> int:
    """Open the window. An optional argument preloads a chapter folder."""
    from PyQt6.QtWidgets import QApplication

    from ..config import default_config
    from .main_window import MainWindow

    from .theme import apply as apply_theme
    from .theme import logo_icon

    argv = list(sys.argv if argv is None else argv)
    _claim_taskbar_identity()
    app = QApplication(argv)
    app.setApplicationName("MangaTL")
    # On the application rather than the window, so the settings panel and
    # every message box carry it too.
    app.setWindowIcon(logo_icon())
    apply_theme(app)

    window = MainWindow(default_config())
    window.show()

    for argument in argv[1:]:
        folder = Path(argument)
        if folder.is_dir():
            window.load_chapter(folder)
            break

    return app.exec()


__all__ = ["main"]
