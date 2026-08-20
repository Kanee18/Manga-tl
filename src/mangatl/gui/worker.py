"""Background workers.

Everything expensive happens here and nothing here touches a widget. Loading
the models alone takes several seconds, and a page takes twenty more, so doing
any of it on the GUI thread would freeze the window for the length of a
chapter. Results travel back as signals, which Qt delivers on the GUI thread.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import QObject, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap

from ..config import Config
from ..runner import Cancelled, ChapterRunner, PageResult
from ..types import RegionStatus


def to_pixmap(image: np.ndarray, width: int | None = None) -> QPixmap:
    """A BGR array as a pixmap, optionally scaled to a width."""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if width is not None and rgb.shape[1] > width:
        scale = width / rgb.shape[1]
        rgb = cv2.resize(
            rgb, (width, max(1, int(rgb.shape[0] * scale))), interpolation=cv2.INTER_AREA
        )
    rgb = np.ascontiguousarray(rgb)
    height, wide, _ = rgb.shape
    # copy(), because QImage does not take ownership of the numpy buffer and
    # the array is free to be collected the moment this function returns.
    return QPixmap.fromImage(
        QImage(rgb.data, wide, height, 3 * wide, QImage.Format.Format_RGB888).copy()
    )


class RunnerHost(QObject):
    """Owns the loaded models, and builds them on first use.

    One instance is shared by every job so the models are loaded once per
    session rather than once per page.
    """

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self._cfg = cfg
        self._runner: ChapterRunner | None = None

    @property
    def cfg(self) -> Config:
        return self._cfg

    def apply(self, cfg: Config) -> None:
        """Swap in new settings, discarding models built from the old ones."""
        self.close()
        self._cfg = cfg

    def runner(self) -> ChapterRunner:
        if self._runner is None:
            self._runner = ChapterRunner(cfg=self._cfg)
        return self._runner

    def close(self) -> None:
        if self._runner is not None:
            self._runner.close()
            self._runner = None


class HealthWorker(QThread):
    """Checks Ollama and the model files without blocking startup."""

    done = pyqtSignal(bool, str)

    def __init__(self, host: RunnerHost) -> None:
        super().__init__()
        self._host = host

    def run(self) -> None:
        try:
            ok, message = self._host.runner().check_ready()
        except Exception as exc:  # noqa: BLE001 -- surfaced in the status bar
            ok, message = False, str(exc)
        self.done.emit(ok, message)


class TranslateWorker(QThread):
    """Runs the pipeline over one or more pages."""

    progress = pyqtSignal(str, float)
    page_done = pyqtSignal(object)          # PageResult
    finished_run = pyqtSignal(int, int)     # pages done, pages attempted
    failed = pyqtSignal(str)

    def __init__(self, host: RunnerHost, pages: list[Path], out_dir: Path) -> None:
        super().__init__()
        self._host = host
        self._pages = pages
        self._out_dir = out_dir
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def _should_stop(self) -> bool:
        return self._stop

    def run(self) -> None:
        total = len(self._pages)
        done = 0
        try:
            self.progress.emit("loading models", 0.0)
            runner = self._host.runner()

            for index, path in enumerate(self._pages):
                if self._stop:
                    break

                def page_progress(label: str, fraction: float, i: int = index) -> None:
                    prefix = f"page {i + 1}/{total} -- " if total > 1 else ""
                    self.progress.emit(f"{prefix}{label}", (i + fraction) / total)

                result = runner.translate_page(
                    path, self._out_dir, page_progress, self._should_stop
                )
                result.page.index = index
                self.page_done.emit(result)
                done += 1
        except Cancelled:
            pass
        except Exception as exc:  # noqa: BLE001 -- reported in the UI
            self.failed.emit(f"{exc.__class__.__name__}: {exc}")
            return
        self.finished_run.emit(done, total)


class RegionWorker(QThread):
    """Reads and translates one region drawn by hand.

    Its own thread for the same reason a page run has one: recognition is
    about half a second and the translator is anywhere from five seconds to
    half a minute, and a window that stops repainting for that long looks
    like it has crashed.

    The region is translated on its own rather than with the page around it,
    which is the one place this pipeline knowingly gives up context. It is the
    right trade here: the rest of the page is already translated and settled,
    and re-running it to give one late caption its neighbours would rewrite
    lines the user may have spent the last ten minutes correcting by hand.
    """

    done = pyqtSignal(object)   # the Region, read and translated
    failed = pyqtSignal(str)

    def __init__(self, host: "RunnerHost", image: np.ndarray, region) -> None:
        super().__init__()
        self._host = host
        self._image = image
        self._region = region

    def run(self) -> None:
        try:
            runner = self._host.runner()
            runner.reader.read(self._image, self._region)
            if self._region.status is RegionStatus.READ:
                runner.translator.translate_page([self._region], runner.memory)
        except Exception as exc:  # noqa: BLE001 -- reported in the UI
            self.failed.emit(f"{exc.__class__.__name__}: {exc}")
            return
        self.done.emit(self._region)


def summarise(result: PageResult) -> str:
    """A one-line account of how a page went."""
    if result.page.error:
        return f"{result.page.name}: {result.page.error}"
    parts = [
        f"{result.page.name}: {result.typeset}/{result.attempted} lettered "
        f"in {result.seconds:.1f}s"
    ]
    if result.skipped:
        parts.append(f"({result.skipped} left as art)")
    failures = len(result.problems)
    if failures:
        parts.append(f"-- {failures} need attention")
    return " ".join(parts)
