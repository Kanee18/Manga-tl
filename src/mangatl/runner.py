"""Driving a page, and a chapter, through the stages.

The stage order here is the product of one decision worth stating: erasure
happens *after* translation. The obvious order -- clear the page, then fill it
in -- means every region is erased before anyone knows whether it can be
replaced, so any later failure leaves a blank balloon. Translating first makes
failure harmless: a region that could not be read, translated or fitted is
simply never erased, and the page ships with that one balloon still in English.

Models are built once per `ChapterRunner` and reused for every page, because
loading them costs several seconds and translating a chapter means doing this
twenty times over.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .config import Config, default_config
from .debug import write_page_debug
from .pipeline.classify import classify
from .pipeline.detect import TextDetector
from .pipeline.inpaint import Inpainter
from .pipeline.ocr import TextReader
from .pipeline.translate import OllamaTranslator, TranslationError, TranslationMemory
from .pipeline.typeset import Typesetter
from .types import Page, RegionStatus

PAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".jxl"}

# Called with (stage label, fraction complete in 0..1).
Progress = Callable[[str, float], None]


def find_pages(chapter: Path) -> list[Path]:
    """Source images in a chapter directory, in the order they are read.

    Sorted so that page2 precedes page10, which plain lexical order does not.
    """

    def key(path: Path) -> tuple:
        parts = []
        digits = ""
        for char in path.stem:
            if char.isdigit():
                digits += char
            else:
                if digits:
                    parts.append((1, int(digits)))
                    digits = ""
                parts.append((0, char.lower()))
        if digits:
            parts.append((1, int(digits)))
        return (tuple(parts), path.stem)

    return sorted(
        (p for p in chapter.iterdir() if p.suffix.lower() in PAGE_SUFFIXES and p.is_file()),
        key=key,
    )


def output_dir_for(chapter: Path, cfg: Config) -> Path:
    return chapter.parent / f"{chapter.name}{cfg.output.suffix}"


@dataclass
class PageResult:
    page: Page
    output: Path | None = None
    seconds: float = 0.0

    @property
    def counts(self) -> dict[str, int]:
        return self.page.stats()

    @property
    def typeset(self) -> int:
        return sum(
            1 for r in self.page.regions if r.status is RegionStatus.TYPESET
        )

    @property
    def skipped(self) -> int:
        """Regions deliberately left alone -- sound effects, and noise."""
        return sum(
            1
            for r in self.page.regions
            if r.status in (
                RegionStatus.SKIPPED_SFX,
                RegionStatus.SKIPPED_EMPTY,
                RegionStatus.SKIPPED_SAME,
                RegionStatus.KEPT_ORIGINAL,
            )
        )

    @property
    def attempted(self) -> int:
        """Regions that were supposed to end up lettered.

        Skipped regions are excluded, because leaving hand-drawn sound effects
        and detector noise alone is the correct outcome rather than a miss.
        Counting them made a page that behaved perfectly report "2/4".
        """
        return len(self.page.regions) - self.skipped

    @property
    def problems(self) -> list[str]:
        return [
            f"{r.id}: {r.status.value}" + (f" -- {r.note}" if r.note else "")
            for r in self.page.regions
            if r.status.is_failure
        ]


class Cancelled(RuntimeError):
    """Raised inside a run when the caller asked it to stop."""


@dataclass
class ChapterRunner:
    """Holds the loaded models and translates pages with them."""

    cfg: Config = field(default_factory=default_config)
    memory: TranslationMemory = field(default_factory=TranslationMemory)

    def __post_init__(self) -> None:
        problems = self.cfg.check_assets()
        if problems:
            raise FileNotFoundError("; ".join(problems))
        self._detector: TextDetector | None = None
        self._reader: TextReader | None = None
        self._painter: Inpainter | None = None
        self._typesetter: Typesetter | None = None
        self._translator: OllamaTranslator | None = None

    # -- lazily built models --------------------------------------------

    @property
    def detector(self) -> TextDetector:
        if self._detector is None:
            self._detector = TextDetector(self.cfg)
        return self._detector

    @property
    def reader(self) -> TextReader:
        if self._reader is None:
            self._reader = TextReader(self.cfg)
        return self._reader

    @property
    def painter(self) -> Inpainter:
        if self._painter is None:
            self._painter = Inpainter(self.cfg)
        return self._painter

    @property
    def typesetter(self) -> Typesetter:
        if self._typesetter is None:
            self._typesetter = Typesetter(self.cfg)
        return self._typesetter

    @property
    def translator(self) -> OllamaTranslator:
        if self._translator is None:
            self._translator = OllamaTranslator(self.cfg)
        return self._translator

    def close(self) -> None:
        if self._translator is not None:
            self._translator.close()
            self._translator = None

    def check_ready(self) -> tuple[bool, str]:
        """Whether a run can start: assets present and Ollama reachable."""
        problems = self.cfg.check_assets()
        if problems:
            return False, "; ".join(problems)
        return self.translator.health()

    # -- one page --------------------------------------------------------

    def translate_page(
        self,
        path: Path,
        out_dir: Path | None = None,
        progress: Progress | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> PageResult:
        """Run every stage over one page and write the result."""
        started = time.perf_counter()
        page = Page(path=path)
        result = PageResult(page=page)

        def step(label: str, fraction: float) -> None:
            if should_stop is not None and should_stop():
                raise Cancelled(f"stopped before {label}")
            if progress is not None:
                progress(label, fraction)

        step("loading", 0.0)
        image = _imread(path)
        if image is None:
            page.error = f"could not read {path.name}"
            result.seconds = time.perf_counter() - started
            return result
        page.original = image

        step("detecting text", 0.10)
        detected = self.detector.detect(image)

        step("reading colours", 0.25)
        page.regions = classify(image, detected.regions, self.cfg)

        step("recognising text", 0.35)
        for region in page.regions:
            self.reader.read(image, region)

        step("translating", 0.50)
        self.translator.translate_page(page.regions, self.memory)

        # Fitting decides what may be erased, so it runs first. Erasing a
        # region whose translation turns out not to fit leaves a blank
        # balloon, which reads as damage; leaving it alone leaves English,
        # which reads as an omission.
        step("fitting lettering", 0.75)
        layouts = self.typesetter.plan(page.regions, image.shape[:2])

        step("erasing lettering", 0.82)
        page.glyph_mask = detected.glyph_mask
        page.clean = self.painter.erase(image, page.regions, detected.glyph_mask)

        step("typesetting", 0.90)
        page.rendered = self.typesetter.draw_page(page.clean, page.regions, layouts)

        if out_dir is not None:
            step("writing", 0.97)
            result.output = self._write(page, out_dir)

        if self.cfg.output.debug:
            write_page_debug(
                page, detected.glyph_mask, _debug_dir(path) / path.parent.name
            )

        step("done", 1.0)
        result.seconds = time.perf_counter() - started
        return result

    def _write(self, page: Page, out_dir: Path) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        cfg = self.cfg.output
        target = out_dir / f"{page.name}.{cfg.format}"
        params = (
            [cv2.IMWRITE_JPEG_QUALITY, cfg.jpeg_quality]
            if cfg.format == "jpg"
            else [cv2.IMWRITE_PNG_COMPRESSION, 6]
        )
        _imwrite(target, page.rendered, params)
        return target

    # -- a chapter -------------------------------------------------------

    def translate_chapter(
        self,
        chapter: Path,
        out_dir: Path | None = None,
        progress: Progress | None = None,
        on_page: Callable[[PageResult], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> Iterator[PageResult]:
        """Translate every page of a chapter, yielding results as they finish.

        The translation memory carries across pages, so a name rendered one
        way on the first page is rendered the same way on the last.
        """
        pages = find_pages(chapter)
        if not pages:
            return
        destination = out_dir if out_dir is not None else output_dir_for(chapter, self.cfg)
        total = len(pages)

        for index, path in enumerate(pages):
            def page_progress(label: str, fraction: float, i: int = index) -> None:
                if progress is not None:
                    progress(
                        f"page {i + 1}/{total} -- {label}",
                        (i + fraction) / total,
                    )

            result = self.translate_page(
                path, destination, page_progress, should_stop
            )
            result.page.index = index
            if on_page is not None:
                on_page(result)
            yield result


def _imread(path: Path) -> np.ndarray | None:
    """Read an image whose path may contain non-ASCII characters.

    cv2.imread goes through a narrow-char API on Windows and returns None for
    any path outside the active code page, which a chapter folder named in
    Indonesian or Japanese very often is.
    """
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return image


def _imwrite(path: Path, image: np.ndarray, params: list[int]) -> None:
    ok, buffer = cv2.imencode(path.suffix, image, params)
    if not ok:
        raise OSError(f"could not encode {path}")
    buffer.tofile(str(path))


def _debug_dir(page_path: Path) -> Path:
    from .config import PROJECT_ROOT

    return PROJECT_ROOT / "debug"
