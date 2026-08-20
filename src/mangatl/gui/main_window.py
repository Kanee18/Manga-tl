"""The application window.

The shape follows the two things the tool is asked to do: translate the page
being looked at, or translate the whole chapter. Everything else on screen
exists to answer "did that come out right?" -- the page list carries each
page's outcome, the preview wipes between before and after, and the detail
panel lists what was read and what it became, region by region, including the
ones that failed and why.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PyQt6.QtCore import QEvent, QSize, Qt, QTimer
from PyQt6.QtGui import QAction, QColor, QIcon, QKeySequence
from PyQt6.QtWidgets import (
    QApplication, QFileDialog, QHBoxLayout, QHeaderView, QLabel,
    QListWidget, QListWidgetItem, QMainWindow, QMessageBox, QProgressBar,
    QPushButton, QSizePolicy, QSplitter, QStatusBar, QTabWidget, QTableWidget,
    QTableWidgetItem, QToolBar, QToolButton, QVBoxLayout, QWidget,
)

from ..config import Config, default_config
from ..languages import pair_for
from ..runner import PageResult, find_pages, output_dir_for
from ..selection import load_skipped, save_skipped
from ..types import RegionStatus
from .editing import PageEditor
from .editor import EditorPanel
from .meters import ResourceStrip
from .preview import SELECT_RADIUS, PagePreview
from .telemetry import TelemetryWorker
from .theme import STATUS_COLOURS, Palette, icon, logo_mark
from .worker import (
    HealthWorker, RegionWorker, RunnerHost, TranslateWorker, summarise, to_pixmap,
)

_THUMB_WIDTH = 96

_STATUS_TINT = {
    status: QColor(STATUS_COLOURS[status.value])
    for status in RegionStatus
    if status.value in STATUS_COLOURS
}


class MainWindow(QMainWindow):
    def __init__(self, cfg: Config | None = None) -> None:
        super().__init__()
        self.resize(1400, 900)

        self._host = RunnerHost(cfg or default_config())
        self._chapter: Path | None = None
        self._pages: list[Path] = []
        self._results: dict[Path, PageResult] = {}
        self._worker: TranslateWorker | None = None
        self._health: HealthWorker | None = None
        self._telemetry: TelemetryWorker | None = None
        self._editor = PageEditor(self._host)
        self._region_worker: RegionWorker | None = None
        self._selected = ""
        # Suppresses tick-state writes while the list is being filled or
        # updated. setIcon and setText both raise itemChanged, so without this
        # every thumbnail loaded would rewrite the selection file.
        self._quiet = False

        self._build_ui()
        self._name_for_pair()
        self._update_actions()
        QApplication.instance().installEventFilter(self)
        QTimer.singleShot(50, self._check_health)
        QTimer.singleShot(0, self._start_telemetry)

    # -- construction ---------------------------------------------------

    def _build_ui(self) -> None:
        self._page_list = QListWidget()
        self._page_list.setIconSize(QSize(_THUMB_WIDTH, 136))
        self._page_list.setMaximumWidth(210)
        self._page_list.setSpacing(1)
        self._page_list.setFrameShape(QListWidget.Shape.NoFrame)
        self._page_list.currentRowChanged.connect(self._on_page_selected)
        self._page_list.itemChanged.connect(self._on_item_changed)
        self._page_list.setToolTip(
            "Untick a page to leave it out of Translate Chapter.\n"
            "Translate Page still works on whichever page is selected."
        )

        check_all = QPushButton(icon("check", Palette.muted, 14), " All")
        check_all.setToolTip("Tick every page")
        check_all.clicked.connect(lambda: self._set_all_checked(True))
        check_none = QPushButton(icon("clear", Palette.muted, 14), " None")
        check_none.setToolTip("Untick every page")
        check_none.clicked.connect(lambda: self._set_all_checked(False))
        for button in (check_all, check_none):
            button.setMaximumWidth(84)

        picker = QHBoxLayout()
        picker.setContentsMargins(0, 0, 0, 0)
        picker.addWidget(check_all)
        picker.addWidget(check_none)
        picker.addStretch(1)

        left = QVBoxLayout()
        left.setContentsMargins(0, 0, 0, 0)
        left.addWidget(self._page_list, 1)
        left.addLayout(picker)
        left_panel = QWidget()
        left_panel.setMaximumWidth(220)
        left_panel.setLayout(left)

        self._preview = PagePreview()

        self._detail = QTableWidget(0, 4)
        self._detail.verticalHeader().setVisible(False)
        self._detail.setWordWrap(True)
        self._detail.setShowGrid(False)
        self._detail.setFrameShape(QTableWidget.Shape.NoFrame)
        self._detail.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        self._detail.itemSelectionChanged.connect(self._on_row_selected)
        header = self._detail.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)

        self._log = QListWidget()
        self._log.setFrameShape(QListWidget.Shape.NoFrame)

        tabs = QTabWidget()
        tabs.addTab(self._detail, "Lines")
        tabs.addTab(self._log, "Log")

        right = QSplitter(Qt.Orientation.Vertical)
        right.addWidget(self._preview)
        right.addWidget(tabs)
        right.setSizes([620, 260])

        self._panel = EditorPanel()
        self._wire_panel()

        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(left_panel)
        split.addWidget(right)
        split.addWidget(self._panel)
        split.setStretchFactor(1, 1)

        # Toggles rather than tick boxes: these two change what the preview
        # shows rather than recording a preference, and a pressed button says
        # "this is the view you are in" where a tick says "this is set".
        def toggle(glyph: str, label: str, slot) -> QToolButton:
            button = QToolButton()
            button.setObjectName("toggle")
            button.setCheckable(True)
            button.setIcon(icon(glyph, Palette.muted, 15))
            button.setText(f" {label}")
            button.setToolButtonStyle(
                Qt.ToolButtonStyle.ToolButtonTextBesideIcon
            )
            button.toggled.connect(slot)
            return button

        self._compare = toggle(
            "compare", "Compare with original", self._preview.set_compare
        )
        self._after = toggle("eye", "Show translated", self._preview.set_show_after)
        self._after.setChecked(True)

        self._selection = QLabel("")
        self._selection.setStyleSheet(f"color: {Palette.faint}; font-size: 12px;")

        controls = QHBoxLayout()
        controls.addWidget(self._compare)
        controls.addWidget(self._after)
        controls.addStretch(1)
        controls.addWidget(self._selection)

        self._progress = QProgressBar()
        self._progress.setRange(0, 1000)
        self._progress.setTextVisible(False)
        self._progress.setFixedHeight(4)

        body = QVBoxLayout()
        body.setContentsMargins(12, 10, 12, 8)
        body.setSpacing(8)
        body.addWidget(split, 1)
        body.addLayout(controls)
        body.addWidget(self._progress)

        central = QWidget()
        central.setLayout(body)
        self.setCentralWidget(central)

        self._build_toolbar()

        # The gauges sit as permanent widgets on the right of the status bar,
        # so the left of it stays free for what the run is doing. Both are
        # answers to "what is it up to"; one is the stage, the other the cost.
        self._resources = ResourceStrip()
        bar = QStatusBar()
        bar.setSizeGripEnabled(False)
        bar.addPermanentWidget(self._resources)
        self.setStatusBar(bar)
        self.statusBar().showMessage("Checking Ollama...")

    def _brand(self) -> QWidget:
        """The logo and the name, at the left of the toolbar.

        A label rather than a button: it is the only thing on the bar that
        does nothing when clicked, and giving it a button's hover would
        promise otherwise.
        """
        mark = QLabel()
        mark.setPixmap(logo_mark(26, Palette.text))
        mark.setStyleSheet("background: transparent; padding: 0 2px 0 4px;")

        name = QLabel("MangaTL")
        name.setStyleSheet(
            f"color: {Palette.text}; background: transparent;"
            "font-size: 14px; font-weight: 600; padding-right: 4px;"
        )

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        row.addWidget(mark)
        row.addWidget(name)

        holder = QWidget()
        holder.setStyleSheet("background: transparent;")
        holder.setLayout(row)
        holder.setToolTip("MangaTL -- offline manga translation")
        return holder

    def _build_toolbar(self) -> None:
        bar = QToolBar("Main")
        bar.setMovable(False)
        bar.setIconSize(QSize(17, 17))
        bar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.addToolBar(bar)
        bar.addWidget(self._brand())
        bar.addSeparator()

        def add(glyph: str, label: str, slot, shortcut: str | None = None) -> QAction:
            action = QAction(icon(glyph, Palette.text), label, self)
            if shortcut:
                action.setShortcut(shortcut)
            action.triggered.connect(slot)
            bar.addAction(action)
            return action

        self._open_action = add("folder", "Open Chapter", self.open_chapter)
        self._open_action.setShortcut(QKeySequence.StandardKey.Open)
        bar.addSeparator()

        self._page_action = add("page", "Translate Page", self.translate_page, "Ctrl+T")
        self._chapter_action = add(
            "chapter", "Translate Chapter", self.translate_chapter, "Ctrl+Shift+T"
        )
        # The two that start work are the reason the window is open, so they
        # are the only things in it drawn in the accent colour.
        for action in (self._page_action, self._chapter_action):
            bar.widgetForAction(action).setObjectName("primary")

        self._stop_action = add("stop", "Stop", self.stop)
        bar.addSeparator()

        self._folder_action = add("output", "Output Folder", self.reveal_output)
        self._settings_action = add("settings", "Settings", self.edit_settings)

        # Not on the toolbar: undo belongs to the panel, and putting it in two
        # places invites the question of which one is authoritative. The
        # shortcut is registered on the window so it works wherever focus is.
        self._undo_action = QAction("Undo", self)
        self._undo_action.setShortcut(QKeySequence.StandardKey.Undo)
        self._undo_action.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
        self._undo_action.triggered.connect(self.undo_edit)
        self.addAction(self._undo_action)

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        # Transparent, or it takes the window colour from the sheet and draws
        # a paler rectangle across the middle of the toolbar.
        spacer.setStyleSheet("background: transparent;")
        bar.addWidget(spacer)
        self._device = QLabel("")
        self._device.setStyleSheet(
            f"color: {Palette.muted}; background: {Palette.raised};"
            f"border: 1px solid {Palette.border}; border-radius: 7px;"
            "padding: 4px 10px; font-size: 12px;"
        )
        bar.addWidget(self._device)

    def _name_for_pair(self) -> None:
        """Say which two languages the window is actually working in.

        The title and the two text columns used to read "English" and
        "Indonesian" whatever the pair was set to, so a Japanese chapter came
        back with its source column headed English -- which is the one place
        someone checking a translation looks to see what was read.
        """
        pair = pair_for(self._host.cfg.translate.pair)
        self._detail.setHorizontalHeaderLabels(
            ["", "Status", pair.source.name, pair.target.name]
        )
        self._panel.set_pair(pair)
        self._retitle()

    def _retitle(self) -> None:
        """The chapter being worked on if there is one, else the pair."""
        if self._chapter is not None:
            self.setWindowTitle(f"MangaTL -- {self._chapter.name}")
            return
        pair = pair_for(self._host.cfg.translate.pair)
        self.setWindowTitle(f"MangaTL -- {pair.source.name} to {pair.target.name}")

    # -- chapter ---------------------------------------------------------

    def open_chapter(self) -> None:
        start = str(self._chapter.parent) if self._chapter else ""
        chosen = QFileDialog.getExistingDirectory(self, "Choose a chapter folder", start)
        if chosen:
            self.load_chapter(Path(chosen))

    def load_chapter(self, chapter: Path) -> None:
        pages = find_pages(chapter)
        if not pages:
            QMessageBox.warning(
                self, "No pages",
                f"{chapter} has no images this tool can read.\n\n"
                "Expected .png, .jpg, .webp, .bmp or .tif files.",
            )
            return

        self._chapter = chapter
        self._pages = pages
        self._results.clear()
        self._preview.clear()
        self._detail.setRowCount(0)
        self._page_list.clear()
        self._adopt(None)

        skipped = load_skipped(chapter)
        self._quiet = True  # bulk fill: do not write the tick state back yet
        for path in pages:
            item = QListWidgetItem(path.name)
            item.setData(Qt.ItemDataRole.UserRole, str(path))
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                Qt.CheckState.Unchecked
                if path.name in skipped
                else Qt.CheckState.Checked
            )
            self._page_list.addItem(item)
        self._quiet = False

        self._retitle()
        self.statusBar().showMessage(
            f"{len(pages)} pages in {chapter.name}. "
            f"Output goes to {output_dir_for(chapter, self._host.cfg).name}"
        )
        self._page_list.setCurrentRow(0)
        self._update_selection_label()
        QTimer.singleShot(0, self._load_thumbnails)
        self._update_actions()

    # -- which pages a chapter run covers ---------------------------------

    def _selected_pages(self) -> list[Path]:
        return [
            self._pages[row]
            for row in range(self._page_list.count())
            if self._page_list.item(row).checkState() == Qt.CheckState.Checked
        ]

    def _skipped_names(self) -> set[str]:
        return {
            self._pages[row].name
            for row in range(self._page_list.count())
            if self._page_list.item(row).checkState() != Qt.CheckState.Checked
        }

    def _on_item_changed(self, item: QListWidgetItem) -> None:
        """Persist the tick state. Also fires for icon and text updates."""
        if self._quiet or self._chapter is None:
            return
        save_skipped(self._chapter, self._skipped_names())
        self._update_selection_label()
        self._update_actions()

    def _set_all_checked(self, checked: bool) -> None:
        if not self._pages:
            return
        state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        self._quiet = True
        for row in range(self._page_list.count()):
            self._page_list.item(row).setCheckState(state)
        self._quiet = False
        self._on_item_changed(self._page_list.item(0))

    def _update_selection_label(self) -> None:
        total = len(self._pages)
        chosen = len(self._selected_pages())
        if not total:
            self._selection.setText("")
        elif chosen == total:
            self._selection.setText(f"All {total} pages will be translated")
        else:
            self._selection.setText(
                f"{chosen} of {total} pages will be translated "
                f"({total - chosen} skipped)"
            )

    def _load_thumbnails(self) -> None:
        self._quiet = True
        try:
            for row in range(self._page_list.count()):
                item = self._page_list.item(row)
                if not item.icon().isNull():
                    continue
                path = Path(item.data(Qt.ItemDataRole.UserRole))
                image = _read(path)
                if image is not None:
                    item.setIcon(QIcon(to_pixmap(image, _THUMB_WIDTH)))
                QApplication.processEvents()
        finally:
            self._quiet = False

    def _wire_panel(self) -> None:
        """Both directions: the panel edits the page, the page fills the panel."""
        panel, preview = self._panel, self._preview
        panel.tool_changed.connect(preview.set_tool)
        panel.brush_changed.connect(preview.set_brush)
        panel.outlines_toggled.connect(preview.set_show_regions)

        panel.text_edited.connect(
            lambda text: self._edit(self._editor.set_text(self._selected, text))
        )
        panel.size_scaled.connect(
            lambda scale: self._edit(self._editor.set_scale(self._selected, scale))
        )
        panel.bold_toggled.connect(
            lambda on: self._edit(self._editor.set_bold(self._selected, on))
        )
        panel.reset_requested.connect(
            lambda: self._edit(self._editor.reset(self._selected))
        )
        panel.save_requested.connect(self.save_page)
        panel.undo_requested.connect(self.undo_edit)
        panel.keep_original_toggled.connect(
            lambda on: self._edit(
                self._editor.set_keep_original(self._selected, on)
            )
        )
        panel.translate_requested.connect(self.translate_region)
        panel.recentre_requested.connect(self._recentre)
        preview.text_nudged.connect(self._on_nudged)

        preview.region_picked.connect(self._select_region)
        preview.text_selected.connect(self._on_text_selected)
        preview.stroke_painted.connect(
            lambda points, radius: self._edit(self._editor.erase(points, radius))
        )

    # -- editing ---------------------------------------------------------

    def _edit(self, changed: bool) -> None:
        """Re-letter the page after a correction, keeping where you were."""
        if not changed or not self._editor.editable:
            return
        page = self._editor.page
        rendered = self._editor.reletter()
        if rendered is None:
            return
        self._preview.set_page(page.original, rendered, keep_view=True)
        self._preview.set_regions(page.regions)
        self._refill_detail()
        # Re-lettering happens while someone is typing, and refilling the text
        # box under a live caret puts it back at the start of the line.
        self._panel.show_region(
            self._editor.region(self._selected), keep_text=self._panel.editing_text
        )
        self._panel.set_undo(self._editor.can_undo, self._editor.undo_label)
        self._panel.set_saveable(True)

    def _select_region(self, region_id: str) -> None:
        self._panel.commit()
        self._selected = region_id
        self._preview.set_selected(region_id)
        self._preview.show_original(None, None)
        self._panel.show_region(self._editor.region(region_id))
        self._highlight_row(region_id)

    def undo_edit(self) -> None:
        """Take back the last correction (Ctrl+Z)."""
        self._panel.commit()
        if not self._editor.can_undo:
            return
        label = self._editor.undo_label
        if self._editor.undo():
            self._edit(True)
            self.statusBar().showMessage(f"Undid {label}")
            if self._editor.region(self._selected) is None:
                self._select_region("")

    def _peek_original(self, on: bool) -> None:
        """Show the selected balloon as it was drawn, while O is held.

        Separate from Keep original, which changes the page. This only
        answers "what was under this?", and that answer is wanted while you
        are looking at it and not after -- so it is a key held down, and
        letting go puts the translation straight back.
        """
        region = self._editor.region(self._selected)
        page = self._editor.page
        if not on or region is None or page is None or page.original is None:
            self._preview.show_original(None, None)
            return
        box = region.box.padded(8, within=page.original.shape[:2])
        self._preview.show_original(box.crop(page.original), box)

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 -- Qt naming
        """Hold O to peek at the original, wherever the focus happens to be.

        Filtered on the application rather than handled on the window, because
        a key only reaches the window when nothing else wants it -- and the
        page list wants letters, for type-ahead. Someone peeking at a balloon
        has just clicked a row in the list as often as not, and a shortcut
        that works from one side of the window and not the other reads as
        broken rather than as focused.
        """
        kind = event.type()
        if kind in (QEvent.Type.KeyPress, QEvent.Type.KeyRelease):
            if event.key() == Qt.Key.Key_O and not event.isAutoRepeat():
                # Not while typing: an O belongs in the translation.
                if self._panel.editing_text:
                    return False
                self._peek_original(kind is QEvent.Type.KeyPress)
                return True
        elif kind is QEvent.Type.WindowDeactivate:
            # Letting go outside the window never arrives, and the page would
            # be left showing lettering it does not have.
            self._peek_original(False)
        return super().eventFilter(watched, event)

    def _on_nudged(self, dx: float, dy: float, merge: bool) -> None:
        self._edit(self._editor.nudge(self._selected, dx, dy, merge))

    def _recentre(self) -> None:
        region = self._editor.region(self._selected)
        if region is not None:
            self._edit(
                self._editor.nudge(
                    self._selected, -region.offset_x, -region.offset_y, False
                )
            )

    def _highlight_row(self, region_id: str) -> None:
        blocked = self._detail.blockSignals(True)
        try:
            for row in range(self._detail.rowCount()):
                cell = self._detail.item(row, 0)
                if cell is not None and cell.text() == region_id:
                    self._detail.selectRow(row)
                    return
            self._detail.clearSelection()
        finally:
            self._detail.blockSignals(blocked)

    def _on_row_selected(self) -> None:
        row = self._detail.currentRow()
        cell = self._detail.item(row, 0) if row >= 0 else None
        if cell is not None and cell.text() != self._selected:
            self._select_region(cell.text())

    def _on_text_selected(self, points) -> None:
        """Mark lettering the detector never found, and select it.

        Selecting does not translate. Which words the stroke landed on is a
        judgement worth seeing before twenty seconds are spent on it -- it
        lands on the wrong balloon, or takes the neighbour too -- so the block
        is outlined and selected, and Translate is a second press.
        """
        if not self._editor.editable or not points:
            return
        # Lettering that was found already, drawn over again. The outlines
        # only show while a tool is active, so this is an easy thing to do and
        # the result would be the same words read and lettered twice.
        found = self._editor.region_under(points)
        if found is not None:
            self._select_region(found.id)
            self._note(f"{found.id} is already there")
            self.statusBar().showMessage(
                f"{found.id} covers that already. Press Translate to read it again."
            )
            return
        region = self._editor.new_region(points, SELECT_RADIUS)
        if region is None:
            self._note("Nothing that reads as lettering under that stroke")
            self.statusBar().showMessage(
                "Nothing there to read -- draw across the words themselves"
            )
            return
        self._edit(True)
        self._select_region(region.id)
        self._note(f"{region.id} selected -- press Translate to read it")
        self.statusBar().showMessage(
            f"{region.id} selected. Press Translate to read and translate it."
        )

    def translate_region(self) -> None:
        """Read and translate the selected region, on a thread of its own."""
        region = self._editor.region(self._selected)
        page = self._editor.page
        if (
            region is None
            or page is None
            or page.original is None
            or self._region_worker is not None
            # A page or chapter run holds the same recogniser and the same
            # Ollama model, and two callers into one ONNX session is not a
            # thing to find out about halfway through a chapter.
            or self._worker is not None
        ):
            self.statusBar().showMessage(
                "Wait for the run to finish before translating one region"
                if self._worker is not None
                else "Already reading a region"
            )
            return
        self._panel.commit()
        self._editor.record_translate(region.id)
        self._panel.set_translating(True)
        self._panel.show_region(region)
        self._note(f"Reading {region.id}...")
        self.statusBar().showMessage(f"Reading and translating {region.id}...")
        self._region_worker = RegionWorker(self._host, page.original, region)
        self._region_worker.done.connect(self._on_region_ready)
        self._region_worker.failed.connect(self._on_failed)
        self._region_worker.finished.connect(self._clear_region_worker)
        self._region_worker.start()

    def _on_region_ready(self, region) -> None:
        self._note(f"{region.id}: {region.status.value} -- {region.text_id or region.text_en}")
        self.statusBar().showMessage(f"{region.id} {region.status.value}")
        # Its own lettering first. The run erased what it lettered and nothing
        # else, so a balloon it never saw still has the artist's words in it,
        # and the translation would be set straight on top of them.
        self._editor.erase_lettering(region.id)
        self._edit(True)
        self._select_region(region.id)

    def _clear_region_worker(self) -> None:
        self._region_worker = None
        self._panel.set_translating(False)

    def save_page(self) -> None:
        if self._chapter is None or not self._editor.editable:
            return
        target = self._editor.save(output_dir_for(self._chapter, self._host.cfg))
        if target is not None:
            self._note(f"Saved {target.name}")
            self.statusBar().showMessage(f"Saved {target}")
            self._panel.set_saveable(False)

    # -- selection -------------------------------------------------------

    def _current_page(self) -> Path | None:
        row = self._page_list.currentRow()
        return self._pages[row] if 0 <= row < len(self._pages) else None

    def _on_page_selected(self, row: int) -> None:
        path = self._current_page()
        if path is None:
            return
        result = self._results.get(path)
        original = result.page.original if result else _read(path)
        rendered = result.page.rendered if result else None
        self._preview.set_page(original, rendered)
        self._compare.setEnabled(rendered is not None)
        self._adopt(result)
        self._update_actions()

    def _adopt(self, result: PageResult | None) -> None:
        """Hand a finished page to the editor, the canvas and the panel."""
        self._editor.open(result)
        self._preview.set_regions(result.page.regions if result else [])
        self._fill_detail(result)
        self._select_region("")
        self._panel.set_undo(self._editor.can_undo, self._editor.undo_label)
        self._panel.set_saveable(self._editor.editable)

    def _refill_detail(self) -> None:
        self._fill_detail(self._editor.result)

    def _fill_detail(self, result: PageResult | None) -> None:
        regions = result.page.regions if result else []
        blocked = self._detail.blockSignals(True)
        self._detail.setRowCount(len(regions))
        for row, region in enumerate(regions):
            cells = [
                region.id,
                region.status.value,
                region.text_en_fixed or region.text_en,
                region.text_id or (region.note or ""),
            ]
            for column, text in enumerate(cells):
                cell = QTableWidgetItem(text)
                cell.setFlags(cell.flags() & ~Qt.ItemFlag.ItemIsEditable)
                if region.status.is_failure:
                    cell.setForeground(QColor(Palette.danger))
                elif region.status in _STATUS_TINT:
                    cell.setForeground(_STATUS_TINT[region.status])
                self._detail.setItem(row, column, cell)
        self._detail.resizeRowsToContents()
        self._detail.blockSignals(blocked)
        if self._selected:
            self._highlight_row(self._selected)

    # -- running ---------------------------------------------------------

    def translate_page(self) -> None:
        path = self._current_page()
        if path is not None:
            self._start([path])

    def translate_chapter(self) -> None:
        chosen = self._selected_pages()
        if not chosen:
            QMessageBox.information(
                self,
                "Nothing selected",
                "Every page is unticked, so a chapter run would do nothing.\n\n"
                "Tick the pages you want translated, or press All.",
            )
            return
        skipped = len(self._pages) - len(chosen)
        if skipped:
            self._note(f"Skipping {skipped} unticked page(s)")
        self._start(chosen)

    def _start(self, pages: list[Path]) -> None:
        if self._worker is not None or self._chapter is None:
            return
        out_dir = output_dir_for(self._chapter, self._host.cfg)
        self._worker = TranslateWorker(self._host, pages, out_dir)
        self._worker.progress.connect(self._on_progress)
        self._worker.page_done.connect(self._on_page_done)
        self._worker.finished_run.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._clear_worker)
        self._note(
            f"Translating {len(pages)} page{'s' if len(pages) != 1 else ''} "
            f"-> {out_dir}"
        )
        self._worker.start()
        self._update_actions()

    def stop(self) -> None:
        if self._worker is not None:
            self._worker.stop()
            self.statusBar().showMessage("Stopping after the current page...")

    def _on_progress(self, label: str, fraction: float) -> None:
        self._progress.setValue(int(fraction * 1000))
        self.statusBar().showMessage(label)

    def _on_page_done(self, result: PageResult) -> None:
        self._results[result.page.path] = result
        self._note(summarise(result))
        for problem in result.problems:
            self._note(f"    {problem}")

        row = self._pages.index(result.page.path)
        item = self._page_list.item(row)
        self._quiet = True
        try:
            failed = any(r.status.is_failure for r in result.page.regions)
            item.setForeground(
                QColor(Palette.warn) if failed else QColor(Palette.ok)
            )
            item.setText(f"{result.page.name}  ({result.typeset})")
            if result.page.rendered is not None:
                item.setIcon(QIcon(to_pixmap(result.page.rendered, _THUMB_WIDTH)))
        finally:
            self._quiet = False

        if self._current_page() == result.page.path:
            self._preview.set_page(result.page.original, result.page.rendered)
            self._compare.setEnabled(True)
            self._adopt(result)

    def _on_finished(self, done: int, total: int) -> None:
        self._progress.setValue(1000 if done == total else 0)
        self.statusBar().showMessage(
            f"Finished {done} of {total} pages." if done == total
            else f"Stopped after {done} of {total} pages."
        )

    def _on_failed(self, message: str) -> None:
        self._progress.setValue(0)
        self._note(f"FAILED: {message}")
        QMessageBox.critical(self, "Run failed", message)

    def _clear_worker(self) -> None:
        self._worker = None
        self._update_actions()

    # -- odds and ends ---------------------------------------------------

    def _start_telemetry(self) -> None:
        if self._telemetry is not None:
            return
        self._telemetry = TelemetryWorker()
        self._telemetry.sampled.connect(self._resources.show_reading)
        self._telemetry.start()

    def _check_health(self) -> None:
        from ..runtime import describe_device

        self._device.setText(describe_device())
        self._health = HealthWorker(self._host)
        self._health.done.connect(self._on_health)
        self._health.start()

    def _on_health(self, ok: bool, message: str) -> None:
        self.statusBar().showMessage(
            message if ok else f"Not ready -- {message}"
        )
        if not ok:
            self._note(f"Not ready: {message}")

    def edit_settings(self) -> None:
        from ..config import load_config, save_overrides
        from ..pipeline.translate import OllamaTranslator
        from .settings import SettingsDialog

        with OllamaTranslator(self._host.cfg) as translator:
            models = translator.list_models()

        dialog = SettingsDialog(self._host.cfg, models, self)
        if dialog.exec() != dialog.DialogCode.Accepted:
            return

        save_overrides(dialog.overrides())
        # Rebuilt rather than patched: a changed device or model means the
        # loaded sessions and the open HTTP client belong to the old settings.
        self._host.apply(load_config())
        self._name_for_pair()
        self._note("Settings saved to settings.local.yaml")
        self.statusBar().showMessage("Settings applied. Rechecking Ollama...")
        if self._chapter is not None:
            self.statusBar().showMessage(
                f"Settings applied. Output now goes to "
                f"{output_dir_for(self._chapter, self._host.cfg).name}"
            )
        self._check_health()

    def reveal_output(self) -> None:
        if self._chapter is None:
            return
        target = output_dir_for(self._chapter, self._host.cfg)
        target.mkdir(parents=True, exist_ok=True)
        import os

        os.startfile(str(target))  # noqa: S606 -- Windows shell open

    def _note(self, text: str) -> None:
        self._log.addItem(text)
        self._log.scrollToBottom()

    def _update_actions(self) -> None:
        running = self._worker is not None
        has_chapter = bool(self._pages)
        self._open_action.setEnabled(not running)
        self._page_action.setEnabled(has_chapter and not running)
        self._chapter_action.setEnabled(has_chapter and not running)
        self._stop_action.setEnabled(running)
        self._folder_action.setEnabled(has_chapter)
        self._settings_action.setEnabled(not running)

    def closeEvent(self, event) -> None:  # noqa: N802
        QApplication.instance().removeEventFilter(self)
        if self._worker is not None:
            self._worker.stop()
            self._worker.wait(15000)
        if self._region_worker is not None:
            self._region_worker.wait(30000)
        if self._telemetry is not None:
            self._telemetry.stop()
            self._telemetry.wait(2000)
        self._host.close()
        super().closeEvent(event)


def _read(path: Path) -> np.ndarray | None:
    from ..runner import _imread

    return _imread(path)
