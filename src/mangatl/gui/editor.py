"""The correction panel, down the right-hand side of the window.

The pipeline gets most of a page right and will not get all of it right: a
name comes back spelled a way the series does not use, a shout is set at
speaking size, a scrap of the original lettering survives the erase, a caption
the detector never saw is still in the source language. Every one of those
is a minute's work by hand and unreachable without somewhere to do it.

So the panel is not a settings screen -- nothing here changes how the *next*
page is translated. It changes this page, in front of you, and re-letters it
straight away. That is why it sits beside the preview rather than behind a
dialog: the judgement being made is visual, and a dialog covering the artwork
would hide the thing being judged.

The corrections, which are what the page actually needs:

    text       retype the translation when the model got it wrong
    size       nudge the fitted size up or down
    bold       weight a line the letterer had drawn heavy
    erase      paint out what the erase stage left behind
    select     draw over lettering the detector missed, then press Translate
    translate  read and translate the selected region again
    original   leave this balloon reading as the artist drew it

Size and weight are adjustments, not commands: they move what the fitter
aims for, and the fitter still refuses to put lettering outside its balloon.

Selecting does not translate on its own. It marks which words, which is a
judgement someone can see is wrong straight away -- the stroke lands on the
wrong balloon, or takes its neighbour too -- where a translation is twenty
seconds of waiting to find out. So the stroke is drawn, the block it landed on
appears outlined and selected, and Translate is a second press.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QButtonGroup, QGroupBox, QHBoxLayout, QLabel, QPlainTextEdit, QPushButton,
    QSlider, QToolButton, QVBoxLayout, QWidget,
)

from ..types import Region
from .preview import ERASE, NUDGE, PAN, SELECT
from .theme import Palette, icon

# How long typing must pause before the page is re-lettered. Long enough not
# to re-letter every keystroke, short enough that it feels like the page is
# following what you type.
_SETTLE_MS = 450

_STEP = 0.1
_MIN_SCALE, _MAX_SCALE = 0.4, 3.0


def _heading(text: str) -> QLabel:
    label = QLabel(text.upper())
    label.setStyleSheet(
        f"color: {Palette.faint}; font-size: 10px; font-weight: 700;"
        "letter-spacing: 0.7px; padding: 2px 0;"
    )
    return label


class EditorPanel(QWidget):
    """Tools and the one selected line's settings."""

    tool_changed = pyqtSignal(str)
    brush_changed = pyqtSignal(int)
    outlines_toggled = pyqtSignal(bool)

    text_edited = pyqtSignal(str)
    size_scaled = pyqtSignal(float)
    bold_toggled = pyqtSignal(bool)
    keep_original_toggled = pyqtSignal(bool)
    translate_requested = pyqtSignal()
    recentre_requested = pyqtSignal()
    reset_requested = pyqtSignal()
    undo_requested = pyqtSignal()
    save_requested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        # Wide enough for the longest label on it -- "Keep original" --
        # to be read rather than elided, on a UI font wider than this
        # machine's happens to be.
        self.setFixedWidth(302)
        self._region: Region | None = None
        # Guards the signal loop: filling the boxes from a region must not
        # read back as the user having edited them.
        self._loading = False
        # A reading is in flight. Kept here rather than only on the button,
        # because selecting another line refills the panel and would otherwise
        # hand back a Translate button while the last one is still running.
        self._translating = False

        self._settle = QTimer(self)
        self._settle.setSingleShot(True)
        self._settle.setInterval(_SETTLE_MS)
        self._settle.timeout.connect(self._push_text)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        layout.addWidget(self._build_tools())
        layout.addWidget(self._build_line())
        layout.addStretch(1)
        layout.addWidget(self._build_actions())
        self.show_region(None)

    # -- construction ---------------------------------------------------

    def _tool_button(self, glyph: str, label: str, tip: str) -> QToolButton:
        button = QToolButton()
        button.setObjectName("toggle")
        button.setCheckable(True)
        button.setIcon(icon(glyph, Palette.muted, 15))
        button.setText(f" {label}")
        button.setToolTip(tip)
        button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        button.setSizePolicy(button.sizePolicy().horizontalPolicy(), button.sizePolicy().verticalPolicy())
        return button

    def _build_tools(self) -> QWidget:
        box = QGroupBox("Tools")
        self._pan = self._tool_button(
            "move", "Pan", "Drag to move around the page, wheel to zoom"
        )
        self._nudge_tool = self._tool_button(
            "text_move", "Move text",
            "Drag a line to reposition it inside its balloon.\n"
            "Arrow keys nudge it a pixel at a time.",
        )
        self._select = self._tool_button(
            "textbox", "Select text",
            "Draw over lettering the detector missed -- a stroke across the\n"
            "words, the way you would point at them. The block it lands on\n"
            "is outlined; press Translate to read it.",
        )
        self._erase = self._tool_button(
            "eraser", "Erase",
            "Paint over anything the erase stage left behind. Balloons are filled with\n"
            "their own ground colour; over artwork the area is rebuilt.",
        )
        self._pan.setChecked(True)

        self._tools = QButtonGroup(self)
        self._tools.setExclusive(True)
        for button, name in (
            (self._pan, PAN), (self._nudge_tool, NUDGE),
            (self._select, SELECT), (self._erase, ERASE),
        ):
            self._tools.addButton(button)
            button.toggled.connect(
                lambda on, tool=name: self._on_tool(on, tool)
            )

        self._brush = QSlider(Qt.Orientation.Horizontal)
        self._brush.setRange(4, 80)
        self._brush.setValue(16)
        self._brush.valueChanged.connect(self._on_brush)
        self._brush_label = QLabel("Brush 16 px")
        self._brush_label.setStyleSheet(f"color: {Palette.faint}; font-size: 11px;")

        self._outlines = QToolButton()
        self._outlines.setObjectName("toggle")
        self._outlines.setCheckable(True)
        self._outlines.setIcon(icon("compare", Palette.muted, 15))
        self._outlines.setText(" Show regions")
        self._outlines.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._outlines.toggled.connect(self.outlines_toggled.emit)

        inner = QVBoxLayout(box)
        inner.setSpacing(6)
        for button in (self._pan, self._nudge_tool, self._select, self._erase):
            inner.addWidget(button)
        inner.addWidget(self._brush_label)
        inner.addWidget(self._brush)
        inner.addWidget(self._outlines)
        return box

    def _build_line(self) -> QWidget:
        box = QGroupBox("Selected line")
        self._which = QLabel("Click a line on the page")
        self._which.setStyleSheet(f"color: {Palette.muted}; font-size: 11px;")
        self._which.setWordWrap(True)

        self._english = QLabel("")
        self._english.setWordWrap(True)
        self._english.setStyleSheet(
            f"color: {Palette.faint}; font-size: 11px; font-style: italic;"
        )

        self._text = QPlainTextEdit()
        self._text.setPlaceholderText("Translation")
        self._text.setFixedHeight(96)
        self._text.textChanged.connect(self._on_typed)

        self._smaller = QToolButton()
        self._smaller.setIcon(icon("minus", Palette.text, 15))
        self._smaller.setToolTip("Set this line smaller")
        self._smaller.clicked.connect(lambda: self._nudge(-_STEP))
        self._bigger = QToolButton()
        self._bigger.setIcon(icon("plus", Palette.text, 15))
        self._bigger.setToolTip("Set this line larger")
        self._bigger.clicked.connect(lambda: self._nudge(+_STEP))
        self._size_label = QLabel("100%")
        self._size_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._size_label.setStyleSheet(f"color: {Palette.text}; font-size: 12px;")

        self._bold = QToolButton()
        self._bold.setObjectName("toggle")
        self._bold.setCheckable(True)
        self._bold.setIcon(icon("bold", Palette.muted, 15))
        self._bold.setText(" Bold")
        self._bold.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._bold.toggled.connect(self._on_bold)

        # Latched, and it changes the page rather than the view of it: this is
        # how a balloon is *kept* in the source language, not how it is peeked
        # at. Peeking is the O key, which shows whatever is under the cursor
        # for as long as it is held and puts nothing back.
        self._original = QToolButton()
        self._original.setObjectName("toggle")
        self._original.setCheckable(True)
        self._original.setIcon(icon("eye", Palette.muted, 15))
        self._original.setText(" Keep original")
        self._original.setToolTip(
            "Leave this balloon reading as it was drawn.\n"
            "Press again to letter the translation back in -- nothing is\n"
            "lost either way. Hold O to peek at the original instead."
        )
        self._original.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._original.toggled.connect(self._on_original)

        self._translate = QPushButton(icon("chapter", Palette.text, 14), " Translate")
        self._translate.setToolTip(
            "Read this region and translate it again.\n"
            "This is what turns a box drawn by hand into lettering."
        )
        self._translate.clicked.connect(self.translate_requested.emit)

        self._centre = QPushButton(icon("target", Palette.muted, 14), " Recentre")
        self._centre.setToolTip("Put this line back where the fitter placed it")
        self._centre.clicked.connect(self.recentre_requested.emit)

        size_row = QHBoxLayout()
        size_row.setSpacing(6)
        size_row.addWidget(QLabel("Size"))
        size_row.addStretch(1)
        size_row.addWidget(self._smaller)
        size_row.addWidget(self._size_label, 1)
        size_row.addWidget(self._bigger)

        self._reset = QPushButton(icon("undo", Palette.muted, 14), " Reset")
        self._reset.setToolTip("Put the size, weight and translation back as they were")
        self._reset.clicked.connect(self.reset_requested.emit)

        inner = QVBoxLayout(box)
        inner.setSpacing(7)
        inner.addWidget(self._which)
        inner.addWidget(self._english)
        inner.addWidget(self._text)
        inner.addWidget(self._translate)
        inner.addLayout(size_row)
        # A row each, not two up. "Keep original" is the longest label in the
        # panel and at half width Qt elides it to "Keep origi...", which is
        # the one label here that has to be read to be understood.
        inner.addWidget(self._bold)
        inner.addWidget(self._original)
        inner.addWidget(self._centre)
        inner.addWidget(self._reset)
        return box

    def _build_actions(self) -> QWidget:
        self._undo = QPushButton(icon("undo", Palette.muted, 15), " Undo")
        self._undo.setToolTip("Take back the last correction (Ctrl+Z)")
        self._undo.clicked.connect(self.undo_requested.emit)
        self._undo.setEnabled(False)

        self._save = QPushButton(icon("save", Palette.text, 15), " Save page")
        self._save.setToolTip("Write this page to the output folder as it looks now")
        self._save.clicked.connect(self.save_requested.emit)
        holder = QWidget()
        row = QVBoxLayout(holder)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        row.addWidget(self._undo)
        row.addWidget(self._save)
        return holder

    def set_pair(self, pair) -> None:
        """Name the two languages the window is working in.

        The panel showed "Indonesian" over the box whatever was set, which is
        the one label someone typing a correction reads to know what is wanted
        of them.
        """
        self._text.setPlaceholderText(pair.target.name)
        self._english_label = pair.source.name

    def set_undo(self, available: bool, label: str = "") -> None:
        self._undo.setEnabled(available)
        self._undo.setText(f" Undo {label}" if label else " Undo")

    # -- state ----------------------------------------------------------

    def show_region(self, region: Region | None, keep_text: bool = False) -> None:
        """Fill the panel from a region, or empty it when nothing is chosen.

        `keep_text` leaves the text box alone. Re-lettering happens *while*
        someone is typing, and refilling the box mid-sentence puts the caret
        back at the start of it -- so the page updated correctly and the next
        keystroke landed in front of everything already typed.
        """
        self._region = region
        self._loading = True
        try:
            editable = region is not None
            self._translate.setEnabled(editable and not self._translating)
            for widget in (
                self._text, self._smaller, self._bigger, self._bold,
                self._original, self._centre, self._reset,
            ):
                widget.setEnabled(editable)

            if region is None:
                self._which.setText("Click a line on the page")
                self._english.setText("")
                self._text.setPlainText("")
                self._size_label.setText("100%")
                self._bold.setChecked(False)
                self._original.setChecked(False)
                return

            hand_set = " · edited" if region.is_hand_set else ""
            self._which.setText(f"{region.id} · {region.status.value}{hand_set}")
            self._english.setText(region.text_en_fixed or region.text_en or "—")
            if not keep_text and self._text.toPlainText() != region.text_id:
                self._text.setPlainText(region.text_id)
            self._size_label.setText(f"{region.size_scale * 100:.0f}%")
            self._bold.setChecked(region.bold)
            self._original.setChecked(region.keep_original)
        finally:
            self._loading = False

    def set_saveable(self, on: bool) -> None:
        self._save.setEnabled(on)

    # -- reactions ------------------------------------------------------

    def _on_tool(self, checked: bool, tool: str) -> None:
        if not checked:
            return
        self._brush.setEnabled(tool == ERASE)
        self._brush_label.setEnabled(tool == ERASE)
        self.tool_changed.emit(tool)

    def _on_brush(self, value: int) -> None:
        self._brush_label.setText(f"Brush {value} px")
        self.brush_changed.emit(value)

    def _on_typed(self) -> None:
        if not self._loading:
            self._settle.start()

    def _push_text(self) -> None:
        if self._region is not None:
            self.text_edited.emit(self._text.toPlainText().strip())

    @property
    def editing_text(self) -> bool:
        """Whether the caret is in the text box and must not be disturbed."""
        return self._text.hasFocus()

    def commit(self) -> None:
        """Apply a pending edit now, rather than when typing settles."""
        if self._settle.isActive():
            self._settle.stop()
            self._push_text()

    def _nudge(self, delta: float) -> None:
        if self._region is None:
            return
        scale = min(_MAX_SCALE, max(_MIN_SCALE, self._region.size_scale + delta))
        self._size_label.setText(f"{scale * 100:.0f}%")
        self.size_scaled.emit(scale)

    def _on_bold(self, on: bool) -> None:
        if not self._loading:
            self.bold_toggled.emit(on)

    def _on_original(self, on: bool) -> None:
        if not self._loading:
            self.keep_original_toggled.emit(on)

    def set_translating(self, busy: bool) -> None:
        """Grey the button out while a reading is in flight."""
        self._translating = busy
        self._translate.setEnabled(not busy and self._region is not None)
        self._translate.setText(" Translating..." if busy else " Translate")
