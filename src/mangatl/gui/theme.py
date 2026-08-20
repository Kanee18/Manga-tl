"""One palette, one stylesheet, and icons drawn rather than shipped.

Qt's default look is a 1990s toolkit look: bevels, hard borders, grey on grey.
The window spends its life next to the manga page it is translating, so it
should recede -- a dark neutral ground, one accent, and contrast spent on the
things that carry meaning (a region that failed, a page that is done) rather
than on chrome.

The icons are SVG strings compiled at run time instead of files under
`assets/`. There are a dozen of them, they are two lines each, and having them
here means they take the palette's colours as arguments: the same glyph is
drawn muted in the status strip and bright on a toolbar button, and neither
needs its own file. It also keeps `assets/` for the things that genuinely
cannot be inlined -- the models and the lettering face.
"""

from __future__ import annotations

import functools
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import QByteArray, Qt
from PyQt6.QtGui import (
    QColor, QFont, QFontDatabase, QIcon, QImage, QPainter, QPixmap,
)
from PyQt6.QtSvg import QSvgRenderer
from PyQt6.QtWidgets import QApplication

from ..config import PROJECT_ROOT

# The one graphic that is a file rather than a path in `_ICONS`: it is
# artwork, not a glyph, and nothing here could draw it.
LOGO_PATH = PROJECT_ROOT / "assets" / "logo" / "logo.jpg"

# Most preferred first. Chosen in code rather than in the sheet because Qt
# reads a stylesheet's font-family as one name, not as a fallback list, so
# "Segoe UI", Inter, sans-serif names a family no machine has and every label
# quietly falls back to the default.
_FONTS = ("Segoe UI", "Inter", "SF Pro Text", "Ubuntu", "Noto Sans", "Arial")


class Palette:
    """The whole colour vocabulary of the window."""

    window = "#15171c"
    surface = "#1b1e25"
    raised = "#22262f"
    hover = "#282d38"
    border = "#2c313c"

    text = "#e6e9ef"
    muted = "#8b93a3"
    faint = "#5d6472"

    accent = "#6f8cff"
    accent_dim = "#3d4a80"

    ok = "#57b37c"
    warn = "#d9a441"
    danger = "#e0705f"


# Region status is the one place colour carries information rather than mood,
# so the mapping lives beside the palette instead of in the window.
STATUS_COLOURS = {
    "typeset": Palette.ok,
    "translated": Palette.ok,
    "skipped_sfx": Palette.faint,
    "skipped_empty": Palette.faint,
    "skipped_same": Palette.faint,
    "kept_original": Palette.warn,
    "read": Palette.muted,
    "detected": Palette.muted,
}

_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
    'stroke="{colour}" stroke-width="{weight}" stroke-linecap="round" '
    'stroke-linejoin="round">{body}</svg>'
)

# Stroke-only, 24x24, so every glyph shares a weight and reads as one set.
_ICONS = {
    "folder": '<path d="M3 8a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2Z"/>',
    "page": (
        '<path d="M14 3H7a1 1 0 0 0-1 1v16a1 1 0 0 0 1 1h10a1 1 0 0 0 1-1V7Z"/>'
        '<path d="M14 3v4h4"/><path d="M9 12h6M9 16h4"/>'
    ),
    "chapter": '<path d="M12 3 3 8l9 5 9-5-9-5Z"/><path d="M3 13l9 5 9-5"/>',
    "stop": '<rect x="6" y="6" width="12" height="12" rx="2"/>',
    "output": (
        '<path d="M3 8a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2Z"/>'
        '<path d="M12 18v-6M9.5 14.5 12 12l2.5 2.5"/>'
    ),
    "settings": (
        '<path d="M4 7h9M19 7h1M4 17h5M15 17h5"/>'
        '<circle cx="16" cy="7" r="2.5"/><circle cx="12" cy="17" r="2.5"/>'
    ),
    "cpu": (
        '<rect x="6" y="6" width="12" height="12" rx="2"/>'
        '<rect x="10" y="10" width="4" height="4" rx="1"/>'
        '<path d="M9.5 2v4M14.5 2v4M9.5 18v4M14.5 18v4M2 9.5h4M2 14.5h4M18 9.5h4M18 14.5h4"/>'
    ),
    "memory": (
        '<rect x="2" y="7" width="20" height="10" rx="1.5"/>'
        '<path d="M6 17v3M12 17v3M18 17v3"/><path d="M6 11h1.5M11 11h1.5M16 11h1.5"/>'
    ),
    "vram": (
        '<rect x="2" y="6" width="20" height="12" rx="2"/>'
        '<circle cx="9" cy="12" r="3"/><path d="M15 10h4M15 14h4"/>'
    ),
    "power": '<polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>',
    "move": (
        '<path d="M12 3v18M3 12h18"/>'
        '<path d="M9.5 5.5 12 3l2.5 2.5M9.5 18.5 12 21l2.5-2.5"/>'
        '<path d="M5.5 9.5 3 12l2.5 2.5M18.5 9.5 21 12l-2.5 2.5"/>'
    ),
    "textbox": (
        '<rect x="3" y="5" width="18" height="14" rx="2"/>'
        '<path d="M8.5 9.5h7M12 9.5v5.5"/>'
    ),
    "eraser": (
        '<path d="M9 20H6l-2.5-2.5a1.5 1.5 0 0 1 0-2.1L13 5.9a1.5 1.5 0 0 1 2.1 0l4.9 4.9'
        'a1.5 1.5 0 0 1 0 2.1L12.5 20"/><path d="M9 20h11"/><path d="m9.5 9.5 5 5"/>'
    ),
    "bold": '<path d="M7.5 4H13a4 4 0 0 1 0 8H7.5Zm0 8H14a4 4 0 0 1 0 8H7.5Z"/>',
    "plus": '<path d="M12 5.5v13M5.5 12h13"/>',
    "minus": '<path d="M5.5 12h13"/>',
    "save": (
        '<path d="M5 3.5h11L19.5 7v13.5H5Z"/><path d="M8.5 3.5v5.5h7V3.5"/>'
        '<rect x="8.5" y="13" width="7" height="7.5"/>'
    ),
    "trash": '<path d="M4 6.5h16M10 6.5V4.5h4v2M6.5 6.5 7.5 20h9l1-13.5"/>',
    "undo": '<path d="M9 8H4V3"/><path d="M4 8a8 8 0 1 1 2.3 9"/>',
    "text_move": (
        '<path d="M5 7h14M9.5 7v10M7.5 17h4"/>'
        '<path d="M17 12v8M14 17l3 3 3-3"/>'
    ),
    "target": (
        '<circle cx="12" cy="12" r="7.5"/><circle cx="12" cy="12" r="1.5"/>'
        '<path d="M12 2v3M12 19v3M2 12h3M19 12h3"/>'
    ),
    "chevron": '<path d="M6 9.5 12 15.5 18 9.5"/>',
    "check": '<path d="M4 12.5 9 17.5 20 6.5"/>',
    "clear": '<path d="M6 6l12 12M18 6 6 18"/>',
    "compare": (
        '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M12 4v16"/>'
    ),
    "eye": (
        '<path d="M2 12s3.5-6 10-6 10 6 10 6-3.5 6-10 6-10-6-10-6Z"/>'
        '<circle cx="12" cy="12" r="2.5"/>'
    ),
}


def icon(name: str, colour: str = Palette.text, size: int = 18, weight: float = 1.8) -> QIcon:
    """One of `_ICONS`, rendered in `colour` at `size` logical pixels."""
    svg = _SVG.format(colour=colour, weight=weight, body=_ICONS[name])
    renderer = QSvgRenderer(QByteArray(svg.encode("utf-8")))
    # Drawn at twice the size and marked as such, so it stays sharp on a
    # scaled display instead of being upsampled from a 18px bitmap.
    ratio = 2
    pixmap = QPixmap(size * ratio, size * ratio)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    renderer.render(painter)
    painter.end()
    pixmap.setDevicePixelRatio(ratio)
    return QIcon(pixmap)


@functools.lru_cache(maxsize=1)
def logo_icon() -> QIcon:
    """The logo as drawn, for the taskbar and the title bar.

    Unaltered, plate and all. Those two places are outside the window's own
    palette -- a taskbar is whatever the desktop theme makes it -- so the mark
    that survives both is the one that brings its own ground.
    """
    return QIcon(str(LOGO_PATH)) if LOGO_PATH.exists() else QIcon()


@functools.lru_cache(maxsize=1)
def _logo_coverage() -> "np.ndarray | None":
    """How much ink covers each pixel of the logo, 0 to 255.

    The file is one dark ink on paper, so its own darkness is the coverage.
    Mapping [60, 200] rather than [0, 255] is what puts the paper's texture --
    it scans at about 250, not at 255 -- firmly at nothing while leaving the
    antialiased edges of the strokes soft. A hard threshold instead gives a
    mark with stepped edges, which at 22px is most of what you see of it.
    """
    if not LOGO_PATH.exists():
        return None
    data = np.fromfile(str(LOGO_PATH), dtype=np.uint8)
    grey = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if grey is None:
        return None
    coverage = (200.0 - grey.astype(np.float32)) / 140.0
    return (np.clip(coverage, 0.0, 1.0) * 255.0).astype(np.uint8)


@functools.lru_cache(maxsize=8)
def logo_mark(height: int, colour: str = Palette.text, opacity: float = 1.0) -> QPixmap:
    """The logo inside the window, drawn in one of the palette's colours.

    Cached, because the preview's empty state draws it on every repaint and
    the widget tracks the mouse: rebuilding a 192px mark per mouse move is
    real work for a picture that never changes.

    The plate is dropped and the ink is re-inked, because the window is dark
    and the logo is a dark mark on a light ground: placed as it comes it is
    either an invisible navy shape or a white card sitting on the toolbar.
    The mark is a single-ink drawing, so re-inking it keeps everything that
    identifies it -- the shape -- and changes only the thing the ground
    decides anyway.
    """
    coverage = _logo_coverage()
    if coverage is None:
        return QPixmap()
    # Drawn at twice the asked-for size and marked as such, like `icon`, so it
    # stays sharp where the display is scaled.
    ratio = 2
    tall = max(1, height * ratio)
    wide = max(1, round(tall * coverage.shape[1] / coverage.shape[0]))
    alpha = cv2.resize(coverage, (wide, tall), interpolation=cv2.INTER_AREA)
    if opacity < 1.0:
        alpha = (alpha.astype(np.float32) * max(0.0, opacity)).astype(np.uint8)

    ink = QColor(colour)
    rgba = np.empty((tall, wide, 4), np.uint8)
    rgba[..., 0], rgba[..., 1], rgba[..., 2] = ink.red(), ink.green(), ink.blue()
    rgba[..., 3] = alpha
    # copy(), because QImage wraps the buffer it is given rather than taking
    # it: without it the array is freed and the mark paints as noise.
    image = QImage(
        rgba.data, wide, tall, 4 * wide, QImage.Format.Format_RGBA8888
    ).copy()
    pixmap = QPixmap.fromImage(image)
    pixmap.setDevicePixelRatio(ratio)
    return pixmap


def _arrow_file() -> str:
    """The dropdown chevron, written out once for the stylesheet to point at.

    Qt stylesheets can only take an image as a URL, so the one glyph that has
    to be referenced from the sheet rather than set on a widget needs a real
    file. It is written to the cache directory rather than shipped, so it
    stays a product of the palette like every other icon here -- change the
    colour and the arrow changes with it.

    Without this there is no arrow at all: styling `QComboBox::drop-down`
    replaces Qt's own, and an editable combo with no arrow is a text field as
    far as anyone can tell. That is exactly how it read -- the model list was
    populated the whole time and looked empty, because nothing said the
    control opened.
    """
    directory = Path(tempfile.gettempdir()) / "mangatl-ui"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"chevron-{Palette.muted.lstrip('#')}.png"
    if not target.exists():
        svg = _SVG.format(colour=Palette.muted, weight=2.2, body=_ICONS["chevron"])
        renderer = QSvgRenderer(QByteArray(svg.encode("utf-8")))
        pixmap = QPixmap(20, 20)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        renderer.render(painter)
        painter.end()
        pixmap.save(str(target))
    # Qt wants forward slashes in a stylesheet url() even on Windows.
    return str(target).replace("\\", "/")


def stylesheet() -> str:
    """The whole window's appearance, as one sheet."""
    p = Palette
    arrow = _arrow_file()
    return f"""
    QWidget {{
        background: {p.window};
        color: {p.text};
        font-size: 13px;
    }}
    QMainWindow::separator {{ background: {p.border}; width: 1px; height: 1px; }}
    /* Both inherit the rule above otherwise, and paint the window colour as a
       rectangle over whatever panel they sit on -- visible as a darker band
       behind every label and tick inside a settings group. */
    QLabel, QCheckBox {{ background: transparent; }}

    /* -- toolbar ---------------------------------------------------------- */
    QToolBar {{
        background: {p.surface};
        border: none;
        border-bottom: 1px solid {p.border};
        padding: 7px 10px;
        spacing: 4px;
    }}
    QToolBar::separator {{
        background: {p.border};
        width: 1px;
        margin: 4px 8px;
    }}
    QToolButton {{
        background: transparent;
        border: 1px solid transparent;
        border-radius: 7px;
        padding: 6px 11px;
        color: {p.text};
    }}
    QToolButton:hover {{ background: {p.hover}; }}
    QToolButton:pressed {{ background: {p.raised}; }}
    QToolButton:disabled {{ color: {p.faint}; }}
    QToolButton:checked {{
        background: {p.accent_dim};
        border-color: {p.accent_dim};
        color: {p.text};
    }}
    QToolButton#toggle {{
        background: {p.surface};
        border: 1px solid {p.border};
        color: {p.muted};
        padding: 5px 12px;
    }}
    QToolButton#toggle:hover {{ background: {p.hover}; color: {p.text}; }}
    QToolButton#toggle:checked {{
        background: {p.accent_dim};
        border-color: {p.accent};
        color: {p.text};
    }}
    QToolButton#toggle:disabled {{ color: {p.faint}; background: {p.window}; }}
    QToolButton#primary {{
        background: {p.accent_dim};
        border: 1px solid {p.accent};
        color: #ffffff;
    }}
    QToolButton#primary:hover {{ background: {p.accent}; }}
    QToolButton#primary:disabled {{
        background: transparent;
        border-color: {p.border};
        color: {p.faint};
    }}

    /* -- panels ----------------------------------------------------------- */
    QListWidget, QTableWidget, QTabWidget::pane, QPlainTextEdit {{
        background: {p.surface};
        border: 1px solid {p.border};
        border-radius: 9px;
    }}
    QListWidget {{ padding: 4px; outline: none; }}
    QListWidget::item {{
        border-radius: 6px;
        padding: 5px 6px;
        margin: 1px 0;
        color: {p.muted};
    }}
    QListWidget::item:hover {{ background: {p.hover}; }}
    QListWidget::item:selected {{ background: {p.accent_dim}; color: {p.text}; }}
    /* The page ticks. Styled separately from QCheckBox: a check indicator
       inside a view is a QListWidget::indicator and inherits nothing from it,
       so left alone these stay Fusion's own and read as a different control
       from the toggles under the preview. */
    QListWidget::indicator {{
        width: 14px; height: 14px;
        border: 1px solid {p.border};
        border-radius: 4px;
        background: {p.raised};
    }}
    QListWidget::indicator:checked {{
        background: {p.accent};
        border-color: {p.accent};
    }}
    QListWidget::indicator:hover {{ border-color: {p.accent}; }}

    QTableWidget {{
        gridline-color: transparent;
        outline: none;
        selection-background-color: {p.hover};
    }}
    QTableWidget::item {{ padding: 5px 8px; border: none; }}
    QHeaderView::section {{
        background: {p.surface};
        color: {p.faint};
        border: none;
        border-bottom: 1px solid {p.border};
        padding: 7px 8px;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
    }}
    QTableCornerButton::section {{ background: {p.surface}; border: none; }}

    QTabBar::tab {{
        background: transparent;
        color: {p.muted};
        border: none;
        border-radius: 7px;
        padding: 6px 14px;
        margin: 4px 2px;
    }}
    QTabBar::tab:selected {{ background: {p.raised}; color: {p.text}; }}
    QTabBar::tab:hover:!selected {{ color: {p.text}; }}

    /* -- controls --------------------------------------------------------- */
    QPushButton {{
        background: {p.raised};
        border: 1px solid {p.border};
        border-radius: 7px;
        padding: 6px 14px;
        color: {p.text};
    }}
    QPushButton:hover {{ background: {p.hover}; border-color: {p.accent_dim}; }}
    QPushButton:disabled {{ color: {p.faint}; background: {p.surface}; }}

    QCheckBox {{ color: {p.muted}; spacing: 7px; padding: 3px; }}
    QCheckBox:hover {{ color: {p.text}; }}
    QCheckBox::indicator {{
        width: 15px; height: 15px;
        border: 1px solid {p.border};
        border-radius: 4px;
        background: {p.raised};
    }}
    QCheckBox::indicator:checked {{
        background: {p.accent};
        border-color: {p.accent};
    }}

    QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {{
        background: {p.raised};
        border: 1px solid {p.border};
        border-radius: 7px;
        padding: 5px 9px;
        selection-background-color: {p.accent_dim};
    }}
    QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus {{
        border-color: {p.accent};
    }}
    QComboBox::drop-down {{
        border: none;
        width: 22px;
        subcontrol-origin: padding;
        subcontrol-position: center right;
    }}
    QComboBox::down-arrow {{ image: url({arrow}); width: 13px; height: 13px; }}
    QComboBox::down-arrow:disabled {{ image: none; }}
    QSpinBox::up-button, QSpinBox::down-button,
    QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
        background: transparent;
        border: none;
        width: 15px;
    }}

    /* -- grouped settings -------------------------------------------------- */
    QGroupBox {{
        background: {p.surface};
        border: 1px solid {p.border};
        border-radius: 9px;
        margin-top: 11px;
        padding: 14px 12px 12px 12px;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top left;
        left: 12px;
        padding: 0 6px;
        color: {p.muted};
        font-size: 11px;
        font-weight: 600;
    }}
    QDialogButtonBox QPushButton {{ min-width: 74px; }}
    QDialogButtonBox QPushButton:default {{
        background: {p.accent_dim};
        border-color: {p.accent};
        color: #ffffff;
    }}
    QDialogButtonBox QPushButton:default:hover {{ background: {p.accent}; }}
    QComboBox QAbstractItemView {{
        background: {p.raised};
        border: 1px solid {p.border};
        selection-background-color: {p.accent_dim};
        outline: none;
    }}

    /* -- progress and status ---------------------------------------------- */
    QProgressBar {{
        background: {p.border};
        border: none;
        border-radius: 3px;
    }}
    QProgressBar::chunk {{ background: {p.accent}; border-radius: 3px; }}

    QStatusBar {{
        background: {p.surface};
        border-top: 1px solid {p.border};
        color: {p.muted};
    }}
    QStatusBar::item {{ border: none; }}
    QToolTip {{
        background: {p.raised};
        color: {p.text};
        border: 1px solid {p.border};
        border-radius: 6px;
        padding: 5px 8px;
    }}

    /* -- scrollbars ------------------------------------------------------- */
    QScrollBar:vertical {{
        background: transparent; width: 10px; margin: 2px;
    }}
    QScrollBar:horizontal {{
        background: transparent; height: 10px; margin: 2px;
    }}
    QScrollBar::handle {{ background: {p.border}; border-radius: 4px; }}
    QScrollBar::handle:hover {{ background: {p.faint}; }}
    QScrollBar::handle:vertical {{ min-height: 28px; }}
    QScrollBar::handle:horizontal {{ min-width: 28px; }}
    QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
    QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

    QSplitter::handle {{ background: transparent; }}
    QSplitter::handle:hover {{ background: {p.accent_dim}; }}
    """


def apply(app: QApplication) -> None:
    """Dress the whole application: one font, one sheet.

    Set on the application rather than the window so the settings dialog and
    every message box are dressed the same way without each asking for it.
    """
    available = set(QFontDatabase.families())
    for family in _FONTS:
        if family in available:
            font = QFont(family)
            font.setPointSizeF(9.5)
            app.setFont(font)
            break
    # Fusion, because the native Windows style ignores much of a stylesheet
    # and draws its own bevels underneath what it does honour.
    app.setStyle("Fusion")
    app.setStyleSheet(stylesheet())
