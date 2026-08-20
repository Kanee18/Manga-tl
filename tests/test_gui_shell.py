"""The window's chrome: its gauges, its icons and its sheet.

None of this changes a translated page, which is exactly why it is worth
holding down -- a gauge that reads the wrong unit, an icon that silently fails
to compile, or a stylesheet with an unresolved placeholder in it all still
launch, and the first anyone hears of them is a window that looks broken.
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("PyQt6")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from mangatl.gui import theme  # noqa: E402
from mangatl.gui.meters import ResourceStrip  # noqa: E402
from mangatl.gui.telemetry import Gauge, Reading  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


# -- what a gauge says -------------------------------------------------------


@pytest.mark.parametrize(
    "gauge,expected",
    [
        (Gauge(37.4, 100.0, "%"), "37%"),
        (Gauge(148.2, 220.0, "W"), "148 W"),
        # Past a gigabyte, mebibytes are unreadable: "6963 / 8192 MiB" is a
        # number to decode, "6.8 / 8 GB" is one to glance at.
        (Gauge(6963, 8192, "MiB"), "6.8 / 8 GB"),
        (Gauge(512, 900, "MiB"), "512 / 900 MiB"),
    ],
)
def test_a_gauge_reads_in_the_unit_it_was_measured_in(gauge, expected):
    assert gauge.text == expected


def test_a_gauge_cannot_report_past_full():
    """Power draw runs over the enforced limit in a boost, and a bar cannot."""
    assert Gauge(240.0, 220.0, "W").fraction == 1.0
    assert Gauge(-1.0, 220.0, "W").fraction == 0.0


def test_a_gauge_with_nothing_to_measure_against_is_empty_not_infinite():
    assert Gauge(5.0, 0.0, "MiB").fraction == 0.0


# -- the strip ---------------------------------------------------------------


def _reading(**kwargs) -> Reading:
    base = dict(cpu=Gauge(20.0, 100.0, "%"), memory=Gauge(8000, 32000, "MiB"))
    return Reading(**{**base, **kwargs})


def test_a_machine_with_no_card_still_shows_its_processor(app):
    """An AMD or Apple machine runs the pipeline on the CPU perfectly well."""
    strip = ResourceStrip()
    strip.show_reading(_reading())
    assert strip.cpu._gauge is not None
    assert strip.vram._gauge is None
    assert strip.power._gauge is None
    assert "not available" in strip.vram.toolTip()


def test_a_card_is_named_in_the_tooltip_rather_than_on_the_strip(app):
    strip = ResourceStrip()
    strip.show_reading(
        _reading(
            vram=Gauge(6963, 8192, "MiB"),
            power=Gauge(148.0, 220.0, "W"),
            gpu_util=Gauge(78.0, 100.0, "%"),
            gpu_name="NVIDIA GeForce RTX 3060 Ti",
        )
    )
    assert "3060 Ti" in strip.vram.toolTip()
    assert "78% busy" in strip.power.toolTip()


def test_a_full_card_is_coloured_as_a_warning_and_a_busy_one_is_not(app):
    """VRAM running out costs minutes a page; a card at its power limit is fine."""
    strip = ResourceStrip()
    strip.show_reading(
        _reading(vram=Gauge(8100, 8192, "MiB"), power=Gauge(219.0, 220.0, "W"))
    )
    assert strip.vram._bar_colour().name() == theme.Palette.danger
    assert strip.power._bar_colour().name() == theme.Palette.accent


# -- chrome ------------------------------------------------------------------


def test_every_icon_compiles(app):
    """An SVG that does not parse yields an empty QIcon and no error at all."""
    for name in theme._ICONS:
        assert not theme.icon(name).isNull(), f"{name} did not render"


def test_the_stylesheet_has_no_placeholder_left_in_it():
    """A stray `{p.accent}` is valid CSS to Qt: it drops the rule and says so
    nowhere, so the widget it styled is simply undressed."""
    import re

    sheet = theme.stylesheet()
    leftover = re.findall(r"\{[A-Za-z_][\w.]*\}", sheet)
    assert not leftover, f"unresolved placeholders: {leftover}"
    assert theme.Palette.accent in sheet


# -- peeking at the original -------------------------------------------------


def _window():
    import numpy as np
    from pathlib import Path

    from mangatl.config import default_config
    from mangatl.gui.main_window import MainWindow
    from mangatl.runner import PageResult
    from mangatl.types import Box, Page, Region, RegionStatus

    window = MainWindow(default_config())
    image = np.full((400, 400, 3), 255, np.uint8)
    page = Page(path=Path("001.png"))
    page.original = image.copy()
    page.original[160:240, 130:270] = 17
    page.clean = image.copy()
    page.rendered = image.copy()
    page.regions.append(
        Region(
            id="r00",
            box=Box(120, 150, 280, 250),
            source_line_height=24.0,
            text_en="LET ME GO!",
            text_id="Lepaskan aku!",
            status=RegionStatus.TYPESET,
        )
    )
    window._adopt(PageResult(page=page))
    window._select_region("r00")
    # Shown and activated, or nothing in it can hold the keyboard focus and
    # the "not while typing" guard has nothing to read.
    window.show()
    window.activateWindow()
    return window


def _tap(app, window, kind):
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QKeyEvent

    app.sendEvent(
        window._page_list,
        QKeyEvent(kind, Qt.Key.Key_O, Qt.KeyboardModifier.NoModifier),
    )
    app.processEvents()


def test_o_peeks_at_the_original_wherever_the_focus_is(app):
    """A key only reaches the window when nothing else wants it.

    The page list wants letters, for type-ahead, and someone peeking at a
    balloon has as often as not just clicked a row in it.
    """
    from PyQt6.QtCore import QEvent

    window = _window()
    assert window._preview._peek is None
    _tap(app, window, QEvent.Type.KeyPress)
    assert window._preview._peek is not None, "held O showed nothing"
    _tap(app, window, QEvent.Type.KeyRelease)
    assert window._preview._peek is None, "letting go left the original up"
    window.close()


def test_o_belongs_in_the_translation_while_it_is_being_typed(app):
    from PyQt6.QtCore import QEvent

    window = _window()
    window._panel._text.setFocus()
    app.processEvents()
    assert window._panel.editing_text, "the caret is not in the text box"
    _tap(app, window, QEvent.Type.KeyPress)
    assert window._preview._peek is None
    window.close()


def test_leaving_the_window_mid_peek_puts_the_page_back(app):
    """Letting go outside the window never arrives."""
    from PyQt6.QtCore import QEvent

    window = _window()
    _tap(app, window, QEvent.Type.KeyPress)
    assert window._preview._peek is not None
    app.sendEvent(window, QEvent(QEvent.Type.WindowDeactivate))
    app.processEvents()
    assert window._preview._peek is None
    window.close()
