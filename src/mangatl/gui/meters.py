"""The strip of gauges along the bottom of the window.

Four small readouts, painted rather than assembled out of labels and progress
bars: a gauge is an icon, a name, a number and a bar, and getting those four
to line up across four instances is less code with a paint method than with
sixteen widgets and a layout for each.

They are deliberately quiet. Nothing here is an alert -- the machine being
busy is what a working translation looks like -- so the bars sit in the accent
colour and only change when a gauge is close enough to full to be about to
cost something. VRAM is the one that really can: the detector, the recogniser
and the inpainter each hold a session and the translator holds a model, and on
an 8 GB card a run that overruns that spills to the CPU and takes minutes
instead of seconds per page.
"""

from __future__ import annotations

from PyQt6.QtCore import QRectF, QSize, Qt
from PyQt6.QtGui import QColor, QFont, QPainter, QPainterPath
from PyQt6.QtWidgets import QHBoxLayout, QWidget

from .telemetry import Gauge, Reading
from .theme import Palette, icon

# Where a gauge stops being information and starts being a warning.
_TIGHT = 0.85
_FULL = 0.95


class Meter(QWidget):
    """One gauge: an icon, a name, a value and a bar."""

    def __init__(self, glyph: str, name: str, pressure: bool = True) -> None:
        super().__init__()
        self._name = name
        self._pressure = pressure
        self._gauge: Gauge | None = None
        self._icon = icon(glyph, Palette.faint, size=13, weight=2.0)
        self.setFixedSize(QSize(124, 32))
        self.setToolTip(f"{name}: waiting for a reading")

    def set_gauge(self, gauge: Gauge | None, detail: str = "") -> None:
        self._gauge = gauge
        self.setToolTip(
            f"{self._name} — {detail or gauge.detail}" if gauge else
            f"{self._name} — not available on this machine"
        )
        self.update()

    def _bar_colour(self) -> QColor:
        if self._gauge is None or not self._pressure:
            return QColor(Palette.accent)
        fraction = self._gauge.fraction
        if fraction >= _FULL:
            return QColor(Palette.danger)
        if fraction >= _TIGHT:
            return QColor(Palette.warn)
        return QColor(Palette.accent)

    def paintEvent(self, event) -> None:  # noqa: N802 -- Qt's spelling
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)

        self._icon.paint(painter, 0, 2, 13, 13)

        label = QFont(self.font())
        label.setPointSizeF(7.5)
        label.setWeight(QFont.Weight.DemiBold)
        label.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 0.6)
        painter.setFont(label)
        painter.setPen(QColor(Palette.faint))
        painter.drawText(
            QRectF(18, 1, 60, 14),
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
            self._name.upper(),
        )

        value = QFont(self.font())
        value.setPointSizeF(8.5)
        value.setWeight(QFont.Weight.DemiBold)
        painter.setFont(value)
        painter.setPen(QColor(Palette.text if self._gauge else Palette.faint))
        painter.drawText(
            QRectF(60, 1, self.width() - 60, 14),
            int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
            self._gauge.text if self._gauge else "—",
        )

        track = QRectF(0, 21, self.width(), 4)
        path = QPainterPath()
        path.addRoundedRect(track, 2, 2)
        painter.fillPath(path, QColor(Palette.border))

        if self._gauge is not None and self._gauge.fraction > 0:
            # A hair of width at the low end, so "busy but barely" still reads
            # as a bar rather than as nothing at all.
            width = max(3.0, track.width() * self._gauge.fraction)
            filled = QPainterPath()
            filled.addRoundedRect(QRectF(0, 21, width, 4), 2, 2)
            painter.fillPath(filled, self._bar_colour())
        painter.end()


class ResourceStrip(QWidget):
    """The four gauges, kept together and fed one reading at a time."""

    def __init__(self) -> None:
        super().__init__()
        self.cpu = Meter("cpu", "CPU")
        self.memory = Meter("memory", "RAM")
        self.vram = Meter("vram", "VRAM")
        # Power is not a pressure: a card drawing its full board limit is a
        # card doing its job, and colouring that red would say the opposite.
        self.power = Meter("power", "GPU PWR", pressure=False)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 4, 0)
        layout.setSpacing(14)
        for meter in (self.cpu, self.memory, self.vram, self.power):
            layout.addWidget(meter)

    def show_reading(self, reading: Reading) -> None:
        self.cpu.set_gauge(reading.cpu)
        self.memory.set_gauge(reading.memory)

        card = reading.gpu_name or "GPU"
        self.vram.set_gauge(
            reading.vram, f"{card}: {reading.vram.detail}" if reading.vram else ""
        )
        detail = ""
        if reading.power is not None:
            detail = f"{card}: {reading.power.detail}"
            if reading.gpu_util is not None:
                detail += f", {reading.gpu_util.used:.0f}% busy"
        self.power.set_gauge(reading.power, detail)
