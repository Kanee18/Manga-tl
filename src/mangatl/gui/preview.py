"""The page preview: a zoomable view, a before/after wipe, and a canvas.

Comparison is a wipe rather than two panes side by side. Lettering has to be
judged against the artwork it sits on -- whether a balloon is fully cleared,
whether the replacement is the size the original was -- and that judgement
needs both versions in the same place on screen, not next to each other at
half width.

The same view is where corrections are made, because the two are the same
act: a translation is judged against the page and fixed on the page. Three
tools share it, and which one is active only changes what dragging does --
clicking always picks the region under the cursor, so choosing a line to edit
never costs a tool change.

    pan     drag to move, wheel to zoom
    select  draw over lettering the detector missed, to have it translated
    erase   paint over lettering the erase stage left behind

Selecting is a stroke over the words rather than a rectangle round them.
Balloons are not rectangles and the lettering in them is not laid out on a
grid, so a box is either too tight, and cuts a line, or generous enough to
take in the balloon edge and its neighbour. A stroke says *these words* the
way a finger would, and the block it lands on is worked out from the ink.

Everything here works in *page* pixels and converts at the edges. The view has
a zoom, an offset and a fit mode, so a stroke drawn on screen means nothing
until it is mapped back through those; keeping page coordinates as the only
ones that leave this class is what stops that mapping leaking into the window
and the pipeline.
"""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import QPoint, QPointF, QRect, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import (
    QColor, QPainter, QPen, QPixmap, QPolygonF, QWheelEvent,
)
from PyQt6.QtWidgets import QWidget

from ..types import Box, Region
from .theme import Palette, logo_mark
from .worker import to_pixmap

PAN, NUDGE, SELECT, ERASE = "pan", "nudge", "select", "erase"

# How wide the selection stroke reads on the page, in page pixels. It is a
# seed rather than an outline -- whatever ink it touches brings its whole
# block along -- so it only has to be wide enough to hit a letter while
# being dragged at speed.
SELECT_RADIUS = 9

# How far the cursor may travel and still count as a click rather than a drag.
_CLICK_SLOP = 4


class PagePreview(QWidget):
    """Shows one page, wipes between versions, and takes corrections on it."""

    wipe_moved = pyqtSignal(float)
    region_picked = pyqtSignal(str)     # region id, or "" for empty space
    text_selected = pyqtSignal(object)        # [(x, y), ...] in page pixels
    stroke_painted = pyqtSignal(object, int)  # [(x, y), ...] page px, radius
    text_nudged = pyqtSignal(float, float, bool)  # dx, dy in page px, merge

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(320, 320)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.OpenHandCursor)

        self._before: QPixmap | None = None
        self._after: QPixmap | None = None
        self._show_after = True
        self._wipe = 1.0          # fraction of width showing the "after"
        self._compare = False

        self._zoom = 1.0
        self._fit = True
        self._offset = QPoint(0, 0)
        self._dragging: QPoint | None = None
        self._pressed_at: QPoint | None = None

        self._tool = PAN
        self._regions: list[Region] = []
        self._selected = ""
        self._show_regions = False
        self._stroke: list[tuple[float, float]] = []
        self._brush = 16
        self._cursor_at: QPointF | None = None
        self._nudge_from: QPointF | None = None
        self._peek: QPixmap | None = None
        self._peek_box: Box | None = None

    # -- content --------------------------------------------------------

    def set_page(
        self, before: np.ndarray | None, after: np.ndarray | None, keep_view: bool = False
    ) -> None:
        self._before = to_pixmap(before) if before is not None else None
        self._after = to_pixmap(after) if after is not None else None
        if not keep_view:
            self._fit = True
            self._zoom = 1.0
            self._offset = QPoint(0, 0)
        self.update()

    def clear(self) -> None:
        self._before = self._after = None
        self._regions = []
        self._selected = ""
        self.update()

    @property
    def has_result(self) -> bool:
        return self._after is not None

    def set_show_after(self, show: bool) -> None:
        self._show_after = show
        self.update()

    def set_compare(self, on: bool) -> None:
        self._compare = on
        self._wipe = 0.5 if on else 1.0
        self.update()

    # -- editing --------------------------------------------------------

    def set_regions(self, regions: list[Region]) -> None:
        self._regions = list(regions)
        self.update()

    def set_selected(self, region_id: str) -> None:
        self._selected = region_id
        self.update()

    def set_tool(self, tool: str) -> None:
        self._tool = tool
        self._stroke = []
        self.setCursor(self._cursor_for(tool))
        self.update()

    @staticmethod
    def _cursor_for(tool: str) -> Qt.CursorShape:
        if tool == PAN:
            return Qt.CursorShape.OpenHandCursor
        if tool == NUDGE:
            return Qt.CursorShape.SizeAllCursor
        return Qt.CursorShape.CrossCursor

    def set_show_regions(self, on: bool) -> None:
        self._show_regions = on
        self.update()

    def set_brush(self, radius: int) -> None:
        self._brush = max(2, radius)
        self.update()

    def show_original(self, patch, box) -> None:
        """Paint one region's untranslated pixels over the result, or stop.

        Only that region, and only while asked. Comparing the whole page is
        the wipe's job; this answers a narrower question -- whether *this*
        balloon should have been translated at all -- and answering it by
        wiping the page means losing sight of the balloon while you decide.
        """
        self._peek = to_pixmap(patch) if patch is not None else None
        self._peek_box = box if patch is not None else None
        self.update()

    # -- view -----------------------------------------------------------

    def fit_to_window(self) -> None:
        self._fit = True
        self._offset = QPoint(0, 0)
        self.update()

    def zoom_by(self, factor: float) -> None:
        self._zoom = float(np.clip(self._zoom * factor, 0.1, 8.0))
        self._fit = False
        self.update()

    def _active(self) -> QPixmap | None:
        if self._compare:
            return self._after or self._before
        if self._show_after and self._after is not None:
            return self._after
        return self._before or self._after

    def _scale_for(self, pixmap: QPixmap) -> float:
        if self._fit:
            return min(
                self.width() / max(1, pixmap.width()),
                self.height() / max(1, pixmap.height()),
            )
        return self._zoom

    def _target(self, pixmap: QPixmap) -> QRectF:
        scale = self._scale_for(pixmap)
        width, height = pixmap.width() * scale, pixmap.height() * scale
        x = (self.width() - width) / 2 + self._offset.x()
        y = (self.height() - height) / 2 + self._offset.y()
        return QRectF(x, y, width, height)

    # -- coordinates ----------------------------------------------------

    def _to_page(self, position: QPointF) -> QPointF | None:
        """A point on screen as a point on the page, or None if it is off it."""
        pixmap = self._active()
        if pixmap is None:
            return None
        target = self._target(pixmap)
        if target.width() <= 0 or target.height() <= 0:
            return None
        x = (position.x() - target.left()) / target.width() * pixmap.width()
        y = (position.y() - target.top()) / target.height() * pixmap.height()
        if not (0 <= x < pixmap.width() and 0 <= y < pixmap.height()):
            return None
        return QPointF(x, y)

    def _to_widget(self, x: float, y: float) -> QPointF:
        pixmap = self._active()
        if pixmap is None:
            return QPointF(0, 0)
        target = self._target(pixmap)
        return QPointF(
            target.left() + x / max(1, pixmap.width()) * target.width(),
            target.top() + y / max(1, pixmap.height()) * target.height(),
        )

    def _rect_for(self, box: Box) -> QRectF:
        top_left = self._to_widget(box.x1, box.y1)
        bottom_right = self._to_widget(box.x2, box.y2)
        return QRectF(top_left, bottom_right)

    def _region_at(self, page: QPointF) -> str:
        """The id of the smallest region covering a page point.

        Smallest, because regions nest: a caption inside a balloon shares its
        space with the balloon's own block, and the one a click means is the
        tighter of the two.
        """
        best, best_area = "", None
        for region in self._regions:
            box = region.box
            if box.x1 <= page.x() < box.x2 and box.y1 <= page.y() < box.y2:
                if best_area is None or box.area < best_area:
                    best, best_area = region.id, box.area
        return best

    # -- painting -------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802 -- Qt naming
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(Palette.window))
        pixmap = self._active()
        if pixmap is None:
            self._draw_empty(painter)
            return

        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        target = self._target(pixmap)

        if self._compare and self._before is not None and self._after is not None:
            split = target.left() + target.width() * self._wipe
            painter.save()
            painter.setClipRect(QRectF(target.left(), target.top(),
                                       split - target.left(), target.height()))
            painter.drawPixmap(target, self._after, QRectF(self._after.rect()))
            painter.restore()

            painter.save()
            painter.setClipRect(QRectF(split, target.top(),
                                       target.right() - split, target.height()))
            painter.drawPixmap(target, self._before, QRectF(self._before.rect()))
            painter.restore()

            painter.setPen(QPen(QColor(Palette.accent), 2))
            painter.drawLine(int(split), int(target.top()), int(split), int(target.bottom()))
            self._label(painter, target, split)
        else:
            painter.drawPixmap(target, pixmap, QRectF(pixmap.rect()))

        if self._peek is not None and self._peek_box is not None:
            painter.drawPixmap(
                self._rect_for(self._peek_box), self._peek, QRectF(self._peek.rect())
            )
            pen = QPen(QColor(Palette.warn), 2)
            painter.setPen(pen)
            painter.drawRect(self._rect_for(self._peek_box))

        self._draw_overlay(painter)
        painter.end()

    def _draw_empty(self, painter: QPainter) -> None:
        """What the largest panel shows before a chapter is opened.

        The logo goes here rather than only on the toolbar because this is the
        one moment the window has nothing to say -- and a mark set faintly in
        an empty frame reads as a resting state, where the same mark next to
        live controls would only be decoration competing with them.
        """
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.setPen(QColor(Palette.faint))
        area = self.rect()
        prompt = "Open a chapter folder to begin"

        mark = logo_mark(96, Palette.faint, opacity=0.55)
        # The logical size, not the pixel one: the mark is drawn at twice the
        # size it occupies so it stays sharp on a scaled display, and laying
        # it out by QPixmap.height() would place it half a logo too low.
        size = mark.deviceIndependentSize()
        line = painter.fontMetrics().height()
        gap = 18
        if mark.isNull() or area.height() < size.height() + gap + line + 40:
            painter.drawText(area, Qt.AlignmentFlag.AlignCenter, prompt)
            return

        # The mark and the line under it are centred as one block, so the
        # panel does not read as a logo with a caption drifting away from it.
        top = area.center().y() - (size.height() + gap + line) / 2
        painter.drawPixmap(
            int(area.center().x() - size.width() / 2), int(top), mark
        )
        painter.drawText(
            QRect(area.left(), int(top + size.height() + gap), area.width(), line),
            Qt.AlignmentFlag.AlignCenter,
            prompt,
        )

    def _draw_overlay(self, painter: QPainter) -> None:
        """Region outlines, the stroke being drawn, and the brush."""
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        if self._show_regions or self._tool != PAN:
            faint = QColor(Palette.accent)
            faint.setAlpha(70)
            painter.setPen(QPen(faint, 1))
            for region in self._regions:
                if region.id != self._selected:
                    painter.drawRect(self._rect_for(region.box))

        for region in self._regions:
            if region.id == self._selected:
                painter.setPen(QPen(QColor(Palette.accent), 2))
                painter.drawRect(self._rect_for(region.box))
                break

        if self._tool == SELECT and self._stroke:
            # A highlighter, not an outline: the stroke says which words, and
            # drawing it as a translucent band over them is how every other
            # program on the machine says the same thing.
            scale = self._scale_for(self._active()) or 1.0
            ink = QColor(Palette.accent)
            ink.setAlpha(90)
            pen = QPen(ink, max(3.0, SELECT_RADIUS * 2 * scale))
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            painter.setPen(pen)
            painter.drawPolyline(
                QPolygonF([self._to_widget(x, y) for x, y in self._stroke])
            )

        if self._tool == ERASE and self._cursor_at is not None:
            radius = self._brush * self._scale_for(self._active())
            painter.setPen(QPen(QColor(Palette.warn), 1))
            painter.drawEllipse(self._cursor_at, radius, radius)

    def _label(self, painter: QPainter, target: QRectF, split: float) -> None:
        painter.setPen(QColor(Palette.accent))
        font = painter.font()
        font.setPointSize(9)
        painter.setFont(font)
        top = int(target.top()) + 14
        if split - target.left() > 60:
            painter.drawText(int(target.left()) + 8, top, "translated")
        if target.right() - split > 60:
            painter.drawText(int(split) + 8, top, "original")

    # -- interaction ----------------------------------------------------

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        self.zoom_by(1.15 if event.angleDelta().y() > 0 else 1 / 1.15)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() != Qt.MouseButton.LeftButton:
            return
        self._pressed_at = event.position().toPoint()
        if self._compare:
            self._set_wipe(event.position().x())
            return
        if self._tool == NUDGE:
            self._nudge_from = event.position()
        elif self._tool in (SELECT, ERASE):
            page = self._to_page(event.position())
            self._stroke = [(page.x(), page.y())] if page is not None else []
        else:
            self._dragging = event.position().toPoint()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        self._cursor_at = event.position()
        if self._compare and event.buttons() & Qt.MouseButton.LeftButton:
            self._set_wipe(event.position().x())
            return
        held = bool(event.buttons() & Qt.MouseButton.LeftButton)
        if self._tool == NUDGE and held and self._nudge_from is not None:
            scale = self._scale_for(self._active()) or 1.0
            delta = event.position() - self._nudge_from
            self._nudge_from = event.position()
            # Reported in page pixels, so a drag means the same distance on
            # the page whatever the view is zoomed to.
            self.text_nudged.emit(delta.x() / scale, delta.y() / scale, True)
        elif self._tool in (SELECT, ERASE) and held:
            page = self._to_page(event.position())
            if page is not None:
                self._stroke.append((page.x(), page.y()))
            self.update()
        elif self._dragging is not None:
            point = event.position().toPoint()
            self._offset += point - self._dragging
            self._dragging = point
            self._fit = False
            self.update()
        elif self._tool == ERASE:
            self.update()  # keep the brush ring under the cursor

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() != Qt.MouseButton.LeftButton:
            return
        moved = (
            self._pressed_at is not None
            and (event.position().toPoint() - self._pressed_at).manhattanLength()
            > _CLICK_SLOP
        )
        self._dragging = None
        self._nudge_from = None
        self.setCursor(self._cursor_for(self._tool))

        if self._tool == SELECT and self._stroke and moved:
            self.text_selected.emit(list(self._stroke))
            self._stroke = []
        elif self._tool == ERASE and self._stroke:
            self.stroke_painted.emit(list(self._stroke), self._brush)
            self._stroke = []
        elif not moved and not self._compare:
            # A click, whatever the tool. Picking the line to work on should
            # never cost a trip back to the tool buttons.
            page = self._to_page(event.position())
            self.region_picked.emit(self._region_at(page) if page else "")

        self._pressed_at = None
        self.update()

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        self.fit_to_window()

    def leaveEvent(self, event) -> None:  # noqa: N802
        self._cursor_at = None
        self.update()

    def _set_wipe(self, x: float) -> None:
        pixmap = self._active()
        if pixmap is None:
            return
        target = self._target(pixmap)
        if target.width() <= 0:
            return
        self._wipe = float(np.clip((x - target.left()) / target.width(), 0.0, 1.0))
        self.wipe_moved.emit(self._wipe)
        self.update()
