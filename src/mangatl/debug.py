"""Overlays for inspecting what a stage actually did.

Written under debug/<chapter>/ when output.debug is on, and used by the tests
and by hand while tuning thresholds. Everything here is read-only with respect
to the page.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .types import Box, Page, Region, RegionKind, RegionStatus

_KIND_COLOR = {
    RegionKind.BALLOON: (60, 200, 60),
    RegionKind.FREE: (230, 160, 40),
    RegionKind.SFX: (60, 60, 230),
}

_STATUS_COLOR = {
    RegionStatus.TYPESET: (60, 200, 60),
    RegionStatus.TRANSLATED: (200, 200, 60),
    RegionStatus.READ: (200, 140, 60),
    RegionStatus.SKIPPED_SFX: (140, 140, 140),
    RegionStatus.SKIPPED_EMPTY: (140, 140, 140),
}


def _label(canvas: np.ndarray, text: str, at: tuple[int, int], color) -> None:
    x, y = at
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    y = max(th + 2, y)
    cv2.rectangle(canvas, (x, y - th - 3), (x + tw + 4, y + 2), color, -1)
    cv2.putText(
        canvas, text, (x + 2, y - 1),
        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA,
    )


def regions_overlay(
    image: np.ndarray,
    regions: list[Region],
    lines: list[Box] | None = None,
    by_status: bool = False,
) -> np.ndarray:
    """Boxes, ids and reading order drawn over the page."""
    canvas = image.copy()
    if lines:
        for box in lines:
            cv2.rectangle(canvas, (box.x1, box.y1), (box.x2, box.y2), (200, 200, 200), 1)

    centers: list[tuple[int, int]] = []
    for region in regions:
        color = (
            _STATUS_COLOR.get(region.status, (60, 60, 230))
            if by_status
            else _KIND_COLOR[region.kind]
        )
        box = region.box
        if region.polygon is not None:
            cv2.polylines(canvas, [region.polygon], True, (255, 120, 0), 2, cv2.LINE_AA)
        cv2.rectangle(canvas, (box.x1, box.y1), (box.x2, box.y2), color, 2)
        tag = region.id if not by_status else f"{region.id} {region.status.value}"
        _label(canvas, tag, (box.x1, box.y1 - 2), color)
        cx, cy = box.center
        centers.append((int(cx), int(cy)))

    for a, b in zip(centers, centers[1:]):
        cv2.arrowedLine(canvas, a, b, (0, 0, 255), 1, cv2.LINE_AA, tipLength=0.02)
    return canvas


def mask_overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """The erase mask tinted over the page, to check glyph coverage."""
    tint = np.zeros_like(image)
    tint[:, :] = (255, 0, 220)
    binary = (mask > 0)[:, :, None]
    return np.where(binary, cv2.addWeighted(image, 1 - alpha, tint, alpha, 0), image)


def side_by_side(*images: np.ndarray, gap: int = 12) -> np.ndarray:
    """Stack pages horizontally on a neutral ground for before/after reading."""
    usable = [im for im in images if im is not None]
    if not usable:
        raise ValueError("nothing to compose")
    height = max(im.shape[0] for im in usable)
    scaled = []
    for im in usable:
        if im.shape[0] != height:
            scale = height / im.shape[0]
            im = cv2.resize(im, (int(im.shape[1] * scale), height))
        scaled.append(im)
    width = sum(im.shape[1] for im in scaled) + gap * (len(scaled) - 1)
    canvas = np.full((height, width, 3), 40, dtype=np.uint8)
    x = 0
    for im in scaled:
        canvas[: im.shape[0], x : x + im.shape[1]] = im
        x += im.shape[1] + gap
    return canvas


def write_page_debug(page: Page, glyph_mask: np.ndarray | None, out_dir: Path) -> None:
    """Dump every overlay for one page."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = page.name
    if page.original is not None:
        cv2.imwrite(
            str(out_dir / f"{stem}_regions.png"),
            regions_overlay(page.original, page.regions, by_status=True),
        )
        if glyph_mask is not None:
            cv2.imwrite(
                str(out_dir / f"{stem}_mask.png"),
                mask_overlay(page.original, glyph_mask),
            )
    if page.clean is not None:
        cv2.imwrite(str(out_dir / f"{stem}_clean.png"), page.clean)

    lines = [
        f"{r.id}  {r.kind.value:8s} {r.status.value:18s} "
        f"box={tuple(r.box)} conf={r.ocr_confidence:.2f}\n"
        f"     EN: {r.text_en}\n"
        f"     ID: {r.text_id}\n"
        for r in page.regions
    ]
    (out_dir / f"{stem}_regions.txt").write_text("".join(lines), encoding="utf-8")
