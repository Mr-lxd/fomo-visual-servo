"""Gaussian centre, offset and size targets for CenterNet-lite at one stride."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, log, sqrt
from typing import Sequence

import numpy as np

VISIBILITIES = ("full", "truncated", "occluded", "ignore")


@dataclass(frozen=True)
class Box:
    """A letterbox-pixel ``xyxy`` box with a zero-based class and a visibility value."""

    class_id: int
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    visibility: str = "full"


@dataclass(frozen=True)
class CenterNetTargets:
    """Dense targets: ``heat [C,G,G]``, ``offset``/``size`` ``[2,G,G]``, masks ``[G,G]``.

    ``heat_weight`` is 0 for cells covered by an ``ignore`` box (unless another
    object has its centre there), else 1. ``reg_mask`` marks centre cells with an
    offset target; ``size_mask`` the subset with a size target (``full`` only).
    """

    heat: np.ndarray
    offset: np.ndarray
    size: np.ndarray
    reg_mask: np.ndarray
    size_mask: np.ndarray
    heat_weight: np.ndarray


def gaussian_radius(height: float, width: float, min_overlap: float = 0.7) -> float:
    """CenterNet reference radius (output-grid units) keeping IoU >= ``min_overlap``."""

    a1 = 1.0
    b1 = height + width
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    r1 = (b1 + sqrt(b1 * b1 - 4 * a1 * c1)) / 2

    a2 = 4.0
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    r2 = (b2 + sqrt(b2 * b2 - 4 * a2 * c2)) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    r3 = (b3 + sqrt(b3 * b3 - 4 * a3 * c3)) / 2
    return min(r1, r2, r3)


def draw_gaussian(heat: np.ndarray, cx: int, cy: int, radius: int) -> None:
    """Max-merge a ``(2r+1)`` Gaussian with ``sigma=(2r+1)/6`` onto one ``[G,G]`` map."""

    diameter = 2 * radius + 1
    sigma = diameter / 6.0
    ys, xs = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    gaussian = np.exp(-(xs * xs + ys * ys) / (2 * sigma * sigma)).astype(np.float32)
    grid = heat.shape[0]
    left, right = min(cx, radius), min(grid - cx, radius + 1)
    top, bottom = min(cy, radius), min(grid - cy, radius + 1)
    window = heat[cy - top : cy + bottom, cx - left : cx + right]
    patch = gaussian[radius - top : radius + bottom, radius - left : radius + right]
    np.maximum(window, patch, out=window)


def build_targets(
    boxes: Sequence[Box],
    *,
    grid_size: int,
    stride: int,
    num_classes: int,
    min_overlap: float = 0.7,
) -> CenterNetTargets:
    """Create CenterNet-lite targets from letterbox-pixel boxes."""

    heat = np.zeros((num_classes, grid_size, grid_size), dtype=np.float32)
    offset = np.zeros((2, grid_size, grid_size), dtype=np.float32)
    size = np.zeros((2, grid_size, grid_size), dtype=np.float32)
    reg_mask = np.zeros((grid_size, grid_size), dtype=np.float32)
    size_mask = np.zeros((grid_size, grid_size), dtype=np.float32)
    ignore = np.zeros((grid_size, grid_size), dtype=bool)

    for box in boxes:
        if box.visibility not in VISIBILITIES:
            raise ValueError("unknown visibility '{}'".format(box.visibility))
        if box.visibility == "ignore":
            x0 = min(max(int(box.x_min // stride), 0), grid_size - 1)
            y0 = min(max(int(box.y_min // stride), 0), grid_size - 1)
            x1 = min(max(ceil(box.x_max / stride) - 1, x0), grid_size - 1)
            y1 = min(max(ceil(box.y_max / stride) - 1, y0), grid_size - 1)
            ignore[y0 : y1 + 1, x0 : x1 + 1] = True
            continue
        if not 0 <= box.class_id < num_classes:
            raise ValueError("class_id is outside the configured classes")
        w_cells = max(box.x_max - box.x_min, 1e-3) / stride
        h_cells = max(box.y_max - box.y_min, 1e-3) / stride
        cx = (box.x_min + box.x_max) / 2.0 / stride
        cy = (box.y_min + box.y_max) / 2.0 / stride
        ix = min(max(int(cx), 0), grid_size - 1)
        iy = min(max(int(cy), 0), grid_size - 1)
        radius = max(0, int(gaussian_radius(h_cells, w_cells, min_overlap)))
        draw_gaussian(heat[box.class_id], ix, iy, radius)
        offset[0, iy, ix] = min(max(cx - ix, 0.0), 1.0)
        offset[1, iy, ix] = min(max(cy - iy, 0.0), 1.0)
        reg_mask[iy, ix] = 1.0
        if box.visibility == "full":
            size[0, iy, ix] = log(w_cells)
            size[1, iy, ix] = log(h_cells)
            size_mask[iy, ix] = 1.0
        else:
            size[:, iy, ix] = 0.0
            size_mask[iy, ix] = 0.0

    heat_weight = np.ones((grid_size, grid_size), dtype=np.float32)
    heat_weight[ignore] = 0.0
    heat_weight[heat.max(axis=0) >= 1.0] = 1.0
    return CenterNetTargets(heat, offset, size, reg_mask, size_mask, heat_weight)
