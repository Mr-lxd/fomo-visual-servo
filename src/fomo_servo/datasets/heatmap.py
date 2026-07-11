"""Stride-space FOMO target creation and decoding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np


class HeatmapError(ValueError):
    """Raised when a heatmap configuration or centroid is invalid."""


class HeatmapCollisionError(HeatmapError):
    """Raised when different foreground classes occupy one grid cell."""


@dataclass(frozen=True)
class HeatmapTarget:
    """FOMO labels at one stride.

    ``class_index`` has shape ``[G,G]`` and dtype ``int64``. Zero is
    background and values ``1..N`` are foreground classes. ``one_hot`` has
    shape ``[1+N,G,G]`` and dtype ``uint8`` for visualisation or explicit
    channel consumers.
    """

    class_index: np.ndarray
    one_hot: np.ndarray
    stride: int
    same_class_collision_count: int
    different_class_collision_count: int = 0


@dataclass(frozen=True)
class GridCentroid:
    """One non-background class-index heatmap cell."""

    class_index: int
    grid_x: int
    grid_y: int


def generate_fomo_heatmap(
    centroids: Iterable[Tuple[float, float, int]],
    input_size: int,
    stride: int,
    num_foreground_classes: int,
    collision_policy: str = "error",
) -> HeatmapTarget:
    """Create class-index and one-hot FOMO labels from letterbox centroids.

    Args:
        centroids: Iterable of ``(x_lb, y_lb, foreground_class_id)`` in
            letterbox input pixels. Class IDs are zero-based foreground IDs.
        input_size: Square letterbox size ``S``.
        stride: Output stride. The grid size is ``G=S/stride``.
        num_foreground_classes: Number of foreground classes ``N``.
        collision_policy: ``error`` rejects different-class collisions; ``keep_first``
            keeps the first label in file order and counts the collision explicitly.

    Returns:
        Background-plus-classes labels with shapes ``[G,G]`` and ``[1+N,G,G]``.
    """

    _require_positive_integer(input_size, "input_size")
    _require_positive_integer(stride, "stride")
    _require_positive_integer(num_foreground_classes, "num_foreground_classes")
    if collision_policy not in {"error", "keep_first"}:
        raise HeatmapError("collision_policy must be 'error' or 'keep_first'")
    if input_size % stride != 0:
        raise HeatmapError("input_size must be divisible by stride")

    grid_size = input_size // stride
    class_index = np.zeros((grid_size, grid_size), dtype=np.int64)
    same_class_collision_count = 0
    different_class_collision_count = 0

    for x, y, foreground_class_id in centroids:
        _require_coordinate(x, "x")
        _require_coordinate(y, "y")
        if (
            isinstance(foreground_class_id, bool)
            or not isinstance(foreground_class_id, int)
            or not 0 <= foreground_class_id < num_foreground_classes
        ):
            raise HeatmapError("foreground_class_id is outside configured classes")

        grid_x = min(max(int(x // stride), 0), grid_size - 1)
        grid_y = min(max(int(y // stride), 0), grid_size - 1)
        encoded_class = foreground_class_id + 1
        existing_class = int(class_index[grid_y, grid_x])

        if existing_class == 0:
            class_index[grid_y, grid_x] = encoded_class
        elif existing_class == encoded_class:
            same_class_collision_count += 1
        elif collision_policy == "keep_first":
            different_class_collision_count += 1
        else:
            raise HeatmapCollisionError(
                "one heatmap cell received different foreground classes: {} and {}"
                .format(existing_class - 1, foreground_class_id)
            )

    one_hot = np.zeros(
        (num_foreground_classes + 1, grid_size, grid_size), dtype=np.uint8
    )
    for class_id in range(num_foreground_classes + 1):
        one_hot[class_id] = (class_index == class_id).astype(np.uint8)

    return HeatmapTarget(
        class_index=class_index,
        one_hot=one_hot,
        stride=stride,
        same_class_collision_count=same_class_collision_count,
        different_class_collision_count=different_class_collision_count,
    )


def decode_class_index_heatmap(class_index: np.ndarray) -> Tuple[GridCentroid, ...]:
    """Return each non-background ``[G,G]`` class-index heatmap cell."""

    if not isinstance(class_index, np.ndarray) or class_index.ndim != 2:
        raise HeatmapError("class_index must have shape [G,G]")

    decoded = []
    for grid_y, grid_x in np.argwhere(class_index > 0):
        decoded.append(
            GridCentroid(
                class_index=int(class_index[grid_y, grid_x]),
                grid_x=int(grid_x),
                grid_y=int(grid_y),
            )
        )
    return tuple(decoded)


def _require_positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise HeatmapError("{} must be a positive integer".format(name))


def _require_coordinate(value: float, name: str) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise HeatmapError("{} must be numeric".format(name))
    if not np.isfinite(value):
        raise HeatmapError("{} must be finite".format(name))
