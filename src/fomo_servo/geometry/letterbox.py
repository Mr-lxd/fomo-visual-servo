"""Reversible square letterbox operations for RGB images and coordinates."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Tuple

import cv2
import numpy as np


class GeometryError(ValueError):
    """Raised when image or coordinate geometry is invalid."""


@dataclass(frozen=True)
class LetterboxTransform:
    """Metadata for an aspect-ratio-preserving original-to-square transform.

    Coordinates passed to and returned from the point methods are pixel-space
    ``(x, y)`` pairs. ``input_size`` is square, and continuous heatmap
    coordinates use ``letterbox_pixels / stride``.
    """

    original_width: int
    original_height: int
    input_size: int
    resized_width: int
    resized_height: int
    scale: float
    pad_left: int
    pad_top: int
    pad_right: int
    pad_bottom: int

    @classmethod
    def from_image_size(
        cls, original_width: int, original_height: int, input_size: int
    ) -> "LetterboxTransform":
        """Build a transform without resizing an image."""

        _require_positive_integer(original_width, "original_width")
        _require_positive_integer(original_height, "original_height")
        _require_positive_integer(input_size, "input_size")

        scale = min(input_size / original_width, input_size / original_height)
        resized_width = max(1, int(round(original_width * scale)))
        resized_height = max(1, int(round(original_height * scale)))
        pad_x = input_size - resized_width
        pad_y = input_size - resized_height
        if pad_x < 0 or pad_y < 0:
            raise GeometryError("letterbox resize exceeded input_size")

        pad_left = pad_x // 2
        pad_top = pad_y // 2
        return cls(
            original_width=original_width,
            original_height=original_height,
            input_size=input_size,
            resized_width=resized_width,
            resized_height=resized_height,
            scale=scale,
            pad_left=pad_left,
            pad_top=pad_top,
            pad_right=pad_x - pad_left,
            pad_bottom=pad_y - pad_top,
        )

    def forward_point(self, x: float, y: float) -> Tuple[float, float]:
        """Map an original-image pixel coordinate to letterbox pixel space."""

        _require_finite(x, "x")
        _require_finite(y, "y")
        return x * self.scale + self.pad_left, y * self.scale + self.pad_top

    def inverse_point(self, x: float, y: float) -> Tuple[float, float]:
        """Map a letterbox pixel coordinate back to original-image pixels."""

        _require_finite(x, "x")
        _require_finite(y, "y")
        return (x - self.pad_left) / self.scale, (y - self.pad_top) / self.scale

    def forward_box(
        self, x_min: float, y_min: float, x_max: float, y_max: float
    ) -> Tuple[float, float, float, float]:
        """Map an original ``(x_min,y_min,x_max,y_max)`` bbox to letterbox pixels."""

        left, top = self.forward_point(x_min, y_min)
        right, bottom = self.forward_point(x_max, y_max)
        return left, top, right, bottom

    def inverse_box(
        self, x_min: float, y_min: float, x_max: float, y_max: float
    ) -> Tuple[float, float, float, float]:
        """Map a letterbox bbox back to original-image pixels."""

        left, top = self.inverse_point(x_min, y_min)
        right, bottom = self.inverse_point(x_max, y_max)
        return left, top, right, bottom

    def letterbox_to_heatmap(
        self, x: float, y: float, stride: int
    ) -> Tuple[float, float]:
        """Map letterbox pixels to continuous stride-space heatmap coordinates."""

        _require_positive_integer(stride, "stride")
        _require_finite(x, "x")
        _require_finite(y, "y")
        return x / stride, y / stride

    def heatmap_to_original(
        self, x: float, y: float, stride: int
    ) -> Tuple[float, float]:
        """Invert continuous heatmap coordinates to original-image pixels."""

        _require_positive_integer(stride, "stride")
        return self.inverse_point(x * stride, y * stride)

    def grid_cell_center_to_original(
        self, grid_x: int, grid_y: int, stride: int
    ) -> Tuple[float, float]:
        """Map a quantized grid-cell centre to original-image pixel coordinates."""

        _require_positive_integer(stride, "stride")
        grid_size = self.input_size // stride
        if self.input_size % stride != 0:
            raise GeometryError("input_size must be divisible by stride")
        if not 0 <= grid_x < grid_size or not 0 <= grid_y < grid_size:
            raise GeometryError("grid coordinate is outside the heatmap")
        return self.inverse_point((grid_x + 0.5) * stride, (grid_y + 0.5) * stride)


def letterbox_rgb(
    image: np.ndarray, input_size: int, pad_value: int = 114
) -> Tuple[np.ndarray, LetterboxTransform]:
    """Letterbox an RGB ``uint8 [H,W,3]`` image to ``uint8 [S,S,3]``.

    The operation preserves aspect ratio, performs no crop, and keeps padding
    metadata in the returned ``LetterboxTransform``.
    """

    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise GeometryError("image must have RGB shape [H,W,3]")
    if image.shape[0] <= 0 or image.shape[1] <= 0:
        raise GeometryError("image dimensions must be positive")
    _require_positive_integer(input_size, "input_size")
    if not isinstance(pad_value, int) or not 0 <= pad_value <= 255:
        raise GeometryError("pad_value must be an integer in [0,255]")

    transform = LetterboxTransform.from_image_size(
        original_width=image.shape[1],
        original_height=image.shape[0],
        input_size=input_size,
    )
    resized = cv2.resize(
        image,
        (transform.resized_width, transform.resized_height),
        interpolation=cv2.INTER_LINEAR,
    )
    output = np.full((input_size, input_size, 3), pad_value, dtype=image.dtype)
    output[
        transform.pad_top : transform.pad_top + transform.resized_height,
        transform.pad_left : transform.pad_left + transform.resized_width,
    ] = resized
    return output, transform


def _require_positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise GeometryError("{} must be a positive integer".format(name))


def _require_finite(value: float, name: str) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not isfinite(value):
        raise GeometryError("{} must be finite".format(name))
