"""Vision-domain frame model shared by realtime consumers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from .protocol import MAX_DECODED_PIXELS, MAX_DIMENSION


class PixelFormat(str, Enum):
    """Raw in-process pixel formats. This is not the RBVS payload codec."""

    BGR8 = "bgr8"


@dataclass(frozen=True)
class VisionFrame:
    """One raw frame captured by the single CameraOwner."""

    frame_id: int
    capture_timestamp_ns: int
    width: int
    height: int
    pixel_format: PixelFormat
    image: Any

    def __post_init__(self) -> None:
        if self.frame_id < 0:
            raise ValueError("frame_id must be non-negative")
        if self.capture_timestamp_ns <= 0:
            raise ValueError("capture_timestamp_ns must be positive")
        if not 1 <= self.width <= MAX_DIMENSION:
            raise ValueError("frame width is outside the supported range")
        if not 1 <= self.height <= MAX_DIMENSION:
            raise ValueError("frame height is outside the supported range")
        if self.width * self.height > MAX_DECODED_PIXELS:
            raise ValueError("frame decoded pixel count exceeds the supported limit")
