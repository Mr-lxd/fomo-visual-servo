"""Visual-sequence stability statistics for selected FOMO targets."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot, sqrt
from statistics import mean, pstdev
from typing import Optional, Tuple

from fomo_servo.postprocess import Detection


def normalize_centroid(
    original_x: float, original_y: float, original_width: int, original_height: int
) -> Tuple[float, float]:
    """Map original pixels to visual-servo coordinates in ``[-1,1]``."""

    if original_width <= 0 or original_height <= 0:
        raise ValueError("original image dimensions must be positive")
    return (
        2.0 * original_x / original_width - 1.0,
        2.0 * original_y / original_height - 1.0,
    )


@dataclass(frozen=True)
class SequenceStatisticsResult:
    """Aggregated stability statistics; these are not MOT identity metrics."""

    processed_frame_count: int
    detected_frame_count: int
    lost_frame_count: int
    detection_availability: float
    target_loss_rate: float
    reacquisition_count: int
    jitter_mean_pixels: float
    jitter_std_pixels: float
    jitter_rms_pixels: float
    jitter_mean_normalized: float
    jitter_std_normalized: float
    jitter_rms_normalized: float


class SequenceStatistics:
    """Track selected-centroid jitter, availability, losses, and reacquisitions."""

    def __init__(self) -> None:
        self.processed_frame_count = 0
        self.detected_frame_count = 0
        self.lost_frame_count = 0
        self.reacquisition_count = 0
        self._previous: Optional[Tuple[float, float]] = None
        self._pixel_jitters: list[float] = []
        self._normalized_jitters: list[float] = []

    def update(
        self,
        status: str,
        detection: Optional[Detection],
        original_width: int,
        original_height: int,
    ) -> None:
        """Consume one processed frame; jitter uses only detected/reacquired selections."""

        self.processed_frame_count += 1
        if status == "lost":
            self.lost_frame_count += 1
            self._previous = None
        elif status == "idle":
            self._previous = None
        if status in {"detected", "reacquired"} and detection is not None:
            self.detected_frame_count += 1
            if status == "reacquired":
                self.reacquisition_count += 1
            current = (detection.original_x, detection.original_y)
            if self._previous is not None:
                self._pixel_jitters.append(hypot(current[0] - self._previous[0], current[1] - self._previous[1]))
                previous_normalized = normalize_centroid(*self._previous, original_width, original_height)
                current_normalized = normalize_centroid(*current, original_width, original_height)
                self._normalized_jitters.append(
                    hypot(current_normalized[0] - previous_normalized[0], current_normalized[1] - previous_normalized[1])
                )
            self._previous = current

    def summary(self) -> SequenceStatisticsResult:
        """Return pixel and normalized jitter moments plus availability/loss rates."""

        pixel = self._moments(self._pixel_jitters)
        normalized = self._moments(self._normalized_jitters)
        count = self.processed_frame_count
        return SequenceStatisticsResult(
            processed_frame_count=count,
            detected_frame_count=self.detected_frame_count,
            lost_frame_count=self.lost_frame_count,
            detection_availability=self.detected_frame_count / count if count else 0.0,
            target_loss_rate=self.lost_frame_count / count if count else 0.0,
            reacquisition_count=self.reacquisition_count,
            jitter_mean_pixels=pixel[0],
            jitter_std_pixels=pixel[1],
            jitter_rms_pixels=pixel[2],
            jitter_mean_normalized=normalized[0],
            jitter_std_normalized=normalized[1],
            jitter_rms_normalized=normalized[2],
        )

    @staticmethod
    def _moments(values: list[float]) -> Tuple[float, float, float]:
        if not values:
            return 0.0, 0.0, 0.0
        return mean(values), pstdev(values) if len(values) > 1 else 0.0, sqrt(mean(value * value for value in values))
