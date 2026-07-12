"""Stateful target tracking independent from stateless heatmap postprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from .detections import Detection, PostprocessError
from .selection import select_target


@dataclass(frozen=True)
class TrackingResult:
    """One tracker update result with a public status and optional selected target."""

    status: str
    detection: Optional[Detection]
    lost_frames: int


class TargetTracker:
    """Track one selected centroid using only visual detections and bounded memory."""

    VALID_STATES = {"idle", "detected", "lost", "reacquired"}

    def __init__(
        self,
        *,
        strategy: str = "highest_confidence",
        max_match_distance: float = 32.0,
        max_lost_frames: int = 5,
        allowed_class_ids: Optional[Sequence[int]] = None,
    ) -> None:
        if max_match_distance < 0:
            raise PostprocessError("max_match_distance must be non-negative")
        if isinstance(max_lost_frames, bool) or not isinstance(max_lost_frames, int) or max_lost_frames < 0:
            raise PostprocessError("max_lost_frames must be a non-negative integer")
        self.strategy = strategy
        self.max_match_distance = float(max_match_distance)
        self.max_lost_frames = max_lost_frames
        self.allowed_class_ids = None if allowed_class_ids is None else tuple(allowed_class_ids)
        self.state = "idle"
        self.lost_frames = 0
        self.previous_centroid: Optional[Tuple[float, float]] = None

    def update(self, detections: Sequence[Detection]) -> TrackingResult:
        """Consume one frame's detections and return ``idle/detected/lost/reacquired``."""

        selected = select_target(
            detections,
            self.strategy,
            previous_centroid=self.previous_centroid,
            max_match_distance=self.max_match_distance,
            allowed_class_ids=self.allowed_class_ids,
        )
        if selected is not None:
            status = "reacquired" if self.state == "lost" else "detected"
            self.state = status
            self.lost_frames = 0
            self.previous_centroid = (selected.original_x, selected.original_y)
            return TrackingResult(status, selected, 0)
        if self.state in {"detected", "reacquired", "lost"}:
            self.lost_frames += 1
            if self.lost_frames > self.max_lost_frames:
                self.state = "idle"
                self.lost_frames = 0
                self.previous_centroid = None
                return TrackingResult("idle", None, 0)
            self.state = "lost"
            return TrackingResult("lost", None, self.lost_frames)
        self.state = "idle"
        return TrackingResult("idle", None, 0)
