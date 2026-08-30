"""Deterministic target selection strategies for centroid detections."""

from __future__ import annotations

from math import hypot
from typing import Optional, Sequence, Tuple

from .detections import Detection, PostprocessError


def select_target(
    detections: Sequence[Detection],
    strategy: str,
    *,
    previous_centroid: Optional[Tuple[float, float]] = None,
    max_match_distance: Optional[float] = None,
    allowed_class_ids: Optional[Sequence[int]] = None,
) -> Optional[Detection]:
    """Select one detection with deterministic confidence/area/distance tie-breaks."""

    candidates = [item for item in detections if _allowed_class(item, allowed_class_ids)]
    if not candidates:
        return None
    if strategy == "highest_confidence":
        return min(candidates, key=lambda item: (-item.confidence, -item.component_area_cells, item.class_id, item.original_y, item.original_x))
    if strategy == "largest_component":
        return min(candidates, key=lambda item: (-item.component_area_cells, -item.confidence, item.class_id, item.original_y, item.original_x))
    if strategy != "nearest_previous":
        raise PostprocessError(
            "strategy must be 'highest_confidence', 'largest_component', or 'nearest_previous'"
        )
    if previous_centroid is None:
        return min(candidates, key=lambda item: (-item.confidence, -item.component_area_cells, item.class_id, item.original_y, item.original_x))
    if max_match_distance is None or max_match_distance < 0:
        raise PostprocessError("max_match_distance must be non-negative for nearest_previous")
    ranked = sorted(
        [
            (
            hypot(item.original_x - previous_centroid[0], item.original_y - previous_centroid[1]),
            item,
            )
            for item in candidates
        ],
        key=lambda pair: pair[0],
    )
    valid = [pair for pair in ranked if pair[0] <= max_match_distance]
    if not valid:
        return None
    return min(valid, key=lambda pair: (pair[0], -pair[1].confidence, -pair[1].component_area_cells, pair[1].class_id, pair[1].original_y, pair[1].original_x))[1]


def _allowed_class(item: Detection, allowed_class_ids: Optional[Sequence[int]]) -> bool:
    if allowed_class_ids is None:
        return True
    return item.class_id in set(allowed_class_ids)
