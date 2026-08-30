from __future__ import annotations

from fomo_servo.postprocess import Detection, select_target


def _detection(class_id: int, confidence: float, area: int, x: float) -> Detection:
    return Detection(
        class_id=class_id,
        class_name=("fish", "crab")[class_id],
        confidence=confidence,
        mean_confidence=confidence - 0.1,
        component_area_cells=area,
        heatmap_x=x / 8.0,
        heatmap_y=1.0,
        input_x=x,
        input_y=8.0,
        original_x=x,
        original_y=8.0,
    )


def test_highest_confidence_and_largest_component_are_deterministic() -> None:
    detections = (_detection(0, 0.8, 2, 10.0), _detection(1, 0.7, 4, 20.0))

    assert select_target(detections, "highest_confidence") == detections[0]
    assert select_target(detections, "largest_component") == detections[1]


def test_nearest_previous_applies_distance_and_class_filter() -> None:
    detections = (_detection(0, 0.8, 2, 10.0), _detection(1, 0.9, 4, 20.0))

    selected = select_target(
        detections,
        "nearest_previous",
        previous_centroid=(11.0, 8.0),
        max_match_distance=5.0,
        allowed_class_ids=(0,),
    )
    assert selected == detections[0]
    assert select_target(
        detections,
        "nearest_previous",
        previous_centroid=(100.0, 100.0),
        max_match_distance=5.0,
    ) is None
