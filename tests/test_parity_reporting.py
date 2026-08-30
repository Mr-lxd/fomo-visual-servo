"""Tests for JSON-safe reports shared by local and EI parity evaluation."""

from __future__ import annotations

from fomo_servo.evaluation.parity_reporting import (
    edge_detections_from_local,
    edge_ground_truths_from_local,
    evaluate_local_parity,
    serialize_edge_evaluation,
)
from fomo_servo.metrics import GroundTruthCentroid
from fomo_servo.postprocess import Detection


def _detection(class_id: int, x: float, y: float) -> Detection:
    return Detection(
        class_id=class_id,
        class_name=("fish", "shark")[class_id],
        confidence=0.9,
        mean_confidence=0.9,
        component_area_cells=1,
        heatmap_x=x / 8.0,
        heatmap_y=y / 8.0,
        input_x=x,
        input_y=y,
        original_x=x,
        original_y=y,
    )


def _ground_truth(class_id: int, x: float, y: float) -> GroundTruthCentroid:
    return GroundTruthCentroid(
        class_id=class_id,
        class_name=("fish", "shark")[class_id],
        original_x=x,
        original_y=y,
        x_min=x - 2.0,
        y_min=y - 2.0,
        x_max=x + 2.0,
        y_max=y + 2.0,
    )


def test_local_records_include_image_matches_and_counts() -> None:
    report = evaluate_local_parity(
        predictions=((_detection(0, 10.0, 10.0),),),
        ground_truths=((_ground_truth(0, 10.0, 10.0),),),
        class_names=("fish", "shark"),
        matching_mode="centroid_in_bbox",
        max_distance_pixels=32.0,
    )

    assert report["true_positives"] == 1
    assert report["false_positives"] == 0
    assert report["false_negatives"] == 0
    assert report["prediction_count"] == 1
    assert report["ground_truth_count"] == 1
    assert report["image_results"][0]["matches"] == [
        {"prediction_index": 0, "ground_truth_index": 0, "distance_pixels": 0.0}
    ]


def test_edge_conversion_and_serialization_keep_original_coordinates() -> None:
    detections = edge_detections_from_local((_detection(1, 12.0, 16.0),))
    ground_truths = edge_ground_truths_from_local((_ground_truth(1, 12.0, 16.0),))

    payload = serialize_edge_evaluation(
        predictions=(detections,),
        ground_truths=(ground_truths,),
        image_sizes=((192, 192),),
        class_names=("fish", "shark"),
        mode="strict_one_to_one",
    )

    assert payload["true_positives"] == 1
    assert payload["prediction_count"] == 1
    assert payload["image_results"][0]["matches"][0]["normalized_distance"] == 0.0
