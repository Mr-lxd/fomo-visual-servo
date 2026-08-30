from __future__ import annotations

import pytest
import torch

from fomo_servo.geometry import LetterboxTransform
from fomo_servo.metrics import (
    CentroidEvaluator,
    GroundTruthCentroid,
    sweep_confidence_thresholds,
)
from fomo_servo.postprocess import Detection


def _prediction(class_id: int, x: float, confidence: float = 0.9) -> Detection:
    return Detection(
        class_id=class_id,
        class_name=("fish", "crab")[class_id],
        confidence=confidence,
        mean_confidence=confidence,
        component_area_cells=1,
        heatmap_x=x / 8.0,
        heatmap_y=1.0,
        input_x=x,
        input_y=8.0,
        original_x=x,
        original_y=8.0,
    )


def _truth(class_id: int, x: float, name: str) -> GroundTruthCentroid:
    return GroundTruthCentroid(
        class_id=class_id,
        class_name=name,
        original_x=x,
        original_y=8.0,
        x_min=x - 3.0,
        y_min=5.0,
        x_max=x + 3.0,
        y_max=11.0,
    )


def test_centroid_matching_is_one_to_one_and_rejects_wrong_class() -> None:
    evaluator = CentroidEvaluator(("fish", "crab"), matching_mode="max_distance_pixels", max_distance_pixels=5.0)
    result = evaluator.evaluate_dataset(
        predictions=((_prediction(0, 10.0), _prediction(0, 10.5), _prediction(1, 20.0)),),
        ground_truths=((_truth(0, 10.0, "fish"), _truth(1, 20.0, "crab")),),
    )

    assert result.centroid_precision == pytest.approx(2.0 / 3.0)
    assert result.centroid_recall == 1.0
    assert result.centroid_f1 == pytest.approx(0.8)
    assert result.count_error_per_image == (1,)
    assert result.mean_count_bias == pytest.approx(1.0)
    assert result.mean_absolute_count_error == pytest.approx(1.0)
    assert result.confusion_matrix.tolist() == [[1, 0], [0, 1]]


def test_centroid_in_bbox_and_threshold_sweep() -> None:
    evaluator = CentroidEvaluator(("fish",), matching_mode="centroid_in_bbox")
    result = evaluator.evaluate_dataset(
        predictions=((_prediction(0, 10.0, 0.9), _prediction(0, 30.0, 0.1)),),
        ground_truths=((_truth(0, 10.0, "fish"),),),
    )
    assert result.centroid_f1 == pytest.approx(2.0 / 3.0)

    logits = (torch.tensor([[[[4.0, 0.0], [0.0, 0.0]], [[4.0, 0.0], [0.0, 0.0]]]]),)
    sweep = sweep_confidence_thresholds(
        logits=logits,
        transforms=(LetterboxTransform.from_image_size(16, 16, 16),),
        ground_truths=((_truth(0, 4.0, "fish"),),),
        class_names=("fish",),
        stride=8,
        thresholds=(0.05, 0.95),
        matching_mode="max_distance_pixels",
        max_distance_pixels=5.0,
    )
    assert sweep.best_threshold in {0.05, 0.95}
