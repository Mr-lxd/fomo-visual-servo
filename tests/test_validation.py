from __future__ import annotations

import torch

from fomo_servo.config import EvaluationConfig, PostprocessConfig
from fomo_servo.evaluation import evaluate_logit_collection
from fomo_servo.geometry import LetterboxTransform
from fomo_servo.metrics import GroundTruthCentroid


def test_validation_report_contains_grid_and_centroid_metrics() -> None:
    logits = (torch.tensor([[[[0.0, 0.0], [0.0, 0.0]], [[5.0, 0.0], [0.0, 0.0]]]]),)
    targets = (torch.tensor([[1, 0], [0, 0]], dtype=torch.int64),)
    transforms = (LetterboxTransform.from_image_size(16, 16, 16),)
    ground_truths = (
        (
            GroundTruthCentroid(
                class_id=0,
                class_name="creature",
                original_x=4.0,
                original_y=4.0,
                x_min=0.0,
                y_min=0.0,
                x_max=8.0,
                y_max=8.0,
            ),
        ),
    )

    report = evaluate_logit_collection(
        logits=logits,
        targets=targets,
        transforms=transforms,
        ground_truths=ground_truths,
        class_names=("creature",),
        stride=8,
        postprocess_config=PostprocessConfig(confidence_threshold=0.05),
        evaluation_config=EvaluationConfig(threshold_sweep=(0.05, 0.95)),
    )

    assert report.grid_metrics.grid_f1 == 1.0
    assert report.centroid_metrics.centroid_f1 == 1.0
    assert report.best_centroid_metrics.centroid_f1 == 1.0
    assert report.best_threshold in {0.05, 0.95}
    payload = report.as_dict(
        class_weight_mode="auto",
        class_weights=(1.0, 25.0),
        class_statistics=(
            {
                "class_id": 0,
                "class_name": "creature",
                "image_count": 1,
                "bbox_count": 1,
                "encoded_centroid_cell_count": 1,
                "same_class_collision_count": 0,
                "different_class_collision_count": 0,
            },
        ),
    )
    assert payload["mean_count_bias"] == 0.0
    assert payload["mean_absolute_count_error"] == 0.0
    assert payload["class_weight_mode"] == "auto"
    assert payload["class_weights"] == [1.0, 25.0]
    assert payload["class_statistics"][0]["encoded_centroid_cell_count"] == 1
