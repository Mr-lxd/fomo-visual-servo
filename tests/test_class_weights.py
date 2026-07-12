"""Tests for manual and automatic FOMO class-weight resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from fomo_servo.datasets import YOLOv5FOMODataset
from fomo_servo.training.class_weights import (
    AutoClassWeightSettings,
    ClassWeightError,
    ClassTrainingStatistics,
    collect_training_heatmap_statistics,
    resolve_auto_class_weights,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = ROOT / "tests" / "fixtures" / "yolo_micro"


def _settings(**overrides: float) -> AutoClassWeightSettings:
    values = {
        "background_weight": 1.0,
        "foreground_base_weight": 25.0,
        "class_balance": "sqrt_inverse_frequency",
        "min_foreground_weight": 12.5,
        "max_foreground_weight": 75.0,
    }
    values.update(overrides)
    return AutoClassWeightSettings(**values)


def _statistics(class_id: int, count: int) -> ClassTrainingStatistics:
    return ClassTrainingStatistics(
        class_id=class_id,
        class_name=("fish", "crab")[class_id],
        image_count=1,
        bbox_count=count,
        encoded_centroid_cell_count=count,
        same_class_collision_count=0,
        different_class_collision_count=0,
    )


def test_auto_weights_keep_balanced_classes_at_base_weight() -> None:
    weights = resolve_auto_class_weights(
        (_statistics(0, 100), _statistics(1, 100)), _settings()
    )

    assert weights == pytest.approx((1.0, 25.0, 25.0))


def test_auto_weights_sqrt_balance_imbalanced_classes() -> None:
    weights = resolve_auto_class_weights(
        (_statistics(0, 100), _statistics(1, 25)), _settings()
    )

    assert weights[1] == pytest.approx(25.0 * (62.5 / 100.0) ** 0.5)
    assert weights[2] == pytest.approx(25.0 * (62.5 / 25.0) ** 0.5)
    assert weights[2] > weights[1]


def test_auto_weights_reject_zero_encoded_class_count() -> None:
    with pytest.raises(ClassWeightError, match="encoded centroid cell count is zero"):
        resolve_auto_class_weights(
            (_statistics(0, 10), _statistics(1, 0)), _settings()
        )


def test_auto_weights_apply_configured_limits() -> None:
    weights = resolve_auto_class_weights(
        (_statistics(0, 100), _statistics(1, 1)), _settings()
    )

    assert weights[1] == pytest.approx(17.7658387729)
    assert weights[2] == 75.0


def test_dataset_statistics_count_encoded_heatmap_cells_and_collisions() -> None:
    dataset = YOLOv5FOMODataset(
        FIXTURE_ROOT,
        split="train",
        input_size=96,
        stride=8,
        class_mode="merge_single",
    )

    statistics = collect_training_heatmap_statistics(dataset)

    assert len(statistics) == 1
    assert statistics[0].image_count > 0
    assert statistics[0].bbox_count > 0
    assert statistics[0].encoded_centroid_cell_count > 0
