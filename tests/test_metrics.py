"""Tests for foreground micro precision, recall, and F1 metrics."""

from __future__ import annotations

import importlib
from typing import Any, Callable

import pytest
import torch


def _metrics_api() -> Callable[[torch.Tensor, torch.Tensor], Any] | None:
    """Return the public metric function without failing import collection."""

    module = importlib.import_module("fomo_servo.metrics")
    return getattr(module, "foreground_micro_metrics", None)


def test_foreground_micro_metrics_count_multiclass_misclassifications() -> None:
    """A wrong foreground class contributes both one FP and one FN."""

    metric_function = _metrics_api()
    assert callable(metric_function), "fomo_servo.metrics.foreground_micro_metrics must exist"
    predictions = torch.tensor([[[0, 1], [2, 1]]], dtype=torch.int64)
    targets = torch.tensor([[[0, 1], [1, 2]]], dtype=torch.int64)

    metrics = metric_function(predictions, targets)

    assert metrics.true_positives == 1
    assert metrics.false_positives == 2
    assert metrics.false_negatives == 2
    assert metrics.precision == pytest.approx(1.0 / 3.0)
    assert metrics.recall == pytest.approx(1.0 / 3.0)
    assert metrics.f1 == pytest.approx(1.0 / 3.0)


def test_foreground_micro_metrics_return_zero_when_no_foreground_exists() -> None:
    """An all-background validation batch has defined zero metrics."""

    metric_function = _metrics_api()
    assert callable(metric_function), "fomo_servo.metrics.foreground_micro_metrics must exist"

    metrics = metric_function(
        torch.zeros(1, 2, 2, dtype=torch.int64),
        torch.zeros(1, 2, 2, dtype=torch.int64),
    )

    assert metrics.precision == 0.0
    assert metrics.recall == 0.0
    assert metrics.f1 == 0.0
