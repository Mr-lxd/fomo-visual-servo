"""Foreground grid-cell classification metrics for FOMO validation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


class MetricError(ValueError):
    """Raised when FOMO prediction and target class-index tensors are invalid."""


@dataclass(frozen=True)
class GridMetrics:
    """Grid-level foreground micro counts and grid precision/recall/F1 values."""

    true_positives: int
    false_positives: int
    false_negatives: int
    grid_precision: float
    grid_recall: float
    grid_f1: float

    @property
    def precision(self) -> float:
        """Backward-compatible alias for ``grid_precision``."""

        return self.grid_precision

    @property
    def recall(self) -> float:
        """Backward-compatible alias for ``grid_recall``."""

        return self.grid_recall

    @property
    def f1(self) -> float:
        """Backward-compatible alias for ``grid_f1``."""

        return self.grid_f1


ForegroundMetrics = GridMetrics


def foreground_micro_metrics(predictions: Tensor, targets: Tensor) -> GridMetrics:
    """Compute foreground micro metrics from class-index tensors [B,G,G].

    Background class zero is excluded. A foreground wrong-class prediction is both a
    false positive for the predicted class and a false negative for the target class.
    """

    if not isinstance(predictions, Tensor) or predictions.ndim != 3:
        raise MetricError("predictions must have shape [B,G,G]")
    if not isinstance(targets, Tensor) or targets.ndim != 3:
        raise MetricError("targets must have shape [B,G,G]")
    if predictions.shape != targets.shape:
        raise MetricError("predictions and targets must have identical shapes")
    if predictions.dtype != torch.int64 or targets.dtype != torch.int64:
        raise MetricError("predictions and targets must have dtype torch.int64")

    target_foreground = targets > 0
    prediction_foreground = predictions > 0
    correct = predictions == targets
    true_positives = int((correct & target_foreground).sum().item())
    false_positives = int((prediction_foreground & ~correct).sum().item())
    false_negatives = int((target_foreground & ~correct).sum().item())
    precision_denominator = true_positives + false_positives
    recall_denominator = true_positives + false_negatives
    precision = (
        true_positives / precision_denominator if precision_denominator else 0.0
    )
    recall = true_positives / recall_denominator if recall_denominator else 0.0
    f1_denominator = precision + recall
    f1 = 2.0 * precision * recall / f1_denominator if f1_denominator else 0.0
    return GridMetrics(
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        grid_precision=precision,
        grid_recall=recall,
        grid_f1=f1,
    )
