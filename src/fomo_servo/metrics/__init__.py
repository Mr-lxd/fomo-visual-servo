"""Public metrics with lazy imports so video deployment stays torch-free."""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORT_MODULES = {
    "ForegroundMetrics": ".classification",
    "GridMetrics": ".classification",
    "MetricError": ".classification",
    "foreground_micro_metrics": ".classification",
    "CentroidEvaluation": ".centroid",
    "CentroidEvaluator": ".centroid",
    "GroundTruthCentroid": ".centroid",
    "ThresholdSweepResult": ".centroid",
    "ground_truths_from_boxes": ".centroid",
    "sweep_confidence_thresholds": ".centroid",
    "SequenceStatistics": ".sequence",
    "SequenceStatisticsResult": ".sequence",
    "normalize_centroid": ".sequence",
    "CentroidPRAUC": ".pr_auc",
    "ClassPRAUC": ".pr_auc",
    "PRAUCError": ".pr_auc",
    "PRPoint": ".pr_auc",
    "centroid_pr_auc": ".pr_auc",
}


def __getattr__(name: str) -> Any:
    """Load a metric family only when one of its public names is requested."""

    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError("module {!r} has no attribute {!r}".format(__name__, name))
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


__all__ = list(_EXPORT_MODULES)
