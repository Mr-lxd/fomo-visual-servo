"""Centroid postprocess threshold-grid precision/recall curve integration."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Mapping, Optional, Sequence

from fomo_servo.metrics.centroid import CentroidEvaluation


class PRAUCError(ValueError):
    """Raised when a threshold-sweep result cannot define a PR curve."""


@dataclass(frozen=True)
class PRPoint:
    """One raw centroid precision/recall point at an applied postprocess threshold."""

    threshold: float
    precision: float
    recall: float


@dataclass(frozen=True)
class ClassPRAUC:
    """Raw class curve and observed-grid trapezoidal area; null means no GT."""

    class_name: str
    ground_truth_count: int
    auc: Optional[float]
    raw_points: tuple[PRPoint, ...]


@dataclass(frozen=True)
class CentroidPRAUC:
    """Macro/micro centroid PR-AUC summary with raw per-class curves."""

    macro_auc: Optional[float]
    micro_auc: Optional[float]
    macro_effective_class_count: int
    per_class: Mapping[str, ClassPRAUC]
    micro_points: tuple[PRPoint, ...]
    integration: str = "trapezoidal_observed_recall_no_envelope"


def centroid_pr_auc(
    results: Mapping[float, CentroidEvaluation], class_names: Sequence[str]
) -> CentroidPRAUC:
    """Compute deterministic observed-grid PR-AUC from centroid sweep results.

    Raw threshold points are sorted ascending by threshold.  Integration sorts
    by recall, retains maximum precision for duplicate recall, and performs a
    trapezoidal integral without endpoint extrapolation or a precision envelope.
    """

    if not results:
        raise PRAUCError("threshold sweep results must not be empty")
    names = tuple(class_names)
    if not names or len(set(names)) != len(names):
        raise PRAUCError("class_names must be unique and non-empty")
    ordered = _ordered_results(results)
    per_class: dict[str, ClassPRAUC] = {}
    macro_values: list[float] = []
    for name in names:
        raw = tuple(
            PRPoint(
                threshold,
                _metric_number(result.per_class_precision_recall_f1, name, "precision"),
                _metric_number(result.per_class_precision_recall_f1, name, "recall"),
            )
            for threshold, result in ordered
        )
        ground_truth_count = int(
            _metric_number(ordered[0][1].per_class_precision_recall_f1, name, "true_positives")
            + _metric_number(ordered[0][1].per_class_precision_recall_f1, name, "false_negatives")
        )
        auc = _observed_auc(raw) if ground_truth_count > 0 else None
        per_class[name] = ClassPRAUC(name, ground_truth_count, auc, raw)
        if auc is not None:
            macro_values.append(auc)
    micro_points = tuple(
        PRPoint(threshold, result.centroid_precision, result.centroid_recall)
        for threshold, result in ordered
    )
    micro_ground_truth_count = ordered[0][1].true_positives + ordered[0][1].false_negatives
    return CentroidPRAUC(
        macro_auc=(sum(macro_values) / len(macro_values) if macro_values else None),
        micro_auc=(_observed_auc(micro_points) if micro_ground_truth_count > 0 else None),
        macro_effective_class_count=len(macro_values),
        per_class=per_class,
        micro_points=micro_points,
    )


def _ordered_results(
    results: Mapping[float, CentroidEvaluation]
) -> tuple[tuple[float, CentroidEvaluation], ...]:
    ordered = []
    for threshold, result in results.items():
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not isfinite(float(threshold)):
            raise PRAUCError("thresholds must be finite numbers")
        ordered.append((float(threshold), result))
    return tuple(sorted(ordered, key=lambda item: item[0]))


def _metric_number(metrics: Mapping[str, Mapping[str, float]], name: str, key: str) -> float:
    try:
        value = metrics[name][key]
    except KeyError as error:
        raise PRAUCError("missing per-class '{}' metric for '{}'".format(key, name)) from error
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(float(value)):
        raise PRAUCError("per-class '{}' metric for '{}' must be finite".format(key, name))
    return float(value)


def _observed_auc(points: Sequence[PRPoint]) -> float:
    """Integrate observed PR coordinates; one unique recall value has zero width."""

    recall_to_precision: dict[float, float] = {}
    for point in points:
        if not (0.0 <= point.precision <= 1.0 and 0.0 <= point.recall <= 1.0):
            raise PRAUCError("precision and recall must be probabilities")
        recall_to_precision[point.recall] = max(
            point.precision, recall_to_precision.get(point.recall, 0.0)
        )
    ordered = sorted(recall_to_precision.items())
    if len(ordered) < 2:
        return 0.0
    area = 0.0
    for (left_recall, left_precision), (right_recall, right_precision) in zip(
        ordered, ordered[1:]
    ):
        area += (right_recall - left_recall) * (left_precision + right_precision) / 2.0
    return float(area)


__all__ = ["CentroidPRAUC", "ClassPRAUC", "PRAUCError", "PRPoint", "centroid_pr_auc"]
