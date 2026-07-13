"""Centroid-level FOMO evaluation with deterministic greedy matching."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot, isfinite
from statistics import mean, median
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor

from fomo_servo.datasets.yolo import AbsoluteBox
from fomo_servo.geometry import LetterboxTransform
from fomo_servo.postprocess import Detection, postprocess_logits


class CentroidMetricError(ValueError):
    """Raised when centroid predictions, ground truth, or matching settings are invalid."""


@dataclass(frozen=True)
class GroundTruthCentroid:
    """One ground-truth centroid and its original-image bbox in pixels."""

    class_id: int
    class_name: str
    original_x: float
    original_y: float
    x_min: float
    y_min: float
    x_max: float
    y_max: float


@dataclass(frozen=True)
class CentroidEvaluation:
    """Aggregated centroid precision/recall, localization, confusion, and count errors."""

    centroid_precision: float
    centroid_recall: float
    centroid_f1: float
    per_class_precision_recall_f1: Mapping[str, Mapping[str, float]]
    confusion_matrix: np.ndarray
    mean_localization_error_pixels: float
    median_localization_error_pixels: float
    count_error_per_image: Tuple[int, ...]
    mean_count_bias: float
    mean_absolute_count_error: float
    true_positives: int
    false_positives: int
    false_negatives: int


@dataclass(frozen=True)
class ThresholdSweepResult:
    """Validation threshold sweep with the best threshold selected by centroid F1."""

    best_threshold: float
    best_result: CentroidEvaluation
    results: Mapping[float, CentroidEvaluation]


class CentroidEvaluator:
    """Match detections to same-class ground truths one-to-one by distance."""

    def __init__(
        self,
        class_names: Sequence[str],
        *,
        matching_mode: str = "centroid_in_bbox",
        max_distance_pixels: float = 32.0,
    ) -> None:
        self.class_names = tuple(class_names)
        if not self.class_names or len(set(self.class_names)) != len(self.class_names):
            raise CentroidMetricError("class_names must contain unique non-empty names")
        if matching_mode not in {"centroid_in_bbox", "max_distance_pixels"}:
            raise CentroidMetricError(
                "matching_mode must be 'centroid_in_bbox' or 'max_distance_pixels'"
            )
        if not isinstance(max_distance_pixels, (int, float)) or isinstance(max_distance_pixels, bool) or not isfinite(float(max_distance_pixels)) or max_distance_pixels < 0:
            raise CentroidMetricError("max_distance_pixels must be finite and non-negative")
        self.matching_mode = matching_mode
        self.max_distance_pixels = float(max_distance_pixels)

    def evaluate_image(
        self,
        predictions: Sequence[Detection],
        ground_truths: Sequence[GroundTruthCentroid],
    ) -> CentroidEvaluation:
        """Evaluate one image; ``count_error_per_image`` contains one signed count."""

        return self.evaluate_dataset((predictions,), (ground_truths,))

    def evaluate_dataset(
        self,
        predictions: Sequence[Sequence[Detection]],
        ground_truths: Sequence[Sequence[GroundTruthCentroid]],
    ) -> CentroidEvaluation:
        """Aggregate image-wise greedy matches without matching objects across images."""

        if len(predictions) != len(ground_truths):
            raise CentroidMetricError("predictions and ground_truths must have equal image counts")
        class_counts = {name: [0, 0, 0] for name in self.class_names}
        confusion = np.zeros((len(self.class_names), len(self.class_names)), dtype=np.int64)
        distances = []
        count_errors = []
        true_positives = false_positives = false_negatives = 0
        for image_predictions, image_ground_truths in zip(predictions, ground_truths):
            _validate_items(image_predictions, image_ground_truths, self.class_names)
            matches, unmatched_predictions, unmatched_ground_truths = self._match(
                image_predictions, image_ground_truths
            )
            true_positives += len(matches)
            false_positives += len(unmatched_predictions)
            false_negatives += len(unmatched_ground_truths)
            count_errors.append(len(image_predictions) - len(image_ground_truths))
            for prediction_index, ground_truth_index, distance in matches:
                prediction = image_predictions[prediction_index]
                ground_truth = image_ground_truths[ground_truth_index]
                class_counts[ground_truth.class_name][0] += 1
                distances.append(distance)
            for prediction_index in unmatched_predictions:
                class_counts[image_predictions[prediction_index].class_name][1] += 1
            for ground_truth_index in unmatched_ground_truths:
                class_counts[image_ground_truths[ground_truth_index].class_name][2] += 1
            for prediction_index, ground_truth_index, _ in self._confusion_matches(
                image_predictions, image_ground_truths
            ):
                confusion[
                    image_ground_truths[ground_truth_index].class_id,
                    image_predictions[prediction_index].class_id,
                ] += 1

        per_class = {}
        for name, (true_positive, false_positive, false_negative) in class_counts.items():
            per_class[name] = _metrics_dict(true_positive, false_positive, false_negative)
        return CentroidEvaluation(
            centroid_precision=_precision(true_positives, false_positives),
            centroid_recall=_recall(true_positives, false_negatives),
            centroid_f1=_f1(
                _precision(true_positives, false_positives),
                _recall(true_positives, false_negatives),
            ),
            per_class_precision_recall_f1=per_class,
            confusion_matrix=confusion,
            mean_localization_error_pixels=mean(distances) if distances else 0.0,
            median_localization_error_pixels=median(distances) if distances else 0.0,
            count_error_per_image=tuple(count_errors),
            mean_count_bias=mean(count_errors) if count_errors else 0.0,
            mean_absolute_count_error=(
                mean(abs(value) for value in count_errors) if count_errors else 0.0
            ),
            true_positives=true_positives,
            false_positives=false_positives,
            false_negatives=false_negatives,
        )

    def _match(
        self,
        predictions: Sequence[Detection],
        ground_truths: Sequence[GroundTruthCentroid],
    ) -> Tuple[Tuple[Tuple[int, int, float], ...], Tuple[int, ...], Tuple[int, ...]]:
        pairs = []
        for prediction_index, prediction in enumerate(predictions):
            for ground_truth_index, ground_truth in enumerate(ground_truths):
                if prediction.class_id != ground_truth.class_id:
                    continue
                distance = hypot(
                    prediction.original_x - ground_truth.original_x,
                    prediction.original_y - ground_truth.original_y,
                )
                if self._valid_pair(prediction, ground_truth, distance):
                    pairs.append((distance, prediction_index, ground_truth_index))
        pairs.sort(key=lambda item: (item[0], item[1], item[2]))
        used_predictions = set()
        used_ground_truths = set()
        matches = []
        for distance, prediction_index, ground_truth_index in pairs:
            if prediction_index in used_predictions or ground_truth_index in used_ground_truths:
                continue
            used_predictions.add(prediction_index)
            used_ground_truths.add(ground_truth_index)
            matches.append((prediction_index, ground_truth_index, distance))
        return (
            tuple(matches),
            tuple(index for index in range(len(predictions)) if index not in used_predictions),
            tuple(index for index in range(len(ground_truths)) if index not in used_ground_truths),
        )

    def _confusion_matches(
        self,
        predictions: Sequence[Detection],
        ground_truths: Sequence[GroundTruthCentroid],
    ) -> Tuple[Tuple[int, int, float], ...]:
        """Assign nearby predictions to ground truths for diagnostic confusion only."""

        pairs = []
        for prediction_index, prediction in enumerate(predictions):
            for ground_truth_index, ground_truth in enumerate(ground_truths):
                distance = hypot(
                    prediction.original_x - ground_truth.original_x,
                    prediction.original_y - ground_truth.original_y,
                )
                if self._valid_pair(prediction, ground_truth, distance):
                    pairs.append((distance, prediction_index, ground_truth_index))
        pairs.sort(key=lambda item: (item[0], item[1], item[2]))
        used_predictions = set()
        used_ground_truths = set()
        matches = []
        for distance, prediction_index, ground_truth_index in pairs:
            if prediction_index in used_predictions or ground_truth_index in used_ground_truths:
                continue
            used_predictions.add(prediction_index)
            used_ground_truths.add(ground_truth_index)
            matches.append((prediction_index, ground_truth_index, distance))
        return tuple(matches)

    def _valid_pair(
        self, prediction: Detection, ground_truth: GroundTruthCentroid, distance: float
    ) -> bool:
        if self.matching_mode == "max_distance_pixels":
            return distance <= self.max_distance_pixels
        return (
            ground_truth.x_min <= prediction.original_x <= ground_truth.x_max
            and ground_truth.y_min <= prediction.original_y <= ground_truth.y_max
        )


def ground_truths_from_boxes(
    boxes: Sequence[AbsoluteBox], class_names: Sequence[str]
) -> Tuple[GroundTruthCentroid, ...]:
    """Convert dataset original-image boxes into evaluator ground truths."""

    names = tuple(class_names)
    output = []
    for box in boxes:
        if not 0 <= box.foreground_class_id < len(names):
            raise CentroidMetricError("ground-truth class ID is outside class_names")
        x, y = box.center
        output.append(
            GroundTruthCentroid(
                class_id=box.foreground_class_id,
                class_name=names[box.foreground_class_id],
                original_x=x,
                original_y=y,
                x_min=box.x_min,
                y_min=box.y_min,
                x_max=box.x_max,
                y_max=box.y_max,
            )
        )
    return tuple(output)


def sweep_confidence_thresholds(
    *,
    logits: Sequence[Tensor],
    transforms: Sequence[LetterboxTransform],
    ground_truths: Sequence[Sequence[GroundTruthCentroid]],
    class_names: Sequence[str],
    stride: int,
    thresholds: Sequence[float],
    matching_mode: str = "centroid_in_bbox",
    max_distance_pixels: float = 32.0,
    class_thresholds: Optional[Sequence[float]] = None,
    component_mode: str = "connected_components",
    confidence_mode: str = "max",
) -> ThresholdSweepResult:
    """Run the same postprocessor at each validation threshold and select max F1."""

    if not thresholds:
        raise CentroidMetricError("thresholds must contain at least one value")
    if len(logits) != len(transforms) or len(logits) != len(ground_truths):
        raise CentroidMetricError("logits, transforms, and ground_truths must have equal lengths")
    evaluator = CentroidEvaluator(
        class_names,
        matching_mode=matching_mode,
        max_distance_pixels=max_distance_pixels,
    )
    results = {}
    for threshold in thresholds:
        detections = []
        for image_logits, transform in zip(logits, transforms):
            if image_logits.ndim == 3:
                image_logits = image_logits.unsqueeze(0)
            if image_logits.ndim != 4 or image_logits.shape[0] != 1:
                raise CentroidMetricError("each logits item must have shape [C,G,G] or [1,C,G,G]")
            detections.append(
                postprocess_logits(
                    image_logits,
                    class_names=class_names,
                    stride=stride,
                    transforms=(transform,),
                    confidence_threshold=float(threshold),
                    class_thresholds=class_thresholds,
                    component_mode=component_mode,
                    confidence_mode=confidence_mode,
                )[0]
            )
        results[float(threshold)] = evaluator.evaluate_dataset(detections, ground_truths)
    best_threshold = min(
        results,
        key=lambda threshold: (
            -results[threshold].centroid_f1,
            threshold,
        ),
    )
    return ThresholdSweepResult(best_threshold, results[best_threshold], results)


def _validate_items(
    predictions: Sequence[Detection],
    ground_truths: Sequence[GroundTruthCentroid],
    class_names: Sequence[str],
) -> None:
    for prediction in predictions:
        if prediction.class_id < 0 or prediction.class_id >= len(class_names):
            raise CentroidMetricError("prediction class ID is outside class_names")
        if prediction.class_name != class_names[prediction.class_id]:
            raise CentroidMetricError("prediction class name does not match class ID")
    for ground_truth in ground_truths:
        if ground_truth.class_id < 0 or ground_truth.class_id >= len(class_names):
            raise CentroidMetricError("ground-truth class ID is outside class_names")
        if ground_truth.class_name != class_names[ground_truth.class_id]:
            raise CentroidMetricError("ground-truth class name does not match class ID")


def _metrics_dict(true_positive: int, false_positive: int, false_negative: int) -> dict[str, float]:
    precision = _precision(true_positive, false_positive)
    recall = _recall(true_positive, false_negative)
    return {
        "precision": precision,
        "recall": recall,
        "f1": _f1(precision, recall),
        "true_positives": float(true_positive),
        "false_positives": float(false_positive),
        "false_negatives": float(false_negative),
    }


def _precision(true_positive: int, false_positive: int) -> float:
    return true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0


def _recall(true_positive: int, false_negative: int) -> float:
    return true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0


def _f1(precision: float, recall: float) -> float:
    return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
