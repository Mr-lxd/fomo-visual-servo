"""Shared fixed-threshold reporting for local and Edge Impulse parity runs."""

from __future__ import annotations

from statistics import mean
from typing import Mapping, Sequence

from fomo_servo.evaluation.edge_impulse import (
    EdgeImpulseCentroidEvaluator,
    EdgeImpulseDetection,
    EdgeImpulseGroundTruth,
    MatchingMode,
)
from fomo_servo.metrics import CentroidEvaluator, GroundTruthCentroid
from fomo_servo.postprocess import Detection


class ParityReportingError(ValueError):
    """Raised when a parity report cannot preserve evaluator semantics."""


def edge_detections_from_local(
    detections: Sequence[Detection],
) -> tuple[EdgeImpulseDetection, ...]:
    """Convert local detections to EI records while retaining original centroids.

    The generated ``input_bbox`` is a one-cell ``8x8`` diagnostic proxy; EI
    matching consumes only ``original_centroid``. Input and original coordinates
    remain pixel ``(x, y)`` pairs.
    """

    return tuple(
        EdgeImpulseDetection(
            class_id=detection.class_id,
            class_name=detection.class_name,
            confidence=detection.confidence,
            input_bbox=(detection.input_x - 4.0, detection.input_y - 4.0, 8.0, 8.0),
            input_centroid=(detection.input_x, detection.input_y),
            original_centroid=(detection.original_x, detection.original_y),
        )
        for detection in detections
    )


def edge_ground_truths_from_local(
    ground_truths: Sequence[GroundTruthCentroid],
) -> tuple[EdgeImpulseGroundTruth, ...]:
    """Convert local original-pixel ground truths to EI centroid ground truths."""

    return tuple(
        EdgeImpulseGroundTruth(
            class_id=ground_truth.class_id,
            class_name=ground_truth.class_name,
            original_centroid=(ground_truth.original_x, ground_truth.original_y),
        )
        for ground_truth in ground_truths
    )


def evaluate_local_parity(
    *,
    predictions: Sequence[Sequence[Detection]],
    ground_truths: Sequence[Sequence[GroundTruthCentroid]],
    class_names: Sequence[str],
    matching_mode: str,
    max_distance_pixels: float,
) -> dict[str, object]:
    """Run the unchanged local evaluator and include per-image match provenance."""

    evaluator = CentroidEvaluator(
        class_names,
        matching_mode=matching_mode,
        max_distance_pixels=max_distance_pixels,
    )
    result = evaluator.evaluate_dataset(predictions, ground_truths)
    image_results = []
    for image_predictions, image_ground_truths in zip(predictions, ground_truths):
        matches, unmatched_predictions, unmatched_ground_truths = evaluator._match(
            image_predictions, image_ground_truths
        )
        image_results.append(
            {
                "matches": [
                    {
                        "prediction_index": prediction_index,
                        "ground_truth_index": ground_truth_index,
                        "distance_pixels": distance,
                    }
                    for prediction_index, ground_truth_index, distance in matches
                ],
                "unmatched_prediction_indices": list(unmatched_predictions),
                "unmatched_ground_truth_indices": list(unmatched_ground_truths),
            }
        )
    per_class = {
        name: dict(values)
        for name, values in result.per_class_precision_recall_f1.items()
    }
    return {
        "evaluator": "local_current",
        "matching_mode": matching_mode,
        "precision": result.centroid_precision,
        "recall": result.centroid_recall,
        "f1": result.centroid_f1,
        "macro_f1": mean(float(values["f1"]) for values in per_class.values()),
        "per_class": per_class,
        "true_positives": result.true_positives,
        "false_positives": result.false_positives,
        "false_negatives": result.false_negatives,
        "prediction_count": sum(len(items) for items in predictions),
        "ground_truth_count": sum(len(items) for items in ground_truths),
        "mean_localization_error_pixels": result.mean_localization_error_pixels,
        "median_localization_error_pixels": result.median_localization_error_pixels,
        "count_error_per_image": list(result.count_error_per_image),
        "mean_count_bias": result.mean_count_bias,
        "mean_absolute_count_error": result.mean_absolute_count_error,
        "non_background_confusion": result.confusion_matrix.tolist(),
        "image_results": image_results,
    }


def serialize_edge_evaluation(
    *,
    predictions: Sequence[Sequence[EdgeImpulseDetection]],
    ground_truths: Sequence[Sequence[EdgeImpulseGroundTruth]],
    image_sizes: Sequence[tuple[int, int]],
    class_names: Sequence[str],
    mode: MatchingMode,
) -> dict[str, object]:
    """Evaluate EI compatibility and return a JSON-safe fixed-threshold report."""

    result = EdgeImpulseCentroidEvaluator(class_names, mode=mode).evaluate_dataset(
        predictions, ground_truths, image_sizes=image_sizes
    )
    return {
        "evaluator": mode,
        "matching_mode": mode,
        "precision": result.precision,
        "recall": result.recall,
        "f1": result.f1,
        "macro_f1": result.macro_f1,
        "per_class": {
            name: dict(values)
            for name, values in result.per_class_precision_recall_f1.items()
        },
        "true_positives": result.true_positives,
        "false_positives": result.false_positives,
        "false_negatives": result.false_negatives,
        "prediction_count": result.prediction_count,
        "ground_truth_count": result.ground_truth_count,
        "mean_localization_error_normalized": result.mean_localization_error_normalized,
        "median_localization_error_normalized": result.median_localization_error_normalized,
        "count_error_per_image": list(result.count_error_per_image),
        "mean_count_bias": result.mean_count_bias,
        "mean_absolute_count_error": result.mean_absolute_count_error,
        "non_background_confusion": result.non_background_confusion.tolist(),
        "image_results": [
            {
                "matches": [
                    {
                        "prediction_index": match.prediction_index,
                        "ground_truth_index": match.ground_truth_index,
                        "normalized_distance": match.normalized_distance,
                    }
                    for match in image.matches
                ],
                "unmatched_prediction_indices": list(image.unmatched_prediction_indices),
                "unmatched_ground_truth_indices": list(image.unmatched_ground_truth_indices),
            }
            for image in result.image_results
        ],
    }


__all__ = [
    "ParityReportingError",
    "edge_detections_from_local",
    "edge_ground_truths_from_local",
    "evaluate_local_parity",
    "serialize_edge_evaluation",
]
