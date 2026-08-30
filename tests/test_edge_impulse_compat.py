"""Behavioral tests for the separately scoped Edge Impulse compatibility mode."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from fomo_servo.evaluation.edge_impulse import (
    EdgeImpulseCentroidEvaluator,
    EdgeImpulseDetection,
    EdgeImpulseGroundTruth,
    EdgeImpulseMetricError,
    decode_edge_impulse_fomo,
    normalized_centroid_distance,
    probabilities_from_logits,
)


CLASS_NAMES = ("fish", "shark")


def _detection(class_id: int, x: float, y: float, confidence: float = 0.8) -> EdgeImpulseDetection:
    return EdgeImpulseDetection(
        class_id=class_id,
        class_name=CLASS_NAMES[class_id],
        confidence=confidence,
        input_bbox=(x - 4.0, y - 4.0, 8.0, 8.0),
        input_centroid=(x, y),
        original_centroid=(x, y),
    )


def _ground_truth(class_id: int, x: float, y: float) -> EdgeImpulseGroundTruth:
    return EdgeImpulseGroundTruth(
        class_id=class_id,
        class_name=CLASS_NAMES[class_id],
        original_centroid=(x, y),
    )


def test_decoder_accepts_exact_threshold_and_merges_diagonal_same_class_cells() -> None:
    probabilities = torch.zeros((3, 3, 3), dtype=torch.float32)
    probabilities[..., 0] = 1.0
    probabilities[0, 0, :] = torch.tensor([0.5, 0.5, 0.0])
    probabilities[1, 1, :] = torch.tensor([0.2, 0.8, 0.0])

    detections = decode_edge_impulse_fomo(
        probabilities, class_names=CLASS_NAMES, input_size=(24, 24), threshold=0.5
    )

    assert len(detections) == 1
    assert detections[0].class_id == 0
    assert detections[0].input_bbox == (0.0, 0.0, 16.0, 16.0)
    assert detections[0].input_centroid == (8.0, 8.0)
    assert detections[0].confidence == pytest.approx(0.8)


def test_decoder_keeps_different_class_adjacent_cells_separate() -> None:
    probabilities = np.zeros((2, 2, 3), dtype=np.float32)
    probabilities[..., 0] = 1.0
    probabilities[0, 0] = (0.4, 0.6, 0.0)
    probabilities[0, 1] = (0.4, 0.0, 0.6)

    detections = decode_edge_impulse_fomo(
        probabilities, class_names=CLASS_NAMES, input_size=(16, 16), threshold=0.5
    )

    assert [(item.class_id, item.input_centroid) for item in detections] == [
        (0, (4.0, 4.0)),
        (1, (12.0, 4.0)),
    ]


def test_logits_are_softmaxed_exactly_once() -> None:
    logits = torch.tensor([[[[0.0]], [[math.log(3.0)]], [[0.0]]]], dtype=torch.float32)

    probabilities = probabilities_from_logits(logits)
    detections = decode_edge_impulse_fomo(
        probabilities, class_names=CLASS_NAMES, input_size=(8, 8), threshold=0.5
    )

    assert probabilities.shape == (1, 1, 3)
    assert float(probabilities.sum()) == pytest.approx(1.0)
    assert len(detections) == 1
    assert detections[0].class_id == 0
    assert detections[0].confidence == pytest.approx(0.6)


@pytest.mark.parametrize(
    ("delta_x", "delta_y", "expected"),
    [
        (38.4 - 1e-5, 0.0, True),
        (38.4, 0.0, True),
        (38.4 + 1e-5, 0.0, False),
        (0.0, 38.4, True),
        (27.15290039756342, 27.15290039756342, True),
        (27.15290039756342 + 1e-4, 27.15290039756342 + 1e-4, False),
    ],
)
def test_normalized_distance_boundary_uses_coordinatewise_normalization(
    delta_x: float, delta_y: float, expected: bool
) -> None:
    distance = normalized_centroid_distance((0.0, 0.0), (delta_x, delta_y), 192, 192)

    assert (distance <= 0.2) is expected


def test_legacy_matching_allows_two_predictions_to_match_one_ground_truth() -> None:
    predictions = ((_detection(0, 95.0, 96.0), _detection(0, 97.0, 96.0)),)
    targets = ((_ground_truth(0, 96.0, 96.0),),)

    legacy = EdgeImpulseCentroidEvaluator(CLASS_NAMES, mode="edge_impulse_legacy").evaluate_dataset(
        predictions, targets, image_sizes=((192, 192),)
    )
    strict = EdgeImpulseCentroidEvaluator(CLASS_NAMES, mode="strict_one_to_one").evaluate_dataset(
        predictions, targets, image_sizes=((192, 192),)
    )

    assert (legacy.true_positives, legacy.false_positives, legacy.false_negatives) == (2, 0, 0)
    assert (strict.true_positives, strict.false_positives, strict.false_negatives) == (1, 1, 0)


def test_two_ground_truths_and_one_prediction_leaves_one_false_negative() -> None:
    report = EdgeImpulseCentroidEvaluator(CLASS_NAMES, mode="edge_impulse_legacy").evaluate_dataset(
        ((_detection(0, 20.0, 20.0),),),
        ((_ground_truth(0, 20.0, 20.0), _ground_truth(0, 160.0, 160.0)),),
        image_sizes=((192, 192),),
    )

    assert (report.true_positives, report.false_positives, report.false_negatives) == (1, 0, 1)


def test_wrong_class_and_no_target_are_false_positive_false_negative_cases() -> None:
    evaluator = EdgeImpulseCentroidEvaluator(CLASS_NAMES, mode="strict_one_to_one")
    wrong_class = evaluator.evaluate_dataset(
        ((_detection(1, 96.0, 96.0),),),
        ((_ground_truth(0, 96.0, 96.0),),),
        image_sizes=((192, 192),),
    )
    no_target = evaluator.evaluate_dataset(
        ((_detection(0, 96.0, 96.0),),), ((),), image_sizes=((192, 192),)
    )

    assert (wrong_class.true_positives, wrong_class.false_positives, wrong_class.false_negatives) == (0, 1, 1)
    assert (no_target.true_positives, no_target.false_positives, no_target.false_negatives) == (0, 1, 0)


def test_decoder_rejects_invalid_probability_tensor() -> None:
    probabilities = np.zeros((2, 2, 3), dtype=np.float32)
    probabilities[0, 0, 1] = -0.1

    with pytest.raises(EdgeImpulseMetricError, match="non-negative"):
        decode_edge_impulse_fomo(
            probabilities, class_names=CLASS_NAMES, input_size=(16, 16), threshold=0.5
        )
