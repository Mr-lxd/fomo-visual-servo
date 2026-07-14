from __future__ import annotations

from math import exp

import pytest
import torch

from fomo_servo.geometry import LetterboxTransform
from fomo_servo.postprocess import (
    PostprocessError,
    postprocess_logits,
    postprocess_probabilities,
)


def _transform() -> LetterboxTransform:
    return LetterboxTransform.from_image_size(32, 16, 32)


def test_connected_components_merge_adjacent_cells_and_weight_centroid() -> None:
    logits = torch.zeros(1, 3, 4, 4)
    logits[:, 0] = -2.0
    logits[0, 1, 1, 1] = 5.0
    logits[0, 1, 1, 2] = 4.0
    logits[0, 2, 3, 3] = 5.0

    detections = postprocess_logits(
        logits,
        class_names=("fish", "crab"),
        stride=8,
        transforms=(_transform(),),
        confidence_threshold=0.8,
    )[0]

    assert len(detections) == 2
    fish = next(item for item in detections if item.class_name == "fish")
    assert fish.class_id == 0
    assert fish.component_area_cells == 2
    probability_1 = exp(5.0) / (exp(-2.0) + exp(5.0) + exp(0.0))
    probability_2 = exp(4.0) / (exp(-2.0) + exp(4.0) + exp(0.0))
    assert fish.heatmap_x == pytest.approx(
        (1.5 * probability_1 + 2.5 * probability_2)
        / (probability_1 + probability_2)
    )
    assert fish.heatmap_y == pytest.approx(1.5)
    assert fish.input_x == pytest.approx(fish.heatmap_x * 8)
    assert fish.original_y == pytest.approx(4.0)
    assert fish.confidence > fish.mean_confidence


def test_background_is_excluded_and_per_class_thresholds_apply() -> None:
    logits = torch.zeros(1, 3, 2, 2)
    logits[0, 0, 0, 0] = 8.0
    logits[0, 1, 0, 1] = 3.0
    logits[0, 2, 1, 1] = 1.0

    detections = postprocess_logits(
        logits,
        class_names=("fish", "crab"),
        stride=8,
        transforms=(LetterboxTransform.from_image_size(16, 16, 16),),
        confidence_threshold=0.2,
        class_thresholds={"fish": 0.95, "crab": 0.4},
    )[0]

    assert [item.class_name for item in detections] == ["crab"]


def test_postprocess_rejects_unsupported_component_mode() -> None:
    with pytest.raises(PostprocessError, match="local_peaks"):
        postprocess_logits(
            torch.zeros(1, 2, 2, 2),
            class_names=("creature",),
            stride=8,
            transforms=(LetterboxTransform.from_image_size(16, 16, 16),),
            confidence_threshold=0.5,
            component_mode="local_peaks",
        )


def test_probability_entrypoint_matches_logits_without_a_second_softmax() -> None:
    logits = torch.tensor(
        [[[[2.0, 0.0], [0.0, 0.0]], [[0.0, 3.0], [0.0, 0.0]]]],
        dtype=torch.float32,
    )
    transform = LetterboxTransform.from_image_size(16, 16, 16)

    from_logits = postprocess_logits(
        logits,
        class_names=("fish",),
        stride=8,
        transforms=(transform,),
        confidence_threshold=0.5,
    )
    from_probabilities = postprocess_probabilities(
        logits.softmax(dim=1),
        class_names=("fish",),
        stride=8,
        transforms=(transform,),
        confidence_threshold=0.5,
    )

    assert from_probabilities == from_logits
