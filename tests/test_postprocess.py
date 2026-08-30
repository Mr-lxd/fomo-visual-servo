from __future__ import annotations

from math import exp

import numpy as np
import pytest
import torch

from fomo_servo.geometry import LetterboxTransform
from fomo_servo.postprocess import (
    PostprocessError,
    postprocess_logits,
    postprocess_numpy_logits,
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


def test_torch_logits_entrypoint_preserves_torch_softmax(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import fomo_servo.postprocess.detections as detections_module

    def reject_numpy_logits(*args, **kwargs):
        raise AssertionError("torch entrypoint must not delegate softmax to NumPy")

    monkeypatch.setattr(
        detections_module, "postprocess_numpy_logits", reject_numpy_logits
    )
    result = postprocess_logits(
        torch.tensor([[[[0.0]], [[2.0]]]], dtype=torch.float32),
        class_names=("creature",),
        stride=8,
        transforms=(LetterboxTransform.from_image_size(8, 8, 8),),
        confidence_threshold=0.4,
    )

    assert len(result[0]) == 1


def test_numpy_logits_match_torch_detection_pipeline() -> None:
    logits = torch.tensor(
        [
            [
                [[-2.0, -2.0], [-2.0, -2.0]],
                [[4.0, 3.0], [-1.0, -1.0]],
                [[-1.0, -1.0], [5.0, -1.0]],
            ]
        ],
        dtype=torch.float32,
    )
    transform = LetterboxTransform.from_image_size(16, 12, 16)
    keyword_arguments = {
        "class_names": ("fish", "crab"),
        "stride": 8,
        "transforms": (transform,),
        "confidence_threshold": 0.4,
        "component_mode": "connected_components",
        "confidence_mode": "max",
    }

    torch_detections = postprocess_logits(logits, **keyword_arguments)[0]
    numpy_detections = postprocess_numpy_logits(
        logits.numpy(), **keyword_arguments
    )[0]

    assert [item.class_id for item in numpy_detections] == [
        item.class_id for item in torch_detections
    ]
    assert [item.class_name for item in numpy_detections] == [
        item.class_name for item in torch_detections
    ]
    for numpy_detection, torch_detection in zip(
        numpy_detections, torch_detections
    ):
        for field in (
            "confidence",
            "mean_confidence",
            "heatmap_x",
            "heatmap_y",
            "input_x",
            "input_y",
            "original_x",
            "original_y",
        ):
            assert getattr(numpy_detection, field) == pytest.approx(
                getattr(torch_detection, field), abs=1e-6
            )


def test_numpy_logits_reject_nonfinite_values() -> None:
    logits = np.zeros((1, 2, 2, 2), dtype=np.float32)
    logits[0, 1, 0, 0] = np.nan

    with pytest.raises(PostprocessError, match="NaN or Inf"):
        postprocess_numpy_logits(
            logits,
            class_names=("creature",),
            stride=8,
            transforms=(LetterboxTransform.from_image_size(16, 16, 16),),
            confidence_threshold=0.4,
        )
