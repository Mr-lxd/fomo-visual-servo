"""Regression tests for separated inference and checkpoint thresholds."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from fomo_servo.config import EvaluationConfig, LossConfig, PostprocessConfig, load_config
from fomo_servo.datasets import FOMOBatch
from fomo_servo.geometry import LetterboxTransform
from fomo_servo.losses import build_classification_loss
from fomo_servo.postprocess import postprocess_logits
from fomo_servo.training.engine import validate_one_epoch
from fomo_servo.training.runtime import TrainingRuntime


def test_locked_threshold_schema_is_parsed_without_legacy_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FOMO_DATASET_ROOT", "data/aquarium_pretrain")
    path = tmp_path / "locked.yaml"
    path.write_text(
        "dataset:\n"
        "  root: ${FOMO_DATASET_ROOT}\n"
        "  classes: [creature]\n"
        "model:\n"
        "  input_size: 96\n"
        "  output_stride: 8\n"
        "postprocess:\n"
        "  inference_threshold: 0.5\n"
        "evaluation:\n"
        "  checkpoint_threshold: 0.5\n"
        "  threshold_sweep:\n"
        "    enabled: true\n"
        "    minimum: 0.05\n"
        "    maximum: 0.95\n"
        "    step: 0.05\n",
        encoding="utf-8",
    )

    config = load_config(path)

    assert config.postprocess.inference_threshold == pytest.approx(0.5)
    assert config.postprocess.confidence_threshold is None
    assert config.evaluation.checkpoint_threshold == pytest.approx(0.5)
    assert config.evaluation.threshold_sweep_enabled is True
    assert config.evaluation.threshold_sweep[0] == pytest.approx(0.05)
    assert config.evaluation.threshold_sweep[-1] == pytest.approx(0.95)


def test_legacy_confidence_threshold_warns_and_does_not_set_checkpoint_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FOMO_DATASET_ROOT", "data/aquarium_pretrain")
    path = tmp_path / "legacy.yaml"
    path.write_text(
        "dataset:\n"
        "  root: ${FOMO_DATASET_ROOT}\n"
        "  classes: [creature]\n"
        "model:\n"
        "  input_size: 96\n"
        "  output_stride: 8\n"
        "postprocess:\n"
        "  confidence_threshold: 0.05\n",
        encoding="utf-8",
    )

    with pytest.warns(DeprecationWarning, match="confidence_threshold"):
        config = load_config(path)

    assert config.postprocess.inference_threshold == pytest.approx(0.05)
    assert config.postprocess.confidence_threshold == pytest.approx(0.05)
    assert config.evaluation.checkpoint_threshold == pytest.approx(0.5)


def test_eight_channel_thresholds_produce_different_detections() -> None:
    logits = torch.full((1, 8, 2, 2), -2.0, dtype=torch.float32)
    logits[0, 0, :, :] = 4.0
    logits[0, 0, 0, 0] = 0.0
    logits[0, 1, 0, 0] = 0.2
    transform = LetterboxTransform.from_image_size(16, 16, 16)
    class_names = (
        "fish",
        "jellyfish",
        "penguin",
        "puffin",
        "shark",
        "starfish",
        "stingray",
    )

    low = postprocess_logits(
        logits,
        class_names=class_names,
        stride=8,
        transforms=(transform,),
        confidence_threshold=0.05,
    )[0]
    high = postprocess_logits(
        logits,
        class_names=class_names,
        stride=8,
        transforms=(transform,),
        confidence_threshold=0.5,
    )[0]

    assert len(low) == 1
    assert low[0].confidence < 0.5
    assert high == ()


def test_direct_legacy_postprocess_constructor_remains_compatible() -> None:
    config = PostprocessConfig(confidence_threshold=0.2)
    assert config.inference_threshold == pytest.approx(0.2)


def test_evaluation_defaults_are_explicit() -> None:
    config = EvaluationConfig()
    assert config.checkpoint_threshold == pytest.approx(0.5)
    assert config.threshold_sweep_enabled is True


def test_epoch_centroid_metric_uses_checkpoint_threshold_not_inference_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[float] = []

    def fake_postprocess(*args: object, **kwargs: object) -> list[tuple[object, ...]]:
        captured.append(float(kwargs["confidence_threshold"]))
        return [()]

    monkeypatch.setattr(
        "fomo_servo.training.engine.postprocess_logits", fake_postprocess
    )

    class ConstantModel(torch.nn.Module):
        def forward(self, images: torch.Tensor) -> torch.Tensor:
            return torch.zeros(images.shape[0], 2, 12, 12)

    batch = FOMOBatch(
        images=torch.zeros(1, 3, 96, 96),
        targets=torch.zeros(1, 12, 12, dtype=torch.int64),
        transforms=(LetterboxTransform.from_image_size(96, 96, 96),),
        original_boxes=((),),
    )
    criterion = build_classification_loss(
        LossConfig(name="focal_cross_entropy", gamma=2.0, class_weights=(1.0, 4.0))
    )
    runtime = TrainingRuntime(
        device=torch.device("cpu"),
        amp_enabled=False,
        num_workers=0,
        pin_memory=False,
        diagnostics=(),
    )

    validate_one_epoch(
        model=ConstantModel(),
        loader=[batch],
        criterion=criterion,
        runtime=runtime,
        class_names=("creature",),
        stride=8,
        postprocess_config=PostprocessConfig(inference_threshold=0.05),
        evaluation_config=EvaluationConfig(checkpoint_threshold=0.5),
        checkpoint_threshold=0.5,
    )
    validate_one_epoch(
        model=ConstantModel(),
        loader=[batch],
        criterion=criterion,
        runtime=runtime,
        class_names=("creature",),
        stride=8,
        postprocess_config=PostprocessConfig(inference_threshold=0.9),
        evaluation_config=EvaluationConfig(checkpoint_threshold=0.5),
        checkpoint_threshold=0.5,
    )

    assert captured == [0.5, 0.5]
