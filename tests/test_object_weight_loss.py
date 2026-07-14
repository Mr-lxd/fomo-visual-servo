"""Tests for the isolated Edge Impulse-style object-weight losses."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn.functional as functional

from fomo_servo.config import LossConfig, load_config
from fomo_servo.losses import LossConfigurationError, build_classification_loss
from fomo_servo.training import run_training


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = ROOT / "tests" / "fixtures" / "yolo_micro"


def _config(
    name: str = "weighted_softmax_ce",
    *,
    object_weight: float = 1.0,
    background_weight: float = 1.0,
    class_weights: tuple[float, ...] | None = None,
    gamma: float = 0.0,
) -> LossConfig:
    return LossConfig(
        name=name,
        gamma=gamma,
        class_weights=class_weights,
        class_weight_mode="disabled" if name in {"weighted_softmax_ce", "ei_weighted_xent_legacy"} else "manual",
        background_weight=background_weight,
        object_weight=object_weight,
    )


def _logits_targets() -> tuple[torch.Tensor, torch.Tensor]:
    logits = torch.tensor(
        [
            [
                [[2.0, -1.0], [0.5, 1.5]],
                [[-0.5, 2.0], [1.0, -1.0]],
                [[0.0, 0.5], [-0.5, 1.0]],
            ]
        ],
        dtype=torch.float32,
    )
    targets = torch.tensor([[[0, 1], [2, 0]]], dtype=torch.int64)
    return logits, targets


def test_object_weight_one_is_standard_softmax_cross_entropy() -> None:
    logits, targets = _logits_targets()
    actual = build_classification_loss(_config(object_weight=1.0))(logits, targets)
    expected = functional.cross_entropy(logits, targets)
    torch.testing.assert_close(actual, expected)


def test_pure_background_is_unaffected_by_object_weight() -> None:
    logits, _ = _logits_targets()
    targets = torch.zeros((1, 2, 2), dtype=torch.int64)
    baseline = build_classification_loss(_config(object_weight=1.0))(logits, targets)
    weighted = build_classification_loss(_config(object_weight=100.0))(logits, targets)
    torch.testing.assert_close(baseline, weighted)


def test_single_foreground_uses_the_exact_target_cell_weight() -> None:
    logits = torch.tensor([[[[0.0]], [[1.0]], [[-1.0]]]], requires_grad=True)
    targets = torch.tensor([[[1]]], dtype=torch.int64)
    criterion = build_classification_loss(_config(object_weight=10.0))
    actual = criterion(logits, targets)
    expected = functional.cross_entropy(logits, targets)
    torch.testing.assert_close(actual, expected)
    assert criterion.target_cell_weights(targets).item() == pytest.approx(10.0)


def test_multiclass_foreground_targets_share_one_object_weight() -> None:
    logits, targets = _logits_targets()
    criterion = build_classification_loss(_config(object_weight=30.0))
    weights = criterion.target_cell_weights(targets)
    assert weights.tolist() == [[[1.0, 30.0], [30.0, 1.0]]]


@pytest.mark.parametrize("object_weight", [10.0, 100.0])
def test_configured_object_weight_is_not_stacked_with_per_class_weights(
    object_weight: float,
) -> None:
    logits, targets = _logits_targets()
    without_class_weights = build_classification_loss(
        _config(object_weight=object_weight, class_weights=None)
    )(logits, targets)
    with_class_weights = build_classification_loss(
        _config(object_weight=object_weight, class_weights=(1.0, 3.0, 7.0))
    )(logits, targets)
    torch.testing.assert_close(without_class_weights, with_class_weights)
    exact_weights = build_classification_loss(
        _config(object_weight=object_weight)
    ).target_cell_weights(targets)
    assert exact_weights[0, 0, 1].item() == pytest.approx(object_weight)


def test_multiclass_batch_assigns_the_same_object_weight_to_each_foreground_class() -> None:
    logits = torch.zeros((2, 4, 1, 1), dtype=torch.float32)
    targets = torch.tensor([[[1]], [[3]]], dtype=torch.int64)
    weights = build_classification_loss(_config(object_weight=30.0)).target_cell_weights(targets)
    assert weights.shape == (2, 1, 1)
    assert weights.flatten().tolist() == [30.0, 30.0]


def test_object_weight_mode_rejects_focal_gamma() -> None:
    with pytest.raises(LossConfigurationError, match="does not support focal gamma"):
        build_classification_loss(_config(object_weight=100.0, gamma=2.0))


def test_ei_legacy_mode_matches_tensorflow_weighted_logistic_reference() -> None:
    logits, targets = _logits_targets()
    criterion = build_classification_loss(
        _config(name="ei_weighted_xent_legacy", object_weight=10.0)
    )
    actual = criterion(logits, targets)
    one_hot = functional.one_hot(targets, num_classes=3).permute(0, 3, 1, 2).to(logits.dtype)
    pos_weight = torch.tensor([1.0, 10.0, 10.0], dtype=logits.dtype).view(1, 3, 1, 1)
    expected = functional.binary_cross_entropy_with_logits(
        logits, one_hot, pos_weight=pos_weight
    )
    torch.testing.assert_close(actual, expected)


def test_object_weight_backward_and_cpu_amp_are_finite() -> None:
    logits, targets = _logits_targets()
    logits.requires_grad_(True)
    criterion = build_classification_loss(_config(object_weight=100.0))
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        loss = criterion(logits, targets)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_yaml_type_and_checkpoint_metadata_record_object_weight(tmp_path: Path) -> None:
    config_path = tmp_path / "object_weight.yaml"
    output_dir = tmp_path / "run"
    config_path.write_text(
        f"""
dataset:
  root: "{FIXTURE_ROOT.as_posix()}"
  train_split: train
  validation_split: val
  classes: [creature]
  class_mode: merge_single
model:
  input_size: 96
  output_stride: 8
loss:
  type: weighted_softmax_ce
  background_weight: 1.0
  object_weight: 100.0
training:
  device: cpu
  amp: false
  num_workers: 0
  pin_memory: false
  batch_size: 2
  epochs: 2
  seed: 123
  output_dir: "{output_dir.as_posix()}"
  epoch_snapshots:
    enabled: true
    interval: 1
    keep_last: null
""".lstrip(),
        encoding="utf-8",
    )
    config = load_config(config_path)
    assert config.loss.name == "weighted_softmax_ce"
    assert config.loss.class_weight_mode == "disabled"
    assert config.loss.class_weights is None
    assert config.loss.object_weight == pytest.approx(100.0)

    run_training(config, device_override="cpu")
    checkpoint = torch.load(output_dir / "last.pt", map_location="cpu", weights_only=False)
    assert checkpoint["loss_type"] == "weighted_softmax_ce"
    assert checkpoint["object_weight"] == pytest.approx(100.0)
    assert checkpoint["background_weight"] == pytest.approx(1.0)
    assert checkpoint["per_class_weights_applied"] is False
    snapshot = torch.load(
        output_dir / "epoch_snapshots" / "epoch_001_weights.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert snapshot["loss"]["loss_type"] == "weighted_softmax_ce"
    assert snapshot["loss"]["object_weight"] == pytest.approx(100.0)


def test_object_weight_smoke_loss_decreases_and_resume_is_reproducible(tmp_path: Path) -> None:
    """A tiny optimizer smoke checks finite loss, snapshots, and resumable state."""

    output_dir = tmp_path / "resume-run"
    config_path = tmp_path / "resume.yaml"
    config_path.write_text(
        f"""
dataset:
  root: "{FIXTURE_ROOT.as_posix()}"
  classes: [creature]
  class_mode: merge_single
model:
  input_size: 96
  output_stride: 8
loss:
  type: weighted_softmax_ce
  object_weight: 100.0
training:
  device: cpu
  amp: false
  num_workers: 0
  pin_memory: false
  batch_size: 2
  epochs: 2
  seed: 123
  output_dir: "{output_dir.as_posix()}"
  resume: null
  epoch_snapshots:
    enabled: true
    interval: 1
    keep_last: null
""".lstrip(),
        encoding="utf-8",
    )
    config = load_config(config_path)
    summary = run_training(config, device_override="cpu")
    assert summary.completed_epochs == 2
    history_lines = (output_dir / "history.csv").read_text(encoding="utf-8").splitlines()
    assert len(history_lines) == 3
    assert all(torch.isfinite(torch.tensor(float(line.split(",")[1]))) for line in history_lines[1:])
    checkpoint = torch.load(output_dir / "last.pt", map_location="cpu", weights_only=False)
    assert isinstance(checkpoint["scaler_state"], dict)
    assert checkpoint["loss_type"] == "weighted_softmax_ce"
    assert checkpoint["object_weight"] == pytest.approx(100.0)

    resumed_path = tmp_path / "resumed.yaml"
    resumed_path.write_text(
        config_path.read_text(encoding="utf-8").replace("epochs: 2", "epochs: 3").replace(
            "resume: null", f"resume: \"{(output_dir / 'last.pt').as_posix()}\""
        ),
        encoding="utf-8",
    )
    resumed = run_training(load_config(resumed_path), device_override="cpu")
    assert resumed.start_epoch == 3
    assert resumed.completed_epochs == 3


def test_weighted_softmax_ce_decreases_in_minimal_gradient_smoke() -> None:
    parameter = torch.nn.Parameter(torch.zeros(3, dtype=torch.float32))
    optimizer = torch.optim.SGD([parameter], lr=0.5)
    criterion = build_classification_loss(_config(object_weight=100.0))
    targets = torch.tensor([[[1]]], dtype=torch.int64)
    losses = []
    for _ in range(8):
        optimizer.zero_grad(set_to_none=True)
        logits = parameter.view(1, 3, 1, 1)
        loss = criterion(logits, targets)
        assert torch.isfinite(loss)
        loss.backward()
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        optimizer.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < losses[0]
