"""CPU smoke tests for deterministic FOMO train/validation, checkpoints, and resume."""

from __future__ import annotations

import csv
import importlib
import json
from pathlib import Path
from typing import Any, Callable

import pytest
import torch

from fomo_servo.config import load_config
from fomo_servo.datasets import YOLOv5FOMODataset


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = ROOT / "tests" / "fixtures" / "yolo_micro"


def _engine_api() -> tuple[
    Callable[..., Any] | None,
    Callable[..., Any] | None,
    Callable[[torch.nn.Module], None] | None,
]:
    """Return optional engine APIs so missing implementation has an assertion failure."""

    try:
        module = importlib.import_module("fomo_servo.training")
    except ModuleNotFoundError:
        return None, None, None
    return (
        getattr(module, "collate_fomo_samples", None),
        getattr(module, "run_training", None),
        getattr(module, "ensure_finite_gradients", None),
    )


def _write_training_config(
    path: Path,
    output_dir: Path,
    *,
    epochs: int,
    resume: Path | None = None,
) -> Path:
    """Write a complete YAML run config for the synthetic two-class YOLO fixture."""

    resume_text = "null" if resume is None else '"{}"'.format(resume.as_posix())
    path.write_text(
        """
dataset:
  root: "{root}"
  train_split: train
  validation_split: val
  classes: [creature]
  class_mode: merge_single
  merged_class_name: creature
model:
  backbone: mobilenet_v2_lite
  width_multiplier: 0.35
  head_channels: 32
  input_size: 96
  output_stride: 8
loss:
  name: focal_cross_entropy
  gamma: 2.0
  class_weights: [1.0, 3.0]
training:
  device: cpu
  amp: false
  num_workers: 0
  pin_memory: false
  batch_size: 2
  epochs: {epochs}
  seed: 123
  output_dir: "{output_dir}"
  resume: {resume}
  early_stopping_patience: 0
  early_stopping_min_delta: 0.0
  optimizer:
    name: adamw
    learning_rate: 0.001
    weight_decay: 0.0
  scheduler:
    name: step_lr
    step_size: 1
    gamma: 0.9
""".format(
            root=FIXTURE_ROOT.as_posix(),
            epochs=epochs,
            output_dir=output_dir.as_posix(),
            resume=resume_text,
        ).lstrip(),
        encoding="utf-8",
    )
    return path


def test_collate_fomo_samples_returns_training_tensor_contract() -> None:
    """Dataset samples must collate to float32 images [B,3,S,S] and int64 targets."""

    collate, _, _ = _engine_api()
    assert callable(collate), "fomo_servo.training.collate_fomo_samples must exist"
    dataset = YOLOv5FOMODataset(
        FIXTURE_ROOT,
        split="train",
        input_size=96,
        stride=8,
        class_mode="merge_single",
    )

    batch = collate([dataset[0], dataset[1]])

    assert batch.images.shape == (2, 3, 96, 96)
    assert batch.images.dtype == torch.float32
    assert batch.targets.shape == (2, 12, 12)
    assert batch.targets.dtype == torch.int64


def test_ensure_finite_gradients_rejects_infinite_gradient() -> None:
    """The optimizer must not step when any parameter gradient is NaN or Inf."""

    _, _, gradient_guard = _engine_api()
    assert callable(gradient_guard), "fomo_servo.training.ensure_finite_gradients must exist"
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    parameter.grad = torch.tensor(float("inf"))
    model = torch.nn.Module()
    model.register_parameter("weight", parameter)

    with pytest.raises(Exception, match="non-finite gradient.*weight"):
        gradient_guard(model)


def test_cpu_two_epoch_smoke_saves_best_last_history_and_resumes(tmp_path: Path) -> None:
    """The fixture dataset must train for two CPU epochs, persist state, then resume."""

    _, run_training, _ = _engine_api()
    assert callable(run_training), "fomo_servo.training.run_training must exist"
    output_dir = tmp_path / "run"
    first_config = load_config(
        _write_training_config(tmp_path / "first.yaml", output_dir, epochs=2)
    )

    first_summary = run_training(first_config, device_override="cpu")

    last_checkpoint = output_dir / "last.pt"
    best_checkpoint = output_dir / "best_val_f1.pt"
    best_grid_checkpoint = output_dir / "best_grid_f1.pt"
    best_centroid_checkpoint = output_dir / "best_centroid_f1.pt"
    history_path = output_dir / "history.csv"
    summary_path = output_dir / "training_summary.json"
    assert first_summary.completed_epochs == 2
    assert first_summary.best_val_f1 >= 0.0
    assert last_checkpoint.is_file()
    assert best_checkpoint.is_file()
    assert best_grid_checkpoint.is_file()
    assert best_centroid_checkpoint.is_file()
    assert summary_path.is_file()
    checkpoint_payload = torch.load(last_checkpoint, map_location="cpu", weights_only=False)
    assert checkpoint_payload["checkpoint_type"] == "last"
    assert checkpoint_payload["selection_metric"] == "last"
    assert checkpoint_payload["selection_threshold"] == pytest.approx(0.5)
    assert checkpoint_payload["centroid_f1"] >= 0.0
    assert checkpoint_payload["class_weight_mode"] == "manual"
    assert checkpoint_payload["class_weights"] == [1.0, 3.0]
    assert len(checkpoint_payload["class_statistics"]) == 1
    training_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert training_summary["class_weight_mode"] == "manual"
    assert training_summary["class_weights"] == [1.0, 3.0]
    assert len(training_summary["class_statistics"]) == 1
    assert training_summary["best_grid_epoch"] >= 1
    assert training_summary["best_centroid_epoch"] >= 1
    assert training_summary["checkpoint_threshold"] == pytest.approx(0.5)
    assert training_summary["best_val_f1_alias_target"] == "best_grid_f1.pt"
    with history_path.open("r", newline="", encoding="utf-8") as history_file:
        first_rows = list(csv.DictReader(history_file))
    assert len(first_rows) == 2
    assert set(
        (
            "train_loss",
            "val_loss",
            "grid_precision",
            "grid_recall",
            "grid_f1",
            "centroid_precision",
            "centroid_recall",
            "centroid_f1",
            "mean_count_bias",
            "mean_absolute_count_error",
        )
    ).issubset(
        first_rows[0]
    )

    resumed_config = load_config(
        _write_training_config(
            tmp_path / "resumed.yaml", output_dir, epochs=3, resume=last_checkpoint
        )
    )
    resumed_summary = run_training(resumed_config, device_override="cpu")

    assert resumed_summary.start_epoch == 3
    assert resumed_summary.completed_epochs == 3
    with history_path.open("r", newline="", encoding="utf-8") as history_file:
        resumed_rows = list(csv.DictReader(history_file))
    assert len(resumed_rows) == 3


def test_experiment_run_writes_reproducibility_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An opted-in run writes a config copy, metadata JSON, and one CSV row."""

    _, run_training, _ = _engine_api()
    assert callable(run_training), "fomo_servo.training.run_training must exist"
    monkeypatch.setattr(
        "fomo_servo.training.engine.git_commit_sha", lambda _: "a" * 40
    )
    monkeypatch.setattr(
        "fomo_servo.training.engine.git_worktree_fingerprint",
        lambda _: (True, "b" * 64),
    )
    output_dir = tmp_path / "experiment-output"
    config_path = _write_training_config(tmp_path / "experiment.yaml", output_dir, epochs=1)
    with config_path.open("a", encoding="utf-8") as config_file:
        config_file.write(
            "\nexperiment:\n"
            "  name: smoke_experiment\n"
            f"  summary_csv: \"{(tmp_path / 'experiments_summary.csv').as_posix()}\"\n"
        )

    config = load_config(config_path)
    summary = run_training(config, device_override="cpu")

    assert summary.completed_epochs == 1
    assert "WARNING: Git worktree is dirty" in capsys.readouterr().out
    assert (output_dir / "config.yaml").read_text(encoding="utf-8") == config_path.read_text(
        encoding="utf-8"
    )
    metadata = json.loads((output_dir / "experiment_metadata.json").read_text(encoding="utf-8"))
    assert metadata["experiment_name"] == "smoke_experiment"
    assert metadata["random_seed"] == 123
    assert metadata["best_epoch"] == 1
    assert metadata["total_training_time_seconds"] >= 0.0
    with (tmp_path / "experiments_summary.csv").open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 1
    assert rows[0]["experiment_name"] == "smoke_experiment"
    assert rows[0]["best_threshold"]
