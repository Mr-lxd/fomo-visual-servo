"""CPU smoke tests for deterministic FOMO train/validation, checkpoints, and resume."""

from __future__ import annotations

import csv
import hashlib
import importlib
import json
from pathlib import Path
from typing import Any, Callable

import pytest
import torch

from fomo_servo.config import load_config
from fomo_servo.datasets import YOLOv5FOMODataset
from fomo_servo.models import build_fomo_model


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


def _write_weights_snapshot(path: Path, model: torch.nn.Module) -> str:
    payload = {
        "checkpoint_kind": "epoch_snapshot",
        "weights_only": True,
        "resumable": False,
        "epoch": 40,
        "seed": 42,
        "model_state": model.state_dict(),
    }
    torch.save(payload, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_initialize_model_weights_strictly_loads_snapshot_and_returns_provenance(
    tmp_path: Path,
) -> None:
    from fomo_servo.training import engine

    initializer = getattr(engine, "initialize_model_weights", None)
    assert callable(initializer), "initialize_model_weights must exist"
    source_model = torch.nn.Linear(3, 2)
    with torch.no_grad():
        source_model.weight.fill_(0.25)
        source_model.bias.fill_(-0.5)
    checkpoint = tmp_path / "epoch_040_weights.pt"
    digest = _write_weights_snapshot(checkpoint, source_model)
    target_model = torch.nn.Linear(3, 2)

    provenance = initializer(target_model, checkpoint, digest)

    assert torch.equal(target_model.weight, source_model.weight)
    assert torch.equal(target_model.bias, source_model.bias)
    assert provenance == {
        "path": str(checkpoint),
        "sha256": digest,
        "checkpoint_kind": "epoch_snapshot",
        "source_epoch": 40,
        "source_seed": 42,
        "strict_missing_keys": [],
        "strict_unexpected_keys": [],
    }


@pytest.mark.parametrize("failure", ["sha", "schema", "state"])
def test_initialize_model_weights_rejects_untrusted_or_incompatible_snapshot(
    tmp_path: Path, failure: str
) -> None:
    from fomo_servo.training import TrainingError, engine

    initializer = getattr(engine, "initialize_model_weights", None)
    assert callable(initializer), "initialize_model_weights must exist"
    source_model = torch.nn.Linear(3, 2)
    checkpoint = tmp_path / "epoch_040_weights.pt"
    digest = _write_weights_snapshot(checkpoint, source_model)
    target_model = torch.nn.Linear(3 if failure != "state" else 4, 2)
    if failure == "sha":
        digest = "0" * 64
    elif failure == "schema":
        torch.save({"model_state": source_model.state_dict()}, checkpoint)
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

    with pytest.raises(
        TrainingError,
        match={
            "sha": "SHA-256 mismatch",
            "schema": "weights-only epoch snapshot",
            "state": "strict model state",
        }[failure],
    ):
        initializer(target_model, checkpoint, digest)


def _write_training_config(
    path: Path,
    output_dir: Path,
    *,
    epochs: int,
    resume: Path | None = None,
    epoch_snapshots_yaml: str = "",
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
{epoch_snapshots_yaml}
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
            epoch_snapshots_yaml=epoch_snapshots_yaml,
        ).lstrip(),
        encoding="utf-8",
    )
    return path


def _write_train_only_config(
    path: Path,
    output_dir: Path,
    *,
    epochs: int,
    initialize_from: Path | None = None,
    initialize_sha256: str | None = None,
    resume: Path | None = None,
    experiment: bool = False,
) -> Path:
    initialization = ""
    if initialize_from is not None and initialize_sha256 is not None:
        initialization = (
            '  initialize_from: "{}"\n'
            "  initialize_sha256: {}\n"
        ).format(initialize_from.as_posix(), initialize_sha256)
    resume_text = "null" if resume is None else '"{}"'.format(resume.as_posix())
    experiment_yaml = ""
    if experiment:
        experiment_yaml = (
            "\nexperiment:\n"
            "  name: train_only_smoke\n"
            '  summary_csv: "{}"\n'.format(
                (output_dir.parent / "train_only_summary.csv").as_posix()
            )
        )
    path.write_text(
        """
dataset:
  root: "{root}"
  train_split: train
  validation_split: null
  classes: [creature]
  class_mode: merge_single
  merged_class_name: creature
model:
  backbone: mobilenet_v2_lite
  width_multiplier: 0.35
  head_channels: 32
  input_size: 96
  output_stride: 8
  pretrained: false
loss:
  name: focal_cross_entropy
  gamma: 2.0
  class_weights: [1.0, 3.0]
postprocess:
  inference_threshold: 0.40
evaluation:
  checkpoint_threshold: 0.40
  threshold_sweep:
    enabled: false
  threshold_calibration:
    enabled: false
training:
  device: cpu
  amp: false
  num_workers: 0
  pin_memory: false
  batch_size: 2
  epochs: {epochs}
  seed: 42
  output_dir: "{output_dir}"
  resume: {resume}
{initialization}  checkpoint_policy: fixed_final_epoch
  early_stopping_patience: 0
  epoch_snapshots:
    enabled: true
    format: weights_only
    interval: {epochs}
    keep_last: 1
  optimizer:
    name: adamw
    learning_rate: 0.0001
    weight_decay: 0.0001
  scheduler:
    name: none
""".format(
            root=FIXTURE_ROOT.as_posix(),
            epochs=epochs,
            output_dir=output_dir.as_posix(),
            initialization=initialization,
            resume=resume_text,
        ).lstrip()
        + experiment_yaml,
        encoding="utf-8",
    )
    return path


def test_train_only_run_skips_validation_and_persists_fixed_final_epoch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fomo_servo.training import engine

    output_dir = tmp_path / "train-only-output"
    base_config = load_config(
        _write_train_only_config(tmp_path / "base.yaml", output_dir, epochs=1)
    )
    checkpoint = tmp_path / "epoch_040_weights.pt"
    digest = _write_weights_snapshot(checkpoint, build_fomo_model(base_config))
    config = load_config(
        _write_train_only_config(
            tmp_path / "train-only.yaml",
            output_dir,
            epochs=1,
            initialize_from=checkpoint,
            initialize_sha256=digest,
            experiment=True,
        )
    )

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("validation/evaluation must not run in train-only mode")

    monkeypatch.setattr(engine, "validate_one_epoch", forbidden)
    monkeypatch.setattr(engine, "evaluate_validation_dataset", forbidden)
    monkeypatch.setattr(engine, "git_commit_sha", lambda _: "a" * 40)
    monkeypatch.setattr(
        engine, "git_worktree_fingerprint", lambda _: (False, "b" * 64)
    )

    summary = engine.run_training(config, device_override="cpu")

    assert summary.start_epoch == 1
    assert summary.completed_epochs == 1
    assert summary.best_val_f1 is None
    assert summary.best_grid_f1 is None
    assert summary.best_centroid_f1 is None
    assert summary.best_epoch is None
    assert summary.final_train_loss > 0.0
    assert summary.checkpoint_policy == "fixed_final_epoch"
    assert summary.initialization["sha256"] == digest
    assert (output_dir / "last.pt").is_file()
    assert (output_dir / "epoch_snapshots/epoch_001_weights.pt").is_file()
    assert not (output_dir / "best_val_f1.pt").exists()
    assert not (output_dir / "best_grid_f1.pt").exists()
    assert not (output_dir / "best_centroid_f1.pt").exists()
    with (output_dir / "history.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert rows[0]["train_loss"]
    assert rows[0]["val_loss"] == ""
    assert rows[0]["grid_f1"] == ""
    last = torch.load(output_dir / "last.pt", map_location="cpu", weights_only=False)
    assert last["checkpoint_policy"] == "fixed_final_epoch"
    assert last["selection_metric"] == "fixed_final_epoch"
    assert last["best_val_f1"] is None
    assert last["grid_f1"] is None
    assert last["initialization"]["sha256"] == digest
    persisted = json.loads(
        (output_dir / "training_summary.json").read_text(encoding="utf-8")
    )
    assert persisted["best_val_f1"] is None
    assert persisted["final_train_loss"] == pytest.approx(summary.final_train_loss)
    assert persisted["initialization"]["source_epoch"] == 40
    metadata = json.loads(
        (output_dir / "experiment_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["checkpoint_policy"] == "fixed_final_epoch"
    assert metadata["validation_split"] is None


def test_train_only_last_checkpoint_resumes_without_fabricated_metrics(
    tmp_path: Path,
) -> None:
    from fomo_servo.training import engine

    output_dir = tmp_path / "train-only-resume"
    first = load_config(
        _write_train_only_config(tmp_path / "first.yaml", output_dir, epochs=1)
    )
    engine.run_training(first, device_override="cpu")
    second = load_config(
        _write_train_only_config(
            tmp_path / "second.yaml",
            output_dir,
            epochs=2,
            resume=output_dir / "last.pt",
        )
    )

    summary = engine.run_training(second, device_override="cpu")

    assert summary.start_epoch == 2
    assert summary.completed_epochs == 2
    assert summary.best_val_f1 is None
    with (output_dir / "history.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert [int(row["epoch"]) for row in rows] == [1, 2]


def test_epoch_snapshot_interval_writes_weights_only_without_changing_legacy_checkpoints(
    tmp_path: Path,
) -> None:
    """v2 snapshots are opt-in and leave full checkpoint protocol intact."""

    _, run_training, _ = _engine_api()
    assert callable(run_training)
    output_dir = tmp_path / "snapshot-run"
    config = load_config(
        _write_training_config(
            tmp_path / "snapshot.yaml",
            output_dir,
            epochs=2,
            epoch_snapshots_yaml=(
                "  epoch_snapshots:\n"
                "    enabled: true\n"
                "    format: weights_only\n"
                "    interval: 2\n"
                "    keep_last: null"
            ),
        )
    )

    run_training(config, device_override="cpu")

    snapshots = sorted((output_dir / "epoch_snapshots").glob("*.pt"))
    assert [path.name for path in snapshots] == ["epoch_002_weights.pt"]
    payload = torch.load(snapshots[0], map_location="cpu", weights_only=False)
    assert payload["checkpoint_kind"] == "epoch_snapshot"
    assert "optimizer_state" not in payload
    assert (output_dir / "last.pt").is_file()
    legacy_payload = torch.load(output_dir / "last.pt", map_location="cpu", weights_only=False)
    assert "optimizer_state" in legacy_payload


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_mapped_resume_rng_state_restores_cpu_torch_state() -> None:
    """CUDA-mapped checkpoints must restore the CPU RNG state without error."""

    from fomo_servo.training.engine import _capture_rng_state, _restore_rng_state

    state = _capture_rng_state()
    cuda_mapped_state = dict(state)
    cuda_mapped_state["torch"] = state["torch"].to("cuda")
    cuda_mapped_state["cuda"] = [item.to("cuda") for item in state["cuda"]]

    _restore_rng_state(cuda_mapped_state)


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
    assert not (output_dir / "epoch_snapshots").exists()
    assert summary_path.is_file()
    checkpoint_payload = torch.load(last_checkpoint, map_location="cpu", weights_only=False)
    assert checkpoint_payload["checkpoint_type"] == "last"
    assert checkpoint_payload["selection_metric"] == "last"
    assert checkpoint_payload["selection_threshold"] == pytest.approx(0.5)
    assert checkpoint_payload["centroid_f1"] >= 0.0
    assert checkpoint_payload["class_weight_mode"] == "manual"
    assert checkpoint_payload["class_weights"] == [1.0, 3.0]
    assert len(checkpoint_payload["class_statistics"]) == 1
    assert checkpoint_payload["augmentation_preset"] is None
    assert "color_jitter" in checkpoint_payload["resolved_augmentation"]
    assert checkpoint_payload["augmentation_stats"]["total_samples"] > 0
    assert checkpoint_payload["model_metadata"]["backbone_name"] == "mobilenet_v2_lite"
    assert checkpoint_payload["model_metadata"]["initialization"] == "pytorch_module_defaults"
    training_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert training_summary["class_weight_mode"] == "manual"
    assert training_summary["class_weights"] == [1.0, 3.0]
    assert len(training_summary["class_statistics"]) == 1
    assert training_summary["best_grid_epoch"] >= 1
    assert training_summary["best_centroid_epoch"] >= 1
    assert training_summary["checkpoint_threshold"] == pytest.approx(0.5)
    assert training_summary["best_val_f1_alias_target"] == "best_grid_f1.pt"
    assert len(training_summary["augmentation_epoch_stats"]) == 2
    assert training_summary["model_metadata"] == checkpoint_payload["model_metadata"]
    assert [item["epoch"] for item in training_summary["augmentation_epoch_stats"]] == [1, 2]
    with history_path.open("r", newline="", encoding="utf-8") as history_file:
        first_rows = list(csv.DictReader(history_file))
    assert len(first_rows) == 2
    assert "augmentation_stats" in first_rows[0]
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
    resumed_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert resumed_summary["augmentation_epoch_stats"][-1]["epoch"] == 3


def test_new_backbone_metadata_is_persisted_in_training_artifacts(
    tmp_path: Path,
) -> None:
    """A real one-epoch CPU run must persist exact new-topology identity."""

    _, run_training, _ = _engine_api()
    assert callable(run_training)
    output_dir = tmp_path / "new-backbone-run"
    config_path = _write_training_config(
        tmp_path / "new-backbone.yaml", output_dir, epochs=1
    )
    payload = config_path.read_text(encoding="utf-8").replace(
        "  backbone: mobilenet_v2_lite\n",
        "  backbone: mobilenet_v2_fomo\n"
        "  cut_point: block_6_expand_relu\n"
        "  pretrained: false\n",
    )
    config_path.write_text(payload, encoding="utf-8")

    summary = run_training(load_config(config_path), device_override="cpu")
    checkpoint = torch.load(
        output_dir / "last.pt", map_location="cpu", weights_only=False
    )
    persisted_summary = json.loads(
        (output_dir / "training_summary.json").read_text(encoding="utf-8")
    )
    expected = {
        "backbone_name": "mobilenet_v2_fomo",
        "width_multiplier": 0.35,
        "cut_point": "block_6_expand_relu",
        "cut_point_output_channels": 96,
        "output_stride": 8,
        "head_channels": 32,
        "pretrained": False,
        "initialization": "pytorch_module_defaults",
        "backbone_parameter_count": 15_840,
        "head_parameter_count": 3_170,
        "parameter_count": 19_010,
        "cut_point_input_channels": 16,
    }
    assert summary.model_metadata == expected
    assert checkpoint["model_metadata"] == expected
    assert persisted_summary["model_metadata"] == expected


def test_training_propagates_epoch_to_train_dataset_and_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The train Dataset receives the current and resumed epoch before loading data."""

    _, run_training, _ = _engine_api()
    assert callable(run_training)
    calls: list[int] = []
    original_set_epoch = YOLOv5FOMODataset.set_epoch

    def recording_set_epoch(dataset: YOLOv5FOMODataset, epoch: int) -> None:
        calls.append(epoch)
        original_set_epoch(dataset, epoch)

    monkeypatch.setattr(YOLOv5FOMODataset, "set_epoch", recording_set_epoch)
    output_dir = tmp_path / "epoch-aware-run"
    first = load_config(_write_training_config(tmp_path / "first.yaml", output_dir, epochs=1))
    run_training(first, device_override="cpu")
    assert calls == [1]

    resumed = load_config(
        _write_training_config(
            tmp_path / "resumed.yaml", output_dir, epochs=2, resume=output_dir / "last.pt"
        )
    )
    run_training(resumed, device_override="cpu")
    assert calls == [1, 2]


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
