"""CLI-level smoke coverage for checkpoint selection protocol v2."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = ROOT / "tests" / "fixtures" / "yolo_micro"


def _write_smoke_config(path: Path, output_dir: Path, *, epochs: int) -> Path:
    """Write a deterministic 96x96 CPU smoke configuration."""

    path.write_text(
        f"""
dataset:
  root: "{FIXTURE_ROOT.as_posix()}"
  train_split: train
  validation_split: val
  classes: [creature]
  class_mode: merge_single
  merged_class_name: creature
  collision_policy: keep_first
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
  output_dir: "{output_dir.as_posix()}"
  resume: null
  early_stopping_patience: 0
  early_stopping_min_delta: 0.0
  checkpoint_criterion: centroid_f1
  epoch_snapshots:
    enabled: true
    format: weights_only
    interval: 1
    keep_last: null
  optimizer:
    name: adamw
    learning_rate: 0.001
    weight_decay: 0.0
  scheduler:
    name: step_lr
    step_size: 1
    gamma: 0.9
evaluation:
  checkpoint_threshold: 0.5
  checkpoint_selection:
    metric: centroid_pr_auc_macro
    split: val
    threshold_grid:
      minimum: 0.5
      maximum: 0.9
      step: 0.2
""".lstrip(),
        encoding="utf-8",
    )
    return path


def _run_script(script_name: str, *arguments: Path | str) -> subprocess.CompletedProcess[str]:
    """Run one repository CLI in the current Python environment."""

    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script_name), *(str(item) for item in arguments)],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


def test_cli_snapshot_selection_candidate_and_resume_smoke(tmp_path: Path) -> None:
    """Exercise train, offline selection, candidate evaluation, and both resume paths."""

    output_dir = tmp_path / "checkpoint-selection-v2-smoke"
    config_path = _write_smoke_config(tmp_path / "smoke.yaml", output_dir, epochs=2)

    trained = _run_script("train.py", "--config", config_path, "--device", "cpu")
    assert trained.returncode == 0, trained.stdout + trained.stderr

    snapshot_dir = output_dir / "epoch_snapshots"
    snapshots = sorted(snapshot_dir.glob("epoch_*_weights.pt"))
    assert [path.name for path in snapshots] == [
        "epoch_001_weights.pt",
        "epoch_002_weights.pt",
    ]
    snapshot_payloads = [
        torch.load(path, map_location="cpu", weights_only=False) for path in snapshots
    ]
    for expected_epoch, payload in enumerate(snapshot_payloads, start=1):
        assert payload["checkpoint_kind"] == "epoch_snapshot"
        assert payload["epoch"] == expected_epoch
        assert payload["weights_only"] is True
        assert payload["resumable"] is False
        assert payload["model_metadata"]["backbone_name"] == "mobilenet_v2_lite"
        assert payload["config_fingerprint"]
        assert payload["dataset_content_hash"]
        assert all(tensor.device.type == "cpu" for tensor in payload["model_state"].values())
        assert "optimizer_state" not in payload
        assert "scheduler_state" not in payload
        assert "scaler_state" not in payload
        assert "rng_state" not in payload
    first_payload = snapshot_payloads[0]

    selection_dir = output_dir / "selection"
    selected = _run_script(
        "evaluate_epoch_snapshots.py",
        "--config",
        config_path,
        "--snapshot-dir",
        snapshot_dir,
        "--device",
        "cpu",
        "--output-dir",
        selection_dir,
    )
    assert selected.returncode == 0, selected.stdout + selected.stderr
    summary = json.loads((selection_dir / "selection_summary.json").read_text(encoding="utf-8"))
    assert summary["selection_metric"] == "centroid_pr_auc_macro"
    assert summary["selected"]["epoch"] in {1, 2}
    assert (selection_dir / "epoch_snapshot_metrics.csv").is_file()
    assert (selection_dir / "epoch_snapshot_metrics.json").is_file()

    primary_candidate = output_dir / "best_centroid_pr_auc_macro.pt"
    sweep_candidate = output_dir / "best_sweep_centroid_f1.pt"
    for candidate in (primary_candidate, sweep_candidate):
        payload = torch.load(candidate, map_location="cpu", weights_only=False)
        assert payload["checkpoint_kind"] == "inference_candidate"
        assert payload["weights_only"] is True
        assert payload["resumable"] is False
        assert payload["source_snapshot"] in {path.name for path in snapshots}
        assert payload["source_snapshot_sha256"]

    candidate_report = _run_script(
        "evaluate.py",
        "--config",
        config_path,
        "--checkpoint",
        primary_candidate,
        "--device",
        "cpu",
        "--output-json",
        selection_dir / "candidate_evaluation.json",
    )
    assert candidate_report.returncode == 0, candidate_report.stdout + candidate_report.stderr
    assert json.loads(
        (selection_dir / "candidate_evaluation.json").read_text(encoding="utf-8")
    )["checkpoint"].endswith("best_centroid_pr_auc_macro.pt")

    candidate_resume = _run_script(
        "train.py",
        "--config",
        config_path,
        "--device",
        "cpu",
        "--resume",
        primary_candidate,
    )
    assert candidate_resume.returncode == 1
    assert "inference/evaluation candidate" in candidate_resume.stderr
    assert "optimizer" in candidate_resume.stderr

    mixed_variants = (
        ("model", "model identity", lambda payload: _with_model_mismatch(payload)),
        ("config", "config fingerprint", lambda payload: _with_config_mismatch(payload)),
        ("dataset", "dataset content hash", lambda payload: _with_dataset_mismatch(payload)),
    )
    for index, (name, error_fragment, mutate) in enumerate(mixed_variants, start=1):
        mixed_dir = tmp_path / ("mixed-" + name) / "epoch_snapshots"
        mixed_dir.mkdir(parents=True)
        for source in snapshots:
            shutil.copyfile(source, mixed_dir / source.name)
        mixed_payload = mutate(dict(first_payload))
        mixed_payload["epoch"] = 900 + index
        mixed_snapshot = mixed_dir / (f"epoch_{900 + index:03d}_weights.pt")
        torch.save(mixed_payload, mixed_snapshot)
        mixed_selection = _run_script(
            "evaluate_epoch_snapshots.py",
            "--config",
            config_path,
            "--snapshot-dir",
            mixed_dir,
            "--device",
            "cpu",
            "--output-dir",
            tmp_path / ("mixed-selection-" + name),
            "--no-write-candidates",
        )
        assert mixed_selection.returncode == 1
        assert error_fragment in mixed_selection.stderr

    _write_smoke_config(config_path, output_dir, epochs=3)
    resumed = _run_script(
        "train.py",
        "--config",
        config_path,
        "--device",
        "cpu",
        "--resume",
        output_dir / "last.pt",
    )
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    resumed_payload = torch.load(output_dir / "last.pt", map_location="cpu", weights_only=False)
    assert resumed_payload["epoch"] == 3
    assert (snapshot_dir / "epoch_003_weights.pt").is_file()


def _with_model_mismatch(payload: dict[str, object]) -> dict[str, object]:
    """Return a snapshot payload with a different model identity."""

    metadata = dict(payload["model_metadata"])
    metadata["backbone_name"] = "different-backbone"
    payload["model_metadata"] = metadata
    return payload


def _with_config_mismatch(payload: dict[str, object]) -> dict[str, object]:
    """Return a snapshot payload with a different configuration fingerprint."""

    payload["config_fingerprint"] = "mixed-config-fingerprint"
    return payload


def _with_dataset_mismatch(payload: dict[str, object]) -> dict[str, object]:
    """Return a snapshot payload with a different dataset content hash."""

    payload["dataset_content_hash"] = "mixed-dataset-content-hash"
    return payload
