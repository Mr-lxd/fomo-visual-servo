"""Tests for the transparent training-configuration preflight CLI."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = ROOT / "tests" / "fixtures" / "yolo_micro"


def _write_config(path: Path, output_dir: Path) -> Path:
    """Write a complete two-epoch CPU config that uses the synthetic fixture dataset."""

    path.write_text(
        """
dataset:
  root: "{fixture_root}"
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
  name: weighted_cross_entropy
  gamma: 0.0
  class_weights: [1.0, 2.0]
training:
  device: auto
  amp: false
  num_workers: 0
  pin_memory: false
  batch_size: 2
  epochs: 2
  seed: 123
  output_dir: "{output_dir}"
  resume: null
  early_stopping_patience: 0
  early_stopping_min_delta: 0.0
  optimizer:
    name: adamw
    learning_rate: 0.001
    weight_decay: 0.0
  scheduler:
    name: none
    step_size: 1
    gamma: 1.0
""".format(
            fixture_root=FIXTURE_ROOT.as_posix(), output_dir=output_dir.as_posix()
        ).lstrip(),
        encoding="utf-8",
    )
    return path


def test_train_cli_applies_explicit_cpu_override(tmp_path: Path) -> None:
    """`--device cpu` must take precedence over the YAML auto setting."""

    output_dir = tmp_path / "cli-run"
    config_path = _write_config(tmp_path / "runtime.yaml", output_dir)
    result = subprocess.run(
        [
            sys.executable,
            "scripts/train.py",
            "--config",
            str(config_path),
            "--device",
            "cpu",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Device: cpu" in result.stdout
    assert "AMP: disabled" in result.stdout
    assert "Training complete" in result.stdout
    assert (output_dir / "last.pt").is_file()
    assert (output_dir / "best_val_f1.pt").is_file()


def test_train_cli_reports_missing_config_file() -> None:
    """An absent configuration path must return a diagnosable non-zero result."""

    result = subprocess.run(
        [
            sys.executable,
            "scripts/train.py",
            "--config",
            "missing-training-config.yaml",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "Unable to read configuration" in result.stderr
