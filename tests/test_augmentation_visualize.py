"""CLI smoke test for the disabled augmentation visualization interface."""

from __future__ import annotations

import subprocess
import sys
import json
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]


def _write_visualization_config(path: Path) -> Path:
    """Write a small config for the synthetic fixture contact-sheet smoke test."""

    path.write_text(
        f"""
dataset:
  root: "{(ROOT / 'tests/fixtures/yolo_micro').as_posix()}"
  train_split: train
  validation_split: val
  classes: [fish, crab]
  class_mode: preserve
model:
  input_size: 96
  output_stride: 8
training:
  seed: 42
augmentation:
  enabled: true
  color_jitter:
    enabled: true
    probability: 1.0
    brightness: 0.2
    contrast: 0.2
    saturation: 0.2
    hue: 0.02
""".lstrip(),
        encoding="utf-8",
    )
    return path


def _write_hflip_visualization_config(path: Path) -> Path:
    """Write an aug02-like config for the synthetic geometric visualization test."""

    path.write_text(
        f"""
dataset:
  root: "{(ROOT / 'tests/fixtures/yolo_micro').as_posix()}"
  train_split: train
  validation_split: val
  classes: [fish, crab]
  class_mode: preserve
model:
  input_size: 96
  output_stride: 8
training:
  seed: 42
augmentation:
  enabled: true
  color_jitter:
    enabled: true
    probability: 1.0
    brightness: 0.2
    contrast: 0.2
    saturation: 0.2
    hue: 0.02
  horizontal_flip:
    enabled: true
    probability: 0.5
  gaussian_blur:
    enabled: false
    probability: 0.0
  gaussian_noise:
    enabled: false
    probability: 0.0
  affine:
    enabled: false
    probability: 0.0
""".lstrip(),
        encoding="utf-8",
    )
    return path


def _write_suite_visualization_config(path: Path) -> Path:
    """Write a compact underwater preset config for the suite smoke test."""

    path.write_text(
        f"""
dataset:
  root: "{(ROOT / 'tests/fixtures/yolo_micro').as_posix()}"
  train_split: train
  validation_split: val
  classes: [fish, crab]
  class_mode: preserve
model:
  input_size: 96
  output_stride: 8
training:
  seed: 42
augmentation:
  enabled: true
  preset: underwater_conservative
  overrides: {{}}
experiment:
  name: augmentation_suite
""".lstrip(),
        encoding="utf-8",
    )
    return path


def test_visualize_augmentations_writes_fixture_panel(tmp_path: Path) -> None:
    """The future visualization entry point must consume no-op dataset outputs."""

    output_path = tmp_path / "augmentation.jpg"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/visualize_augmentations.py",
            "--dataset-root",
            "tests/fixtures/yolo_micro",
            "--split",
            "train",
            "--index",
            "0",
            "--input-size",
            "96",
            "--output",
            str(output_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    panel = cv2.imread(str(output_path), cv2.IMREAD_COLOR)
    assert panel is not None
    assert panel.shape == (192, 192, 3)


def test_visualize_augmentations_writes_color_contact_sheet_and_json(
    tmp_path: Path,
) -> None:
    """The config-driven mode emits deterministic audit artifacts for fixture images."""

    output_dir = tmp_path / "visualization"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/visualize_augmentations.py",
            "--config",
            str(_write_visualization_config(tmp_path / "aug01.yaml")),
            "--dataset-root",
            "tests/fixtures/yolo_micro",
            "--split",
            "train",
            "--num-images",
            "2",
            "--input-size",
            "96",
            "--output-dir",
            str(output_dir),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    contact_sheet = output_dir / "color_jitter_contact_sheet.jpg"
    samples_json = output_dir / "color_jitter_samples.json"
    assert contact_sheet.is_file()
    assert samples_json.is_file()
    records = json.loads(samples_json.read_text(encoding="utf-8"))
    assert len(records) >= 2 * 8
    assert all("brightness_factor" in record for record in records)


def test_visualize_augmentations_writes_hflip_contact_sheet_and_geometry_json(
    tmp_path: Path,
) -> None:
    """The aug02 mode emits forced-flip panels and original/transformed geometry."""

    output_dir = tmp_path / "hflip_visualization"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/visualize_augmentations.py",
            "--config",
            str(_write_hflip_visualization_config(tmp_path / "aug02.yaml")),
            "--dataset-root",
            "tests/fixtures/yolo_micro",
            "--split",
            "train",
            "--num-images",
            "2",
            "--input-size",
            "96",
            "--output-dir",
            str(output_dir),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    contact_sheet = output_dir / "horizontal_flip_contact_sheet.jpg"
    samples_json = output_dir / "horizontal_flip_samples.json"
    assert contact_sheet.is_file()
    assert samples_json.is_file()
    records = json.loads(samples_json.read_text(encoding="utf-8"))
    assert len(records) == 2 * 6
    forced = [record for record in records if record["case"] == "forced_flip"]
    assert len(forced) == 2
    assert all(record["horizontal_flip_applied"] for record in forced)
    assert all("original_boxes" in record for record in records)
    assert all("flipped_centroids" in record for record in records)


def test_visualize_augmentations_writes_full_suite_outputs(tmp_path: Path) -> None:
    """The suite mode emits epoch, preset, affine and metadata artifacts."""

    output_dir = tmp_path / "suite_visualization"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/visualize_augmentations.py",
            "--config",
            str(_write_suite_visualization_config(tmp_path / "suite.yaml")),
            "--split",
            "train",
            "--num-images",
            "2",
            "--output-dir",
            str(output_dir),
            "--suite",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    expected = (
        "rng_across_epochs_contact_sheet.jpg",
        "photometric_preset_contact_sheet.jpg",
        "underwater_conservative_contact_sheet.jpg",
        "affine_geometry_contact_sheet.jpg",
        "augmentation_samples.json",
    )
    assert all((output_dir / name).is_file() for name in expected)
    records = json.loads((output_dir / "augmentation_samples.json").read_text(encoding="utf-8"))
    assert len(records) == 2 * 6
    assert all(not Path(record["relative_image_path"]).is_absolute() for record in records)
    assert all("sample_seed" in record for record in records)
    assert all("dropped_bbox_count" in record for record in records)
