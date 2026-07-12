"""Tests for disabled FOMO augmentation configuration and dataset equivalence."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from fomo_servo.config import load_config
from fomo_servo.datasets import YOLOv5FOMODataset


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = ROOT / "tests" / "fixtures" / "yolo_micro"


def _write_config(path: Path, augmentation: str) -> Path:
    """Write a minimal project config containing one augmentation block."""

    path.write_text(
        f"""
dataset:
  root: "{FIXTURE_ROOT.as_posix()}"
  train_split: train
  validation_split: val
  classes: [fish, crab]
  class_mode: preserve
model:
  input_size: 96
  output_stride: 8
augmentation:
{augmentation}
""".lstrip(),
        encoding="utf-8",
    )
    return path


def _disabled_yaml() -> str:
    """Return the complete all-disabled augmentation schema."""

    return """  enabled: false
  color_jitter:
    enabled: false
    probability: 0.0
    brightness: 0.0
    contrast: 0.0
    saturation: 0.0
    hue: 0.0
  horizontal_flip:
    enabled: false
    probability: 0.0
  gaussian_blur:
    enabled: false
    probability: 0.0
  gaussian_noise:
    enabled: false
    probability: 0.0
  affine:
    enabled: false
    probability: 0.0
"""


def _augmentation_api() -> tuple[Any, Any, Any]:
    """Return the public no-op pipeline API with clear missing-API assertions."""

    try:
        module = importlib.import_module("fomo_servo.datasets.augmentation")
    except ModuleNotFoundError:
        return None, None, None
    return (
        getattr(module, "AugmentationPipeline", None),
        getattr(module, "AugmentationNotImplementedError", None),
        getattr(module, "AugmentationResult", None),
    )


def _sample_equal(left: Any, right: Any) -> None:
    """Assert all required disabled-pipeline sample outputs are identical."""

    np.testing.assert_array_equal(left.image, right.image)
    np.testing.assert_array_equal(left.original_image, right.original_image)
    np.testing.assert_array_equal(left.augmented_image, right.augmented_image)
    np.testing.assert_array_equal(left.letterbox_image, right.letterbox_image)
    assert left.original_boxes == right.original_boxes
    assert left.augmented_boxes == right.augmented_boxes
    assert left.letterbox_boxes == right.letterbox_boxes
    assert left.transform == right.transform
    np.testing.assert_array_equal(
        left.heatmap.class_index, right.heatmap.class_index
    )
    np.testing.assert_array_equal(left.heatmap.one_hot, right.heatmap.one_hot)
    assert left.heatmap.same_class_collision_count == right.heatmap.same_class_collision_count
    assert (
        left.heatmap.different_class_collision_count
        == right.heatmap.different_class_collision_count
    )


def test_disabled_augmentation_schema_loads_all_requested_fields(tmp_path: Path) -> None:
    """The complete schema must be available without enabling any operation."""

    config = load_config(_write_config(tmp_path / "disabled.yaml", _disabled_yaml()))

    assert hasattr(config, "augmentation"), "augmentation config must be available"
    augmentation = config.augmentation
    assert augmentation.enabled is False
    for name in (
        "color_jitter",
        "horizontal_flip",
        "gaussian_blur",
        "gaussian_noise",
        "affine",
    ):
        operation = getattr(augmentation, name)
        assert operation.enabled is False
        assert operation.probability == 0.0


def test_augmentation_rejects_probability_outside_unit_interval(tmp_path: Path) -> None:
    """Configuration errors must identify invalid augmentation probabilities."""

    invalid = _disabled_yaml().replace("probability: 0.0", "probability: 1.1", 1)
    with pytest.raises(Exception, match="probability"):
        load_config(_write_config(tmp_path / "invalid.yaml", invalid))


def test_disabled_pipeline_matches_legacy_dataset_output_elementwise(
    tmp_path: Path,
) -> None:
    """All image, box, transform, heatmap, and collision outputs must be unchanged."""

    config = load_config(_write_config(tmp_path / "disabled.yaml", _disabled_yaml()))
    pipeline_type, _, _ = _augmentation_api()
    assert callable(pipeline_type), "AugmentationPipeline must be importable"
    legacy = YOLOv5FOMODataset(
        FIXTURE_ROOT,
        split="train",
        input_size=96,
        stride=8,
        class_mode="preserve",
    )[3]
    framework = YOLOv5FOMODataset(
        FIXTURE_ROOT,
        split="train",
        input_size=96,
        stride=8,
        class_mode="preserve",
        augmentation=config.augmentation,
        train_split="train",
    )[3]

    _sample_equal(legacy, framework)


def test_validation_and_test_pipelines_are_always_noop(tmp_path: Path) -> None:
    """Non-training splits must not apply augmentation even when a config is supplied."""

    config = load_config(_write_config(tmp_path / "disabled.yaml", _disabled_yaml()))
    pipeline_type, _, _ = _augmentation_api()
    assert callable(pipeline_type), "AugmentationPipeline must be importable"
    image = np.zeros((12, 20, 3), dtype=np.uint8)
    boxes: tuple[Any, ...] = ()
    train_disabled = pipeline_type(config.augmentation, is_train=True)
    validation = pipeline_type(config.augmentation, is_train=False)
    test = pipeline_type(config.augmentation, is_train=False)

    train_result = train_disabled.apply(image, boxes, np.random.default_rng(42))
    validation_result = validation.apply(image, boxes, np.random.default_rng(7))
    test_result = test.apply(image, boxes, np.random.default_rng(99))

    np.testing.assert_array_equal(train_result.image, validation_result.image)
    np.testing.assert_array_equal(validation_result.image, test_result.image)
    assert train_result.boxes == validation_result.boxes == test_result.boxes == boxes


def test_validation_dataset_forces_noop_even_if_global_augmentation_is_enabled(
    tmp_path: Path,
) -> None:
    """The dataset split gate prevents validation from entering future algorithms."""

    enabled = _disabled_yaml().replace("enabled: false", "enabled: true", 1)
    config = load_config(_write_config(tmp_path / "enabled.yaml", enabled))
    validation = YOLOv5FOMODataset(
        FIXTURE_ROOT,
        split="val",
        input_size=96,
        stride=8,
        class_mode="preserve",
        augmentation=config.augmentation,
        train_split=config.dataset.train_split,
    )
    disabled = YOLOv5FOMODataset(
        FIXTURE_ROOT,
        split="val",
        input_size=96,
        stride=8,
        class_mode="preserve",
        augmentation=config.augmentation.disabled(),
        train_split=config.dataset.train_split,
    )
    _sample_equal(validation[0], disabled[0])


def test_disabled_pipeline_does_not_consume_random_state(tmp_path: Path) -> None:
    """The no-op path remains deterministic under identical seed and worker seed inputs."""

    config = load_config(_write_config(tmp_path / "disabled.yaml", _disabled_yaml()))
    pipeline_type, _, _ = _augmentation_api()
    assert callable(pipeline_type), "AugmentationPipeline must be importable"
    pipeline = pipeline_type(config.augmentation, is_train=True)
    image = np.zeros((12, 20, 3), dtype=np.uint8)
    boxes: tuple[Any, ...] = ()
    rng_a = np.random.default_rng(42)
    rng_b = np.random.default_rng(42)

    result_a = pipeline.apply(image, boxes, rng_a)
    result_b = pipeline.apply(image, boxes, rng_b)

    np.testing.assert_array_equal(result_a.image, result_b.image)
    assert result_a.boxes == result_b.boxes
    assert rng_a.random() == rng_b.random()


def test_disabled_dataset_is_deterministic_for_repeated_worker_style_reads(
    tmp_path: Path,
) -> None:
    """Repeated train reads remain identical under the worker-seeded no-op path."""

    config = load_config(_write_config(tmp_path / "disabled.yaml", _disabled_yaml()))
    dataset = YOLOv5FOMODataset(
        FIXTURE_ROOT,
        split="train",
        input_size=96,
        stride=8,
        class_mode="preserve",
        augmentation=config.augmentation,
        train_split="train",
    )
    _sample_equal(dataset[0], dataset[0])


def test_enabled_future_augmentation_fails_explicitly(tmp_path: Path) -> None:
    """No future algorithm may be silently treated as implemented in this phase."""

    enabled = _disabled_yaml().replace("enabled: false", "enabled: true", 1)
    config = load_config(_write_config(tmp_path / "enabled.yaml", enabled))
    pipeline_type, error_type, _ = _augmentation_api()
    assert callable(pipeline_type), "AugmentationPipeline must be importable"
    assert isinstance(error_type, type), "AugmentationNotImplementedError must exist"

    with pytest.raises(error_type, match="not implemented"):
        pipeline_type(config.augmentation, is_train=True).apply(
            np.zeros((12, 20, 3), dtype=np.uint8), (), np.random.default_rng(42)
        )
