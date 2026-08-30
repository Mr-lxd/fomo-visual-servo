"""TDD contracts for the online augmentation suite."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence
import warnings

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from fomo_servo.config import (
    AffineConfig,
    AugmentationConfig,
    AugmentationProbabilityConfig,
    ColorJitterConfig,
    GaussianBlurConfig,
    GaussianNoiseConfig,
    ConfigurationError,
    load_config,
)
from fomo_servo.datasets import AbsoluteBox, AugmentationPipeline, YOLOv5FOMODataset
from fomo_servo.datasets.rng import make_sample_rng, stable_sample_seed


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = ROOT / "tests" / "fixtures" / "yolo_micro"


def _disabled_probability() -> AugmentationProbabilityConfig:
    return AugmentationProbabilityConfig(enabled=False, probability=0.0)


def _suite_config(
    *,
    enabled: bool = True,
    preset: str | None = "custom",
    color: ColorJitterConfig | None = None,
    hflip: AugmentationProbabilityConfig | None = None,
    blur: GaussianBlurConfig | None = None,
    noise: GaussianNoiseConfig | None = None,
    affine: AffineConfig | None = None,
) -> AugmentationConfig:
    return AugmentationConfig(
        enabled=enabled,
        preset=preset,
        color_jitter=color or ColorJitterConfig(),
        horizontal_flip=hflip or _disabled_probability(),
        gaussian_blur=blur or GaussianBlurConfig(),
        gaussian_noise=noise or GaussianNoiseConfig(),
        affine=affine or AffineConfig(),
    )


def _image() -> np.ndarray:
    y, x = np.indices((32, 40))
    return np.stack(
        ((x * 7 + y * 3) % 256, (x * 11 + y * 5) % 256, (x * 13 + y * 17) % 256),
        axis=-1,
    ).astype(np.uint8)


def _boxes() -> tuple[AbsoluteBox, ...]:
    return (
        AbsoluteBox(0, 2.0, 4.0, 14.0, 18.0),
        AbsoluteBox(1, 24.0, 10.0, 38.0, 30.0),
    )


def _signature(sample: Any) -> tuple[Any, ...]:
    metadata = sample.augmentation_metadata
    return (
        sample.image_path.name,
        metadata.epoch,
        metadata.sample_index,
        metadata.sample_seed,
        metadata.color_jitter_applied,
        metadata.brightness_factor,
        metadata.contrast_factor,
        metadata.saturation_factor,
        metadata.hue_shift,
        metadata.horizontal_flip_applied,
        hashlib.sha256(sample.augmented_image.tobytes()).hexdigest(),
    )


def _identity_collate(samples: Sequence[Any]) -> list[Any]:
    return list(samples)


def _dataset(seed: int = 42) -> YOLOv5FOMODataset:
    return YOLOv5FOMODataset(
        FIXTURE_ROOT,
        split="train",
        input_size=96,
        stride=8,
        class_mode="preserve",
        augmentation=_suite_config(
            color=ColorJitterConfig(
                enabled=True,
                probability=1.0,
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.02,
            ),
            hflip=AugmentationProbabilityConfig(enabled=True, probability=0.5),
        ),
        train_split="train",
        augmentation_seed=seed,
    )


def _write_preset_config(path: Path, augmentation: str) -> Path:
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
training:
  seed: 42
augmentation:
  enabled: true
  preset: {augmentation}
  overrides: {{}}
""".lstrip(),
        encoding="utf-8",
    )
    return path


def test_stable_sample_seed_is_worker_independent_and_epoch_aware() -> None:
    first = stable_sample_seed(42, 0, 7)
    assert first == stable_sample_seed(42, 0, 7)
    assert first != stable_sample_seed(42, 1, 7)
    assert first != stable_sample_seed(43, 0, 7)
    assert make_sample_rng(42, 0, 7).random() == make_sample_rng(42, 0, 7).random()


def test_dataset_same_epoch_repeats_and_next_epoch_changes() -> None:
    dataset = _dataset()
    dataset.set_epoch(0)
    first = _signature(dataset[0])
    repeated = _signature(dataset[0])
    dataset.set_epoch(1)
    next_epoch = _signature(dataset[0])

    assert first == repeated
    assert first != next_epoch
    assert next_epoch[1:4] == (1, 0, next_epoch[3])


def test_dataset_metadata_contains_epoch_index_and_sample_seed() -> None:
    dataset = _dataset(seed=123)
    dataset.set_epoch(5)
    metadata = dataset[1].augmentation_metadata

    assert metadata.epoch == 5
    assert metadata.sample_index == 1
    assert metadata.sample_seed == stable_sample_seed(123, 5, 1)
    assert isinstance(metadata.color_jitter_applied, bool)
    assert isinstance(metadata.horizontal_flip_applied, bool)


def test_worker_counts_produce_same_per_index_results() -> None:
    results = []
    for workers in (0, 2, 4):
        dataset = _dataset()
        dataset.set_epoch(3)
        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=workers,
            persistent_workers=False,
            collate_fn=_identity_collate,
        )
        results.append([_signature(batch[0]) for batch in loader])

    assert results[0] == results[1] == results[2]


def test_same_full_loader_seed_reproduces_and_different_seed_changes() -> None:
    def collect(seed: int) -> list[tuple[Any, ...]]:
        torch.manual_seed(seed)
        dataset = _dataset(seed=seed)
        dataset.set_epoch(2)
        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=True,
            num_workers=2,
            persistent_workers=False,
            generator=torch.Generator().manual_seed(seed),
            collate_fn=_identity_collate,
        )
        return [_signature(batch[0]) for batch in loader]

    first = collect(42)
    second = collect(42)
    different = collect(43)
    assert first == second
    assert first != different


def test_validation_dataset_is_noop_for_repeated_reads() -> None:
    dataset = _dataset()
    validation = YOLOv5FOMODataset(
        FIXTURE_ROOT,
        split="val",
        input_size=96,
        stride=8,
        class_mode="preserve",
        augmentation=dataset.augmentation,
        train_split="train",
        augmentation_seed=42,
    )
    first = validation[0]
    for _ in range(4):
        current = validation[0]
        np.testing.assert_array_equal(current.augmented_image, first.augmented_image)
        assert current.augmentation_metadata.applied is False
        assert current.augmentation_metadata.color_jitter_applied is False
        assert current.augmentation_metadata.horizontal_flip_applied is False


def test_none_preset_is_fully_disabled(tmp_path: Path) -> None:
    config = load_config(_write_preset_config(tmp_path / "none.yaml", "none"))
    assert config.augmentation.preset == "none"
    assert config.augmentation.enabled is True
    assert not config.augmentation.color_jitter.enabled
    assert not config.augmentation.horizontal_flip.enabled
    assert not config.augmentation.gaussian_blur.enabled
    assert not config.augmentation.gaussian_noise.enabled
    assert not config.augmentation.affine.enabled


def test_photometric_preset_expands_exact_parameters(tmp_path: Path) -> None:
    config = load_config(
        _write_preset_config(tmp_path / "photometric.yaml", "photometric")
    )
    assert config.augmentation.color_jitter.probability == pytest.approx(0.8)
    assert config.augmentation.gaussian_blur.probability == pytest.approx(0.15)
    assert config.augmentation.gaussian_blur.kernel_sizes == (3, 5)
    assert config.augmentation.gaussian_noise.std_min == pytest.approx(2.0)
    assert not config.augmentation.horizontal_flip.enabled
    assert not config.augmentation.affine.enabled


def test_underwater_preset_expands_geometry_parameters(tmp_path: Path) -> None:
    config = load_config(
        _write_preset_config(tmp_path / "underwater.yaml", "underwater_conservative")
    )
    assert config.augmentation.horizontal_flip.probability == pytest.approx(0.5)
    assert config.augmentation.affine.probability == pytest.approx(0.30)
    assert config.augmentation.affine.scale_min == pytest.approx(0.90)
    assert config.augmentation.affine.scale_max == pytest.approx(1.10)
    assert config.augmentation.affine.translate_fraction == pytest.approx(0.05)
    assert config.augmentation.affine.rotation_degrees == pytest.approx(5.0)


def test_unknown_preset_and_override_are_diagnostic(tmp_path: Path) -> None:
    unknown = _write_preset_config(tmp_path / "unknown.yaml", "not_a_preset")
    with pytest.raises(ConfigurationError, match="unknown.*preset|preset"):
        load_config(unknown)

    override = _write_preset_config(tmp_path / "override.yaml", "custom")
    text = override.read_text(encoding="utf-8").replace(
        "overrides: {}", "overrides:\n    affine.unknown_field: 1"
    )
    override.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigurationError, match="override|unknown"):
        load_config(override)


def test_custom_preset_applies_known_overrides(tmp_path: Path) -> None:
    config_path = _write_preset_config(tmp_path / "custom.yaml", "custom")
    text = config_path.read_text(encoding="utf-8").replace(
        "overrides: {}",
        "overrides:\n    color_jitter.enabled: true\n    color_jitter.probability: 1.0\n    color_jitter.brightness: 0.3",
    )
    config_path.write_text(text, encoding="utf-8")
    config = load_config(config_path)
    assert config.augmentation.color_jitter.enabled is True
    assert config.augmentation.color_jitter.probability == pytest.approx(1.0)
    assert config.augmentation.color_jitter.brightness == pytest.approx(0.3)


def test_invalid_blur_and_affine_parameters_are_diagnostic(tmp_path: Path) -> None:
    config_path = _write_preset_config(tmp_path / "invalid.yaml", "custom")
    text = config_path.read_text(encoding="utf-8").replace(
        "overrides: {}",
        "overrides:\n    gaussian_blur.kernel_sizes: [2]\n    affine.border_value: 300",
    )
    config_path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigurationError, match="kernel_sizes|border_value"):
        load_config(config_path)


def test_disabled_global_flag_overrides_enabled_preset(tmp_path: Path) -> None:
    config_path = _write_preset_config(tmp_path / "disabled.yaml", "underwater_conservative")
    text = config_path.read_text(encoding="utf-8").replace("enabled: true", "enabled: false", 1)
    config_path.write_text(text, encoding="utf-8")
    config = load_config(config_path)
    result = AugmentationPipeline(config.augmentation, is_train=True).apply(
        _image(), _boxes(), np.random.default_rng(7)
    )
    np.testing.assert_array_equal(result.image, _image())
    assert result.boxes == _boxes()
    assert result.applied is False


def test_gaussian_blur_preserves_shape_dtype_range_and_geometry() -> None:
    config = _suite_config(
        blur=GaussianBlurConfig(
            enabled=True,
            probability=1.0,
            kernel_sizes=(3,),
            sigma_min=0.5,
            sigma_max=0.5,
        )
    )
    result = AugmentationPipeline(config, is_train=True).apply(
        _image(), _boxes(), np.random.default_rng(11)
    )
    assert result.image.shape == _image().shape
    assert result.image.dtype == np.uint8
    assert int(result.image.min()) >= 0 and int(result.image.max()) <= 255
    assert result.boxes == _boxes()
    assert result.metadata.gaussian_blur_applied is True
    assert result.metadata.blur_kernel == 3
    assert result.metadata.blur_sigma == pytest.approx(0.5)


def test_gaussian_noise_is_reproducible_and_preserves_contract() -> None:
    config = _suite_config(
        noise=GaussianNoiseConfig(
            enabled=True, probability=1.0, std_min=4.0, std_max=4.0
        )
    )
    pipeline = AugmentationPipeline(config, is_train=True)
    first = pipeline.apply(_image(), _boxes(), np.random.default_rng(12))
    second = pipeline.apply(_image(), _boxes(), np.random.default_rng(12))
    np.testing.assert_array_equal(first.image, second.image)
    assert first.image.shape == _image().shape
    assert first.image.dtype == np.uint8
    assert int(first.image.min()) >= 0 and int(first.image.max()) <= 255
    assert first.boxes == _boxes()
    assert first.metadata.noise_std == pytest.approx(4.0)


def test_affine_identity_is_exact_noop() -> None:
    config = _suite_config(
        affine=AffineConfig(
            enabled=True,
            probability=1.0,
            scale_min=1.0,
            scale_max=1.0,
            translate_fraction=0.0,
            rotation_degrees=0.0,
            min_visibility=0.25,
            border_value=114,
        )
    )
    result = AugmentationPipeline(config, is_train=True).apply(
        _image(), _boxes(), np.random.default_rng(13)
    )
    np.testing.assert_array_equal(result.image, _image())
    assert result.boxes == _boxes()
    assert result.metadata.affine_applied is False


def test_affine_translation_moves_boxes_and_centroids() -> None:
    config = _suite_config(
        affine=AffineConfig(
            enabled=True,
            probability=1.0,
            scale_min=1.0,
            scale_max=1.0,
            translate_fraction=0.1,
            rotation_degrees=0.0,
            min_visibility=0.0,
            border_value=114,
        )
    )
    result = AugmentationPipeline(config, is_train=True).apply(
        _image(), _boxes(), np.random.default_rng(14)
    )
    assert result.metadata.affine_applied is True
    assert len(result.boxes) == len(_boxes())
    assert result.boxes != _boxes()
    assert all(0.0 <= box.x_min <= box.x_max <= 40.0 for box in result.boxes)
    assert all(0.0 <= box.y_min <= box.y_max <= 32.0 for box in result.boxes)


def test_affine_scale_and_rotation_change_geometry_without_invalid_boxes() -> None:
    scale_config = _suite_config(
        affine=AffineConfig(
            enabled=True,
            probability=1.0,
            scale_min=1.1,
            scale_max=1.1,
            translate_fraction=0.0,
            rotation_degrees=0.0,
            min_visibility=0.0,
            border_value=114,
        )
    )
    rotation_config = _suite_config(
        affine=AffineConfig(
            enabled=True,
            probability=1.0,
            scale_min=1.0,
            scale_max=1.0,
            translate_fraction=0.0,
            rotation_degrees=5.0,
            min_visibility=0.0,
            border_value=114,
        )
    )
    scaled = AugmentationPipeline(scale_config, is_train=True).apply(
        _image(), _boxes(), np.random.default_rng(18)
    )
    rotated = AugmentationPipeline(rotation_config, is_train=True).apply(
        _image(), _boxes(), np.random.default_rng(19)
    )
    assert scaled.boxes != _boxes()
    assert rotated.boxes != _boxes()
    for result in (scaled, rotated):
        assert result.image.shape == _image().shape
        assert result.image.dtype == np.uint8
        assert all(0.0 <= box.x_min <= box.x_max <= 40.0 for box in result.boxes)
        assert all(0.0 <= box.y_min <= box.y_max <= 32.0 for box in result.boxes)


def test_affine_clipping_and_visibility_drop_are_recorded() -> None:
    config = _suite_config(
        affine=AffineConfig(
            enabled=True,
            probability=1.0,
            scale_min=1.0,
            scale_max=1.0,
            translate_fraction=1.0,
            rotation_degrees=0.0,
            min_visibility=0.75,
            border_value=114,
        )
    )
    result = AugmentationPipeline(config, is_train=True).apply(
        _image(), _boxes(), np.random.default_rng(15)
    )
    assert result.metadata.clipped_bbox_count >= 0
    assert result.metadata.dropped_bbox_count >= 0
    assert result.metadata.pre_augmentation_object_count == len(_boxes())
    assert result.metadata.post_augmentation_object_count == len(result.boxes)
    assert all(0.0 <= box.x_min <= box.x_max <= 40.0 for box in result.boxes)
    assert all(0.0 <= box.y_min <= box.y_max <= 32.0 for box in result.boxes)


def test_affine_dataset_rebuilds_heatmap_and_collision_metadata() -> None:
    config = _suite_config(
        affine=AffineConfig(
            enabled=True,
            probability=1.0,
            scale_min=1.0,
            scale_max=1.0,
            translate_fraction=0.05,
            rotation_degrees=0.0,
            min_visibility=0.0,
            border_value=114,
        )
    )
    dataset = YOLOv5FOMODataset(
        FIXTURE_ROOT,
        split="train",
        input_size=96,
        stride=8,
        class_mode="preserve",
        augmentation=config,
        train_split="train",
        augmentation_seed=42,
    )
    sample = dataset.get_sample(0, np.random.default_rng(17))
    assert sample.heatmap.class_index.shape == (12, 12)
    assert sample.augmentation_metadata.same_class_collision_count == (
        sample.heatmap.same_class_collision_count
    )
    assert sample.augmentation_metadata.different_class_collision_count == (
        sample.heatmap.different_class_collision_count
    )
    assert sample.augmentation_metadata.post_augmentation_object_count == len(
        sample.augmented_boxes
    )


def test_disabled_blur_noise_affine_are_elementwise_noop() -> None:
    config = _suite_config()
    result = AugmentationPipeline(config, is_train=True).apply(
        _image(), _boxes(), np.random.default_rng(16)
    )
    np.testing.assert_array_equal(result.image, _image())
    assert result.boxes == _boxes()
    assert result.applied is False


def test_legacy_configs_keep_locked_non_augmentation_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FOMO_DATASET_ROOT", str(FIXTURE_ROOT))
    base = load_config(ROOT / "configs/experiments/aug00_none_locked.yaml")
    aug01 = load_config(ROOT / "configs/experiments/aug01_color.yaml")
    aug02 = load_config(ROOT / "configs/experiments/aug02_color_hflip.yaml")

    def comparable(config: Any) -> dict[str, Any]:
        value = asdict(config)
        value.pop("source_path")
        value.pop("augmentation")
        value["experiment"].pop("name")
        value["training"].pop("output_dir")
        return value

    assert comparable(base) == comparable(aug01) == comparable(aug02)


def test_augmentation_suite_config_only_changes_augmentation_and_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FOMO_DATASET_ROOT", str(FIXTURE_ROOT))
    base = load_config(ROOT / "configs/experiments/aug00_none_locked.yaml")
    suite = load_config(ROOT / "configs/experiments/augmentation_suite.yaml")

    def comparable(config: Any) -> dict[str, Any]:
        value = asdict(config)
        value.pop("source_path")
        value.pop("augmentation")
        value["experiment"].pop("name")
        value["training"].pop("output_dir")
        return value

    assert comparable(base) == comparable(suite)
    assert suite.augmentation.preset == "underwater_conservative"


def test_aug03_config_is_locked_except_preset_and_output_without_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FOMO_DATASET_ROOT", str(FIXTURE_ROOT))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        base = load_config(ROOT / "configs/experiments/aug00_none_locked.yaml")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        aug03 = load_config(
            ROOT / "configs/experiments/aug03_underwater_conservative.yaml"
        )

    assert not [warning for warning in caught if issubclass(warning.category, DeprecationWarning)]

    def comparable(config: Any) -> dict[str, Any]:
        value = asdict(config)
        value.pop("source_path")
        value.pop("augmentation")
        value["experiment"].pop("name")
        value["training"].pop("output_dir")
        return value

    assert comparable(base) == comparable(aug03)
    assert aug03.augmentation.preset == "underwater_conservative"
    assert aug03.augmentation.horizontal_flip.enabled is True
    assert aug03.augmentation.gaussian_blur.kernel_sizes == (3, 5)
    assert aug03.augmentation.gaussian_noise.std_min == pytest.approx(2.0)
    assert aug03.augmentation.affine.min_visibility == pytest.approx(0.25)
