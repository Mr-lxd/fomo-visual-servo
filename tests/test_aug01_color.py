"""Behavioral and experiment-isolation tests for aug01 color jitter."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from fomo_servo.config import (
    AugmentationConfig,
    AugmentationProbabilityConfig,
    ColorJitterConfig,
    load_config,
)
from fomo_servo.datasets import YOLOv5FOMODataset
from fomo_servo.datasets.augmentation import (
    AugmentationPipeline,
    ColorJitterFactors,
    apply_color_jitter,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = ROOT / "tests" / "fixtures" / "yolo_micro"


def _config(
    *,
    enabled: bool = True,
    color_enabled: bool = True,
    probability: float = 1.0,
    brightness: float = 0.2,
    contrast: float = 0.2,
    saturation: float = 0.2,
    hue: float = 0.02,
) -> AugmentationConfig:
    """Build an aug01-like config without changing unrelated project settings."""

    disabled_operation = AugmentationProbabilityConfig(enabled=False, probability=0.0)
    return AugmentationConfig(
        enabled=enabled,
        color_jitter=ColorJitterConfig(
            enabled=color_enabled,
            probability=probability,
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
        ),
        horizontal_flip=disabled_operation,
        gaussian_blur=disabled_operation,
        gaussian_noise=disabled_operation,
        affine=disabled_operation,
    )


def _image() -> np.ndarray:
    """Return a non-uniform RGB ``uint8 [18,24,3]`` synthetic image."""

    y, x = np.indices((18, 24))
    return np.stack(
        (
            (x * 11 + y * 3) % 256,
            (x * 5 + y * 17) % 256,
            (x * 19 + y * 7) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)


def _identity_collate(samples: Sequence[Any]) -> list[Any]:
    """Keep worker-produced FOMO samples picklable for RNG repeatability tests."""

    return list(samples)


def _collect_worker_images(seed: int) -> list[np.ndarray]:
    dataset = YOLOv5FOMODataset(
        FIXTURE_ROOT,
        split="train",
        input_size=96,
        stride=8,
        class_mode="preserve",
        augmentation=_config(),
        train_split="train",
        augmentation_seed=42,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        generator=torch.Generator().manual_seed(seed),
        collate_fn=_identity_collate,
    )
    return [batch[0].image.copy() for batch in loader]


def test_global_disabled_color_jitter_is_exact_noop() -> None:
    pipeline = AugmentationPipeline(_config(enabled=False), is_train=True)
    image = _image()
    result = pipeline.apply(image, (), np.random.default_rng(5))

    np.testing.assert_array_equal(result.image, image)
    assert result.applied is False
    assert result.metadata.brightness_factor == 1.0
    assert result.metadata.hue_shift == 0.0


def test_color_jitter_operation_disabled_is_exact_noop() -> None:
    pipeline = AugmentationPipeline(_config(color_enabled=False), is_train=True)
    result = pipeline.apply(_image(), (), np.random.default_rng(5))

    np.testing.assert_array_equal(result.image, _image())
    assert result.applied is False


def test_probability_zero_is_exact_noop() -> None:
    pipeline = AugmentationPipeline(_config(probability=0.0), is_train=True)
    image = _image()
    result = pipeline.apply(image, (), np.random.default_rng(5))

    np.testing.assert_array_equal(result.image, image)
    assert result.applied is False


def test_zero_color_parameters_are_exact_noop_even_at_probability_one() -> None:
    pipeline = AugmentationPipeline(
        _config(
            probability=1.0,
            brightness=0.0,
            contrast=0.0,
            saturation=0.0,
            hue=0.0,
        ),
        is_train=True,
    )
    image = _image()
    result = pipeline.apply(image, (), np.random.default_rng(5))

    np.testing.assert_array_equal(result.image, image)
    assert result.applied is False


def test_fixed_seed_repeats_color_jitter_elementwise() -> None:
    pipeline = AugmentationPipeline(_config(), is_train=True)
    left = pipeline.apply(_image(), (), np.random.default_rng(17))
    right = pipeline.apply(_image(), (), np.random.default_rng(17))

    np.testing.assert_array_equal(left.image, right.image)
    assert left.metadata == right.metadata
    assert left.applied is True


def test_different_seed_changes_color_jitter_image() -> None:
    pipeline = AugmentationPipeline(_config(), is_train=True)
    left = pipeline.apply(_image(), (), np.random.default_rng(17))
    right = pipeline.apply(_image(), (), np.random.default_rng(18))

    assert not np.array_equal(left.image, right.image)


def test_validation_and_test_are_always_noop() -> None:
    image = _image()
    for split_pipeline in (
        AugmentationPipeline(_config(), is_train=False),
        AugmentationPipeline(_config(), is_train=False),
    ):
        result = split_pipeline.apply(image, (), np.random.default_rng(17))
        np.testing.assert_array_equal(result.image, image)
        assert result.applied is False


@pytest.mark.parametrize(
    ("rgb_value", "channel"),
    [((255, 0, 0), 0), ((0, 255, 0), 1), ((0, 0, 255), 2)],
)
def test_rgb_channel_order_is_preserved(
    rgb_value: tuple[int, int, int], channel: int
) -> None:
    image = np.tile(np.asarray(rgb_value, dtype=np.uint8), (8, 8, 1))
    result = apply_color_jitter(
        image,
        ColorJitterFactors(
            brightness_factor=1.0,
            contrast_factor=1.0,
            saturation_factor=0.5,
            hue_shift=0.0,
        ),
    )

    assert int(result[0, 0, channel]) == int(result[0, 0].max())
    assert result[0, 0, channel] > result[0, 0, (channel + 1) % 3]


def test_brightness_extremes_clip_without_overflow() -> None:
    image = np.asarray([[[0, 128, 255], [255, 128, 0]]], dtype=np.uint8)
    for factor in (0.8, 1.2):
        result = apply_color_jitter(
            image,
            ColorJitterFactors(factor, 1.0, 1.0, 0.0),
        )
        assert result.shape == image.shape
        assert result.dtype == image.dtype
        assert int(result.min()) >= 0
        assert int(result.max()) <= 255


def test_contrast_preserves_shape_and_dtype() -> None:
    result = apply_color_jitter(
        _image(), ColorJitterFactors(1.0, 1.2, 1.0, 0.0)
    )
    assert result.shape == (18, 24, 3)
    assert result.dtype == np.uint8


def test_saturation_and_hue_preserve_valid_pixel_range() -> None:
    result = apply_color_jitter(
        _image(), ColorJitterFactors(1.0, 1.0, 1.2, 0.02)
    )
    assert int(result.min()) >= 0
    assert int(result.max()) <= 255


def test_color_jitter_does_not_change_geometry_or_heatmaps() -> None:
    disabled = YOLOv5FOMODataset(
        FIXTURE_ROOT,
        split="train",
        input_size=96,
        stride=8,
        class_mode="preserve",
        augmentation=_config(enabled=False),
        train_split="train",
        augmentation_seed=42,
    )[3]
    colored_dataset = YOLOv5FOMODataset(
        FIXTURE_ROOT,
        split="train",
        input_size=96,
        stride=8,
        class_mode="preserve",
        augmentation=_config(),
        train_split="train",
        augmentation_seed=42,
    )
    colored = colored_dataset.get_sample(3, np.random.default_rng(7))

    assert colored.original_boxes == disabled.original_boxes
    assert colored.augmented_boxes == disabled.augmented_boxes
    assert colored.letterbox_boxes == disabled.letterbox_boxes
    assert colored.transform == disabled.transform
    np.testing.assert_array_equal(
        colored.heatmap.class_index, disabled.heatmap.class_index
    )
    np.testing.assert_array_equal(colored.heatmap.one_hot, disabled.heatmap.one_hot)
    assert colored.heatmap.same_class_collision_count == disabled.heatmap.same_class_collision_count
    assert (
        colored.heatmap.different_class_collision_count
        == disabled.heatmap.different_class_collision_count
    )
    assert colored.augmentation_metadata.applied is True


def test_fixed_seed_and_worker_seed_repeat_augmented_samples() -> None:
    first = _collect_worker_images(42)
    second = _collect_worker_images(42)

    assert len(first) == len(second)
    for left, right in zip(first, second):
        np.testing.assert_array_equal(left, right)


def test_aug01_resolved_config_only_drifts_allowed_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FOMO_DATASET_ROOT", str(FIXTURE_ROOT))
    base = load_config(ROOT / "configs/experiments/aug00_none_locked.yaml")
    aug01 = load_config(ROOT / "configs/experiments/aug01_color.yaml")

    def comparable(config: Any) -> dict[str, Any]:
        value = asdict(config)
        value.pop("source_path")
        value.pop("augmentation")
        value["experiment"].pop("name")
        value["training"].pop("output_dir")
        return value

    assert comparable(base) == comparable(aug01)
    assert aug01.experiment.name == "aug01_color"
    assert aug01.training.output_dir.as_posix().endswith(
        "outputs/experiments/aug01_color"
    )
    assert aug01.augmentation.enabled is True
    assert aug01.augmentation.color_jitter.probability == pytest.approx(0.8)
    assert aug01.augmentation.horizontal_flip.enabled is False
