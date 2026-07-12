"""Tests for aug02 train-only horizontal flip geometry and isolation."""

from __future__ import annotations

from dataclasses import asdict
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
from fomo_servo.datasets import (
    AbsoluteBox,
    AugmentationPipeline,
    YOLOv5FOMODataset,
    flip_boxes_horizontally,
    generate_fomo_heatmap,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = ROOT / "tests" / "fixtures" / "yolo_micro"


def _config(
    *,
    enabled: bool = True,
    flip_enabled: bool = True,
    flip_probability: float = 1.0,
    color_enabled: bool = False,
) -> AugmentationConfig:
    disabled = AugmentationProbabilityConfig(enabled=False, probability=0.0)
    return AugmentationConfig(
        enabled=enabled,
        color_jitter=ColorJitterConfig(
            enabled=color_enabled,
            probability=1.0,
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.02,
        ),
        horizontal_flip=AugmentationProbabilityConfig(
            enabled=flip_enabled, probability=flip_probability
        ),
        gaussian_blur=disabled,
        gaussian_noise=disabled,
        affine=disabled,
    )


def _image() -> np.ndarray:
    """Return an RGB ``uint8 [8,12,3]`` image with a distinct left/right pattern."""

    image = np.zeros((8, 12, 3), dtype=np.uint8)
    image[:, :4] = (255, 0, 0)
    image[:, 4:8] = (0, 255, 0)
    image[:, 8:] = (0, 0, 255)
    return image


def _boxes() -> tuple[AbsoluteBox, ...]:
    return (
        AbsoluteBox(2, 0.0, 1.0, 3.0, 5.0),
        AbsoluteBox(0, 4.0, 2.0, 8.0, 6.0),
        AbsoluteBox(1, 5.0, 0.0, 7.0, 8.0),
        AbsoluteBox(3, 9.0, 3.0, 12.0, 7.0),
    )


def _identity_collate(samples: Sequence[Any]) -> list[Any]:
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


def test_disabled_augmentation_is_exact_noop() -> None:
    image = _image()
    boxes = _boxes()
    result = AugmentationPipeline(_config(enabled=False), is_train=True).apply(
        image, boxes, np.random.default_rng(1)
    )

    np.testing.assert_array_equal(result.image, image)
    assert result.boxes == boxes
    assert result.metadata.horizontal_flip_applied is False


def test_horizontal_flip_disabled_is_exact_noop() -> None:
    image = _image()
    boxes = _boxes()
    result = AugmentationPipeline(
        _config(flip_enabled=False), is_train=True
    ).apply(image, boxes, np.random.default_rng(1))

    np.testing.assert_array_equal(result.image, image)
    assert result.boxes == boxes
    assert result.metadata.horizontal_flip_applied is False


def test_horizontal_flip_probability_zero_is_exact_noop() -> None:
    image = _image()
    boxes = _boxes()
    result = AugmentationPipeline(
        _config(flip_probability=0.0), is_train=True
    ).apply(image, boxes, np.random.default_rng(1))

    np.testing.assert_array_equal(result.image, image)
    assert result.boxes == boxes
    assert result.metadata.horizontal_flip_applied is False


def test_horizontal_flip_probability_one_always_executes() -> None:
    image = _image()
    boxes = _boxes()
    result = AugmentationPipeline(_config(), is_train=True).apply(
        image, boxes, np.random.default_rng(1)
    )

    np.testing.assert_array_equal(result.image, image[:, ::-1, :])
    assert result.boxes == flip_boxes_horizontally(boxes, image.shape[1])
    assert result.metadata.horizontal_flip_applied is True


def test_fixed_seed_repeats_horizontal_flip_result() -> None:
    pipeline = AugmentationPipeline(_config(flip_probability=0.5), is_train=True)
    left = pipeline.apply(_image(), _boxes(), np.random.default_rng(42))
    right = pipeline.apply(_image(), _boxes(), np.random.default_rng(42))

    np.testing.assert_array_equal(left.image, right.image)
    assert left.boxes == right.boxes
    assert left.metadata == right.metadata


def test_validation_and_test_pipelines_never_flip() -> None:
    image = _image()
    boxes = _boxes()
    for pipeline in (
        AugmentationPipeline(_config(), is_train=False),
        AugmentationPipeline(_config(), is_train=False),
    ):
        result = pipeline.apply(image, boxes, np.random.default_rng(1))
        np.testing.assert_array_equal(result.image, image)
        assert result.boxes == boxes
        assert result.metadata.horizontal_flip_applied is False


def test_normalized_center_formula_and_xyxy_continuous_coordinate_convention() -> None:
    width = 100.0
    original = AbsoluteBox(0, 10.0, 12.0, 30.0, 28.0)
    flipped = flip_boxes_horizontally((original,), width)[0]

    original_center = (original.x_min + original.x_max) / (2.0 * width)
    flipped_center = (flipped.x_min + flipped.x_max) / (2.0 * width)
    assert flipped_center == pytest.approx(1.0 - original_center)
    assert flipped.y_min == original.y_min
    assert flipped.y_max == original.y_max
    assert flipped.x_max - flipped.x_min == original.x_max - original.x_min


def test_left_right_center_and_multiclass_order_are_preserved() -> None:
    boxes = (
        AbsoluteBox(4, 0.0, 1.0, 10.0, 5.0),
        AbsoluteBox(2, 45.0, 2.0, 55.0, 6.0),
        AbsoluteBox(1, 90.0, 3.0, 100.0, 7.0),
    )
    flipped = flip_boxes_horizontally(boxes, 100.0)

    assert [box.foreground_class_id for box in flipped] == [4, 2, 1]
    assert flipped[0].x_min == 90.0 and flipped[0].x_max == 100.0
    assert flipped[1].x_min == 45.0 and flipped[1].x_max == 55.0
    assert flipped[2].x_min == 0.0 and flipped[2].x_max == 10.0
    for before, after in zip(boxes, flipped):
        assert after.y_min == before.y_min
        assert after.y_max == before.y_max


def test_boundary_boxes_remain_inside_image() -> None:
    flipped = flip_boxes_horizontally(
        (AbsoluteBox(0, 0.0, 0.0, 100.0, 50.0),), 100.0
    )[0]
    assert 0.0 <= flipped.x_min <= flipped.x_max <= 100.0
    assert 0.0 <= flipped.y_min <= flipped.y_max <= 50.0


def test_two_horizontal_flips_restore_image_and_boxes() -> None:
    image = _image()
    boxes = _boxes()
    pipeline = AugmentationPipeline(_config(), is_train=True)
    once = pipeline.apply(image, boxes, np.random.default_rng(1))
    twice = pipeline.apply(once.image, once.boxes, np.random.default_rng(1))

    np.testing.assert_array_equal(twice.image, image)
    for before, after in zip(boxes, twice.boxes):
        assert after == before


def test_color_jitter_plus_flip_preserves_target_count_and_classes() -> None:
    result = AugmentationPipeline(
        _config(color_enabled=True), is_train=True
    ).apply(_image(), _boxes(), np.random.default_rng(4))

    assert len(result.boxes) == len(_boxes())
    assert [box.foreground_class_id for box in result.boxes] == [
        box.foreground_class_id for box in _boxes()
    ]
    assert result.metadata.horizontal_flip_applied is True


def test_flip_dataset_heatmap_mirrors_centroids_before_stride_quantization() -> None:
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
    sample = dataset.get_sample(3, np.random.default_rng(4))
    width = sample.original_image.shape[1]
    expected_centroids = []
    for box in sample.original_boxes:
        flipped = flip_boxes_horizontally((box,), width)[0]
        x, y = sample.transform.forward_point(*flipped.center)
        expected_centroids.append((x, y, flipped.foreground_class_id))
    expected = generate_fomo_heatmap(
        expected_centroids,
        input_size=96,
        stride=8,
        num_foreground_classes=dataset.num_foreground_classes,
        collision_policy="keep_first",
    )

    assert sample.augmented_boxes == flip_boxes_horizontally(
        sample.original_boxes, width
    )
    np.testing.assert_array_equal(sample.heatmap.class_index, expected.class_index)
    np.testing.assert_array_equal(sample.heatmap.one_hot, expected.one_hot)


def test_flip_recomputes_same_and_different_class_collisions() -> None:
    same = flip_boxes_horizontally(
        (AbsoluteBox(0, 0.0, 0.0, 8.0, 8.0), AbsoluteBox(0, 0.0, 0.0, 8.0, 8.0)),
        96.0,
    )
    same_target = generate_fomo_heatmap(
        [(box.center[0], box.center[1], box.foreground_class_id) for box in same],
        input_size=96,
        stride=8,
        num_foreground_classes=2,
        collision_policy="keep_first",
    )
    different = flip_boxes_horizontally(
        (AbsoluteBox(0, 0.0, 0.0, 8.0, 8.0), AbsoluteBox(1, 0.0, 0.0, 8.0, 8.0)),
        96.0,
    )
    different_target = generate_fomo_heatmap(
        [
            (box.center[0], box.center[1], box.foreground_class_id)
            for box in different
        ],
        input_size=96,
        stride=8,
        num_foreground_classes=2,
        collision_policy="keep_first",
    )

    assert same_target.same_class_collision_count == 1
    assert same_target.different_class_collision_count == 0
    assert different_target.same_class_collision_count == 0
    assert different_target.different_class_collision_count == 1


def test_fixed_worker_seed_repeats_flip_dataset_samples() -> None:
    first = _collect_worker_images(42)
    second = _collect_worker_images(42)

    for left, right in zip(first, second):
        np.testing.assert_array_equal(left, right)


def test_aug02_resolved_config_only_changes_flip_and_allowed_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FOMO_DATASET_ROOT", str(FIXTURE_ROOT))
    aug01 = load_config(ROOT / "configs/experiments/aug01_color.yaml")
    aug02 = load_config(ROOT / "configs/experiments/aug02_color_hflip.yaml")

    def comparable(config: Any) -> dict[str, Any]:
        value = asdict(config)
        value.pop("source_path")
        value.pop("augmentation")
        value["experiment"].pop("name")
        value["training"].pop("output_dir")
        return value

    assert comparable(aug01) == comparable(aug02)
    assert aug02.experiment.name == "aug02_color_hflip"
    assert aug02.training.output_dir.as_posix().endswith(
        "outputs/experiments/aug02_color_hflip"
    )
    assert asdict(aug02.augmentation.color_jitter) == asdict(
        aug01.augmentation.color_jitter
    )
    assert aug02.augmentation.horizontal_flip.enabled is True
    assert aug02.augmentation.horizontal_flip.probability == pytest.approx(0.5)
    for name in ("gaussian_blur", "gaussian_noise", "affine"):
        assert getattr(aug02.augmentation, name).enabled is False
        assert getattr(aug02.augmentation, name).probability == 0.0
