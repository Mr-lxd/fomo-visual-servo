"""Train-only online augmentation pipeline in the locked suite order."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Any, Optional, Sequence

import cv2
import numpy as np

from fomo_servo.config import (
    AffineConfig,
    AugmentationConfig,
    GaussianBlurConfig,
    GaussianNoiseConfig,
)


class AugmentationNotImplementedError(NotImplementedError):
    """Raised when an augmentation outside the supported suite is requested."""


@dataclass(frozen=True)
class ColorJitterFactors:
    """One sampled color transform for an RGB ``uint8 [H,W,3]`` image."""

    brightness_factor: float = 1.0
    contrast_factor: float = 1.0
    saturation_factor: float = 1.0
    hue_shift: float = 0.0

    @classmethod
    def neutral(cls) -> "ColorJitterFactors":
        return cls()

    def validate(self) -> None:
        for name, value in (
            ("brightness_factor", self.brightness_factor),
            ("contrast_factor", self.contrast_factor),
            ("saturation_factor", self.saturation_factor),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(value)
                or value <= 0.0
            ):
                raise ValueError("{} must be a finite positive number".format(name))
        if (
            isinstance(self.hue_shift, bool)
            or not isinstance(self.hue_shift, (int, float))
            or not isfinite(self.hue_shift)
        ):
            raise ValueError("hue_shift must be a finite number")


@dataclass(frozen=True)
class AugmentationMetadata:
    """Lightweight audit metadata that never enters training tensors."""

    applied: bool = False
    epoch: int = 0
    sample_index: int = -1
    sample_seed: Optional[int] = None
    color_jitter_applied: bool = False
    brightness_factor: float = 1.0
    contrast_factor: float = 1.0
    saturation_factor: float = 1.0
    hue_shift: float = 0.0
    horizontal_flip_applied: bool = False
    gaussian_blur_applied: bool = False
    blur_kernel: Optional[int] = None
    blur_sigma: Optional[float] = None
    gaussian_noise_applied: bool = False
    noise_std: Optional[float] = None
    affine_applied: bool = False
    affine_scale: Optional[float] = None
    affine_translate_x: Optional[float] = None
    affine_translate_y: Optional[float] = None
    affine_rotation: Optional[float] = None
    clipped_bbox_count: int = 0
    dropped_bbox_count: int = 0
    pre_augmentation_object_count: int = 0
    post_augmentation_object_count: int = 0
    same_class_collision_count: int = 0
    different_class_collision_count: int = 0

    @classmethod
    def neutral(
        cls,
        *,
        epoch: int = 0,
        sample_index: int = -1,
        sample_seed: Optional[int] = None,
        object_count: int = 0,
    ) -> "AugmentationMetadata":
        return cls(
            epoch=epoch,
            sample_index=sample_index,
            sample_seed=sample_seed,
            pre_augmentation_object_count=object_count,
            post_augmentation_object_count=object_count,
        )


# Backward-compatible public name used by aug01/aug02 callers.
ColorJitterMetadata = AugmentationMetadata


@dataclass(frozen=True)
class AugmentationResult:
    """One pre-letterbox RGB image and its transformed pixel-space boxes."""

    image: np.ndarray
    boxes: tuple[Any, ...]
    applied: bool
    metadata: AugmentationMetadata = field(default_factory=AugmentationMetadata.neutral)


def apply_color_jitter(image: np.ndarray, factors: ColorJitterFactors) -> np.ndarray:
    """Apply fixed RGB color factors to a ``uint8 [H,W,3]`` image."""

    _validate_image(image)
    if not isinstance(factors, ColorJitterFactors):
        raise TypeError("factors must be a ColorJitterFactors instance")
    factors.validate()
    if factors == ColorJitterFactors.neutral():
        return image.copy()
    rgb = image.astype(np.float32) / 255.0
    rgb = np.clip(rgb * float(factors.brightness_factor), 0.0, 1.0)
    if factors.contrast_factor != 1.0:
        mean = float(rgb.mean())
        rgb = np.clip((rgb - mean) * float(factors.contrast_factor) + mean, 0.0, 1.0)
    if factors.saturation_factor != 1.0 or factors.hue_shift != 0.0:
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        hsv[..., 1] = np.clip(hsv[..., 1] * float(factors.saturation_factor), 0.0, 1.0)
        hsv[..., 0] = np.mod(hsv[..., 0] + float(factors.hue_shift) * 360.0, 360.0)
        rgb = np.clip(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB), 0.0, 1.0)
    return np.rint(rgb * 255.0).astype(image.dtype, copy=False)


def flip_boxes_horizontally(boxes: Sequence[Any], image_width: float) -> tuple[Any, ...]:
    """Mirror continuous xyxy boxes using ``x_min'=W-x_max`` and ``x_max'=W-x_min``."""

    if (
        isinstance(image_width, bool)
        or not isinstance(image_width, (int, float))
        or not isfinite(image_width)
        or image_width <= 0.0
    ):
        raise ValueError("image_width must be a finite positive number")
    if not isinstance(boxes, Sequence):
        raise TypeError("boxes must be a sequence")
    flipped = []
    for box in boxes:
        try:
            flipped.append(
                type(box)(
                    box.foreground_class_id,
                    image_width - box.x_max,
                    box.y_min,
                    image_width - box.x_min,
                    box.y_max,
                )
            )
        except AttributeError as error:
            raise TypeError("boxes must contain objects with foreground_class_id and xyxy fields") from error
    return tuple(flipped)


class AugmentationPipeline:
    """Apply hflip, affine, color, blur and noise before letterbox."""

    def __init__(self, config: AugmentationConfig, *, is_train: bool) -> None:
        if not isinstance(config, AugmentationConfig):
            raise TypeError("config must be an AugmentationConfig instance")
        if not isinstance(is_train, bool):
            raise TypeError("is_train must be a boolean")
        self.config = config
        self.is_train = is_train

    @classmethod
    def disabled(cls) -> "AugmentationPipeline":
        return cls(AugmentationConfig.disabled(), is_train=False)

    def apply(
        self,
        image: np.ndarray,
        boxes: Sequence[Any],
        rng: Optional[np.random.Generator] = None,
        *,
        epoch: int = 0,
        sample_index: int = -1,
        sample_seed: Optional[int] = None,
    ) -> AugmentationResult:
        """Apply the fixed online order to RGB ``uint8 [H,W,3]`` and xyxy boxes."""

        _validate_image(image)
        if not isinstance(boxes, Sequence):
            raise TypeError("augmentation boxes must be a sequence")
        original_count = len(boxes)
        if not self.is_train or not self.config.enabled:
            return _no_op_result(image, boxes, epoch, sample_index, sample_seed)
        if not _any_operation_active(self.config):
            return _no_op_result(image, boxes, epoch, sample_index, sample_seed)
        if not isinstance(rng, np.random.Generator):
            raise TypeError("an explicit numpy.random.Generator is required for augmentation")

        augmented_image = image.copy()
        augmented_boxes = tuple(boxes)
        factors = ColorJitterFactors.neutral()
        metadata = {
            "epoch": epoch,
            "sample_index": sample_index,
            "sample_seed": sample_seed,
            "pre_augmentation_object_count": original_count,
            "post_augmentation_object_count": original_count,
        }

        flip = self.config.horizontal_flip
        if _operation_active(flip) and float(rng.random()) < flip.probability:
            augmented_image = np.ascontiguousarray(augmented_image[:, ::-1, :])
            augmented_boxes = flip_boxes_horizontally(augmented_boxes, augmented_image.shape[1])
            metadata["horizontal_flip_applied"] = True

        affine = _as_affine_config(self.config.affine)
        if _operation_active(affine) and float(rng.random()) < affine.probability:
            augmented_image, augmented_boxes, affine_values = _apply_affine(
                augmented_image, augmented_boxes, affine, rng
            )
            metadata.update(affine_values)

        color = self.config.color_jitter
        if _color_operation_active(color) and float(rng.random()) < color.probability:
            factors = ColorJitterFactors(
                brightness_factor=float(rng.uniform(1.0 - color.brightness, 1.0 + color.brightness)),
                contrast_factor=float(rng.uniform(1.0 - color.contrast, 1.0 + color.contrast)),
                saturation_factor=float(rng.uniform(1.0 - color.saturation, 1.0 + color.saturation)),
                hue_shift=float(rng.uniform(-color.hue, color.hue)),
            )
            augmented_image = apply_color_jitter(augmented_image, factors)
            metadata.update(
                color_jitter_applied=True,
                brightness_factor=factors.brightness_factor,
                contrast_factor=factors.contrast_factor,
                saturation_factor=factors.saturation_factor,
                hue_shift=factors.hue_shift,
            )

        blur = _as_blur_config(self.config.gaussian_blur)
        if _operation_active(blur) and float(rng.random()) < blur.probability:
            kernel = int(rng.choice(blur.kernel_sizes))
            sigma = float(rng.uniform(blur.sigma_min, blur.sigma_max))
            augmented_image = cv2.GaussianBlur(
                augmented_image, (kernel, kernel), sigmaX=sigma, borderType=cv2.BORDER_REPLICATE
            )
            metadata.update(gaussian_blur_applied=True, blur_kernel=kernel, blur_sigma=sigma)

        noise = _as_noise_config(self.config.gaussian_noise)
        if _operation_active(noise) and float(rng.random()) < noise.probability:
            std = float(rng.uniform(noise.std_min, noise.std_max))
            noisy = augmented_image.astype(np.float32) + rng.normal(0.0, std, augmented_image.shape)
            augmented_image = np.rint(np.clip(noisy, 0.0, 255.0)).astype(np.uint8)
            metadata.update(gaussian_noise_applied=True, noise_std=std)

        metadata["post_augmentation_object_count"] = len(augmented_boxes)
        applied = any(
            metadata.get(name, False)
            for name in (
                "color_jitter_applied",
                "horizontal_flip_applied",
                "gaussian_blur_applied",
                "gaussian_noise_applied",
                "affine_applied",
            )
        )
        metadata["applied"] = applied
        return AugmentationResult(
            image=augmented_image,
            boxes=augmented_boxes,
            applied=applied,
            metadata=AugmentationMetadata(**metadata),
        )


def _operation_active(config: Any) -> bool:
    return bool(getattr(config, "enabled", False) and getattr(config, "probability", 0.0) > 0.0)


def _color_operation_active(config: Any) -> bool:
    return _operation_active(config) and any(
        getattr(config, name, 0.0) != 0.0
        for name in ("brightness", "contrast", "saturation", "hue")
    )


def _any_operation_active(config: AugmentationConfig) -> bool:
    return _color_operation_active(config.color_jitter) or any(
        _operation_active(getattr(config, name))
        for name in ("horizontal_flip", "gaussian_blur", "gaussian_noise", "affine")
    )


def _as_blur_config(config: Any) -> GaussianBlurConfig:
    if isinstance(config, GaussianBlurConfig):
        return config
    return GaussianBlurConfig(enabled=getattr(config, "enabled", False), probability=getattr(config, "probability", 0.0))


def _as_noise_config(config: Any) -> GaussianNoiseConfig:
    if isinstance(config, GaussianNoiseConfig):
        return config
    return GaussianNoiseConfig(enabled=getattr(config, "enabled", False), probability=getattr(config, "probability", 0.0))


def _as_affine_config(config: Any) -> AffineConfig:
    if isinstance(config, AffineConfig):
        return config
    return AffineConfig(enabled=getattr(config, "enabled", False), probability=getattr(config, "probability", 0.0))


def _apply_affine(
    image: np.ndarray,
    boxes: Sequence[Any],
    config: AffineConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, tuple[Any, ...], dict[str, Any]]:
    """Apply one affine matrix and return clipped continuous boxes."""

    height, width = image.shape[:2]
    scale = float(rng.uniform(config.scale_min, config.scale_max))
    translate_x = float(rng.uniform(-config.translate_fraction, config.translate_fraction) * width)
    translate_y = float(rng.uniform(-config.translate_fraction, config.translate_fraction) * height)
    rotation = float(rng.uniform(-config.rotation_degrees, config.rotation_degrees))
    values: dict[str, Any] = {
        "affine_scale": scale,
        "affine_translate_x": translate_x,
        "affine_translate_y": translate_y,
        "affine_rotation": rotation,
        "clipped_bbox_count": 0,
        "dropped_bbox_count": 0,
    }
    if scale == 1.0 and translate_x == 0.0 and translate_y == 0.0 and rotation == 0.0:
        return image.copy(), tuple(boxes), values
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), rotation, scale)
    matrix[0, 2] += translate_x
    matrix[1, 2] += translate_y
    warped = cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(config.border_value, config.border_value, config.border_value),
    )
    transformed = []
    for box in boxes:
        corners = np.asarray(
            [[box.x_min, box.y_min, 1.0], [box.x_max, box.y_min, 1.0],
             [box.x_max, box.y_max, 1.0], [box.x_min, box.y_max, 1.0]],
            dtype=np.float64,
        )
        points = corners @ matrix.T
        raw_x_min, raw_y_min = points[:, 0].min(), points[:, 1].min()
        raw_x_max, raw_y_max = points[:, 0].max(), points[:, 1].max()
        raw_area = max(0.0, raw_x_max - raw_x_min) * max(0.0, raw_y_max - raw_y_min)
        x_min = min(float(width), max(0.0, float(raw_x_min)))
        y_min = min(float(height), max(0.0, float(raw_y_min)))
        x_max = min(float(width), max(0.0, float(raw_x_max)))
        y_max = min(float(height), max(0.0, float(raw_y_max)))
        clipped_area = max(0.0, x_max - x_min) * max(0.0, y_max - y_min)
        if clipped_area < raw_area - 1e-9:
            values["clipped_bbox_count"] += 1
        visibility = clipped_area / raw_area if raw_area > 0.0 else 0.0
        if visibility < config.min_visibility or x_max - x_min < 1.0 or y_max - y_min < 1.0:
            values["dropped_bbox_count"] += 1
            continue
        transformed.append(type(box)(box.foreground_class_id, x_min, y_min, x_max, y_max))
    values["affine_applied"] = True
    return warped, tuple(transformed), values


def _no_op_result(
    image: np.ndarray,
    boxes: Sequence[Any],
    epoch: int,
    sample_index: int,
    sample_seed: Optional[int],
) -> AugmentationResult:
    return AugmentationResult(
        image=image.copy(),
        boxes=tuple(boxes),
        applied=False,
        metadata=AugmentationMetadata.neutral(
            epoch=epoch,
            sample_index=sample_index,
            sample_seed=sample_seed,
            object_count=len(boxes),
        ),
    )


def _validate_image(image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("augmentation image must have shape [H,W,3]")
    if image.shape[0] <= 0 or image.shape[1] <= 0:
        raise ValueError("augmentation image dimensions must be positive")
    if image.dtype != np.uint8:
        raise ValueError("augmentation image must have dtype uint8")


__all__ = [
    "AugmentationMetadata",
    "AugmentationNotImplementedError",
    "AugmentationPipeline",
    "AugmentationResult",
    "ColorJitterFactors",
    "ColorJitterMetadata",
    "apply_color_jitter",
    "flip_boxes_horizontally",
]
