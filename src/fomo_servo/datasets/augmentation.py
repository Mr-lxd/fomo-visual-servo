"""Train-only augmentation pipeline with the aug01 RGB color-jitter operation."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Any, Optional, Sequence

import cv2
import numpy as np

from fomo_servo.config import AugmentationConfig


class AugmentationNotImplementedError(NotImplementedError):
    """Raised when an augmentation outside the current experiment is enabled."""


@dataclass(frozen=True)
class ColorJitterFactors:
    """One sampled color transform for an RGB ``uint8 [H,W,3]`` image."""

    brightness_factor: float = 1.0
    contrast_factor: float = 1.0
    saturation_factor: float = 1.0
    hue_shift: float = 0.0

    @classmethod
    def neutral(cls) -> "ColorJitterFactors":
        """Return factors that must produce an elementwise image copy."""

        return cls()

    def validate(self) -> None:
        """Validate multiplicative factors and normalized hue shift."""

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
class ColorJitterMetadata:
    """Audit metadata for one color-jitter decision and its sampled factors."""

    applied: bool = False
    brightness_factor: float = 1.0
    contrast_factor: float = 1.0
    saturation_factor: float = 1.0
    hue_shift: float = 0.0

    @classmethod
    def neutral(cls) -> "ColorJitterMetadata":
        """Return metadata for a skipped or mathematically neutral operation."""

        return cls()


@dataclass(frozen=True)
class AugmentationResult:
    """Image/bbox result before letterbox.

    ``image`` is RGB ``uint8 [H,W,3]`` and ``boxes`` are unchanged pixel-space
    bbox objects. ``metadata`` records color-jitter sampling without entering
    the training collate tensors.
    """

    image: np.ndarray
    boxes: tuple[Any, ...]
    applied: bool
    metadata: ColorJitterMetadata = field(default_factory=ColorJitterMetadata.neutral)


def apply_color_jitter(
    image: np.ndarray, factors: ColorJitterFactors
) -> np.ndarray:
    """Apply fixed color factors to an RGB ``uint8 [H,W,3]`` image.

    Brightness and contrast operate in float RGB. Saturation and normalized hue
    operate in explicit OpenCV RGB/HSV space, where ``hue_shift=1`` is one full
    hue cycle. The returned array has the input shape and dtype with pixels
    clipped to ``[0,255]`` before conversion back to ``uint8``.
    """

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
        rgb = np.clip(
            (rgb - mean) * float(factors.contrast_factor) + mean,
            0.0,
            1.0,
        )

    if factors.saturation_factor != 1.0 or factors.hue_shift != 0.0:
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        hsv[..., 1] = np.clip(
            hsv[..., 1] * float(factors.saturation_factor), 0.0, 1.0
        )
        hsv[..., 0] = np.mod(
            hsv[..., 0] + float(factors.hue_shift) * 360.0,
            360.0,
        )
        rgb = np.clip(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB), 0.0, 1.0)

    return np.rint(rgb * 255.0).astype(image.dtype, copy=False)


class AugmentationPipeline:
    """Apply aug01 color jitter only on train samples before letterbox."""

    def __init__(self, config: AugmentationConfig, *, is_train: bool) -> None:
        if not isinstance(config, AugmentationConfig):
            raise TypeError("config must be an AugmentationConfig instance")
        if not isinstance(is_train, bool):
            raise TypeError("is_train must be a boolean")
        self.config = config
        self.is_train = is_train

    @classmethod
    def disabled(cls) -> "AugmentationPipeline":
        """Return a non-training no-op pipeline for validation/test callers."""

        return cls(AugmentationConfig.disabled(), is_train=False)

    def apply(
        self,
        image: np.ndarray,
        boxes: Sequence[Any],
        rng: Optional[np.random.Generator] = None,
    ) -> AugmentationResult:
        """Apply configured color jitter to RGB image/bboxes before letterbox.

        Args:
            image: Original RGB ``uint8 [H,W,3]`` image.
            boxes: Original-image pixel-coordinate bbox objects, unchanged here.
            rng: Explicit ``numpy.random.Generator`` used for probability and
                factor sampling; no global RNG is consulted.

        Returns:
            ``AugmentationResult`` with RGB ``uint8 [H,W,3]`` and unchanged boxes.

        Raises:
            AugmentationNotImplementedError: If a non-color augmentation is enabled.
        """

        _validate_image(image)
        if not isinstance(boxes, Sequence):
            raise TypeError("augmentation boxes must be a sequence")
        if not self.is_train or not self.config.enabled:
            return _no_op_result(image, boxes)
        if _non_color_operation_enabled(self.config):
            raise AugmentationNotImplementedError(
                "only color_jitter is implemented in aug01_color"
            )
        color_config = self.config.color_jitter
        if not color_config.enabled or _color_parameters_are_neutral(color_config):
            return _no_op_result(image, boxes)
        if color_config.probability <= 0.0:
            return _no_op_result(image, boxes)
        if not isinstance(rng, np.random.Generator):
            raise TypeError(
                "an explicit numpy.random.Generator is required for color jitter"
            )

        if float(rng.random()) >= color_config.probability:
            return _no_op_result(image, boxes)
        factors = ColorJitterFactors(
            brightness_factor=float(
                rng.uniform(1.0 - color_config.brightness, 1.0 + color_config.brightness)
            ),
            contrast_factor=float(
                rng.uniform(1.0 - color_config.contrast, 1.0 + color_config.contrast)
            ),
            saturation_factor=float(
                rng.uniform(1.0 - color_config.saturation, 1.0 + color_config.saturation)
            ),
            hue_shift=float(rng.uniform(-color_config.hue, color_config.hue)),
        )
        jittered = apply_color_jitter(image, factors)
        metadata = ColorJitterMetadata(
            applied=True,
            brightness_factor=factors.brightness_factor,
            contrast_factor=factors.contrast_factor,
            saturation_factor=factors.saturation_factor,
            hue_shift=factors.hue_shift,
        )
        return AugmentationResult(
            image=jittered,
            boxes=tuple(boxes),
            applied=True,
            metadata=metadata,
        )


def _no_op_result(image: np.ndarray, boxes: Sequence[Any]) -> AugmentationResult:
    return AugmentationResult(
        image=image.copy(),
        boxes=tuple(boxes),
        applied=False,
        metadata=ColorJitterMetadata.neutral(),
    )


def _color_parameters_are_neutral(config: Any) -> bool:
    return (
        config.brightness == 0.0
        and config.contrast == 0.0
        and config.saturation == 0.0
        and config.hue == 0.0
    )


def _non_color_operation_enabled(config: AugmentationConfig) -> bool:
    return any(
        getattr(config, name).enabled
        for name in ("horizontal_flip", "gaussian_blur", "gaussian_noise", "affine")
    )


def _validate_image(image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("augmentation image must have shape [H,W,3]")
    if image.shape[0] <= 0 or image.shape[1] <= 0:
        raise ValueError("augmentation image dimensions must be positive")
    if image.dtype != np.uint8:
        raise ValueError("augmentation image must have dtype uint8")
