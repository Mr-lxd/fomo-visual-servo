"""Shared RGB preprocessing and NumPy-logit prediction assembly."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import numpy as np

from fomo_servo.geometry import LetterboxTransform, letterbox_rgb
from fomo_servo.postprocess import Detection, postprocess_numpy_logits


class PreprocessingError(ValueError):
    """Raised when an RGB image or preprocessed tensor violates the fixed contract."""


@dataclass(frozen=True)
class PreparedImage:
    """One RGB image and its fixed NCHW FP32 letterbox representation.

    ``original_image`` and ``letterbox_image`` are RGB uint8 ``[H,W,3]``.
    ``input_tensor`` is contiguous normalized RGB float32 ``[1,3,S,S]``.
    """

    original_image: np.ndarray
    letterbox_image: np.ndarray
    transform: LetterboxTransform
    input_tensor: np.ndarray


@dataclass(frozen=True)
class ImagePrediction:
    """One image's shared letterbox metadata and original-coordinate detections."""

    original_image: np.ndarray
    letterbox_image: np.ndarray
    transform: LetterboxTransform
    detections: tuple[Detection, ...]


def preprocess_rgb_image(image: np.ndarray, *, input_size: int) -> PreparedImage:
    """Letterbox RGB uint8 ``[H,W,3]`` into NCHW RGB FP32 ``[1,3,S,S]``.

    Pixel values are divided by 255 into ``[0,1]``. The existing reversible
    :func:`letterbox_rgb` implementation supplies resize and padding semantics.
    """

    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise PreprocessingError("image must have RGB shape [H,W,3]")
    if image.dtype != np.uint8:
        raise PreprocessingError("image must have dtype uint8")
    letterbox_image, transform = letterbox_rgb(image, input_size)
    channels_first = letterbox_image.transpose(2, 0, 1)
    input_tensor = np.ascontiguousarray(
        channels_first[np.newaxis, ...], dtype=np.float32
    )
    input_tensor /= np.float32(255.0)
    return PreparedImage(image, letterbox_image, transform, input_tensor)


def prediction_from_numpy_logits(
    prepared: PreparedImage,
    logits: np.ndarray,
    *,
    class_names: Sequence[str],
    output_stride: int,
    confidence_threshold: float,
    class_thresholds: Optional[Sequence[float] | Mapping[int | str, float]],
    component_mode: str,
    confidence_mode: str,
) -> ImagePrediction:
    """Postprocess raw NumPy logits ``[1,1+N,G,G]`` into original pixels."""

    if not isinstance(prepared, PreparedImage):
        raise PreprocessingError("prepared must be a PreparedImage")
    detections = postprocess_numpy_logits(
        logits,
        class_names=class_names,
        stride=output_stride,
        transforms=(prepared.transform,),
        confidence_threshold=confidence_threshold,
        class_thresholds=class_thresholds,
        component_mode=component_mode,
        confidence_mode=confidence_mode,
    )[0]
    return ImagePrediction(
        prepared.original_image,
        prepared.letterbox_image,
        prepared.transform,
        detections,
    )


__all__ = [
    "ImagePrediction",
    "PreparedImage",
    "PreprocessingError",
    "prediction_from_numpy_logits",
    "preprocess_rgb_image",
]
