"""Stateless logits-to-centroid FOMO postprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from types import ModuleType
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from fomo_servo.geometry import LetterboxTransform

from .connected_components import find_connected_components


class PostprocessError(ValueError):
    """Raised when logits, thresholds, geometry, or postprocess settings are invalid."""


def _require_torch() -> ModuleType:
    """Import torch lazily for the torch-tensor-only entry points.

    NumPy and ORT-only deployments import this module without ever probing for
    torch; only callers of :func:`postprocess_logits` and
    :func:`postprocess_probabilities` pay for the torch import.
    """

    try:
        import torch
    except ModuleNotFoundError as error:
        raise PostprocessError(
            "PyTorch is required for torch-tensor postprocessing; "
            "use the NumPy entry points on ORT-only deployments"
        ) from error
    return torch


ThresholdMapping = Mapping[Union[int, str], float]
ThresholdSpec = Union[Sequence[float], ThresholdMapping]


@dataclass(frozen=True)
class Detection:
    """One FOMO component detection in heatmap, input, and original coordinates.

    ``heatmap_x/y`` are continuous grid coordinates. ``input_x/y`` are letterbox
    pixels. ``original_x/y`` are clipped original-image pixels. ``confidence`` is
    the component maximum by default; ``mean_confidence`` is retained for diagnosis.
    """

    class_id: int
    class_name: str
    confidence: float
    mean_confidence: float
    component_area_cells: int
    heatmap_x: float
    heatmap_y: float
    input_x: float
    input_y: float
    original_x: float
    original_y: float

    def as_dict(self) -> dict[str, Any]:
        """Serialize the detection using the public JSON field names."""

        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "mean_confidence": self.mean_confidence,
            "component_area_cells": self.component_area_cells,
            "heatmap_x": self.heatmap_x,
            "heatmap_y": self.heatmap_y,
            "input_x": self.input_x,
            "input_y": self.input_y,
            "original_x": self.original_x,
            "original_y": self.original_y,
        }


def postprocess_logits(
    logits: Any,
    *,
    class_names: Sequence[str],
    stride: int,
    transforms: Sequence[LetterboxTransform],
    confidence_threshold: float,
    class_thresholds: Optional[ThresholdSpec] = None,
    component_mode: str = "connected_components",
    confidence_mode: str = "max",
) -> Tuple[Tuple[Detection, ...], ...]:
    """Convert logits ``[B,1+N,G,G]`` into per-image centroid detections.

    Softmax is performed on a detached float32 tensor. Background channel zero is
    never emitted. Components use 8-neighbor connectivity and probability-weighted
    grid centers ``(grid_x+0.5, grid_y+0.5)``.
    """

    torch = _require_torch()
    if not isinstance(logits, torch.Tensor) or logits.ndim != 4:
        raise PostprocessError("logits must have shape [B,1+N,G,G]")
    if not logits.is_floating_point():
        raise PostprocessError("logits must have a floating-point dtype")
    detached = logits.detach().float()
    if not bool(torch.isfinite(detached).all().item()):
        raise PostprocessError("logits contain NaN or Inf")
    probabilities = torch.softmax(detached, dim=1)
    return postprocess_numpy_probabilities(
        probabilities.cpu().numpy(),
        class_names=class_names,
        stride=stride,
        transforms=transforms,
        confidence_threshold=confidence_threshold,
        class_thresholds=class_thresholds,
        component_mode=component_mode,
        confidence_mode=confidence_mode,
    )


def postprocess_numpy_logits(
    logits: np.ndarray,
    *,
    class_names: Sequence[str],
    stride: int,
    transforms: Sequence[LetterboxTransform],
    confidence_threshold: float,
    class_thresholds: Optional[ThresholdSpec] = None,
    component_mode: str = "connected_components",
    confidence_mode: str = "max",
) -> Tuple[Tuple[Detection, ...], ...]:
    """Convert NumPy raw logits ``[B,1+N,G,G]`` into centroid detections.

    Values are evaluated as float32. A numerically stable channel softmax is
    applied before the shared NumPy probability/connected-component pipeline.
    """

    if not isinstance(logits, np.ndarray) or logits.ndim != 4:
        raise PostprocessError("logits must have shape [B,1+N,G,G]")
    if not np.issubdtype(logits.dtype, np.floating):
        raise PostprocessError("logits must have a floating-point dtype")
    values = logits.astype(np.float32, copy=False)
    if not bool(np.isfinite(values).all()):
        raise PostprocessError("logits contain NaN or Inf")
    shifted = values - values.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    probabilities = exponentials / exponentials.sum(axis=1, keepdims=True)
    return postprocess_numpy_probabilities(
        probabilities,
        class_names=class_names,
        stride=stride,
        transforms=transforms,
        confidence_threshold=confidence_threshold,
        class_thresholds=class_thresholds,
        component_mode=component_mode,
        confidence_mode=confidence_mode,
    )


def postprocess_probabilities(
    probabilities: Any,
    *,
    class_names: Sequence[str],
    stride: int,
    transforms: Sequence[LetterboxTransform],
    confidence_threshold: float,
    class_thresholds: Optional[ThresholdSpec] = None,
    component_mode: str = "connected_components",
    confidence_mode: str = "max",
) -> Tuple[Tuple[Detection, ...], ...]:
    """Convert FOMO probabilities ``[B,1+N,G,G]`` into centroid detections.

    This is the no-extra-softmax entry point for deployable models whose graph
    already emits softmax probabilities, such as the inspected EI TFLite model.
    Channel zero is background. Per-cell channel sums must be one within
    ``1e-4`` and every value must be finite and non-negative.
    """

    torch = _require_torch()
    if not isinstance(probabilities, torch.Tensor) or probabilities.ndim != 4:
        raise PostprocessError("probabilities must have shape [B,1+N,G,G]")
    if not probabilities.is_floating_point():
        raise PostprocessError("probabilities must have a floating-point dtype")
    detached = probabilities.detach().float()
    if not bool(torch.isfinite(detached).all().item()):
        raise PostprocessError("probabilities contain NaN or Inf")
    return postprocess_numpy_probabilities(
        detached.cpu().numpy(),
        class_names=class_names,
        stride=stride,
        transforms=transforms,
        confidence_threshold=confidence_threshold,
        class_thresholds=class_thresholds,
        component_mode=component_mode,
        confidence_mode=confidence_mode,
    )


def postprocess_numpy_probabilities(
    probabilities: np.ndarray,
    *,
    class_names: Sequence[str],
    stride: int,
    transforms: Sequence[LetterboxTransform],
    confidence_threshold: float,
    class_thresholds: Optional[ThresholdSpec] = None,
    component_mode: str = "connected_components",
    confidence_mode: str = "max",
) -> Tuple[Tuple[Detection, ...], ...]:
    """Convert NumPy probabilities ``[B,1+N,G,G]`` into centroid detections."""

    if not isinstance(probabilities, np.ndarray) or probabilities.ndim != 4:
        raise PostprocessError("probabilities must have shape [B,1+N,G,G]")
    if not np.issubdtype(probabilities.dtype, np.floating):
        raise PostprocessError("probabilities must have a floating-point dtype")
    if not isinstance(stride, int) or isinstance(stride, bool) or stride <= 0:
        raise PostprocessError("stride must be a positive integer")
    names = _validate_class_names(class_names)
    batch_size, channels, grid_height, grid_width = probabilities.shape
    if channels != len(names) + 1:
        raise PostprocessError("probability channel count must equal 1 + len(class_names)")
    if grid_height <= 0 or grid_width <= 0 or grid_height != grid_width:
        raise PostprocessError("probability heatmap must have positive square shape [G,G]")
    if len(transforms) != batch_size:
        raise PostprocessError("one LetterboxTransform is required for every batch image")
    thresholds = _resolve_thresholds(confidence_threshold, class_thresholds, names)
    if component_mode == "local_peaks":
        raise PostprocessError("local_peaks mode is reserved for a future extension")
    if component_mode != "connected_components":
        raise PostprocessError(
            "component_mode must be 'connected_components' or 'local_peaks'"
        )
    if confidence_mode not in {"max", "mean"}:
        raise PostprocessError("confidence_mode must be 'max' or 'mean'")

    probability_array = probabilities.astype(np.float32, copy=False)
    if not bool(np.isfinite(probability_array).all()):
        raise PostprocessError("probabilities contain NaN or Inf")
    if bool((probability_array < 0.0).any()):
        raise PostprocessError("probabilities must be non-negative")
    if not bool(
        np.allclose(
            probability_array.sum(axis=1),
            np.ones_like(probability_array[:, 0]),
            atol=1e-4,
            rtol=1e-4,
        )
    ):
        raise PostprocessError("probabilities must sum to one across classes per heatmap cell")
    output = []
    for batch_index in range(batch_size):
        transform = transforms[batch_index]
        expected_input_size = grid_height * stride
        if (
            transform.input_size != expected_input_size
            or grid_width * stride != transform.input_size
        ):
            raise PostprocessError(
                "LetterboxTransform input_size must equal heatmap grid size times stride"
            )
        image_detections = []
        for class_id, threshold in enumerate(thresholds):
            probability_map = probability_array[batch_index, class_id + 1]
            mask = probability_map >= threshold
            for component in find_connected_components(mask, connectivity=8):
                weights = np.asarray(
                    [probability_map[grid_y, grid_x] for grid_x, grid_y in component.cells],
                    dtype=np.float64,
                )
                weight_sum = float(weights.sum())
                if not isfinite(weight_sum) or weight_sum <= 0.0:
                    raise PostprocessError("component probability weights must sum to a positive finite value")
                grid_x_values = np.asarray(
                    [grid_x + 0.5 for grid_x, _ in component.cells], dtype=np.float64
                )
                grid_y_values = np.asarray(
                    [grid_y + 0.5 for _, grid_y in component.cells], dtype=np.float64
                )
                heatmap_x = float(np.dot(weights, grid_x_values) / weight_sum)
                heatmap_y = float(np.dot(weights, grid_y_values) / weight_sum)
                input_x = heatmap_x * stride
                input_y = heatmap_y * stride
                original_x, original_y = transform.inverse_point(input_x, input_y)
                original_x = float(np.clip(original_x, 0.0, transform.original_width - 1.0))
                original_y = float(np.clip(original_y, 0.0, transform.original_height - 1.0))
                maximum = float(weights.max())
                mean = float(weights.mean())
                confidence = maximum if confidence_mode == "max" else mean
                image_detections.append(
                    Detection(
                        class_id=class_id,
                        class_name=names[class_id],
                        confidence=confidence,
                        mean_confidence=mean,
                        component_area_cells=component.area,
                        heatmap_x=heatmap_x,
                        heatmap_y=heatmap_y,
                        input_x=float(input_x),
                        input_y=float(input_y),
                        original_x=original_x,
                        original_y=original_y,
                    )
                )
        output.append(tuple(image_detections))
    return tuple(output)


def _validate_class_names(class_names: Sequence[str]) -> Tuple[str, ...]:
    if not isinstance(class_names, Sequence) or isinstance(class_names, (str, bytes)):
        raise PostprocessError("class_names must be a sequence of strings")
    names = tuple(class_names)
    if not names or any(not isinstance(name, str) or not name.strip() for name in names):
        raise PostprocessError("class_names must contain non-empty strings")
    if len(set(names)) != len(names):
        raise PostprocessError("class_names must not contain duplicates")
    return names


def _validate_threshold(value: float, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PostprocessError("{} must be a finite probability in [0,1]".format(label))
    value = float(value)
    if not isfinite(value) or not 0.0 <= value <= 1.0:
        raise PostprocessError("{} must be a finite probability in [0,1]".format(label))
    return value


def _resolve_thresholds(
    global_threshold: float,
    class_thresholds: Optional[ThresholdSpec],
    class_names: Tuple[str, ...],
) -> Tuple[float, ...]:
    thresholds = [_validate_threshold(global_threshold, "confidence_threshold") for _ in class_names]
    if class_thresholds is None:
        return tuple(thresholds)
    if isinstance(class_thresholds, Mapping):
        for key, value in class_thresholds.items():
            if isinstance(key, bool) or not isinstance(key, (int, str)):
                raise PostprocessError("class threshold keys must be class IDs or names")
            if isinstance(key, int):
                class_id = key
            else:
                if key not in class_names:
                    raise PostprocessError("unknown class threshold name '{}'".format(key))
                class_id = class_names.index(key)
            if not 0 <= class_id < len(class_names):
                raise PostprocessError("class threshold ID is outside class_names")
            thresholds[class_id] = _validate_threshold(value, "class_thresholds[{}]".format(key))
        return tuple(thresholds)
    if isinstance(class_thresholds, (str, bytes)) or not isinstance(class_thresholds, Sequence):
        raise PostprocessError("class_thresholds must be a sequence or mapping")
    if len(class_thresholds) != len(class_names):
        raise PostprocessError("class_thresholds length must equal len(class_names)")
    return tuple(
        _validate_threshold(value, "class_thresholds[{}]".format(index))
        for index, value in enumerate(class_thresholds)
    )
