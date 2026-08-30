"""Shared checkpoint loading and single-image FOMO inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from fomo_servo.config import ProjectConfig

from .preprocessing import (
    ImagePrediction,
    PreprocessingError,
    preprocess_rgb_image,
)


class InferenceError(RuntimeError):
    """Raised when checkpoint, image, device, or inference contracts are invalid."""


def load_inference_model(
    config: ProjectConfig, checkpoint: Path, device_request: Any
) -> tuple[Any, Any]:
    """Load an existing FOMO checkpoint without changing model architecture or weights."""

    import torch
    from fomo_servo.models import build_fomo_model
    from fomo_servo.runtime import resolve_device

    device = resolve_device(device_request)
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_file():
        raise InferenceError("checkpoint does not exist: {}".format(checkpoint_path))
    model = build_fomo_model(config).to(device)
    try:
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except (OSError, RuntimeError, ValueError) as error:
        raise InferenceError("unable to load checkpoint '{}': {}".format(checkpoint_path, error)) from error
    if isinstance(payload, dict) and "model_state" in payload:
        state_dict = payload["model_state"]
    elif isinstance(payload, dict):
        state_dict = payload
    else:
        raise InferenceError("checkpoint must contain a model_state mapping")
    try:
        model.load_state_dict(state_dict)
    except (RuntimeError, TypeError, ValueError) as error:
        raise InferenceError("checkpoint model state is incompatible with config: {}".format(error)) from error
    model.eval()
    return model, device


def predict_rgb_image(
    model: Any,
    image: np.ndarray,
    *,
    config: ProjectConfig,
    device: Any,
    confidence_threshold: float | None = None,
) -> ImagePrediction:
    """Letterbox RGB ``uint8 [H,W,3]`` and return detections in original pixels."""

    import torch
    from fomo_servo.postprocess import postprocess_logits

    try:
        prepared = preprocess_rgb_image(image, input_size=config.model.input_size)
    except PreprocessingError as error:
        raise InferenceError("invalid input image: {}".format(error)) from error
    tensor = torch.from_numpy(prepared.input_tensor).to(device)
    with torch.no_grad():
        logits = model(tensor)
    threshold = (
        config.postprocess.inference_threshold
        if confidence_threshold is None
        else confidence_threshold
    )
    detections = postprocess_logits(
        logits,
        class_names=config.dataset.class_names,
        stride=config.model.output_stride,
        transforms=(prepared.transform,),
        confidence_threshold=threshold,
        class_thresholds=config.postprocess.class_thresholds,
        component_mode=config.postprocess.component_mode,
        confidence_mode=config.postprocess.confidence_mode,
    )[0]
    return ImagePrediction(
        prepared.original_image,
        prepared.letterbox_image,
        prepared.transform,
        detections,
    )


def read_rgb_image(path: Path) -> np.ndarray:
    """Read a filesystem image as RGB ``uint8 [H,W,3]``."""

    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise InferenceError("unable to read image: {}".format(path))
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
