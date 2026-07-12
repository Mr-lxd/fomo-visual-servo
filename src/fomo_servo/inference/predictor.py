"""Shared checkpoint loading and single-image FOMO inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch

from fomo_servo.config import ProjectConfig
from fomo_servo.geometry import LetterboxTransform, letterbox_rgb
from fomo_servo.models import FOMONet, build_fomo_model
from fomo_servo.postprocess import Detection, postprocess_logits
from fomo_servo.runtime import DeviceRequest, resolve_device


class InferenceError(RuntimeError):
    """Raised when checkpoint, image, device, or inference contracts are invalid."""


@dataclass(frozen=True)
class ImagePrediction:
    """One image's letterbox metadata and postprocessed detections."""

    original_image: np.ndarray
    letterbox_image: np.ndarray
    transform: LetterboxTransform
    detections: tuple[Detection, ...]


def load_inference_model(
    config: ProjectConfig, checkpoint: Path, device_request: DeviceRequest
) -> tuple[FOMONet, torch.device]:
    """Load an existing FOMO checkpoint without changing model architecture or weights."""

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
    model: FOMONet,
    image: np.ndarray,
    *,
    config: ProjectConfig,
    device: torch.device,
    confidence_threshold: float | None = None,
) -> ImagePrediction:
    """Letterbox RGB ``uint8 [H,W,3]`` and return detections in original pixels."""

    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise InferenceError("image must have RGB shape [H,W,3]")
    letterbox_image, transform = letterbox_rgb(image, config.model.input_size)
    normalized = np.ascontiguousarray(letterbox_image.transpose(2, 0, 1), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(normalized).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
    threshold = (
        config.postprocess.confidence_threshold
        if confidence_threshold is None
        else confidence_threshold
    )
    detections = postprocess_logits(
        logits,
        class_names=config.dataset.class_names,
        stride=config.model.output_stride,
        transforms=(transform,),
        confidence_threshold=threshold,
        class_thresholds=config.postprocess.class_thresholds,
        component_mode=config.postprocess.component_mode,
        confidence_mode=config.postprocess.confidence_mode,
    )[0]
    return ImagePrediction(image, letterbox_image, transform, detections)


def read_rgb_image(path: Path) -> np.ndarray:
    """Read a filesystem image as RGB ``uint8 [H,W,3]``."""

    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise InferenceError("unable to read image: {}".format(path))
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
