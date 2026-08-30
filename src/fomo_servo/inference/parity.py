"""Full RGB preprocessing, logits, and detection parity for PyTorch and ORT."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from .ort_predictor import OnnxRuntimePredictor
from .preprocessing import (
    ImagePrediction,
    prediction_from_numpy_logits,
    preprocess_rgb_image,
)


class PipelineParityError(RuntimeError):
    """Raised when inputs or backend outputs cannot be compared safely."""


def compare_rgb_image_pipeline(
    pytorch_model: Any,
    ort_predictor: OnnxRuntimePredictor,
    image: np.ndarray,
    *,
    pytorch_contract: Any = None,
    logits_rtol: float,
    logits_atol: float,
    detection_atol: float,
) -> dict[str, object]:
    """Compare both complete pipelines for one RGB uint8 ``[H,W,3]`` image.

    The report separately covers letterbox/NCHW preprocessing, raw logits
    ``[1,1+N,G,G]``, and all postprocessed detection fields in original pixels.
    """

    for name, value in (
        ("logits_rtol", logits_rtol),
        ("logits_atol", logits_atol),
        ("detection_atol", detection_atol),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
            or value < 0.0
        ):
            raise PipelineParityError("{} must be a finite non-negative number".format(name))
    if not isinstance(ort_predictor, OnnxRuntimePredictor):
        raise PipelineParityError("ort_predictor must be an OnnxRuntimePredictor")
    try:
        import torch
        from fomo_servo.postprocess import postprocess_logits
    except ImportError as error:
        raise PipelineParityError("PyTorch is required for pipeline parity") from error

    contract = ort_predictor.contract
    source_contract = contract if pytorch_contract is None else pytorch_contract
    pytorch_prepared = preprocess_rgb_image(
        image, input_size=tuple(source_contract.input_shape)[-1]
    )
    ort_prepared = preprocess_rgb_image(image, input_size=contract.input_shape[-1])
    preprocessing_passed = bool(
        pytorch_prepared.transform == ort_prepared.transform
        and np.array_equal(
            pytorch_prepared.letterbox_image, ort_prepared.letterbox_image
        )
        and np.array_equal(pytorch_prepared.input_tensor, ort_prepared.input_tensor)
    )

    try:
        with torch.inference_mode():
            pytorch_output = pytorch_model(
                torch.from_numpy(pytorch_prepared.input_tensor)
            )
    except Exception as error:
        raise PipelineParityError("PyTorch inference failed: {}".format(error)) from error
    if not isinstance(pytorch_output, torch.Tensor):
        raise PipelineParityError("PyTorch model must return a torch.Tensor")
    pytorch_logits = pytorch_output.detach().cpu().numpy()
    ort_logits = ort_predictor.predict_logits(ort_prepared.input_tensor)
    if pytorch_logits.shape == ort_logits.shape:
        difference = np.abs(pytorch_logits - ort_logits)
        max_absolute_error = float(difference.max())
        mean_absolute_error = float(difference.mean())
        logits_passed = bool(
            np.allclose(
                pytorch_logits,
                ort_logits,
                rtol=float(logits_rtol),
                atol=float(logits_atol),
            )
        )
    else:
        max_absolute_error = float("inf")
        mean_absolute_error = float("inf")
        logits_passed = False

    pytorch_detections = postprocess_logits(
        pytorch_output,
        class_names=tuple(source_contract.class_names),
        stride=int(source_contract.output_stride),
        transforms=(pytorch_prepared.transform,),
        confidence_threshold=float(source_contract.confidence_threshold),
        class_thresholds=source_contract.class_thresholds,
        component_mode=str(source_contract.component_mode),
        confidence_mode=str(source_contract.confidence_mode),
    )[0]
    pytorch_prediction = ImagePrediction(
        pytorch_prepared.original_image,
        pytorch_prepared.letterbox_image,
        pytorch_prepared.transform,
        pytorch_detections,
    )
    ort_prediction = prediction_from_numpy_logits(
        ort_prepared,
        ort_logits,
        class_names=contract.class_names,
        output_stride=contract.output_stride,
        confidence_threshold=contract.confidence_threshold,
        class_thresholds=contract.class_thresholds,
        component_mode=contract.component_mode,
        confidence_mode=contract.confidence_mode,
    )
    detections_passed, detection_max_error = _compare_detections(
        pytorch_prediction.detections,
        ort_prediction.detections,
        atol=float(detection_atol),
    )
    return {
        "passed": preprocessing_passed and logits_passed and detections_passed,
        "preprocessing": {
            "passed": preprocessing_passed,
            "input_shape": list(pytorch_prepared.input_tensor.shape),
            "input_dtype": str(pytorch_prepared.input_tensor.dtype),
            "letterbox_sha256": _array_sha256(pytorch_prepared.letterbox_image),
            "input_tensor_sha256": _array_sha256(pytorch_prepared.input_tensor),
            "transform": {
                "original_width": pytorch_prepared.transform.original_width,
                "original_height": pytorch_prepared.transform.original_height,
                "input_size": pytorch_prepared.transform.input_size,
                "scale": pytorch_prepared.transform.scale,
                "pad_left": pytorch_prepared.transform.pad_left,
                "pad_top": pytorch_prepared.transform.pad_top,
                "pad_right": pytorch_prepared.transform.pad_right,
                "pad_bottom": pytorch_prepared.transform.pad_bottom,
            },
        },
        "logits": {
            "passed": logits_passed,
            "pytorch_shape": list(pytorch_logits.shape),
            "onnxruntime_shape": list(ort_logits.shape),
            "rtol": float(logits_rtol),
            "atol": float(logits_atol),
            "max_absolute_error": max_absolute_error,
            "mean_absolute_error": mean_absolute_error,
        },
        "detections": {
            "passed": detections_passed,
            "atol": float(detection_atol),
            "max_numeric_error": detection_max_error,
            "pytorch_count": len(pytorch_prediction.detections),
            "onnxruntime_count": len(ort_prediction.detections),
            "pytorch": [item.as_dict() for item in pytorch_prediction.detections],
            "onnxruntime": [item.as_dict() for item in ort_prediction.detections],
        },
    }


def _compare_detections(
    pytorch_detections: tuple[Any, ...],
    ort_detections: tuple[Any, ...],
    *,
    atol: float,
) -> tuple[bool, float]:
    if len(pytorch_detections) != len(ort_detections):
        return False, float("inf")
    numeric_fields = (
        "confidence",
        "mean_confidence",
        "heatmap_x",
        "heatmap_y",
        "input_x",
        "input_y",
        "original_x",
        "original_y",
    )
    maximum_error = 0.0
    for pytorch_detection, ort_detection in zip(
        pytorch_detections, ort_detections
    ):
        if (
            pytorch_detection.class_id != ort_detection.class_id
            or pytorch_detection.class_name != ort_detection.class_name
            or pytorch_detection.component_area_cells
            != ort_detection.component_area_cells
        ):
            return False, float("inf")
        for field in numeric_fields:
            error = abs(
                float(getattr(pytorch_detection, field))
                - float(getattr(ort_detection, field))
            )
            maximum_error = max(maximum_error, error)
            if error > atol:
                return False, maximum_error
    return True, maximum_error


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


__all__ = ["PipelineParityError", "compare_rgb_image_pipeline"]
