"""Explicit TensorFlow Lite / LiteRT tensor inspection and invocation helpers."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np


class TFLiteRuntimeError(RuntimeError):
    """Raised when no supported interpreter or tensor contract is available."""


def create_tflite_interpreter(model_path: Path) -> tuple[Any, str]:
    """Create an interpreter for one local ``.tflite`` file.

    Backends are tried in the explicitly documented order: TensorFlow Lite,
    standalone ``tflite_runtime``, then the smaller official LiteRT package.
    No backend error is suppressed: an aggregate diagnostic is raised if all
    imports or initializations fail.
    """

    path = Path(model_path)
    if not path.is_file():
        raise TFLiteRuntimeError("TFLite model does not exist: {}".format(path))
    candidates = (
        ("tensorflow.lite", "Interpreter", "tensorflow.lite.Interpreter"),
        ("tflite_runtime.interpreter", "Interpreter", "tflite_runtime.interpreter.Interpreter"),
        ("ai_edge_litert.interpreter", "Interpreter", "ai_edge_litert.interpreter.Interpreter"),
    )
    failures = []
    for module_name, class_name, backend_name in candidates:
        try:
            module = importlib.import_module(module_name)
            interpreter_class = getattr(module, class_name)
            return interpreter_class(model_path=str(path)), backend_name
        except (ImportError, AttributeError, OSError, RuntimeError, ValueError) as error:
            failures.append("{}: {}".format(backend_name, error))
    raise TFLiteRuntimeError(
        "no supported TFLite interpreter is available; install the project 'tflite' "
        "extra or TensorFlow. Details: {}".format(" | ".join(failures))
    )


def inspect_tflite_interpreter(interpreter: Any, *, model_sha256: str) -> dict[str, object]:
    """Allocate tensors and serialize one-input/one-output TFLite metadata."""

    if not isinstance(model_sha256, str) or len(model_sha256) != 64:
        raise TFLiteRuntimeError("model_sha256 must be a 64-character SHA-256 string")
    try:
        interpreter.allocate_tensors()
        inputs = interpreter.get_input_details()
        outputs = interpreter.get_output_details()
    except (AttributeError, RuntimeError, ValueError) as error:
        raise TFLiteRuntimeError("unable to allocate or inspect TFLite tensors: {}".format(error)) from error
    if not isinstance(inputs, list) or len(inputs) != 1:
        raise TFLiteRuntimeError("exactly one TFLite input tensor is required")
    if not isinstance(outputs, list) or len(outputs) != 1:
        raise TFLiteRuntimeError("exactly one TFLite output tensor is required")
    operators = _operator_names(interpreter)
    return {
        "model_sha256": model_sha256.lower(),
        "input": _tensor_detail_dict(inputs[0]),
        "output": _tensor_detail_dict(outputs[0]),
        "operators": operators,
        "graph_has_softmax": "SOFTMAX" in operators,
    }


def prepare_tflite_input(
    image_rgb: np.ndarray,
    input_detail: Mapping[str, Any],
    *,
    float_scale: float = 1.0,
) -> np.ndarray:
    """Prepare an RGB ``uint8 [H,W,3]`` image for one inspected input tensor.

    Float input tensors receive RGB values multiplied by explicit
    ``float_scale`` (default raw ``[0,255]``). Callers must select a non-default
    scale only from recorded model/DSP evidence. Integer tensors use the actual
    tensor quantization scale and zero point and fail if they are absent.
    """

    if not isinstance(image_rgb, np.ndarray) or image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise TFLiteRuntimeError("TFLite preprocessing expects RGB image shape [H,W,3]")
    if image_rgb.dtype != np.uint8:
        raise TFLiteRuntimeError("TFLite preprocessing expects uint8 RGB pixels")
    shape = np.asarray(input_detail.get("shape"), dtype=np.int64).tolist()
    if len(shape) != 4 or shape[0] != 1 or shape[3] != 3 or shape[1:3] != list(image_rgb.shape[:2]):
        raise TFLiteRuntimeError(
            "TFLite input shape {} is incompatible with preprocessed RGB shape {}".format(
                shape, list(image_rgb.shape)
            )
        )
    try:
        dtype = np.dtype(input_detail.get("dtype"))
    except TypeError as error:
        raise TFLiteRuntimeError("TFLite input detail has no valid dtype") from error
    if np.issubdtype(dtype, np.floating):
        if not isinstance(float_scale, (int, float)) or isinstance(float_scale, bool) or not np.isfinite(float_scale) or float_scale <= 0.0:
            raise TFLiteRuntimeError("float_scale must be a finite positive number")
        return (image_rgb.astype(dtype, copy=False) * float(float_scale))[np.newaxis, ...]
    if not np.issubdtype(dtype, np.integer):
        raise TFLiteRuntimeError("unsupported TFLite input dtype: {}".format(dtype.name))
    quantization = input_detail.get("quantization")
    if not isinstance(quantization, tuple) or len(quantization) != 2:
        raise TFLiteRuntimeError("integer TFLite input requires (scale, zero_point) quantization")
    scale, zero_point = quantization
    if not isinstance(scale, (int, float)) or scale <= 0.0 or not isinstance(zero_point, (int, np.integer)):
        raise TFLiteRuntimeError("integer TFLite input has invalid quantization parameters")
    info = np.iinfo(dtype)
    quantized = np.rint(image_rgb.astype(np.float32) / float(scale) + int(zero_point))
    return np.clip(quantized, info.min, info.max).astype(dtype)[np.newaxis, ...]


def output_looks_like_probabilities(output: np.ndarray, *, atol: float = 1e-4) -> bool:
    """Return whether an NHWC foreground/background tensor is softmax-like."""

    values = np.asarray(output)
    if values.ndim != 4 or values.shape[0] != 1 or values.shape[-1] < 2:
        return False
    if not np.issubdtype(values.dtype, np.floating) or not np.isfinite(values).all():
        return False
    if bool((values < -atol).any()) or bool((values > 1.0 + atol).any()):
        return False
    return bool(np.allclose(values.sum(axis=-1), 1.0, atol=atol, rtol=atol))


def _tensor_detail_dict(detail: Mapping[str, Any]) -> dict[str, object]:
    try:
        dtype = np.dtype(detail["dtype"])
    except (KeyError, TypeError) as error:
        raise TFLiteRuntimeError("TFLite tensor detail has no valid dtype") from error
    shape = np.asarray(detail.get("shape"), dtype=np.int64).tolist()
    shape_signature = np.asarray(detail.get("shape_signature", shape), dtype=np.int64).tolist()
    quantization = detail.get("quantization", (0.0, 0))
    if not isinstance(quantization, tuple) or len(quantization) != 2:
        raise TFLiteRuntimeError("TFLite tensor quantization detail is invalid")
    raw_parameters = detail.get("quantization_parameters", {})
    if not isinstance(raw_parameters, Mapping):
        raise TFLiteRuntimeError("TFLite tensor quantization_parameters must be a mapping")
    return {
        "name": str(detail.get("name", "")),
        "index": int(detail.get("index", -1)),
        "shape": [int(value) for value in shape],
        "shape_signature": [int(value) for value in shape_signature],
        "dtype": dtype.name,
        "quantization": {"scale": float(quantization[0]), "zero_point": int(quantization[1])},
        "quantization_parameters": {
            "scales": [float(value) for value in np.asarray(raw_parameters.get("scales", ())).reshape(-1)],
            "zero_points": [int(value) for value in np.asarray(raw_parameters.get("zero_points", ())).reshape(-1)],
            "quantized_dimension": int(raw_parameters.get("quantized_dimension", 0)),
        },
    }


def _operator_names(interpreter: Any) -> list[str]:
    """Read the available interpreter operator details, failing visibly if absent."""

    operation_getter = getattr(interpreter, "_get_ops_details", None)
    if not callable(operation_getter):
        raise TFLiteRuntimeError("interpreter does not expose operator details")
    try:
        details = operation_getter()
    except (AttributeError, RuntimeError, ValueError) as error:
        raise TFLiteRuntimeError("unable to inspect TFLite operators: {}".format(error)) from error
    if not isinstance(details, list):
        raise TFLiteRuntimeError("TFLite operator details must be a list")
    names = []
    for detail in details:
        if not isinstance(detail, Mapping) or not isinstance(detail.get("op_name"), str):
            raise TFLiteRuntimeError("TFLite operator detail has no op_name")
        names.append(detail["op_name"])
    return names


__all__ = [
    "TFLiteRuntimeError",
    "create_tflite_interpreter",
    "inspect_tflite_interpreter",
    "output_looks_like_probabilities",
    "prepare_tflite_input",
]
