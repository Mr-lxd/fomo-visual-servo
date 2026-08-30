"""Torch-free ONNX Runtime predictor using the exported artifact sidecar."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from .preprocessing import (
    ImagePrediction,
    prediction_from_numpy_logits,
    preprocess_rgb_image,
)


class OrtPredictorError(RuntimeError):
    """Raised when an ONNX artifact, sidecar contract, or inference call is invalid."""


@dataclass(frozen=True)
class OrtModelContract:
    """Validated fixed-shape runtime and postprocess contract from export JSON."""

    artifact_name: str
    source_experiment_config: str
    source_experiment_config_sha256: str
    export_config_file: str
    export_config_sha256: str
    checkpoint_file: str
    checkpoint_sha256: str
    checkpoint_epoch: int
    checkpoint_seed: int
    parameter_count: int
    config_fingerprint: str
    validation_threshold: float
    validation_threshold_usage: str
    onnx_file: str
    onnx_sha256: str
    onnx_size_bytes: int
    onnx_checker: str
    opset: int
    input_name: str
    input_shape: tuple[int, int, int, int]
    input_dtype: str
    input_color_order: str
    input_value_range: tuple[float, float]
    output_name: str
    output_shape: tuple[int, int, int, int]
    output_dtype: str
    output_semantic: str
    output_stride: int
    class_names: tuple[str, ...]
    confidence_threshold: float
    class_thresholds: Optional[Sequence[float] | Mapping[int | str, float]]
    component_mode: str
    confidence_mode: str
    selection_strategy: str
    max_match_distance_pixels: float
    max_lost_frames: int
    allowed_class_ids: Optional[tuple[int, ...]]
    pytorch_version: str
    onnx_version: str
    onnxruntime_version: str
    exported_at_utc: str
    parity_passed: bool
    parity_seed: int
    parity_rtol: float
    parity_atol: float
    parity_max_absolute_error: float
    parity_mean_absolute_error: float


class OnnxRuntimePredictor:
    """Run fixed batch-one NCHW RGB FP32 inference through ORT CPU."""

    def __init__(self, session: Any, contract: OrtModelContract) -> None:
        self.session = session
        self.contract = contract
        _validate_session_contract(session, contract)

    @classmethod
    def from_files(cls, onnx_path: Path, report_path: Path) -> "OnnxRuntimePredictor":
        """Validate an ONNX/report pair and create a CPUExecutionProvider session."""

        model_path = Path(onnx_path)
        if not model_path.is_file():
            raise OrtPredictorError("ONNX model does not exist: {}".format(model_path))
        contract = load_ort_model_contract(report_path)
        if model_path.name != contract.onnx_file:
            raise OrtPredictorError(
                "ONNX filename mismatch: report expects '{}', got '{}'".format(
                    contract.onnx_file, model_path.name
                )
            )
        actual_sha256 = _sha256_file(model_path)
        if actual_sha256 != contract.onnx_sha256:
            raise OrtPredictorError(
                "ONNX SHA-256 mismatch: expected {}, got {}".format(
                    contract.onnx_sha256, actual_sha256
                )
            )
        actual_size = model_path.stat().st_size
        if actual_size != contract.onnx_size_bytes:
            raise OrtPredictorError(
                "ONNX size mismatch: expected {}, got {}".format(
                    contract.onnx_size_bytes, actual_size
                )
            )
        try:
            import onnxruntime as ort
        except ImportError as error:
            raise OrtPredictorError(
                "ONNX Runtime is unavailable; install the project deployment extra"
            ) from error
        try:
            session = ort.InferenceSession(
                str(model_path), providers=["CPUExecutionProvider"]
            )
        except Exception as error:
            raise OrtPredictorError(
                "unable to create ONNX Runtime CPU session: {}".format(error)
            ) from error
        return cls(session, contract)

    def predict_logits(self, input_tensor: np.ndarray) -> np.ndarray:
        """Return raw float32 logits for normalized NCHW RGB ``[1,3,S,S]``."""

        if not isinstance(input_tensor, np.ndarray):
            raise OrtPredictorError("input tensor must be a numpy.ndarray")
        if tuple(input_tensor.shape) != self.contract.input_shape:
            raise OrtPredictorError(
                "input shape mismatch: expected {}, got {}".format(
                    list(self.contract.input_shape), list(input_tensor.shape)
                )
            )
        if input_tensor.dtype != np.float32:
            raise OrtPredictorError("input tensor dtype must be float32")
        if not bool(np.isfinite(input_tensor).all()):
            raise OrtPredictorError("input tensor contains NaN or Inf")
        lower, upper = self.contract.input_value_range
        if bool((input_tensor < lower).any()) or bool((input_tensor > upper).any()):
            raise OrtPredictorError(
                "input tensor values must be within [{},{}]".format(lower, upper)
            )
        contiguous = np.ascontiguousarray(input_tensor)
        try:
            outputs = self.session.run(
                [self.contract.output_name],
                {self.contract.input_name: contiguous},
            )
        except Exception as error:
            raise OrtPredictorError("ONNX Runtime inference failed: {}".format(error)) from error
        if not isinstance(outputs, list) or len(outputs) != 1:
            raise OrtPredictorError("ONNX Runtime must return exactly one logits output")
        logits = outputs[0]
        if not isinstance(logits, np.ndarray):
            raise OrtPredictorError("ONNX Runtime logits output must be a numpy.ndarray")
        if tuple(logits.shape) != self.contract.output_shape:
            raise OrtPredictorError(
                "output shape mismatch: expected {}, got {}".format(
                    list(self.contract.output_shape), list(logits.shape)
                )
            )
        if logits.dtype != np.float32:
            raise OrtPredictorError("ONNX Runtime logits output dtype must be float32")
        if not bool(np.isfinite(logits).all()):
            raise OrtPredictorError("ONNX Runtime logits contain NaN or Inf")
        return logits

    def predict_rgb_image(
        self,
        image: np.ndarray,
        *,
        confidence_threshold: Optional[float] = None,
    ) -> ImagePrediction:
        """Run shared letterbox/logits/postprocess inference on RGB uint8 pixels."""

        prepared = preprocess_rgb_image(
            image, input_size=self.contract.input_shape[-1]
        )
        logits = self.predict_logits(prepared.input_tensor)
        threshold = (
            self.contract.confidence_threshold
            if confidence_threshold is None
            else confidence_threshold
        )
        return prediction_from_numpy_logits(
            prepared,
            logits,
            class_names=self.contract.class_names,
            output_stride=self.contract.output_stride,
            confidence_threshold=threshold,
            class_thresholds=self.contract.class_thresholds,
            component_mode=self.contract.component_mode,
            confidence_mode=self.contract.confidence_mode,
        )


def load_ort_model_contract(path: Path) -> OrtModelContract:
    """Load and validate the self-contained JSON sidecar without importing torch."""

    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except OSError as error:
        raise OrtPredictorError("unable to read ONNX report '{}': {}".format(source, error)) from error
    except json.JSONDecodeError as error:
        raise OrtPredictorError("unable to parse ONNX report '{}': {}".format(source, error)) from error
    root = _mapping(payload, "ONNX report")
    postprocess = _mapping(root.get("postprocess"), "postprocess")
    parity = _mapping(root.get("parity"), "parity")
    input_shape = _shape(root.get("input_shape"), "input_shape")
    output_shape = _shape(root.get("output_shape"), "output_shape")
    class_names = _class_names(root.get("class_names"))
    output_stride = _positive_int(root.get("output_stride"), "output_stride")
    if input_shape[0:2] != (1, 3) or input_shape[2] != input_shape[3]:
        raise OrtPredictorError("input_shape must be fixed batch-one NCHW RGB")
    if input_shape[2] % output_stride != 0:
        raise OrtPredictorError("input_shape must be divisible by output_stride")
    expected_output = (
        1,
        len(class_names) + 1,
        input_shape[2] // output_stride,
        input_shape[3] // output_stride,
    )
    if output_shape != expected_output:
        raise OrtPredictorError(
            "output_shape mismatch: expected {}, got {}".format(
                list(expected_output), list(output_shape)
            )
        )
    if _text(root.get("input_dtype"), "input_dtype") != "float32":
        raise OrtPredictorError("input_dtype must be float32")
    if _text(root.get("input_color_order"), "input_color_order") != "RGB":
        raise OrtPredictorError("input_color_order must be RGB")
    if _text(root.get("output_dtype"), "output_dtype") != "float32":
        raise OrtPredictorError("output_dtype must be float32")
    if _text(root.get("output_semantic"), "output_semantic") != "raw_logits":
        raise OrtPredictorError("output_semantic must be raw_logits")
    value_range = _float_pair(root.get("input_value_range"), "input_value_range")
    if value_range != (0.0, 1.0):
        raise OrtPredictorError("input_value_range must be [0.0,1.0]")
    opset = _positive_int(root.get("onnx_opset"), "onnx_opset")
    if opset != 17:
        raise OrtPredictorError("onnx_opset must be 17")
    class_thresholds = _class_thresholds(
        postprocess.get("class_thresholds"), class_names
    )
    component_mode = _text(
        postprocess.get("component_mode"), "postprocess.component_mode"
    )
    if component_mode != "connected_components":
        raise OrtPredictorError(
            "postprocess.component_mode must be 'connected_components'"
        )
    confidence_mode = _text(
        postprocess.get("confidence_mode"), "postprocess.confidence_mode"
    )
    if confidence_mode not in {"max", "mean"}:
        raise OrtPredictorError("postprocess.confidence_mode must be 'max' or 'mean'")
    selection_strategy = _text(
        postprocess.get("selection_strategy"), "postprocess.selection_strategy"
    )
    if selection_strategy not in {
        "highest_confidence",
        "largest_component",
        "nearest_previous",
    }:
        raise OrtPredictorError("postprocess.selection_strategy is unsupported")
    allowed = postprocess.get("allowed_class_ids")
    if allowed is not None:
        if not isinstance(allowed, Sequence) or isinstance(allowed, (str, bytes)):
            raise OrtPredictorError("postprocess.allowed_class_ids must be null or a sequence")
        allowed_class_ids = tuple(
            _nonnegative_int(value, "postprocess.allowed_class_ids") for value in allowed
        )
        if len(set(allowed_class_ids)) != len(allowed_class_ids):
            raise OrtPredictorError("postprocess.allowed_class_ids must not contain duplicates")
        if any(class_id >= len(class_names) for class_id in allowed_class_ids):
            raise OrtPredictorError(
                "postprocess.allowed_class_ids contains an ID outside class_names"
            )
    else:
        allowed_class_ids = None
    confidence_threshold = _probability(
        postprocess.get("confidence_threshold"),
        "postprocess.confidence_threshold",
    )
    validation_threshold = _probability(
        root.get("validation_threshold"), "validation_threshold"
    )
    if validation_threshold != confidence_threshold:
        raise OrtPredictorError(
            "validation_threshold must equal postprocess.confidence_threshold"
        )
    validation_threshold_usage = _text(
        root.get("validation_threshold_usage"), "validation_threshold_usage"
    )
    if validation_threshold_usage != "provenance_only_raw_logits_export":
        raise OrtPredictorError(
            "validation_threshold_usage must be 'provenance_only_raw_logits_export'"
        )
    onnx_checker = _text(root.get("onnx_checker"), "onnx_checker")
    if onnx_checker != "passed":
        raise OrtPredictorError("onnx_checker must be 'passed'")
    parity_passed = _boolean(parity.get("passed"), "parity.passed")
    if not parity_passed:
        raise OrtPredictorError("parity.passed must be true")
    return OrtModelContract(
        artifact_name=_text(root.get("artifact_name"), "artifact_name"),
        source_experiment_config=_text(
            root.get("source_experiment_config"), "source_experiment_config"
        ),
        source_experiment_config_sha256=_sha256(
            root.get("source_experiment_config_sha256"),
            "source_experiment_config_sha256",
        ),
        export_config_file=_text(
            root.get("export_config_file"), "export_config_file"
        ),
        export_config_sha256=_sha256(
            root.get("export_config_sha256"), "export_config_sha256"
        ),
        checkpoint_file=_text(root.get("checkpoint_file"), "checkpoint_file"),
        checkpoint_sha256=_sha256(
            root.get("checkpoint_sha256"), "checkpoint_sha256"
        ),
        checkpoint_epoch=_positive_int(root.get("epoch"), "epoch"),
        checkpoint_seed=_nonnegative_int(root.get("seed"), "seed"),
        parameter_count=_positive_int(
            root.get("parameter_count"), "parameter_count"
        ),
        config_fingerprint=_sha256(
            root.get("config_fingerprint"), "config_fingerprint"
        ),
        validation_threshold=validation_threshold,
        validation_threshold_usage=validation_threshold_usage,
        onnx_file=_text(root.get("onnx_file"), "onnx_file"),
        onnx_sha256=_sha256(root.get("onnx_sha256"), "onnx_sha256"),
        onnx_size_bytes=_positive_int(
            root.get("onnx_size_bytes"), "onnx_size_bytes"
        ),
        onnx_checker=onnx_checker,
        opset=opset,
        input_name=_text(root.get("input_name"), "input_name"),
        input_shape=input_shape,
        input_dtype="float32",
        input_color_order="RGB",
        input_value_range=value_range,
        output_name=_text(root.get("output_name"), "output_name"),
        output_shape=output_shape,
        output_dtype="float32",
        output_semantic="raw_logits",
        output_stride=output_stride,
        class_names=class_names,
        confidence_threshold=confidence_threshold,
        class_thresholds=class_thresholds,
        component_mode=component_mode,
        confidence_mode=confidence_mode,
        selection_strategy=selection_strategy,
        max_match_distance_pixels=_positive_float(
            postprocess.get("max_match_distance_pixels"),
            "postprocess.max_match_distance_pixels",
        ),
        max_lost_frames=_nonnegative_int(
            postprocess.get("max_lost_frames"), "postprocess.max_lost_frames"
        ),
        allowed_class_ids=allowed_class_ids,
        pytorch_version=_text(root.get("pytorch_version"), "pytorch_version"),
        onnx_version=_text(root.get("onnx_version"), "onnx_version"),
        onnxruntime_version=_text(
            root.get("onnxruntime_version"), "onnxruntime_version"
        ),
        exported_at_utc=_text(root.get("exported_at_utc"), "exported_at_utc"),
        parity_passed=parity_passed,
        parity_seed=_nonnegative_int(parity.get("input_seed"), "parity.input_seed"),
        parity_rtol=_nonnegative_float(parity.get("rtol"), "parity.rtol"),
        parity_atol=_nonnegative_float(parity.get("atol"), "parity.atol"),
        parity_max_absolute_error=_nonnegative_float(
            parity.get("max_absolute_error"), "parity.max_absolute_error"
        ),
        parity_mean_absolute_error=_nonnegative_float(
            parity.get("mean_absolute_error"), "parity.mean_absolute_error"
        ),
    )


def _validate_session_contract(session: Any, contract: OrtModelContract) -> None:
    try:
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        providers = session.get_providers()
    except Exception as error:
        raise OrtPredictorError("unable to inspect ONNX Runtime session: {}".format(error)) from error
    if "CPUExecutionProvider" not in providers:
        raise OrtPredictorError("ONNX Runtime session must enable CPUExecutionProvider")
    if (
        len(inputs) != 1
        or inputs[0].name != contract.input_name
        or tuple(inputs[0].shape) != contract.input_shape
        or inputs[0].type != "tensor(float)"
    ):
        raise OrtPredictorError("ONNX Runtime input contract does not match report")
    if (
        len(outputs) != 1
        or outputs[0].name != contract.output_name
        or tuple(outputs[0].shape) != contract.output_shape
        or outputs[0].type != "tensor(float)"
    ):
        raise OrtPredictorError("ONNX Runtime output contract does not match report")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise OrtPredictorError("unable to hash ONNX model '{}': {}".format(path, error)) from error
    return digest.hexdigest()


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise OrtPredictorError("{} must be a mapping".format(name))
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise OrtPredictorError("{} must be a non-empty string".format(name))
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise OrtPredictorError("{} must be a positive integer".format(name))
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise OrtPredictorError("{} must be a non-negative integer".format(name))
    return value


def _positive_float(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise OrtPredictorError("{} must be a finite positive number".format(name))
    return float(value)


def _nonnegative_float(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(float(value))
        or float(value) < 0.0
    ):
        raise OrtPredictorError("{} must be a finite non-negative number".format(name))
    return float(value)


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise OrtPredictorError("{} must be a boolean".format(name))
    return value


def _probability(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise OrtPredictorError("{} must be a finite probability".format(name))
    return float(value)


def _shape(value: Any, name: str) -> tuple[int, int, int, int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 4:
        raise OrtPredictorError("{} must contain four positive integers".format(name))
    result = tuple(_positive_int(item, name) for item in value)
    return result  # type: ignore[return-value]


def _float_pair(value: Any, name: str) -> tuple[float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise OrtPredictorError("{} must contain two finite numbers".format(name))
    result = tuple(_finite_float(item, name) for item in value)
    return result  # type: ignore[return-value]


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OrtPredictorError("{} must contain finite numbers".format(name))
    result = float(value)
    if not isfinite(result):
        raise OrtPredictorError("{} must contain finite numbers".format(name))
    return result


def _class_names(value: Any) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise OrtPredictorError("class_names must be a non-empty sequence")
    names = tuple(_text(item, "class_names") for item in value)
    if len(set(names)) != len(names):
        raise OrtPredictorError("class_names must not contain duplicates")
    return names


def _class_thresholds(
    value: Any, class_names: tuple[str, ...]
) -> Optional[tuple[float, ...] | dict[int | str, float]]:
    if value is None:
        return None
    if isinstance(value, Mapping):
        thresholds: dict[int | str, float] = {}
        for raw_key, raw_threshold in value.items():
            key: int | str
            if isinstance(raw_key, bool) or not isinstance(raw_key, (int, str)):
                raise OrtPredictorError(
                    "postprocess.class_thresholds keys must be class IDs or names"
                )
            if isinstance(raw_key, str) and raw_key.isdigit():
                key = int(raw_key)
            else:
                key = raw_key
            if isinstance(key, int):
                if not 0 <= key < len(class_names):
                    raise OrtPredictorError(
                        "postprocess.class_thresholds contains an ID outside class_names"
                    )
            elif key not in class_names:
                raise OrtPredictorError(
                    "postprocess.class_thresholds contains an unknown class name"
                )
            thresholds[key] = _probability(
                raw_threshold, "postprocess.class_thresholds"
            )
        return thresholds
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise OrtPredictorError(
            "postprocess.class_thresholds must be null, a sequence, or mapping"
        )
    if len(value) != len(class_names):
        raise OrtPredictorError(
            "postprocess.class_thresholds length must equal len(class_names)"
        )
    return tuple(
        _probability(item, "postprocess.class_thresholds") for item in value
    )


def _sha256(value: Any, name: str) -> str:
    text = _text(value, name).lower()
    if re.fullmatch(r"[0-9a-f]{64}", text) is None:
        raise OrtPredictorError("{} must be a hexadecimal SHA-256".format(name))
    return text


__all__ = [
    "OnnxRuntimePredictor",
    "OrtModelContract",
    "OrtPredictorError",
    "load_ort_model_contract",
]
