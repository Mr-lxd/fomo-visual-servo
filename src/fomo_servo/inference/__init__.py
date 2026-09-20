"""Public image and latest-frame video inference helpers."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "FramePacket",
    "ImagePrediction",
    "InferenceError",
    "LatestFrameBuffer",
    "LatestFrameReader",
    "OnnxRuntimePredictor",
    "OrtModelContract",
    "OrtPredictorError",
    "OutputPathError",
    "PipelineParityError",
    "PreparedImage",
    "PreprocessingError",
    "load_inference_model",
    "load_ort_model_contract",
    "prediction_from_numpy_logits",
    "compare_rgb_image_pipeline",
    "predict_rgb_image",
    "preprocess_rgb_image",
    "read_rgb_image",
    "validate_output_paths",
]


_EXPORT_MODULES = {
    "FramePacket": ".video",
    "ImagePrediction": ".predictor",
    "InferenceError": ".predictor",
    "LatestFrameBuffer": ".video",
    "LatestFrameReader": ".video",
    "OnnxRuntimePredictor": ".ort_predictor",
    "OrtModelContract": ".ort_predictor",
    "OrtPredictorError": ".ort_predictor",
    "OutputPathError": ".path_safety",
    "PipelineParityError": ".parity",
    "PreparedImage": ".preprocessing",
    "PreprocessingError": ".preprocessing",
    "load_inference_model": ".predictor",
    "load_ort_model_contract": ".ort_predictor",
    "prediction_from_numpy_logits": ".preprocessing",
    "compare_rgb_image_pipeline": ".parity",
    "predict_rgb_image": ".predictor",
    "preprocess_rgb_image": ".preprocessing",
    "read_rgb_image": ".predictor",
    "validate_output_paths": ".path_safety",
}


def __getattr__(name: str) -> Any:
    """Resolve a public inference API without importing unrelated predictors."""

    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError("module {!r} has no attribute {!r}".format(__name__, name))
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value
