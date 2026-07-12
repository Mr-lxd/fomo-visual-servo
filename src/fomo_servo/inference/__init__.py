"""Future inference interfaces; no inference pipeline is implemented yet."""
"""Public image and latest-frame video inference helpers."""

from .predictor import ImagePrediction, InferenceError, load_inference_model, predict_rgb_image, read_rgb_image
from .video import FramePacket, LatestFrameBuffer, LatestFrameReader

__all__ = [
    "FramePacket",
    "ImagePrediction",
    "InferenceError",
    "LatestFrameBuffer",
    "LatestFrameReader",
    "load_inference_model",
    "predict_rgb_image",
    "read_rgb_image",
]
