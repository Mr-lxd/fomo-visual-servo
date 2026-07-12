from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from fomo_servo.config import load_config
from fomo_servo.inference import predict_rgb_image
from fomo_servo.postprocess import Detection
from scripts.predict_video import CSV_COLUMNS


class _FixedLogitModel(nn.Module):
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros(images.shape[0], 2, 12, 12, device=images.device)
        logits[:, 0] = -2.0
        logits[:, 1, 5, 6] = 6.0
        return logits


class _ThresholdBoundaryModel(nn.Module):
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return equal foreground/background logits, so probability is exactly 0.5."""

        return torch.zeros(images.shape[0], 2, 12, 12, device=images.device)


def test_predict_rgb_image_preserves_letterbox_and_returns_original_centroid(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
dataset:
  root: data
  classes: [creature]
model:
  input_size: 96
  output_stride: 8
""".lstrip(),
        encoding="utf-8",
    )
    config = load_config(config_path)
    image = np.zeros((48, 96, 3), dtype=np.uint8)

    prediction = predict_rgb_image(
        _FixedLogitModel(), image, config=config, device=torch.device("cpu")
    )

    assert prediction.letterbox_image.shape == (96, 96, 3)
    assert len(prediction.detections) == 1
    assert 0.0 <= prediction.detections[0].original_x <= 95.0
    assert 0.0 <= prediction.detections[0].original_y <= 47.0


def test_predict_rgb_image_defaults_to_inference_threshold(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
dataset:
  root: data
  classes: [creature]
model:
  input_size: 96
  output_stride: 8
postprocess:
  inference_threshold: 0.6
""".lstrip(),
        encoding="utf-8",
    )
    config = load_config(config_path)
    image = np.zeros((96, 96, 3), dtype=np.uint8)

    default_prediction = predict_rgb_image(
        _ThresholdBoundaryModel(), image, config=config, device=torch.device("cpu")
    )
    explicit_prediction = predict_rgb_image(
        _ThresholdBoundaryModel(),
        image,
        config=config,
        device=torch.device("cpu"),
        confidence_threshold=0.5,
    )

    assert default_prediction.detections == ()
    assert len(explicit_prediction.detections) == 1


def test_detection_and_video_csv_schemas_are_stable() -> None:
    required_detection_fields = {
        "class_id", "class_name", "confidence", "mean_confidence",
        "component_area_cells", "heatmap_x", "heatmap_y", "input_x", "input_y",
        "original_x", "original_y",
    }
    detection = Detection(
        class_id=0,
        class_name="creature",
        confidence=0.9,
        mean_confidence=0.8,
        component_area_cells=1,
        heatmap_x=1.0,
        heatmap_y=1.0,
        input_x=8.0,
        input_y=8.0,
        original_x=8.0,
        original_y=8.0,
    )
    assert required_detection_fields.issubset(set(detection.as_dict()))
    assert {
        "frame_index", "timestamp", "status", "class_id", "class_name", "confidence",
        "original_x", "original_y", "normalized_x", "normalized_y",
        "detection_count", "lost_frames",
    }.issubset(set(CSV_COLUMNS))
