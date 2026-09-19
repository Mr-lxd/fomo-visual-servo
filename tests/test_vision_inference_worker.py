from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import get_type_hints

import pytest

from fomo_servo.postprocess import Detection
from fomo_servo.vision.frame_hub import FrameHub
from fomo_servo.vision.inference_worker import (
    InferenceResult,
    InferenceState,
    InferenceWorker,
)


def test_inference_state_uses_authoritative_lowercase_values() -> None:
    assert InferenceState.DISABLED.value == "disabled"
    assert InferenceState.STARTING.value == "starting"
    assert InferenceState.RUNNING.value == "running"
    assert InferenceState.FAILED.value == "failed"


def test_inference_result_is_frozen_and_reuses_detection_type() -> None:
    result = InferenceResult(
        frame_id=7,
        capture_timestamp_ns=100,
        inference_started_ns=110,
        inference_finished_ns=120,
        detections=(),
    )

    assert result.detections == ()
    assert get_type_hints(InferenceResult)["detections"] == tuple[Detection, ...]
    with pytest.raises(FrozenInstanceError):
        result.frame_id = 8  # type: ignore[misc]


def test_worker_exposes_disabled_status_before_start(tmp_path: Path) -> None:
    worker = InferenceWorker(
        FrameHub(),
        onnx_path=tmp_path / "model.onnx",
        report_path=tmp_path / "report.json",
        predictor_factory=lambda *_args, **_kwargs: None,
        clock_ns=lambda: 1,
        wait_timeout=0.1,
        fps_window_size=5,
    )

    assert worker.status() == {
        "state": "disabled",
        "artifact_name": None,
        "model_sha256": None,
        "confidence_threshold": None,
        "latest_frame_id": None,
        "capture_timestamp_ns": None,
        "inference_fps": None,
        "latency_ms": None,
        "detection_count": None,
        "last_error": None,
        "processed_frames": 0,
        "skipped_frames": 0,
    }
    assert worker.latest_result() is None
    worker.start()
    worker.stop()
