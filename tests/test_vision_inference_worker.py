from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace
import threading
import time
from typing import get_type_hints

import pytest

from fomo_servo.postprocess import Detection
from fomo_servo.vision.frame import PixelFormat, VisionFrame
from fomo_servo.vision.frame_hub import FrameHub
from fomo_servo.vision.inference_worker import (
    InferenceResult,
    InferenceState,
    InferenceWorker,
)


EXPECTED_CONTRACT = {
    "artifact_name": "d2_mobilenet_v2_fomo_seed42_epoch40",
    "checkpoint_seed": 42,
    "checkpoint_epoch": 40,
    "confidence_threshold": 0.40,
    "onnx_sha256": "3dea74511bf2c44844192e75594fd53d4c4ce941f8b53b15767e020832bf9b08",
    "input_shape": (1, 3, 192, 192),
    "input_dtype": "float32",
    "input_color_order": "RGB",
    "input_value_range": (0.0, 1.0),
    "output_shape": (1, 8, 24, 24),
    "output_dtype": "float32",
    "output_semantic": "raw_logits",
    "opset": 17,
}


class FakePredictor:
    def __init__(self, contract: object) -> None:
        self.contract = contract


@pytest.fixture
def fake_contract() -> SimpleNamespace:
    return SimpleNamespace(**EXPECTED_CONTRACT)


@pytest.fixture
def fake_predictor_factory(fake_contract: SimpleNamespace):
    calls: list[tuple[int, Path, Path]] = []

    def factory(onnx_path: Path, report_path: Path) -> FakePredictor:
        calls.append((threading.get_ident(), onnx_path, report_path))
        return FakePredictor(fake_contract)

    factory.calls = calls  # type: ignore[attr-defined]
    return factory


def _wait_for_state(worker: InferenceWorker, expected: InferenceState) -> dict:
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        status = worker.status()
        if status["state"] == expected.value:
            return status
        time.sleep(0.001)
    pytest.fail(f"worker did not reach {expected.value}: {worker.status()}")


def _worker(tmp_path: Path, hub: FrameHub, factory) -> InferenceWorker:
    return InferenceWorker(
        hub,
        onnx_path=tmp_path / "model.onnx",
        report_path=tmp_path / "report.json",
        predictor_factory=factory,
        wait_timeout=0.01,
        fps_window_size=5,
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


def test_worker_initializes_in_worker_thread_and_publishes_running_identity(
    tmp_path: Path, fake_contract: SimpleNamespace
) -> None:
    main_thread_id = threading.get_ident()
    factory_started = threading.Event()
    release_factory = threading.Event()
    factory_thread_ids: list[int] = []

    def blocking_factory(_onnx_path: Path, _report_path: Path) -> FakePredictor:
        factory_thread_ids.append(threading.get_ident())
        factory_started.set()
        assert release_factory.wait(2.0)
        return FakePredictor(fake_contract)

    worker = _worker(tmp_path, FrameHub(), blocking_factory)
    worker.start()
    assert factory_started.wait(2.0)

    starting = worker.status()
    assert starting["state"] == InferenceState.STARTING.value
    assert starting["artifact_name"] is None
    assert starting["model_sha256"] is None
    assert starting["confidence_threshold"] is None

    release_factory.set()
    running = _wait_for_state(worker, InferenceState.RUNNING)
    assert factory_thread_ids == [factory_thread_ids[0]]
    assert factory_thread_ids[0] != main_thread_id
    assert running["artifact_name"] == EXPECTED_CONTRACT["artifact_name"]
    assert running["model_sha256"] == EXPECTED_CONTRACT["onnx_sha256"]
    assert running["confidence_threshold"] == EXPECTED_CONTRACT["confidence_threshold"]
    assert running["last_error"] is None
    worker.stop()


def test_worker_fences_cached_frames_after_model_validation(
    tmp_path: Path, fake_contract: SimpleNamespace
) -> None:
    hub = FrameHub()
    hub.publish(VisionFrame(3, 100, 2, 2, PixelFormat.BGR8, object()))
    factory_started = threading.Event()
    release_factory = threading.Event()

    def blocking_factory(_onnx_path: Path, _report_path: Path) -> FakePredictor:
        factory_started.set()
        assert release_factory.wait(2.0)
        return FakePredictor(fake_contract)

    worker = _worker(tmp_path, hub, blocking_factory)
    worker.start()
    assert factory_started.wait(2.0)
    hub.publish(VisionFrame(4, 101, 2, 2, PixelFormat.BGR8, object()))
    release_factory.set()
    _wait_for_state(worker, InferenceState.RUNNING)

    assert worker._after_frame_id == 4
    worker.stop()


def test_worker_initialization_exception_is_failed_without_retry(
    tmp_path: Path,
) -> None:
    calls = 0

    def failing_factory(_onnx_path: Path, _report_path: Path) -> FakePredictor:
        nonlocal calls
        calls += 1
        raise RuntimeError("factory boom")

    worker = _worker(tmp_path, FrameHub(), failing_factory)
    worker.start()
    failed = _wait_for_state(worker, InferenceState.FAILED)

    assert failed["last_error"] == "factory boom"
    assert failed["artifact_name"] is None
    assert failed["model_sha256"] is None
    assert calls == 1
    worker.stop()


@pytest.mark.parametrize(
    ("field", "mismatched_value"),
    [
        ("artifact_name", "wrong_artifact"),
        ("checkpoint_seed", 43),
        ("checkpoint_epoch", 41),
        ("confidence_threshold", 0.41),
        ("onnx_sha256", "0" * 64),
        ("input_shape", (1, 3, 224, 224)),
        ("input_dtype", "float64"),
        ("input_color_order", "BGR"),
        ("input_value_range", (0.0, 255.0)),
        ("output_shape", (1, 8, 12, 12)),
        ("output_dtype", "float64"),
        ("output_semantic", "probabilities"),
        ("opset", 16),
    ],
)
def test_worker_rejects_each_mismatched_contract_field(
    tmp_path: Path,
    field: str,
    mismatched_value: object,
) -> None:
    contract = SimpleNamespace(**EXPECTED_CONTRACT)
    setattr(contract, field, mismatched_value)

    worker = _worker(tmp_path, FrameHub(), lambda *_args: FakePredictor(contract))
    worker.start()
    failed = _wait_for_state(worker, InferenceState.FAILED)

    assert failed["state"] == InferenceState.FAILED.value
    assert failed["state"] != InferenceState.RUNNING.value
    assert field in failed["last_error"]
    assert failed["artifact_name"] is None
    assert failed["model_sha256"] is None
    assert failed["confidence_threshold"] is None
    worker.stop()
