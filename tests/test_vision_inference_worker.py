from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace
import threading
import time
from typing import get_type_hints

import numpy as np
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
        self.received_images: list[np.ndarray] = []

    def predict_rgb_image(self, image: np.ndarray) -> SimpleNamespace:
        self.received_images.append(image)
        detection = Detection(
            class_id=0,
            class_name="creature",
            confidence=0.9,
            mean_confidence=0.8,
            component_area_cells=1,
            heatmap_x=0.5,
            heatmap_y=0.5,
            input_x=1.0,
            input_y=1.0,
            original_x=1.0,
            original_y=1.0,
        )
        return SimpleNamespace(detections=(detection,))


@pytest.fixture
def fake_contract() -> SimpleNamespace:
    return SimpleNamespace(**EXPECTED_CONTRACT)


def _wait_for_state(worker: InferenceWorker, expected: InferenceState) -> dict:
    deadline = time.monotonic() + 5.0
    with worker._state_condition:
        while True:
            status = worker.status()
            if status["state"] == expected.value:
                return status
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                break
            worker._state_condition.wait(remaining)
    pytest.fail(f"worker did not reach {expected.value}: {worker.status()}")


def _wait_for_latest_result(worker: InferenceWorker) -> InferenceResult:
    deadline = time.monotonic() + 2.0
    with worker._state_condition:
        while worker._latest_result is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                break
            worker._state_condition.wait(remaining)
        if worker._latest_result is not None:
            return worker._latest_result
    pytest.fail("worker did not publish an inference result")


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


def test_worker_uses_monotonic_clock_by_default_and_preserves_explicit_clock(
    tmp_path: Path,
) -> None:
    default_worker = InferenceWorker(
        FrameHub(),
        onnx_path=tmp_path / "default.onnx",
        report_path=tmp_path / "default.json",
        predictor_factory=lambda *_args: None,
    )
    explicit_clock = lambda: 123
    explicit_worker = InferenceWorker(
        FrameHub(),
        onnx_path=tmp_path / "explicit.onnx",
        report_path=tmp_path / "explicit.json",
        predictor_factory=lambda *_args: None,
        clock_ns=explicit_clock,
    )

    assert default_worker._clock_ns is time.monotonic_ns
    assert explicit_worker._clock_ns is explicit_clock


def test_worker_owns_private_state_condition_for_lifecycle_waits(
    tmp_path: Path,
) -> None:
    worker = InferenceWorker(
        FrameHub(),
        onnx_path=tmp_path / "model.onnx",
        report_path=tmp_path / "report.json",
        predictor_factory=lambda *_args: None,
    )

    assert isinstance(worker._state_condition, threading.Condition)


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


def test_start_clears_stale_latest_result_for_new_generation(
    tmp_path: Path, fake_contract: SimpleNamespace
) -> None:
    factory_started = threading.Event()
    release_factory = threading.Event()

    def blocking_factory(_onnx_path: Path, _report_path: Path) -> FakePredictor:
        factory_started.set()
        assert release_factory.wait(2.0)
        return FakePredictor(fake_contract)

    worker = _worker(tmp_path, FrameHub(), blocking_factory)
    worker._latest_result = InferenceResult(
        frame_id=99,
        capture_timestamp_ns=100,
        inference_started_ns=101,
        inference_finished_ns=102,
        detections=(),
    )
    worker.start()
    try:
        assert factory_started.wait(2.0)
        assert worker.status()["state"] == InferenceState.STARTING.value
        assert worker.latest_result() is None
    finally:
        release_factory.set()
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


def test_worker_publishes_rgb_inference_result_bound_to_source_frame(
    tmp_path: Path, fake_contract: SimpleNamespace
) -> None:
    hub = FrameHub()
    source_image = np.array([[[11, 22, 33]]], dtype=np.uint8)
    source_image_before = source_image.copy()
    frame = VisionFrame(
        frame_id=5,
        capture_timestamp_ns=123456,
        width=1,
        height=1,
        pixel_format=PixelFormat.BGR8,
        image=source_image,
    )
    clock_values = iter((1001, 1002))
    predictor_holder: list[FakePredictor] = []

    def factory(_onnx_path: Path, _report_path: Path) -> FakePredictor:
        predictor = FakePredictor(fake_contract)
        predictor_holder.append(predictor)
        return predictor

    worker = _worker(tmp_path, hub, factory)
    worker._clock_ns = lambda: next(clock_values)
    worker.start()
    try:
        _wait_for_state(worker, InferenceState.RUNNING)
        hub.publish(frame)

        result = _wait_for_latest_result(worker)
        assert len(predictor_holder) == 1
        assert len(predictor_holder[0].received_images) == 1
        received_rgb = predictor_holder[0].received_images[0]
        np.testing.assert_array_equal(received_rgb, np.array([[[33, 22, 11]]], dtype=np.uint8))
        assert received_rgb is not source_image
        np.testing.assert_array_equal(source_image, source_image_before)
        assert result.frame_id == frame.frame_id
        assert result.capture_timestamp_ns == frame.capture_timestamp_ns
        assert result.inference_started_ns == 1001
        assert result.inference_finished_ns == 1002
        assert isinstance(result.detections, tuple)
        assert isinstance(result.detections[0], Detection)
    finally:
        worker.stop()


def test_worker_fences_processing_exception_and_enters_failed(
    tmp_path: Path, fake_contract: SimpleNamespace
) -> None:
    class FailingPredictor(FakePredictor):
        def predict_rgb_image(self, image: np.ndarray) -> SimpleNamespace:
            raise RuntimeError("predictor boom")

    hub = FrameHub()
    worker = _worker(
        tmp_path,
        hub,
        lambda *_args: FailingPredictor(fake_contract),
    )
    worker.start()
    try:
        _wait_for_state(worker, InferenceState.RUNNING)
        hub.publish(
            VisionFrame(
                frame_id=6,
                capture_timestamp_ns=123457,
                width=1,
                height=1,
                pixel_format=PixelFormat.BGR8,
                image=np.array([[[11, 22, 33]]], dtype=np.uint8),
            )
        )

        failed = _wait_for_state(worker, InferenceState.FAILED)
        assert failed["last_error"] == "predictor boom"
        assert worker.latest_result() is None
    finally:
        worker.stop()


def test_worker_discards_result_when_stop_requested_before_publish(
    tmp_path: Path, fake_contract: SimpleNamespace
) -> None:
    class StopBeforePublishPredictor(FakePredictor):
        def __init__(self, contract: object) -> None:
            super().__init__(contract)
            self.ready_to_return = threading.Event()
            self.release_return = threading.Event()

        def predict_rgb_image(self, image: np.ndarray) -> SimpleNamespace:
            prediction = super().predict_rgb_image(image)
            self.ready_to_return.set()
            assert self.release_return.wait(2.0)
            return prediction

    hub = FrameHub()
    predictor_holder: list[StopBeforePublishPredictor] = []

    def factory(_onnx_path: Path, _report_path: Path) -> StopBeforePublishPredictor:
        predictor = StopBeforePublishPredictor(fake_contract)
        predictor_holder.append(predictor)
        return predictor

    worker = _worker(tmp_path, hub, factory)
    worker.start()
    try:
        _wait_for_state(worker, InferenceState.RUNNING)
        class TrackingStopEvent(threading.Event):
            def __init__(self, lock: threading.RLock) -> None:
                super().__init__()
                self.set_called = threading.Event()
                self.set_under_worker_lock = False
                self._worker_lock = lock

            def set(self) -> None:
                self.set_under_worker_lock = self._worker_lock._is_owned()
                self.set_called.set()
                super().set()

        stop_event = TrackingStopEvent(worker._lock)
        with worker._lock:
            worker._stop_event = stop_event
        hub.publish(
            VisionFrame(
                frame_id=7,
                capture_timestamp_ns=123458,
                width=1,
                height=1,
                pixel_format=PixelFormat.BGR8,
                image=np.array([[[11, 22, 33]]], dtype=np.uint8),
            )
        )
        assert predictor_holder[0].ready_to_return.wait(2.0)

        stop_finished = threading.Event()
        stopper = threading.Thread(
            target=lambda: (worker.stop(), stop_finished.set()), daemon=True
        )
        stopper.start()
        assert stop_event.set_called.wait(2.0)
        assert not predictor_holder[0].release_return.is_set()
        predictor_holder[0].release_return.set()

        assert stop_finished.wait(2.0)
        stopper.join(2.0)
        assert stop_event.set_under_worker_lock
        assert worker.latest_result() is None
    finally:
        predictor = predictor_holder[0] if predictor_holder else None
        if predictor is not None:
            predictor.release_return.set()
        worker.stop()


def test_stop_during_factory_initialization_never_publishes_running(
    tmp_path: Path, fake_contract: SimpleNamespace
) -> None:
    factory_started = threading.Event()
    release_factory = threading.Event()
    factory_release_seen = threading.Event()
    stop_finished = threading.Event()

    def blocking_factory(_onnx_path: Path, _report_path: Path) -> FakePredictor:
        factory_started.set()
        assert release_factory.wait(2.0)
        assert worker.status()["state"] == InferenceState.STARTING.value
        factory_release_seen.set()
        return FakePredictor(fake_contract)

    worker = _worker(tmp_path, FrameHub(), blocking_factory)
    worker.start()
    assert factory_started.wait(2.0)
    assert worker.status()["state"] == InferenceState.STARTING.value

    stopper = threading.Thread(
        target=lambda: (worker.stop(), stop_finished.set()), daemon=True
    )
    stopper.start()
    assert worker._stop_event.wait(2.0)
    release_factory.set()
    assert factory_release_seen.wait(2.0)
    assert stop_finished.wait(2.0)
    stopper.join(2.0)

    stopped = worker.status()
    assert stopped["state"] == InferenceState.DISABLED.value
    assert stopped["artifact_name"] is None
    assert stopped["model_sha256"] is None
    assert stopped["confidence_threshold"] is None
    assert stopped["last_error"] is None


def test_stop_during_validation_returns_worker_to_disabled(
    tmp_path: Path, fake_contract: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    validation_started = threading.Event()
    release_validation = threading.Event()
    validation_release_seen = threading.Event()
    stop_finished = threading.Event()
    original_validate = InferenceWorker._validate_contract

    def blocking_validate(contract: object) -> None:
        validation_started.set()
        assert release_validation.wait(2.0)
        assert worker.status()["state"] == InferenceState.STARTING.value
        validation_release_seen.set()
        original_validate(contract)

    monkeypatch.setattr(
        InferenceWorker, "_validate_contract", staticmethod(blocking_validate)
    )
    worker = _worker(tmp_path, FrameHub(), lambda *_args: FakePredictor(fake_contract))
    worker.start()
    assert validation_started.wait(2.0)

    stopper = threading.Thread(
        target=lambda: (worker.stop(), stop_finished.set()), daemon=True
    )
    stopper.start()
    assert worker._stop_event.wait(2.0)
    release_validation.set()
    assert validation_release_seen.wait(2.0)
    assert stop_finished.wait(2.0)
    stopper.join(2.0)

    assert worker.status()["state"] == InferenceState.DISABLED.value


def test_stop_running_idle_worker_returns_to_disabled(
    tmp_path: Path, fake_contract: SimpleNamespace
) -> None:
    worker = _worker(tmp_path, FrameHub(), lambda *_args: FakePredictor(fake_contract))
    worker.start()
    _wait_for_state(worker, InferenceState.RUNNING)

    worker.stop()

    stopped = worker.status()
    assert stopped["state"] == InferenceState.DISABLED.value
    assert stopped["artifact_name"] is None
    assert stopped["model_sha256"] is None
    assert stopped["confidence_threshold"] is None


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
        ("input_shape", (True, 3, 192, 192)),
        ("input_dtype", "float64"),
        ("input_color_order", "BGR"),
        ("input_value_range", (0.0, 255.0)),
        ("input_value_range", (0, 1)),
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
