from __future__ import annotations

from dataclasses import FrozenInstanceError, is_dataclass
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
    ModelIdentity,
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

EXPECTED_MODEL_IDENTITY = ModelIdentity(
    artifact_name=EXPECTED_CONTRACT["artifact_name"],
    onnx_sha256=EXPECTED_CONTRACT["onnx_sha256"],
    confidence_threshold=EXPECTED_CONTRACT["confidence_threshold"],
)


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


def _wait_for_status(worker: InferenceWorker, predicate) -> dict:
    deadline = time.monotonic() + 2.0
    with worker._state_condition:
        while True:
            status = worker.status()
            if predicate(status):
                return status
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                break
            worker._state_condition.wait(remaining)
    pytest.fail(f"worker status did not satisfy predicate: {worker.status()}")


def _frame(frame_id: int, capture_timestamp_ns: int = 1) -> VisionFrame:
    return VisionFrame(
        frame_id=frame_id,
        capture_timestamp_ns=capture_timestamp_ns,
        width=1,
        height=1,
        pixel_format=PixelFormat.BGR8,
        image=np.array([[[frame_id, 0, 0]]], dtype=np.uint8),
    )


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
        latency_ms=0.00002,
        model_identity=EXPECTED_MODEL_IDENTITY,
        detections=(),
    )

    assert result.detections == ()
    assert get_type_hints(InferenceResult)["detections"] == tuple[Detection, ...]
    assert get_type_hints(InferenceResult)["model_identity"] is ModelIdentity
    assert is_dataclass(result.model_identity)
    with pytest.raises(FrozenInstanceError):
        result.frame_id = 8  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.model_identity.artifact_name = "changed"  # type: ignore[misc]


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
        latency_ms=0.000002,
        model_identity=EXPECTED_MODEL_IDENTITY,
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


def test_start_resets_generation_state_and_fences_cached_frame(
    tmp_path: Path, fake_contract: SimpleNamespace
) -> None:
    hub = FrameHub()
    factory_calls = 0
    second_factory_started = threading.Event()
    release_second_factory = threading.Event()
    clock_values = iter((1_000, 2_000, 3_000, 4_000, 5_000, 6_000))

    def factory(_onnx_path: Path, _report_path: Path) -> FakePredictor:
        nonlocal factory_calls
        factory_calls += 1
        if factory_calls == 2:
            second_factory_started.set()
            assert release_second_factory.wait(2.0)
        return FakePredictor(fake_contract)

    worker = InferenceWorker(
        hub,
        onnx_path=tmp_path / "model.onnx",
        report_path=tmp_path / "report.json",
        predictor_factory=factory,
        clock_ns=lambda: next(clock_values),
        wait_timeout=0.01,
        fps_window_size=5,
    )
    worker.start()
    try:
        _wait_for_state(worker, InferenceState.RUNNING)
        hub.publish(_frame(1, 1_000))
        _wait_for_status(worker, lambda status: status["latest_frame_id"] == 1)
        hub.publish(_frame(2, 2_000))
        old_status = _wait_for_status(
            worker,
            lambda status: status["latest_frame_id"] == 2
            and status["processed_frames"] == 2,
        )
        assert old_status["inference_fps"] is not None

        worker.stop()
        with worker._lock:
            worker._last_error = "old generation error"
        hub.publish(_frame(3, 3_000))

        worker.start()
        assert second_factory_started.wait(2.0)
        starting = worker.status()
        assert starting["state"] == InferenceState.STARTING.value
        assert starting["latest_frame_id"] is None
        assert starting["capture_timestamp_ns"] is None
        assert starting["inference_fps"] is None
        assert starting["latency_ms"] is None
        assert starting["detection_count"] is None
        assert starting["last_error"] is None
        assert starting["processed_frames"] == 0
        assert starting["skipped_frames"] == 0
        assert starting["artifact_name"] is None
        assert starting["model_sha256"] is None
        assert starting["confidence_threshold"] is None
        assert worker._after_frame_id is None

        release_second_factory.set()
        _wait_for_state(worker, InferenceState.RUNNING)
        assert worker._after_frame_id == 3
        hub.publish(_frame(4, 4_000))
        new_result = _wait_for_latest_result(worker)
        assert new_result.frame_id == 4
        assert worker.status()["processed_frames"] == 1
        assert worker.status()["skipped_frames"] == 0
        assert factory_calls == 2
    finally:
        release_second_factory.set()
        worker.stop()


def test_failed_generation_restarts_only_on_explicit_start(
    tmp_path: Path, fake_contract: SimpleNamespace
) -> None:
    hub = FrameHub()
    factory_calls = 0
    second_factory_started = threading.Event()
    release_second_factory = threading.Event()
    failure_state_visible = threading.Event()
    release_failure_handler = threading.Event()

    def factory(_onnx_path: Path, _report_path: Path) -> FakePredictor:
        nonlocal factory_calls
        factory_calls += 1
        if factory_calls == 1:
            raise RuntimeError("first generation failed")
        second_factory_started.set()
        assert release_second_factory.wait(2.0)
        return FakePredictor(fake_contract)

    worker = _worker(tmp_path, hub, factory)
    original_set_state = worker._set_state_locked

    def hold_failed_generation(state: InferenceState) -> None:
        original_set_state(state)
        if state == InferenceState.FAILED:
            failure_state_visible.set()
            worker._lock.release()
            try:
                assert release_failure_handler.wait(2.0)
            finally:
                worker._lock.acquire()

    worker._set_state_locked = hold_failed_generation  # type: ignore[method-assign]
    worker.start()
    first_thread = worker._thread
    try:
        assert failure_state_visible.wait(2.0)
        assert worker.status()["state"] == InferenceState.FAILED.value
        assert factory_calls == 1
        assert not second_factory_started.is_set()

        worker.start()
        assert second_factory_started.wait(2.0)
        release_failure_handler.set()
        assert first_thread is not None
        first_thread.join(2.0)
        release_second_factory.set()
        _wait_for_state(worker, InferenceState.RUNNING)
        hub.publish(_frame(10, 10_000))
        result = _wait_for_latest_result(worker)

        assert factory_calls == 2
        assert result.frame_id == 10
        assert worker.status()["last_error"] is None
    finally:
        release_failure_handler.set()
        release_second_factory.set()
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
        capture_timestamp_ns=1_000_000,
        width=1,
        height=1,
        pixel_format=PixelFormat.BGR8,
        image=source_image,
    )
    clock_values = iter((1_001_000, 1_002_500))
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
        assert result.inference_started_ns == 1_001_000
        assert result.inference_finished_ns == 1_002_500
        assert result.latency_ms == 0.0025
        assert result.model_identity == EXPECTED_MODEL_IDENTITY
        assert result.model_identity.onnx_sha256 == EXPECTED_CONTRACT["onnx_sha256"]
        assert isinstance(result.detections, tuple)
        assert isinstance(result.detections[0], Detection)
        status = worker.status()
        assert status["latest_frame_id"] == frame.frame_id
        assert status["capture_timestamp_ns"] == frame.capture_timestamp_ns
        assert status["latency_ms"] == result.latency_ms
        assert status["detection_count"] == len(result.detections)
        assert status["inference_fps"] is None
        assert status["processed_frames"] == 1
        assert status["skipped_frames"] == 0
        assert "processing_ms" not in status
    finally:
        worker.stop()


def test_worker_skips_replaced_frames_while_predictor_is_blocked(
    tmp_path: Path, fake_contract: SimpleNamespace
) -> None:
    class BlockingPredictor(FakePredictor):
        def __init__(self, contract: object) -> None:
            super().__init__(contract)
            self.frame_100_started = threading.Event()
            self.release_frame_100 = threading.Event()
            self.received_frame_ids: list[int] = []

        def predict_rgb_image(self, image: np.ndarray) -> SimpleNamespace:
            self.received_frame_ids.append(int(image[0, 0, 2]))
            if len(self.received_frame_ids) == 1:
                self.frame_100_started.set()
                assert self.release_frame_100.wait(2.0)
            return super().predict_rgb_image(image)

    hub = FrameHub()
    hub.publish(_frame(99, 1_000))
    predictor_holder: list[BlockingPredictor] = []

    def factory(_onnx_path: Path, _report_path: Path) -> BlockingPredictor:
        predictor = BlockingPredictor(fake_contract)
        predictor_holder.append(predictor)
        return predictor

    worker = _worker(tmp_path, hub, factory)
    worker.start()
    try:
        _wait_for_state(worker, InferenceState.RUNNING)
        assert worker._after_frame_id == 99

        hub.publish(_frame(100, 1_001))
        assert predictor_holder[0].frame_100_started.wait(2.0)
        hub.publish(_frame(101, 1_002))
        hub.publish(_frame(102, 1_003))
        hub.publish(_frame(103, 1_004))

        predictor_holder[0].release_frame_100.set()
        status = _wait_for_status(
            worker,
            lambda current: current["latest_frame_id"] == 103
            and current["processed_frames"] == 2,
        )

        assert predictor_holder[0].received_frame_ids == [100, 103]
        assert status["processed_frames"] == 2
        assert status["skipped_frames"] == 2
    finally:
        predictor = predictor_holder[0] if predictor_holder else None
        if predictor is not None:
            predictor.release_frame_100.set()
        worker.stop()


def test_worker_inference_fps_uses_bounded_completion_window(
    tmp_path: Path, fake_contract: SimpleNamespace
) -> None:
    hub = FrameHub()
    clock_values = iter(
        (
            1_000_000,
            2_000_000,
            3_000_000,
            4_000_000,
            5_000_000,
            8_000_000,
        )
    )
    worker = InferenceWorker(
        hub,
        onnx_path=tmp_path / "model.onnx",
        report_path=tmp_path / "report.json",
        predictor_factory=lambda *_args: FakePredictor(fake_contract),
        clock_ns=lambda: next(clock_values),
        wait_timeout=0.01,
        fps_window_size=2,
    )
    worker.start()
    try:
        _wait_for_state(worker, InferenceState.RUNNING)

        hub.publish(_frame(1, 1))
        first = _wait_for_status(worker, lambda status: status["latest_frame_id"] == 1)
        assert first["inference_fps"] is None

        hub.publish(_frame(2, 1))
        second = _wait_for_status(worker, lambda status: status["latest_frame_id"] == 2)
        assert second["inference_fps"] == pytest.approx(500.0)

        hub.publish(_frame(3, 1))
        third = _wait_for_status(worker, lambda status: status["latest_frame_id"] == 3)
        assert third["inference_fps"] == pytest.approx(250.0)
    finally:
        worker.stop()


def test_worker_inference_fps_is_none_when_completion_elapsed_is_nonpositive(
    tmp_path: Path, fake_contract: SimpleNamespace
) -> None:
    hub = FrameHub()
    clock_values = iter((1, 2, 3, 2))
    worker = InferenceWorker(
        hub,
        onnx_path=tmp_path / "model.onnx",
        report_path=tmp_path / "report.json",
        predictor_factory=lambda *_args: FakePredictor(fake_contract),
        clock_ns=lambda: next(clock_values),
        wait_timeout=0.01,
        fps_window_size=2,
    )
    worker.start()
    try:
        _wait_for_state(worker, InferenceState.RUNNING)
        hub.publish(_frame(1, 1))
        _wait_for_status(worker, lambda status: status["latest_frame_id"] == 1)
        hub.publish(_frame(2, 1))
        status = _wait_for_status(worker, lambda status: status["latest_frame_id"] == 2)
        assert status["inference_fps"] is None
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


def test_worker_fences_frame_hub_wait_exception_and_enters_failed(
    tmp_path: Path, fake_contract: SimpleNamespace
) -> None:
    class FailingHub(FrameHub):
        def wait_for_newer(self, *_args: object, **_kwargs: object) -> None:
            raise RuntimeError("frame hub boom")

    worker = _worker(
        tmp_path,
        FailingHub(),
        lambda *_args: FakePredictor(fake_contract),
    )
    worker.start()
    try:
        failed = _wait_for_state(worker, InferenceState.FAILED)
        assert failed["last_error"] == "frame hub boom"
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


def test_worker_missing_model_failure_is_terminal_without_framehub_activity(
    tmp_path: Path,
) -> None:
    class RecordingHub(FrameHub):
        def __init__(self) -> None:
            super().__init__()
            self.snapshot_calls = 0
            self.wait_calls = 0

        def snapshot(self):
            self.snapshot_calls += 1
            return super().snapshot()

        def wait_for_newer(self, *args: object, **kwargs: object):
            self.wait_calls += 1
            return super().wait_for_newer(*args, **kwargs)

    hub = RecordingHub()
    factory_calls: list[tuple[Path, Path]] = []
    factory_finished = threading.Event()

    def missing_model_factory(onnx_path: Path, report_path: Path) -> FakePredictor:
        factory_calls.append((onnx_path, report_path))
        factory_finished.set()
        raise FileNotFoundError("model or sidecar missing")

    worker = _worker(tmp_path, hub, missing_model_factory)
    worker.start()
    try:
        assert factory_finished.wait(2.0)
        failed = _wait_for_status(
            worker,
            lambda status: status["state"] == InferenceState.FAILED.value
            and status["last_error"] == "model or sidecar missing",
        )
        thread = worker._thread
        assert thread is not None
        thread.join(2.0)

        assert not thread.is_alive()
        assert failed["artifact_name"] is None
        assert failed["model_sha256"] is None
        assert factory_calls == [
            (tmp_path / "model.onnx", tmp_path / "report.json")
        ]
        assert hub.snapshot_calls == 0
        assert hub.wait_calls == 0
    finally:
        worker.stop()


def test_runtime_predictor_failure_preserves_published_result(
    tmp_path: Path, fake_contract: SimpleNamespace
) -> None:
    class FailOnSecondPrediction(FakePredictor):
        def __init__(self, contract: object) -> None:
            super().__init__(contract)
            self.prediction_count = 0
            self.second_prediction_started = threading.Event()
            self.release_second_prediction = threading.Event()

        def predict_rgb_image(self, image: np.ndarray) -> SimpleNamespace:
            self.prediction_count += 1
            if self.prediction_count == 2:
                self.second_prediction_started.set()
                assert self.release_second_prediction.wait(2.0)
                raise RuntimeError("predictor boom after result")
            return super().predict_rgb_image(image)

    predictor_holder: list[FailOnSecondPrediction] = []

    def factory(_onnx_path: Path, _report_path: Path) -> FailOnSecondPrediction:
        predictor = FailOnSecondPrediction(fake_contract)
        predictor_holder.append(predictor)
        return predictor

    hub = FrameHub()
    worker = _worker(tmp_path, hub, factory)
    worker.start()
    try:
        _wait_for_state(worker, InferenceState.RUNNING)
        hub.publish(_frame(1, 1_000))
        published_result = _wait_for_latest_result(worker)

        hub.publish(_frame(2, 2_000))
        assert predictor_holder[0].second_prediction_started.wait(2.0)
        predictor_holder[0].release_second_prediction.set()

        failed = _wait_for_status(
            worker,
            lambda status: status["state"] == InferenceState.FAILED.value
            and status["last_error"] == "predictor boom after result",
        )

        assert worker.latest_result() is published_result
        assert failed["latest_frame_id"] == published_result.frame_id
        assert failed["capture_timestamp_ns"] == published_result.capture_timestamp_ns
        assert failed["processed_frames"] == 1
        assert failed["last_error"] == "predictor boom after result"
    finally:
        if predictor_holder:
            predictor_holder[0].release_second_prediction.set()
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
