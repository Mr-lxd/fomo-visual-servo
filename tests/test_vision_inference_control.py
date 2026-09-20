from __future__ import annotations

from dataclasses import FrozenInstanceError, is_dataclass
import logging
import threading
from typing import Any

import pytest

import fomo_servo.vision.inference_control as control_module
from fomo_servo.vision.inference_control import (
    InferenceActionResult,
    InferenceControl,
    InferenceControlError,
)


class _FakeWorker:
    def __init__(self, state: str = "disabled") -> None:
        self.state = state
        self.last_error: str | None = (
            "predictor failed" if state == "failed" else None
        )
        self.start_calls = 0
        self.request_stop_calls = 0
        self.stop_calls = 0
        self.reset_disabled_calls = 0
        self.start_error: BaseException | None = None
        self.stop_error: BaseException | None = None
        self.reset_error: BaseException | None = None
        self.stop_started = threading.Event()
        self.stop_finished = threading.Event()
        self.reset_finished = threading.Event()
        self.release_stop = threading.Event()
        self.block_stop = False

    def status(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "artifact_name": None,
            "model_sha256": None,
            "confidence_threshold": None,
            "latest_frame_id": None,
            "capture_timestamp_ns": None,
            "processed_frames": 0,
            "skipped_frames": 0,
            "inference_fps": None,
            "latency_ms": None,
            "detection_count": None,
            "last_error": self.last_error,
        }

    def start(self) -> None:
        self.start_calls += 1
        if self.start_error is not None:
            raise self.start_error
        self.state = "starting"
        self.last_error = None

    def request_stop(self) -> None:
        self.request_stop_calls += 1

    def stop(self) -> None:
        self.stop_calls += 1
        self.stop_started.set()
        try:
            if self.block_stop:
                assert self.release_stop.wait(2.0)
            if self.stop_error is not None:
                raise self.stop_error
            if self.state != "failed":
                self.state = "disabled"
                self.last_error = None
        finally:
            self.stop_finished.set()

    def reset_disabled(self) -> None:
        self.reset_disabled_calls += 1
        try:
            if self.reset_error is not None:
                raise self.reset_error
            self.state = "disabled"
            self.last_error = None
        finally:
            self.reset_finished.set()


def _control(
    worker: _FakeWorker | None,
    *,
    live: dict[str, bool] | None = None,
    camera: dict[str, bool] | None = None,
) -> tuple[InferenceControl, dict[str, bool], dict[str, bool]]:
    live = {"value": True} if live is None else live
    camera = {"value": True} if camera is None else camera
    control = InferenceControl(
        worker,  # type: ignore[arg-type]
        is_live=lambda: live["value"],
        camera_running=lambda: camera["value"],
    )
    control.open_actions()
    return control, live, camera


def _finish(control: InferenceControl, worker: _FakeWorker | None) -> None:
    if worker is not None:
        worker.release_stop.set()
    control.begin_shutdown()
    control.finish_shutdown()


def _expect_error(
    expected_code: str,
    expected_status: int,
    callback,
) -> None:
    with pytest.raises(InferenceControlError) as raised:
        callback()
    assert raised.value.code == expected_code
    assert raised.value.http_status == expected_status


def test_action_result_is_frozen_http_mapping_value() -> None:
    result = InferenceActionResult(202, "inference/start", "accepted")

    assert is_dataclass(result)
    assert result.http_status == 202
    with pytest.raises(FrozenInstanceError):
        result.outcome = "changed"  # type: ignore[misc]


def test_unconfigured_start_and_stop_follow_frozen_compatibility_table() -> None:
    control, _live, _camera = _control(None)
    try:
        status = control.status()
        assert status == {
            "configured": False,
            "control_supported": True,
            "operation": None,
            "state": "disabled",
            "artifact_name": None,
            "model_sha256": None,
            "confidence_threshold": None,
            "latest_frame_id": None,
            "capture_timestamp_ns": None,
            "processed_frames": 0,
            "skipped_frames": 0,
            "inference_fps": None,
            "latency_ms": None,
            "detection_count": None,
            "last_error": None,
        }
        _expect_error(
            "inference_not_configured",
            409,
            control.start_inference,
        )
        assert control.stop_inference() == InferenceActionResult(
            200,
            "inference/stop",
            "already_disabled",
        )
    finally:
        _finish(control, None)


def test_disabled_start_checks_live_camera_then_creates_one_generation() -> None:
    worker = _FakeWorker()
    live = {"value": False}
    camera = {"value": True}
    control, live, camera = _control(worker, live=live, camera=camera)
    try:
        _expect_error("vision_not_live", 409, control.start_inference)
        assert worker.start_calls == 0

        live["value"] = True
        camera["value"] = False
        _expect_error("camera_not_running", 409, control.start_inference)
        assert worker.start_calls == 0

        camera["value"] = True
        assert control.start_inference() == InferenceActionResult(
            202,
            "inference/start",
            "accepted",
        )
        assert worker.start_calls == 1
        assert control.status()["state"] == "starting"
        assert control.start_inference() == InferenceActionResult(
            200,
            "inference/start",
            "already_starting",
        )
        assert worker.start_calls == 1

        worker.state = "running"
        assert control.start_inference() == InferenceActionResult(
            200,
            "inference/start",
            "already_running",
        )
        assert worker.start_calls == 1
    finally:
        _finish(control, worker)


def test_invalid_worker_state_precedes_live_and_camera_admission_checks() -> None:
    worker = _FakeWorker("corrupt")
    control, _live, _camera = _control(
        worker,
        live={"value": False},
        camera={"value": False},
    )
    try:
        _expect_error("invalid_inference_state", 409, control.start_inference)
        _expect_error("invalid_inference_state", 409, control.stop_inference)
    finally:
        _finish(control, worker)


def test_stopping_is_async_busy_and_uses_one_non_daemon_coordinator() -> None:
    worker = _FakeWorker("running")
    worker.block_stop = True
    control, _live, _camera = _control(worker)
    try:
        assert control.stop_inference() == InferenceActionResult(
            202,
            "inference/stop",
            "accepted",
        )
        assert worker.stop_started.wait(2.0)
        coordinator = control._coordinator_thread
        assert coordinator is not None
        assert coordinator.daemon is False
        assert control.status()["operation"] == "stopping"

        _expect_error("inference_busy", 409, control.start_inference)
        assert control.stop_inference() == InferenceActionResult(
            200,
            "inference/stop",
            "already_stopping",
        )
        assert worker.request_stop_calls == 1

        worker.release_stop.set()
        assert worker.reset_finished.wait(2.0)
        assert control.status()["operation"] is None
        assert control.status()["state"] == "disabled"
        assert worker.reset_disabled_calls == 1
    finally:
        _finish(control, worker)


def test_failed_start_retries_after_old_generation_join_once() -> None:
    worker = _FakeWorker("failed")
    worker.block_stop = True
    control, _live, _camera = _control(worker)
    try:
        assert control.start_inference() == InferenceActionResult(
            202,
            "inference/start",
            "accepted",
        )
        assert worker.stop_started.wait(2.0)
        assert control.status()["operation"] == "retrying"
        assert control.start_inference() == InferenceActionResult(
            200,
            "inference/start",
            "already_retrying",
        )
        _expect_error("inference_busy", 409, control.stop_inference)
        assert worker.stop_calls == 1
        assert worker.start_calls == 0

        worker.release_stop.set()
        assert worker.stop_finished.wait(2.0)
        with control._condition:
            assert control._condition.wait_for(
                lambda: worker.start_calls == 1,
                timeout=2.0,
            )
        assert worker.stop_calls == 1
        assert control.status()["operation"] is None
        assert control.status()["state"] == "starting"
    finally:
        _finish(control, worker)


def test_retry_camera_loss_does_not_start_and_waits_for_new_manual_action() -> None:
    worker = _FakeWorker("failed")
    worker.block_stop = True
    camera = {"value": True}
    control, _live, camera = _control(worker, camera=camera)
    try:
        assert control.start_inference().outcome == "accepted"
        assert worker.stop_started.wait(2.0)
        camera["value"] = False
        worker.release_stop.set()
        assert worker.stop_finished.wait(2.0)
        with control._condition:
            assert control._condition.wait_for(
                lambda: control.status()["operation"] is None,
                timeout=2.0,
            )

        status = control.status()
        assert worker.start_calls == 0
        assert status["state"] == "failed"
        assert status["operation"] is None
        assert "camera" in status["last_error"]

        camera["value"] = True
        assert control.start_inference().outcome == "accepted"
        with control._condition:
            assert control._condition.wait_for(
                lambda: worker.stop_calls == 2,
                timeout=2.0,
            )
    finally:
        _finish(control, worker)


def test_cleanup_failure_latches_error_and_blocks_all_new_actions(
    caplog: pytest.LogCaptureFixture,
) -> None:
    worker = _FakeWorker("running")
    worker.stop_error = RuntimeError("worker join failure")
    control, _live, _camera = _control(worker)
    caplog.set_level(logging.ERROR, logger=control_module.__name__)
    try:
        assert control.stop_inference().outcome == "accepted"
        assert worker.stop_finished.wait(2.0)
        with control._condition:
            assert control._condition.wait_for(
                lambda: "worker join failure"
                in str(control.status()["last_error"]),
                timeout=2.0,
            )

        status = control.status()
        assert status["operation"] == "stopping"
        assert "worker join failure" in status["last_error"]
        assert "worker join failure" in caplog.text
        _expect_error("inference_control_failed", 409, control.start_inference)
        _expect_error("inference_control_failed", 409, control.stop_inference)
    finally:
        worker.stop_error = None
        _finish(control, worker)


def test_begin_shutdown_prevents_retry_from_starting_after_cleanup() -> None:
    worker = _FakeWorker("failed")
    worker.block_stop = True
    control, _live, _camera = _control(worker)
    try:
        assert control.start_inference().outcome == "accepted"
        assert worker.stop_started.wait(2.0)
        control.begin_shutdown()
        worker.release_stop.set()
        assert worker.stop_finished.wait(2.0)
        control.finish_shutdown()
        assert worker.start_calls == 0
        _expect_error("service_shutting_down", 503, control.start_inference)
        _expect_error("service_shutting_down", 503, control.stop_inference)
    finally:
        worker.release_stop.set()


def test_coordinator_thread_start_failure_rolls_back_without_stop_signal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = _FakeWorker("running")
    control, _live, _camera = _control(worker)

    def fail_thread_start(_thread: threading.Thread) -> None:
        raise RuntimeError("coordinator thread start failed")

    monkeypatch.setattr(control_module.threading.Thread, "start", fail_thread_start)
    try:
        _expect_error("internal_error", 500, control.stop_inference)
        assert worker.request_stop_calls == 0
        assert control.status()["operation"] is None
        assert control._coordinator_thread is None
    finally:
        _finish(control, worker)


def test_status_remains_available_while_coordinator_waits_for_stop_join() -> None:
    worker = _FakeWorker("running")
    worker.block_stop = True
    control, _live, _camera = _control(worker)
    reader_done = threading.Event()
    reader_result: dict[str, Any] = {}
    reader: threading.Thread | None = None
    try:
        assert control.stop_inference().outcome == "accepted"
        assert worker.stop_started.wait(2.0)

        def read_status() -> None:
            reader_result.update(control.status())
            reader_done.set()

        reader = threading.Thread(target=read_status, daemon=False)
        reader.start()
        assert reader_done.wait(1.0)
        assert reader_result["operation"] == "stopping"
    finally:
        worker.release_stop.set()
        if reader is not None:
            reader.join(2.0)
        _finish(control, worker)


def test_failed_worker_has_no_background_retry() -> None:
    worker = _FakeWorker("failed")
    control, _live, _camera = _control(worker)
    try:
        first = control.status()
        second = control.status()

        assert first["state"] == "failed"
        assert second["state"] == "failed"
        assert worker.start_calls == 0
        assert worker.stop_calls == 0
        assert control._coordinator_thread is None
    finally:
        _finish(control, worker)


def test_failed_clear_error_stops_to_clean_disabled_without_losing_configuration() -> None:
    worker = _FakeWorker("failed")
    control, _live, _camera = _control(worker)
    try:
        assert control.stop_inference() == InferenceActionResult(
            202,
            "inference/stop",
            "accepted",
        )
        assert worker.reset_finished.wait(2.0)

        status = control.status()
        assert status["configured"] is True
        assert status["state"] == "disabled"
        assert status["operation"] is None
        assert status["last_error"] is None
        assert worker.stop_calls == 1
        assert worker.reset_disabled_calls == 1
    finally:
        _finish(control, worker)


def test_ten_controlled_start_stop_cycles_have_no_generation_leak() -> None:
    worker = _FakeWorker()
    control, _live, _camera = _control(worker)
    try:
        for generation in range(1, 11):
            assert control.start_inference() == InferenceActionResult(
                202,
                "inference/start",
                "accepted",
            )
            assert worker.start_calls == generation
            worker.state = "running"

            assert control.stop_inference().outcome == "accepted"
            with control._condition:
                assert control._condition.wait_for(
                    lambda: control.status()["operation"] is None,
                    timeout=2.0,
                )
            status = control.status()
            assert status["state"] == "disabled"
            assert status["latest_frame_id"] is None
            assert status["processed_frames"] == 0
            assert status["skipped_frames"] == 0
            assert status["last_error"] is None

        assert worker.stop_calls == 10
        assert worker.reset_disabled_calls == 10
        coordinator = control._coordinator_thread
        assert coordinator is not None
    finally:
        _finish(control, worker)


def test_concurrent_start_stop_retry_never_admits_more_than_one_operation() -> None:
    worker = _FakeWorker("failed")
    worker.block_stop = True
    control, _live, _camera = _control(worker)
    barrier = threading.Barrier(3)
    outcomes: list[object] = []
    errors: list[InferenceControlError] = []

    def invoke(callback) -> None:
        try:
            barrier.wait(timeout=2.0)
            outcomes.append(callback())
        except InferenceControlError as error:
            errors.append(error)

    start_thread = threading.Thread(
        target=lambda: invoke(control.start_inference),
        daemon=False,
    )
    stop_thread = threading.Thread(
        target=lambda: invoke(control.stop_inference),
        daemon=False,
    )
    try:
        start_thread.start()
        stop_thread.start()
        barrier.wait(timeout=2.0)
        assert worker.stop_started.wait(2.0)
        coordinator = control._coordinator_thread
        assert coordinator is not None
        assert worker.request_stop_calls == 1
        assert len(outcomes) + len(errors) == 2
        assert len(errors) == 1
        assert errors[0].code == "inference_busy"

        worker.release_stop.set()
        start_thread.join(2.0)
        stop_thread.join(2.0)
        with control._condition:
            assert control._condition.wait_for(
                lambda: control.status()["operation"] is None,
                timeout=2.0,
            )
        assert worker.start_calls <= 1
        assert control._coordinator_thread is coordinator
    finally:
        worker.release_stop.set()
        start_thread.join(2.0)
        stop_thread.join(2.0)
        _finish(control, worker)


@pytest.mark.parametrize("state", ["disabled", "starting", "running", "failed"])
def test_shutdown_reclaims_worker_in_disabled_starting_running_and_failed(
    state: str,
) -> None:
    worker = _FakeWorker(state)
    control, _live, _camera = _control(worker)

    control.begin_shutdown()
    control.finish_shutdown()

    assert worker.request_stop_calls >= 1
    assert worker.stop_calls == 1
    _expect_error("service_shutting_down", 503, control.start_inference)
    _expect_error("service_shutting_down", 503, control.stop_inference)


def test_shutdown_reclaims_worker_while_stopping_without_starting_new_generation() -> None:
    worker = _FakeWorker("running")
    worker.block_stop = True
    control, _live, _camera = _control(worker)
    try:
        assert control.stop_inference().outcome == "accepted"
        assert worker.stop_started.wait(2.0)
        control.begin_shutdown()
        worker.release_stop.set()
        control.finish_shutdown()

        assert worker.start_calls == 0
        assert worker.reset_disabled_calls == 1
    finally:
        worker.release_stop.set()
