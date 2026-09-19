from __future__ import annotations

import json
import threading
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import pytest

import fomo_servo.vision.service as service_module
from fomo_servo.vision.mode import VisionMode
from fomo_servo.vision.service import VisionService, VisionServiceConfig


class _LiveCapture:
    def __init__(self) -> None:
        self.opened = True
        self.release_count = 0
        self.read_count = 0
        self.set_calls: list[tuple[int, float]] = []

    def isOpened(self) -> bool:
        return self.opened

    def set(self, prop: int, value: float) -> bool:
        self.set_calls.append((prop, value))
        return True

    def get(self, prop: int) -> float:
        values = {
            cv2.CAP_PROP_FRAME_WIDTH: 4.0,
            cv2.CAP_PROP_FRAME_HEIGHT: 3.0,
            cv2.CAP_PROP_FPS: 25.0,
            cv2.CAP_PROP_FOURCC: float(
                ord("Y")
                | (ord("U") << 8)
                | (ord("Y") << 16)
                | (ord("V") << 24)
            ),
        }
        return values.get(prop, 0.0)

    def read(self):
        if not self.opened:
            return False, None
        time.sleep(0.003)
        self.read_count += 1
        return True, np.full(
            (3, 4, 3),
            self.read_count % 255,
            dtype=np.uint8,
        )

    def release(self) -> None:
        self.release_count += 1
        self.opened = False


def _json_request(port: int, path: str, method: str = "GET") -> dict:
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        method=method,
        data=b"" if method == "POST" else None,
    )
    with urllib.request.urlopen(request, timeout=2.0) as response:
        assert response.status == 200
        return json.loads(response.read())


class _RecordingInferenceWorker:
    instances: list["_RecordingInferenceWorker"] = []
    event_log: list[str] = []

    def __init__(self, hub, onnx_path, report_path) -> None:
        self.events = type(self).event_log
        self.onnx_path = onnx_path
        self.report_path = report_path
        self.failed = False
        self.stop_count = 0
        self.stop_error: BaseException | None = None
        type(self).instances.append(self)

    def start(self) -> None:
        self.events.append("inference.start")

    def stop(self) -> None:
        self.stop_count += 1
        self.events.append("inference.stop")
        if self.stop_error is not None:
            raise self.stop_error


class _RecordingModeManager:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.mode = VisionMode.OFF

    def set_mode(self, mode: VisionMode) -> None:
        self.events.append(f"mode.{mode.value}")
        self.mode = mode

    def shutdown(self) -> None:
        self.events.append("mode.shutdown")
        self.mode = VisionMode.OFF


class _RecordingControlServer:
    def __init__(self, events: list[str], error: BaseException | None = None) -> None:
        self.events = events
        self.error = error

    def start(self) -> None:
        self.events.append("control.start")
        if self.error is not None:
            raise self.error

    def stop(self) -> None:
        self.events.append("control.stop")


class _RecordingCaptureManager:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def shutdown(self) -> None:
        self.events.append("capture.shutdown")


def _service_with_lifecycle_doubles(
    monkeypatch,
    tmp_path: Path,
    events: list[str],
    *,
    control_error: BaseException | None = None,
) -> VisionService:
    _RecordingInferenceWorker.instances.clear()
    _RecordingInferenceWorker.event_log = events
    monkeypatch.setattr(
        service_module,
        "InferenceWorker",
        _RecordingInferenceWorker,
    )
    service = VisionService(
        VisionServiceConfig(
            inference_onnx=tmp_path / "model.onnx",
            inference_report=tmp_path / "report.json",
            capture_output_root=tmp_path,
            capture_min_free_bytes=0,
        )
    )
    service.mode_manager = _RecordingModeManager(events)
    service.control_server = _RecordingControlServer(
        events,
        error=control_error,
    )
    service.capture_manager = _RecordingCaptureManager(events)
    return service


@pytest.mark.parametrize(
    "config_kwargs",
    [
        {"inference_onnx": Path("model.onnx")},
        {"inference_report": Path("report.json")},
    ],
)
def test_inference_paths_must_be_supplied_as_a_pair_before_camera_start(
    monkeypatch,
    config_kwargs: dict[str, Path],
) -> None:
    camera_starts: list[str] = []

    class _CameraOwnerThatMustNotStart:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def start(self) -> None:
            camera_starts.append("camera.start")

    monkeypatch.setattr(
        service_module,
        "CameraOwner",
        _CameraOwnerThatMustNotStart,
    )

    with pytest.raises(ValueError, match="inference_onnx and inference_report"):
        VisionService(VisionServiceConfig(**config_kwargs))

    assert camera_starts == []


def test_inference_worker_is_not_constructed_when_inference_is_disabled(
    monkeypatch,
) -> None:
    _RecordingInferenceWorker.instances.clear()
    monkeypatch.setattr(
        service_module,
        "InferenceWorker",
        _RecordingInferenceWorker,
    )

    VisionService(VisionServiceConfig())

    assert _RecordingInferenceWorker.instances == []


def test_configured_inference_constructs_exactly_one_worker(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _RecordingInferenceWorker.instances.clear()
    monkeypatch.setattr(
        service_module,
        "InferenceWorker",
        _RecordingInferenceWorker,
    )

    VisionService(
        VisionServiceConfig(
            inference_onnx=tmp_path / "model.onnx",
            inference_report=tmp_path / "report.json",
        )
    )

    assert len(_RecordingInferenceWorker.instances) == 1
    worker = _RecordingInferenceWorker.instances[0]
    assert worker.onnx_path == tmp_path / "model.onnx"
    assert worker.report_path == tmp_path / "report.json"


def test_start_live_orders_mode_worker_then_control(
    monkeypatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    service = _service_with_lifecycle_doubles(monkeypatch, tmp_path, events)

    service.start_live()

    assert events == [
        "mode.live",
        "inference.start",
        "control.start",
    ]


def test_async_worker_failure_does_not_roll_back_live_mode(
    monkeypatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    service = _service_with_lifecycle_doubles(monkeypatch, tmp_path, events)
    worker = _RecordingInferenceWorker.instances[0]
    failure_condition = threading.Condition()
    start_returned = threading.Event()

    def fail_asynchronously() -> None:
        worker.events.append("inference.start")
        def fail_after_start_returns() -> None:
            start_returned.wait()
            with failure_condition:
                worker.failed = True
                failure_condition.notify_all()

        worker.failure_thread = threading.Thread(
            target=fail_after_start_returns,
            name="test-inference-failure",
        )
        worker.failure_thread.start()

    worker.start = fail_asynchronously  # type: ignore[method-assign]

    service.start_live()
    start_returned.set()
    with failure_condition:
        assert failure_condition.wait_for(lambda: worker.failed, timeout=1.0)
    worker.failure_thread.join(timeout=1.0)

    assert worker.failed is True
    assert service.mode_manager.mode is VisionMode.LIVE
    assert "mode.shutdown" not in events
    assert events == [
        "mode.live",
        "inference.start",
        "control.start",
    ]


def test_start_live_is_idempotent_when_mode_is_already_live(
    monkeypatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    service = _service_with_lifecycle_doubles(monkeypatch, tmp_path, events)

    service.start_live()
    service.start_live()

    assert events == [
        "mode.live",
        "inference.start",
        "control.start",
    ]


def test_control_start_failure_rolls_back_mode_and_worker(
    monkeypatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    service = _service_with_lifecycle_doubles(
        monkeypatch,
        tmp_path,
        events,
        control_error=RuntimeError("control start failed"),
    )
    worker = _RecordingInferenceWorker.instances[0]

    with pytest.raises(RuntimeError, match="control start failed"):
        service.start_live()

    assert worker.stop_count == 1
    assert events == [
        "mode.live",
        "inference.start",
        "control.start",
        "inference.stop",
        "mode.shutdown",
    ]


def test_control_start_failure_preserves_original_when_worker_cleanup_fails(
    monkeypatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    service = _service_with_lifecycle_doubles(
        monkeypatch,
        tmp_path,
        events,
        control_error=RuntimeError("control start failed"),
    )
    worker = _RecordingInferenceWorker.instances[0]
    worker.stop_error = RuntimeError("worker stop failed")

    with pytest.raises(RuntimeError, match="control start failed"):
        service.start_live()

    assert worker.stop_count == 1
    assert events == [
        "mode.live",
        "inference.start",
        "control.start",
        "inference.stop",
        "mode.shutdown",
    ]


def test_shutdown_orders_control_capture_worker_then_mode(
    monkeypatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    service = _service_with_lifecycle_doubles(monkeypatch, tmp_path, events)
    worker = _RecordingInferenceWorker.instances[0]
    events.clear()

    service.shutdown()

    assert worker.stop_count == 1
    assert events == [
        "control.stop",
        "capture.shutdown",
        "inference.stop",
        "mode.shutdown",
    ]
def test_service_snapshot_reuses_single_camera_owner(tmp_path: Path) -> None:
    capture = _LiveCapture()
    factory_calls = 0

    def factory(_source):
        nonlocal factory_calls
        factory_calls += 1
        return capture

    service = VisionService(
        VisionServiceConfig(
            width=4,
            height=3,
            fps=25.0,
            port=0,
            control_port=0,
            capture_output_root=tmp_path,
            capture_min_free_bytes=0,
        ),
        capture_factory=factory,
    )

    try:
        service.start_live()
        deadline = time.monotonic() + 1.0
        while service.camera_owner.frames_captured < 8:
            assert time.monotonic() < deadline
            time.sleep(0.005)

        control_port = service.control_server.bound_port
        assert control_port is not None
        status = _json_request(
            control_port,
            "/api/v1/vision/status",
        )
        assert status["camera"]["running"] is True
        assert status["camera"]["observed_fps"] == 25.0
        assert status["camera"]["measured_capture_fps"] is not None
        assert status["camera"]["measured_capture_fps"] > 0.0
        before = status["camera"]["latest_frame_id"]

        snapshot = _json_request(
            control_port,
            "/api/v1/vision/snapshot",
            method="POST",
        )
        assert snapshot["ok"] is True
        assert snapshot["capture"]["snapshot_count"] == 1
        assert factory_calls == 1

        session_dir = Path(snapshot["capture"]["session_dir"])
        files = list((session_dir / "frames").glob("*.jpg"))
        assert len(files) == 1

        metadata = json.loads(
            (session_dir / "metadata.json").read_text(encoding="utf-8")
        )
        assert metadata["snapshots"][0]["frame_id"] > before
    finally:
        service.shutdown()

    assert capture.release_count == 1
    assert factory_calls == 1
    metadata = json.loads(
        next(tmp_path.glob("*/*/metadata.json")).read_text(encoding="utf-8")
    )
    assert metadata["closed"] is True
