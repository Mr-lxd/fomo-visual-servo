from __future__ import annotations

import json
import socket
import threading
import time
from types import SimpleNamespace
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import pytest

import fomo_servo.vision.service as service_module
from fomo_servo.vision.mode import VisionMode
from fomo_servo.vision.inference_worker import InferenceState, InferenceWorker
from fomo_servo.vision.service import VisionService, VisionServiceConfig


class _LiveCapture:
    def __init__(self) -> None:
        self.opened = True
        self.release_count = 0
        self.read_count = 0
        self.factory_calls = 0
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


def _http_request(
    port: int,
    path: str,
    *,
    method: str = "GET",
    body: bytes | None = None,
) -> tuple[int, dict]:
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        method=method,
        data=body,
    )
    try:
        with urllib.request.urlopen(request, timeout=2.0) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


_INFERENCE_CONTRACT = {
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


class _ServicePredictor:
    def __init__(self, contract: object, *, block_prediction: bool = False) -> None:
        self.contract = contract
        self.block_prediction = block_prediction
        self.prediction_started = threading.Event()
        self.release_prediction = threading.Event()

    def predict_rgb_image(self, _image: np.ndarray) -> SimpleNamespace:
        self.prediction_started.set()
        if self.block_prediction:
            assert self.release_prediction.wait(2.0)
        return SimpleNamespace(detections=())


def _live_service_with_fake_inference_worker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    predictor_factory,
) -> tuple[VisionService, _LiveCapture, list[InferenceWorker]]:
    workers: list[InferenceWorker] = []

    def worker_factory(
        hub, onnx_path, report_path, *, result_sink=None
    ) -> InferenceWorker:
        worker = InferenceWorker(
            hub,
            onnx_path,
            report_path,
            predictor_factory=predictor_factory,
            result_sink=result_sink,
            wait_timeout=0.01,
        )
        workers.append(worker)
        return worker

    monkeypatch.setattr(service_module, "InferenceWorker", worker_factory)
    capture = _LiveCapture()
    def capture_factory(_source):
        capture.factory_calls += 1
        return capture

    service = VisionService(
        VisionServiceConfig(
            width=4,
            height=3,
            fps=25.0,
            port=0,
            control_port=0,
            detection_port=0,
            capture_output_root=tmp_path,
            capture_min_free_bytes=0,
            inference_onnx=tmp_path / "model.onnx",
            inference_report=tmp_path / "report.json",
        ),
        capture_factory=capture_factory,
    )
    assert len(workers) == 1
    return service, capture, workers


class _RecordingInferenceWorker:
    instances: list["_RecordingInferenceWorker"] = []
    event_log: list[str] = []

    def __init__(self, hub, onnx_path, report_path, *, result_sink=None) -> None:
        self.events = type(self).event_log
        self.onnx_path = onnx_path
        self.report_path = report_path
        self.failed = False
        self.result_sink = result_sink
        self.state = "disabled"
        self.stop_count = 0
        self.request_stop_count = 0
        self.reset_disabled_count = 0
        self.stop_error: BaseException | None = None
        type(self).instances.append(self)

    def start(self) -> None:
        self.events.append("inference.start")
        self.state = "starting"

    def request_stop(self) -> None:
        self.request_stop_count += 1
        self.events.append("inference.request_stop")

    def reset_disabled(self) -> None:
        self.reset_disabled_count += 1
        self.events.append("inference.reset_disabled")
        self.state = "disabled"

    def status(self) -> dict:
        identity_is_validated = self.state == "running"
        return {
            "state": self.state,
            "artifact_name": "recording-model" if identity_is_validated else None,
            "model_sha256": "b" * 64 if identity_is_validated else None,
            "confidence_threshold": 0.4 if identity_is_validated else None,
            "latest_frame_id": None,
            "capture_timestamp_ns": None,
            "processed_frames": 0,
            "skipped_frames": 0,
            "inference_fps": None,
            "latency_ms": None,
            "detection_count": None,
            "last_error": None,
        }

    def stop(self) -> None:
        self.stop_count += 1
        self.events.append("inference.stop")
        self.state = "disabled"
        if self.stop_error is not None:
            raise self.stop_error


class _RecordingModeManager:
    def __init__(
        self,
        events: list[str],
        shutdown_error: BaseException | None = None,
    ) -> None:
        self.events = events
        self.mode = VisionMode.OFF
        self.shutdown_error = shutdown_error
        self.shutdown_count = 0

    def set_mode(self, mode: VisionMode) -> None:
        self.events.append(f"mode.{mode.value}")
        self.mode = mode

    def shutdown(self) -> None:
        self.shutdown_count += 1
        self.events.append("mode.shutdown")
        self.mode = VisionMode.OFF
        if self.shutdown_error is not None:
            raise self.shutdown_error


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


class _RecordingDetectionServer:
    def __init__(
        self,
        events: list[str],
        error: BaseException | None = None,
    ) -> None:
        self.events = events
        self.error = error
        self.bound_port: int | None = None
        self.start_count = 0
        self.stop_count = 0

    def start(self) -> "_RecordingDetectionServer":
        self.start_count += 1
        self.events.append("detection.start")
        if self.error is not None:
            raise self.error
        self.bound_port = 47012
        return self

    def stop(self) -> None:
        self.stop_count += 1
        self.events.append("detection.stop")
        self.bound_port = None


class _RecordingCameraOwner:
    @property
    def is_running(self) -> bool:
        return True


def _service_with_lifecycle_doubles(
    monkeypatch,
    tmp_path: Path,
    events: list[str],
    *,
    control_error: BaseException | None = None,
    mode_error: BaseException | None = None,
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
            detection_port=0,
            capture_output_root=tmp_path,
            capture_min_free_bytes=0,
        )
    )
    service.mode_manager = _RecordingModeManager(
        events,
        shutdown_error=mode_error,
    )
    service.control_server = _RecordingControlServer(
        events,
        error=control_error,
    )
    service.capture_manager = _RecordingCaptureManager(events)
    service.camera_owner = _RecordingCameraOwner()  # type: ignore[assignment]
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


def test_disabled_inference_is_exposed_in_service_status(tmp_path: Path) -> None:
    service = VisionService(
        VisionServiceConfig(
            control_port=0,
            capture_output_root=tmp_path,
            capture_min_free_bytes=0,
        )
    )

    status = service.control_server._status_payload()
    inference = status["inference"]

    assert inference == {
        "configured": False,
        "control_supported": True,
        "detection_stream_supported": True,
        "detection_stream_port": 47012,
        "detection_stream_version": 1,
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
        "vision_process_rss_bytes": inference["vision_process_rss_bytes"],
        "system_total_memory_bytes": inference["system_total_memory_bytes"],
    }


def test_configured_inference_status_is_read_from_single_worker(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _RecordingInferenceWorker.instances.clear()
    monkeypatch.setattr(
        service_module,
        "InferenceWorker",
        _RecordingInferenceWorker,
    )

    service = VisionService(
        VisionServiceConfig(
            inference_onnx=tmp_path / "model.onnx",
            inference_report=tmp_path / "report.json",
            control_port=0,
            capture_output_root=tmp_path,
            capture_min_free_bytes=0,
        )
    )

    status = service.control_server._status_payload()
    inference = status["inference"]

    assert inference == {
        **_RecordingInferenceWorker.instances[0].status(),
        "configured": True,
        "control_supported": True,
        "detection_stream_supported": True,
        "detection_stream_port": 47012,
        "detection_stream_version": 1,
        "operation": None,
        "vision_process_rss_bytes": inference["vision_process_rss_bytes"],
        "system_total_memory_bytes": inference["system_total_memory_bytes"],
    }
    assert len(_RecordingInferenceWorker.instances) == 1


def test_configured_service_stays_disabled_until_manual_start(
    monkeypatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    service = _service_with_lifecycle_doubles(monkeypatch, tmp_path, events)

    service.start_live()

    assert events == ["mode.live", "control.start"]
    inference = service._inference_status()
    assert inference == {
        "configured": True,
        "control_supported": True,
        "detection_stream_supported": True,
        "detection_stream_port": service.detection_server.bound_port,
        "detection_stream_version": 1,
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
        "vision_process_rss_bytes": inference["vision_process_rss_bytes"],
        "system_total_memory_bytes": inference["system_total_memory_bytes"],
    }

    accepted = service.start_inference()

    assert accepted.http_status == 202
    assert accepted.action == "inference/start"
    assert accepted.outcome == "accepted"
    assert events == ["mode.live", "control.start", "inference.start"]


def test_configured_service_stays_disabled_until_http_start_and_factory_runs_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    factory_calls = 0
    factory_started = threading.Event()
    release_factory = threading.Event()

    def predictor_factory(_onnx_path: Path, _report_path: Path) -> _ServicePredictor:
        nonlocal factory_calls
        factory_calls += 1
        factory_started.set()
        assert release_factory.wait(2.0)
        return _ServicePredictor(SimpleNamespace(**_INFERENCE_CONTRACT))

    service, capture, workers = _live_service_with_fake_inference_worker(
        monkeypatch,
        tmp_path,
        predictor_factory,
    )
    worker = workers[0]
    try:
        service.start_live()
        control_port = service.control_server.bound_port
        assert control_port is not None
        assert factory_calls == 0

        before_start = _json_request(control_port, "/api/v1/vision/status")
        assert before_start["inference"]["configured"] is True
        assert before_start["inference"]["state"] == "disabled"
        assert before_start["inference"]["operation"] is None
        assert before_start["inference"]["artifact_name"] is None

        code, acknowledgement = _http_request(
            control_port,
            "/api/v1/vision/inference/start",
            method="POST",
            body=b"{}",
        )
        assert code == 202
        assert acknowledgement == {
            "ok": True,
            "action": "inference/start",
            "outcome": "accepted",
        }
        assert factory_started.wait(2.0)
        assert factory_calls == 1

        starting = _json_request(control_port, "/api/v1/vision/status")
        assert starting["inference"]["state"] == "starting"
        assert starting["inference"]["artifact_name"] is None

        duplicate_code, duplicate = _http_request(
            control_port,
            "/api/v1/vision/inference/start",
            method="POST",
            body=b"{}",
        )
        assert duplicate_code == 200
        assert duplicate == {
            "ok": True,
            "action": "inference/start",
            "outcome": "already_starting",
        }
        assert factory_calls == 1

        release_factory.set()
        with worker._state_condition:
            assert worker._state_condition.wait_for(
                lambda: worker.status()["state"] == InferenceState.RUNNING.value,
                timeout=2.0,
            )
        running = _json_request(control_port, "/api/v1/vision/status")
        assert running["inference"]["state"] == "running"
        assert capture.release_count == 0
    finally:
        release_factory.set()
        service.shutdown()

    assert capture.release_count == 1
    assert capture.factory_calls == 1
    assert factory_calls == 1


def test_live_service_streams_metadata_for_exact_inference_source_frame(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FirstThenBlockPredictor(_ServicePredictor):
        def __init__(self) -> None:
            super().__init__(SimpleNamespace(**_INFERENCE_CONTRACT))
            self.calls = 0
            self.second_started = threading.Event()
            self.release_second = threading.Event()

        def predict_rgb_image(self, _image: np.ndarray) -> SimpleNamespace:
            self.calls += 1
            if self.calls == 2:
                self.second_started.set()
                assert self.release_second.wait(2.0)
            return SimpleNamespace(detections=())

    predictor = FirstThenBlockPredictor()
    service, _capture, workers = _live_service_with_fake_inference_worker(
        monkeypatch,
        tmp_path,
        lambda _onnx, _report: predictor,
    )
    worker = workers[0]
    client: socket.socket | None = None
    try:
        service.start_live()
        detection_port = service.detection_server.bound_port
        assert detection_port is not None
        client = socket.create_connection(
            ("127.0.0.1", detection_port),
            timeout=1.0,
        )
        assert service.detection_server.client_connected.wait(timeout=1.0)

        accepted = service.start_inference()
        assert accepted.http_status == 202
        with worker._state_condition:
            assert worker._state_condition.wait_for(
                lambda: worker.latest_result() is not None,
                timeout=2.0,
            )

        client.settimeout(2.0)
        raw = bytearray()
        while not raw.endswith(b"\n"):
            chunk = client.recv(4096)
            assert chunk
            raw.extend(chunk)
        metadata = json.loads(raw)

        result = worker.latest_result()
        assert result is not None
        assert metadata["frame_id"] == result.frame_id
        assert metadata["capture_timestamp_ns"] == result.capture_timestamp_ns
        assert metadata["width"] == result.frame_width == 4
        assert metadata["height"] == result.frame_height == 3
        assert metadata["coordinate_space"] == "original_frame_pixels"
        assert metadata["detections"] == []
        assert predictor.second_started.wait(timeout=2.0)
    finally:
        predictor.release_second.set()
        if client is not None:
            client.close()
        service.shutdown()


def test_unconfigured_service_http_actions_keep_camera_and_capture_available(
    tmp_path: Path,
) -> None:
    capture = _LiveCapture()
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
        capture_factory=lambda _source: capture,
    )
    try:
        service.start_live()
        assert service.inference_worker is None
        control_port = service.control_server.bound_port
        assert control_port is not None

        code, start = _http_request(
            control_port,
            "/api/v1/vision/inference/start",
            method="POST",
            body=b"{}",
        )
        assert code == 409
        assert start["error"] == "inference_not_configured"

        code, stop = _http_request(
            control_port,
            "/api/v1/vision/inference/stop",
            method="POST",
            body=b"{}",
        )
        assert code == 200
        assert stop == {
            "ok": True,
            "action": "inference/stop",
            "outcome": "already_disabled",
        }

        status = _json_request(control_port, "/api/v1/vision/status")
        assert status["inference"]["configured"] is False
        assert status["inference"]["state"] == "disabled"
        assert status["camera"]["running"] is True

        code, snapshot = _http_request(
            control_port,
            "/api/v1/vision/snapshot",
            method="POST",
            body=b"",
        )
        assert code == 200
        assert snapshot["capture"]["snapshot_count"] == 1
    finally:
        service.shutdown()


def test_http_stop_returns_before_blocked_prediction_and_preserves_capture_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    predictors: list[_ServicePredictor] = []

    def predictor_factory(_onnx_path: Path, _report_path: Path) -> _ServicePredictor:
        predictor = _ServicePredictor(
            SimpleNamespace(**_INFERENCE_CONTRACT),
            block_prediction=True,
        )
        predictors.append(predictor)
        return predictor

    service, capture, workers = _live_service_with_fake_inference_worker(
        monkeypatch,
        tmp_path,
        predictor_factory,
    )
    worker = workers[0]
    try:
        service.start_live()
        control_port = service.control_server.bound_port
        assert control_port is not None

        disabled_code, disabled_snapshot = _http_request(
            control_port,
            "/api/v1/vision/snapshot",
            method="POST",
            body=b"",
        )
        assert disabled_code == 200
        assert disabled_snapshot["capture"]["snapshot_count"] == 1

        start_code, start = _http_request(
            control_port,
            "/api/v1/vision/inference/start",
            method="POST",
            body=b"{}",
        )
        assert start_code == 202
        assert start["outcome"] == "accepted"
        with worker._state_condition:
            assert worker._state_condition.wait_for(
                lambda: worker.status()["state"] == InferenceState.RUNNING.value,
                timeout=2.0,
            )
        assert len(predictors) == 1
        assert predictors[0].prediction_started.wait(2.0)

        record_code, recording = _http_request(
            control_port,
            "/api/v1/vision/recording/start",
            method="POST",
            body=b"",
        )
        assert record_code == 200
        assert recording["capture"]["recording"] is True
        with service.capture_manager._condition:
            assert service.capture_manager._condition.wait_for(
                lambda: service.capture_manager.status()["recorded_frames"] > 0,
                timeout=2.0,
            )
        before_stop = _json_request(control_port, "/api/v1/vision/status")["capture"]

        stop_code, stop = _http_request(
            control_port,
            "/api/v1/vision/inference/stop",
            method="POST",
            body=b"{}",
        )
        assert stop_code == 202
        assert stop == {
            "ok": True,
            "action": "inference/stop",
            "outcome": "accepted",
        }
        stopping = _json_request(control_port, "/api/v1/vision/status")
        assert stopping["inference"]["operation"] == "stopping"
        assert stopping["capture"]["recording"] is True

        stopping_snapshot_code, stopping_snapshot = _http_request(
            control_port,
            "/api/v1/vision/snapshot",
            method="POST",
            body=b"",
        )
        assert stopping_snapshot_code == 200
        assert stopping_snapshot["capture"]["snapshot_count"] == 2

        predictors[0].release_prediction.set()
        with service.inference_control._condition:
            assert service.inference_control._condition.wait_for(
                lambda: service._inference_status()["operation"] is None,
                timeout=2.0,
            )

        final = _json_request(control_port, "/api/v1/vision/status")
        inference = final["inference"]
        assert inference["state"] == "disabled"
        assert inference["operation"] is None
        assert inference["artifact_name"] is None
        assert inference["latest_frame_id"] is None
        assert inference["capture_timestamp_ns"] is None
        assert inference["inference_fps"] is None
        assert inference["latency_ms"] is None
        assert inference["detection_count"] is None
        assert inference["processed_frames"] == 0
        assert inference["skipped_frames"] == 0
        assert inference["last_error"] is None
        assert final["capture"]["session_id"] == before_stop["session_id"]
        assert final["capture"]["snapshot_count"] >= before_stop["snapshot_count"]
        assert final["capture"]["recorded_frames"] >= before_stop["recorded_frames"]
        assert capture.release_count == 0
    finally:
        for predictor in predictors:
            predictor.release_prediction.set()
        service.shutdown()

    assert capture.release_count == 1
    assert capture.factory_calls == 1


def _write_valid_looking_report(report_path: Path, model_name: str) -> None:
    report_path.write_text(
        json.dumps(
            {
                "artifact_name": "d2_mobilenet_v2_fomo_seed42_epoch40",
                "source_experiment_config": "config.yaml",
                "source_experiment_config_sha256": "c" * 64,
                "export_config_file": "export.yaml",
                "export_config_sha256": "d" * 64,
                "checkpoint_file": "checkpoint.pt",
                "checkpoint_sha256": "e" * 64,
                "epoch": 40,
                "seed": 42,
                "parameter_count": 1,
                "config_fingerprint": "f" * 64,
                "validation_threshold": 0.4,
                "validation_threshold_usage": "provenance_only_raw_logits_export",
                "onnx_file": model_name,
                "onnx_sha256": "3dea74511bf2c44844192e75594fd53d4c4ce941f8b53b15767e020832bf9b08",
                "onnx_size_bytes": 1,
                "onnx_checker": "passed",
                "onnx_opset": 17,
                "input_name": "images",
                "input_shape": [1, 3, 192, 192],
                "input_dtype": "float32",
                "input_color_order": "RGB",
                "input_value_range": [0.0, 1.0],
                "output_name": "output",
                "output_shape": [1, 8, 24, 24],
                "output_dtype": "float32",
                "output_semantic": "raw_logits",
                "output_stride": 8,
                "class_names": [
                    "fish_tuna",
                    "jellyfish",
                    "class_2",
                    "class_3",
                    "class_4",
                    "class_5",
                    "class_6",
                ],
                "postprocess": {
                    "class_thresholds": [0.4] * 7,
                    "component_mode": "connected_components",
                    "confidence_mode": "max",
                    "selection_strategy": "highest_confidence",
                    "allowed_class_ids": None,
                    "confidence_threshold": 0.4,
                    "max_match_distance_pixels": 1.0,
                    "max_lost_frames": 0,
                },
                "pytorch_version": "test",
                "onnx_version": "test",
                "onnxruntime_version": "test",
                "exported_at_utc": "test",
                "parity": {
                    "passed": True,
                    "input_seed": 42,
                    "rtol": 0.0,
                    "atol": 0.0,
                    "max_absolute_error": 0.0,
                    "mean_absolute_error": 0.0,
                },
            }
        ),
        encoding="utf-8",
    )


@pytest.mark.parametrize("artifact_kind", ["missing", "invalid"])
def test_invalid_inference_artifact_fails_only_after_manual_start_and_keeps_live_service(
    tmp_path: Path,
    artifact_kind: str,
) -> None:
    capture = _LiveCapture()
    model_path = tmp_path / "model.onnx"
    report_path = tmp_path / "report.json"
    _write_valid_looking_report(report_path, model_path.name)
    if artifact_kind == "invalid":
        model_path.write_bytes(b"not an ONNX model")

    service = VisionService(
        VisionServiceConfig(
            width=4,
            height=3,
            fps=25.0,
            port=0,
            control_port=0,
            capture_output_root=tmp_path,
            capture_min_free_bytes=0,
            inference_onnx=model_path,
            inference_report=report_path,
        ),
        capture_factory=lambda _source: capture,
    )

    try:
        service.start_live()
        assert service.mode_manager.mode is VisionMode.LIVE
        assert service.camera_owner.is_running
        control_port = service.control_server.bound_port
        assert control_port is not None

        before_start = _json_request(control_port, "/api/v1/vision/status")
        assert before_start["inference"]["state"] == "disabled"
        assert before_start["inference"]["configured"] is True

        accepted = service.start_inference()
        assert accepted.http_status == 202
        worker = service.inference_worker
        assert worker is not None
        with worker._state_condition:
            assert worker._state_condition.wait_for(
                lambda: worker.status()["state"] == "failed",
                timeout=2.0,
            )
        failed_status = _json_request(control_port, "/api/v1/vision/status")
        assert set(failed_status["inference"]) == {
            "configured",
            "control_supported",
            "detection_stream_supported",
            "detection_stream_port",
            "detection_stream_version",
            "operation",
            "state",
            "artifact_name",
            "model_sha256",
            "confidence_threshold",
            "latest_frame_id",
            "capture_timestamp_ns",
            "processed_frames",
            "skipped_frames",
            "inference_fps",
            "latency_ms",
            "detection_count",
            "last_error",
            "vision_process_rss_bytes",
            "system_total_memory_bytes",
        }
        assert failed_status["inference"]["last_error"]
        assert failed_status["inference"]["artifact_name"] is None
        assert failed_status["inference"]["model_sha256"] is None
        assert failed_status["inference"]["confidence_threshold"] is None
        assert failed_status["inference"]["latest_frame_id"] is None
        assert failed_status["inference"]["capture_timestamp_ns"] is None
        assert failed_status["inference"]["processed_frames"] == 0
        assert failed_status["inference"]["skipped_frames"] == 0
        assert failed_status["inference"]["inference_fps"] is None
        assert failed_status["inference"]["latency_ms"] is None
        assert failed_status["inference"]["detection_count"] is None
        assert failed_status["camera"]["running"] is True
        assert service.mode_manager.mode is VisionMode.LIVE

        snapshot = _json_request(
            control_port,
            "/api/v1/vision/snapshot",
            method="POST",
        )
        assert snapshot["ok"] is True
    finally:
        service.shutdown()


def test_start_live_orders_mode_then_control_without_auto_start(
    monkeypatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    service = _service_with_lifecycle_doubles(monkeypatch, tmp_path, events)

    service.start_live()

    assert events == [
        "mode.live",
        "control.start",
    ]


def test_detection_server_lifecycle_is_explicit_and_independent(
    monkeypatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    service = _service_with_lifecycle_doubles(monkeypatch, tmp_path, events)
    detection = _RecordingDetectionServer(events)
    service.detection_server = detection  # type: ignore[assignment]

    service.start_live()
    assert events == [
        "mode.live",
        "detection.start",
        "control.start",
    ]
    assert detection.start_count == 1
    assert detection.bound_port == 47012

    events.clear()
    service.shutdown()

    assert detection.stop_count == 1
    assert "detection.stop" in events
    assert events.index("control.stop") < events.index("detection.stop")
    assert events.index("detection.stop") < events.index("capture.shutdown")
    assert detection.bound_port is None


def test_detection_start_failure_rolls_back_live_service(
    monkeypatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    service = _service_with_lifecycle_doubles(monkeypatch, tmp_path, events)
    detection = _RecordingDetectionServer(
        events,
        error=RuntimeError("detection start failed"),
    )
    service.detection_server = detection  # type: ignore[assignment]
    worker = _RecordingInferenceWorker.instances[0]

    with pytest.raises(RuntimeError, match="detection start failed"):
        service.start_live()

    assert worker.stop_count == 1
    assert detection.start_count == 1
    assert detection.stop_count == 1
    assert events == [
        "mode.live",
        "detection.start",
        "inference.request_stop",
        "control.stop",
        "detection.stop",
        "inference.request_stop",
        "inference.stop",
        "mode.shutdown",
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
    failure_thread: threading.Thread | None = None

    def fail_asynchronously() -> None:
        nonlocal failure_thread
        worker.events.append("inference.start")
        worker.state = "starting"

        def fail_after_start_returns() -> None:
            start_returned.wait()
            with failure_condition:
                worker.failed = True
                worker.state = "failed"
                worker.last_error = "asynchronous worker failure"
                failure_condition.notify_all()

        failure_thread = threading.Thread(
            target=fail_after_start_returns,
            name="test-inference-failure",
            daemon=False,
        )
        failure_thread.start()

    worker.start = fail_asynchronously  # type: ignore[method-assign]

    try:
        service.start_live()
        accepted = service.start_inference()
        assert accepted.http_status == 202
        start_returned.set()
        with failure_condition:
            assert failure_condition.wait_for(
                lambda: worker.failed,
                timeout=1.0,
            )

        assert worker.failed is True
        assert service.mode_manager.mode is VisionMode.LIVE
        assert "mode.shutdown" not in events
        assert events == [
            "mode.live",
            "control.start",
            "inference.start",
        ]
    finally:
        start_returned.set()
        assert failure_thread is not None
        failure_thread.join(timeout=1.0)
        assert not failure_thread.is_alive()


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
        "control.start",
        "inference.request_stop",
        "control.stop",
        "inference.request_stop",
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
        "control.start",
        "inference.request_stop",
        "control.stop",
        "inference.request_stop",
        "inference.stop",
        "mode.shutdown",
    ]


def test_control_start_failure_preserves_original_when_mode_rollback_fails(
    monkeypatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    service = _service_with_lifecycle_doubles(
        monkeypatch,
        tmp_path,
        events,
        control_error=RuntimeError("control start failed"),
        mode_error=RuntimeError("mode shutdown failed"),
    )
    worker = _RecordingInferenceWorker.instances[0]

    with pytest.raises(RuntimeError, match="control start failed"):
        service.start_live()

    assert worker.stop_count == 1
    assert service.mode_manager.shutdown_count == 1
    assert events == [
        "mode.live",
        "control.start",
        "inference.request_stop",
        "control.stop",
        "inference.request_stop",
        "inference.stop",
        "mode.shutdown",
    ]


def test_start_live_rejects_reuse_after_terminal_startup_rollback(
    monkeypatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    service = _service_with_lifecycle_doubles(monkeypatch, tmp_path, events)

    class _FlakyControlServer:
        def __init__(self) -> None:
            self.start_calls = 0

        def start(self) -> None:
            self.start_calls += 1
            events.append("control.start")
            if self.start_calls == 1:
                raise RuntimeError("first listener start failed")

        def stop(self) -> None:
            events.append("control.stop")

    control_server = _FlakyControlServer()
    service.control_server = control_server  # type: ignore[assignment]
    try:
        with pytest.raises(RuntimeError, match="first listener start failed"):
            service.start_live()
        events_after_rollback = list(events)

        with pytest.raises(RuntimeError, match="new VisionService"):
            service.start_live()

        assert control_server.start_calls == 1
        assert events == events_after_rollback
    finally:
        service.shutdown()


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
        "inference.request_stop",
        "control.stop",
        "capture.shutdown",
        "inference.request_stop",
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
