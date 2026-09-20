from __future__ import annotations

import json
import socket
import threading
import urllib.error
import urllib.request
from typing import Callable

import numpy as np
import pytest

import fomo_servo.capture.control as control_module
from fomo_servo.capture.control import VisionControlServer
from fomo_servo.vision.camera_owner import CameraFacts
from fomo_servo.vision.frame import PixelFormat, VisionFrame
from fomo_servo.vision.frame_hub import FrameHub
from fomo_servo.vision.inference_control import (
    InferenceActionResult,
    InferenceControlError,
)


class _Manager:
    def __init__(self) -> None:
        self.started = 0
        self.stopped = 0

    def status(self) -> dict:
        return {
            "state": "idle",
            "recording": False,
            "session_id": None,
            "recorded_frames": 7,
            "snapshot_count": 3,
        }

    def snapshot(self, frame, facts) -> dict:
        return {"state": "idle", "frame_id": frame.frame_id}

    def start_recording(self, facts) -> dict:
        self.started += 1
        return {"state": "recording", "recording": True}

    def stop_recording(self) -> dict:
        self.stopped += 1
        return {"state": "idle", "recording": False}


def _facts() -> CameraFacts:
    return CameraFacts(
        source="fake-camera",
        observed_width=4,
        observed_height=3,
        observed_fps=25.0,
        observed_fourcc="YUYV",
    )


def _frame() -> VisionFrame:
    return VisionFrame(
        frame_id=9,
        capture_timestamp_ns=109,
        width=4,
        height=3,
        pixel_format=PixelFormat.BGR8,
        image=np.zeros((3, 4, 3), dtype=np.uint8),
    )


def _inference_status() -> dict:
    return {
        "configured": True,
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


def _server(
    *,
    start: Callable[[], InferenceActionResult] | None = None,
    stop: Callable[[], InferenceActionResult] | None = None,
    inference_status_provider: Callable[[], dict] | None = _inference_status,
) -> tuple[VisionControlServer, _Manager]:
    manager = _Manager()
    hub = FrameHub()
    hub.publish(_frame())
    server = VisionControlServer(
        manager,  # type: ignore[arg-type]
        hub,
        facts_provider=_facts,
        camera_running=lambda: True,
        measured_fps_provider=lambda: 25.0,
        inference_status_provider=inference_status_provider,
        inference_start_callback=start,
        inference_stop_callback=stop,
        bind_host="127.0.0.1",
        port=0,
    ).start()
    assert server.bound_port is not None
    return server, manager


def _request(
    port: int,
    path: str,
    *,
    method: str = "GET",
    body: bytes | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict]:
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=body,
        headers={} if headers is None else headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=2.0) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def _raw_response(port: int, request: bytes) -> tuple[int, dict]:
    with socket.create_connection(("127.0.0.1", port), timeout=2.0) as client:
        client.settimeout(2.0)
        client.sendall(request)
        chunks: list[bytes] = []
        while True:
            chunk = client.recv(4096)
            if not chunk:
                break
            chunks.append(chunk)
    raw = b"".join(chunks)
    headers, body = raw.split(b"\r\n\r\n", 1)
    status = int(headers.split(b"\r\n", 1)[0].split()[1])
    return status, json.loads(body)


@pytest.mark.parametrize("body", [b"", b"{}", b" \r\n { \t } "])
def test_inference_routes_accept_only_empty_body_or_empty_json_object(
    body: bytes,
) -> None:
    calls: list[str] = []

    def start() -> InferenceActionResult:
        calls.append("start")
        return InferenceActionResult(202, "inference/start", "accepted")

    server, _manager = _server(start=start)
    try:
        assert server.bound_port is not None
        code, payload = _request(
            server.bound_port,
            "/api/v1/vision/inference/start",
            method="POST",
            body=body,
            headers={"Content-Type": "application/json"},
        )
        assert code == 202
        assert payload == {
            "ok": True,
            "action": "inference/start",
            "outcome": "accepted",
        }
        assert calls == ["start"]
    finally:
        server.stop()


@pytest.mark.parametrize(
    "body",
    [
        b" ",
        b"[]",
        b"null",
        b"true",
        b"42",
        b"{",
        b'{"model":"forbidden"}',
        b'{"path":"forbidden"}',
        b'{"threshold":0.4}',
    ],
)
def test_inference_route_rejects_nonempty_or_nonobject_json(body: bytes) -> None:
    calls: list[str] = []

    def start() -> InferenceActionResult:
        calls.append("start")
        return InferenceActionResult(202, "inference/start", "accepted")

    server, _manager = _server(start=start)
    try:
        assert server.bound_port is not None
        code, payload = _request(
            server.bound_port,
            "/api/v1/vision/inference/start",
            method="POST",
            body=body,
            headers={"Content-Type": "application/json"},
        )
        assert code == 400
        assert payload["ok"] is False
        assert payload["error"] == "invalid_request"
        assert payload["message"]
        assert calls == []
    finally:
        server.stop()


def test_inference_route_rejects_oversize_and_invalid_framing_before_callback() -> None:
    calls: list[str] = []

    def start() -> InferenceActionResult:
        calls.append("start")
        return InferenceActionResult(202, "inference/start", "accepted")

    server, _manager = _server(start=start)
    try:
        assert server.bound_port is not None
        oversized = (
            b"POST /api/v1/vision/inference/start HTTP/1.1\r\n"
            b"Host: 127.0.0.1\r\n"
            b"Content-Length: 1025\r\n"
            b"Connection: close\r\n\r\n"
        )
        code, payload = _raw_response(server.bound_port, oversized)
        assert code == 413
        assert payload["error"] == "request_too_large"

        unbounded_numeric_length = (
            b"POST /api/v1/vision/inference/start HTTP/1.1\r\n"
            b"Host: 127.0.0.1\r\n"
            + b"Content-Length: "
            + (b"9" * 5000)
            + b"\r\nConnection: close\r\n\r\n"
        )
        code, payload = _raw_response(server.bound_port, unbounded_numeric_length)
        assert code == 413
        assert payload["error"] == "request_too_large"

        invalid_length = (
            b"POST /api/v1/vision/inference/start HTTP/1.1\r\n"
            b"Host: 127.0.0.1\r\n"
            b"Content-Length: invalid\r\n"
            b"Connection: close\r\n\r\n"
        )
        code, payload = _raw_response(server.bound_port, invalid_length)
        assert code == 400
        assert payload["error"] == "invalid_request"

        transfer_encoded = (
            b"POST /api/v1/vision/inference/start HTTP/1.1\r\n"
            b"Host: 127.0.0.1\r\n"
            b"Transfer-Encoding: chunked\r\n"
            b"Connection: close\r\n\r\n"
        )
        code, payload = _raw_response(server.bound_port, transfer_encoded)
        assert code == 400
        assert payload["error"] == "invalid_request"
        assert calls == []
    finally:
        server.stop()


def test_inference_route_rejects_truncated_or_timed_out_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def start() -> InferenceActionResult:
        calls.append("start")
        return InferenceActionResult(202, "inference/start", "accepted")

    monkeypatch.setattr(
        control_module,
        "INFERENCE_BODY_READ_TIMEOUT_SECONDS",
        0.05,
    )
    server, _manager = _server(start=start)
    try:
        assert server.bound_port is not None
        truncated = (
            b"POST /api/v1/vision/inference/start HTTP/1.1\r\n"
            b"Host: 127.0.0.1\r\n"
            b"Content-Length: 4\r\n"
            b"Connection: close\r\n\r\n{}"
        )
        code, payload = _raw_response(server.bound_port, truncated)
        assert code == 400
        assert payload["error"] == "invalid_request"

        timeout_request = (
            b"POST /api/v1/vision/inference/start HTTP/1.1\r\n"
            b"Host: 127.0.0.1\r\n"
            b"Content-Length: 2\r\n"
            b"Connection: close\r\n\r\n"
        )
        code, payload = _raw_response(server.bound_port, timeout_request)
        assert code == 400
        assert payload["error"] == "invalid_request"
        assert calls == []
    finally:
        server.stop()


def test_inference_post_is_only_an_ack_and_get_status_is_authoritative() -> None:
    calls: list[str] = []

    def start() -> InferenceActionResult:
        calls.append("start")
        return InferenceActionResult(202, "inference/start", "accepted")

    server, manager = _server(start=start)
    try:
        assert server.bound_port is not None
        code, acknowledgement = _request(
            server.bound_port,
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
        assert "camera" not in acknowledgement
        assert "capture" not in acknowledgement
        assert "inference" not in acknowledgement

        code, status = _request(server.bound_port, "/api/v1/vision/status")
        assert code == 200
        assert status["camera"]["latest_frame_id"] == 9
        assert status["capture"] == manager.status()
        assert status["inference"] == _inference_status()
        assert calls == ["start"]
    finally:
        server.stop()


def test_inference_control_errors_keep_stable_code_and_readable_message() -> None:
    def start() -> InferenceActionResult:
        raise InferenceControlError(
            "inference_busy",
            409,
            "inference is stopping",
        )

    server, _manager = _server(start=start)
    try:
        assert server.bound_port is not None
        code, payload = _request(
            server.bound_port,
            "/api/v1/vision/inference/start",
            method="POST",
            body=b"",
        )
        assert code == 409
        assert payload == {
            "ok": False,
            "error": "inference_busy",
            "message": "inference is stopping",
        }
    finally:
        server.stop()


def test_standalone_server_without_callbacks_keeps_legacy_status_and_fallbacks() -> None:
    server, _manager = _server(
        inference_status_provider=None,
    )
    try:
        assert server.bound_port is not None
        code, status = _request(server.bound_port, "/api/v1/vision/status")
        assert code == 200
        assert "inference" not in status

        code, start = _request(
            server.bound_port,
            "/api/v1/vision/inference/start",
            method="POST",
            body=b"{}",
        )
        assert code == 409
        assert start["error"] == "inference_not_configured"

        code, stop = _request(
            server.bound_port,
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
    finally:
        server.stop()


def test_inference_body_validation_does_not_change_existing_capture_post_shape() -> None:
    server, manager = _server()
    try:
        assert server.bound_port is not None
        code, payload = _request(
            server.bound_port,
            "/api/v1/vision/recording/stop",
            method="POST",
            body=b'{"not":"inference"}',
            headers={"Content-Type": "application/json"},
        )
        assert code == 200
        assert payload == {
            "ok": True,
            "capture": {"state": "idle", "recording": False},
        }
        assert manager.stopped == 1
    finally:
        server.stop()


def test_get_status_runs_while_a_separate_http_action_handler_is_active() -> None:
    entered = threading.Event()
    release = threading.Event()
    post_done = threading.Event()
    response: dict[str, object] = {}

    def stop() -> InferenceActionResult:
        entered.set()
        assert release.wait(2.0)
        return InferenceActionResult(202, "inference/stop", "accepted")

    server, _manager = _server(stop=stop)
    post_thread: threading.Thread | None = None
    try:
        assert server.bound_port is not None

        def post_stop() -> None:
            response["value"] = _request(
                server.bound_port,
                "/api/v1/vision/inference/stop",
                method="POST",
                body=b"{}",
            )
            post_done.set()

        post_thread = threading.Thread(target=post_stop, daemon=False)
        post_thread.start()
        assert entered.wait(2.0)
        code, status = _request(server.bound_port, "/api/v1/vision/status")
        assert code == 200
        assert status["inference"]["operation"] is None
        assert not post_done.is_set()
        release.set()
        assert post_done.wait(2.0)
        assert response["value"] == (
            202,
            {"ok": True, "action": "inference/stop", "outcome": "accepted"},
        )
    finally:
        release.set()
        if post_thread is not None:
            post_thread.join(2.0)
        server.stop()
