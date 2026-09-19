from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request

import numpy as np
import pytest

from fomo_servo.capture.control import VisionControlServer
from fomo_servo.vision.camera_owner import CameraFacts
from fomo_servo.vision.frame import PixelFormat, VisionFrame
from fomo_servo.vision.frame_hub import FrameHub


def _facts() -> CameraFacts:
    return CameraFacts(
        source="/dev/video0",
        observed_width=4,
        observed_height=3,
        observed_fps=25.0,
        observed_fourcc="YUYV",
    )


def _frame(frame_id: int) -> VisionFrame:
    return VisionFrame(
        frame_id=frame_id,
        capture_timestamp_ns=frame_id + 100,
        width=4,
        height=3,
        pixel_format=PixelFormat.BGR8,
        image=np.zeros((3, 4, 3), dtype=np.uint8),
    )


class _Manager:
    def __init__(self) -> None:
        self.snapshots: list[int] = []
        self.started = 0
        self.stopped = 0

    def status(self) -> dict:
        return {
            "state": "idle",
            "recording": False,
            "session_id": None,
        }

    def snapshot(self, frame, facts) -> dict:
        assert facts.observed_width == 4
        self.snapshots.append(frame.frame_id)
        return {"state": "idle", "snapshot_count": len(self.snapshots)}

    def start_recording(self, facts) -> dict:
        assert facts.observed_fps == 25.0
        self.started += 1
        return {"state": "recording", "recording": True}

    def stop_recording(self) -> dict:
        self.stopped += 1
        return {"state": "idle", "recording": False}
def _request(port: int, path: str, method: str = "GET") -> tuple[int, dict]:
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        method=method,
        data=b"" if method == "POST" else None,
    )
    try:
        with urllib.request.urlopen(request, timeout=2.0) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def test_status_and_recording_actions_use_capture_control_only() -> None:
    manager = _Manager()
    hub = FrameHub()
    hub.publish(_frame(1))
    server = VisionControlServer(
        manager,  # type: ignore[arg-type]
        hub,
        facts_provider=_facts,
        camera_running=lambda: True,
        measured_fps_provider=lambda: 13.75,
        bind_host="127.0.0.1",
        port=0,
    ).start()
    assert server.bound_port is not None

    try:
        code, status = _request(
            server.bound_port,
            "/api/v1/vision/status",
        )
        assert code == 200
        assert status["camera"]["latest_frame_id"] == 1
        assert status["camera"]["observed_fps"] == 25.0
        assert status["camera"]["measured_capture_fps"] == 13.75
        code, started = _request(
            server.bound_port,
            "/api/v1/vision/recording/start",
            method="POST",
        )
        assert code == 200
        assert started["capture"]["recording"] is True
        assert manager.started == 1

        code, stopped = _request(
            server.bound_port,
            "/api/v1/vision/recording/stop",
            method="POST",
        )
        assert code == 200
        assert stopped["capture"]["recording"] is False
        assert manager.stopped == 1
    finally:
        server.stop()


def test_snapshot_waits_for_frame_newer_than_request_floor() -> None:
    manager = _Manager()
    hub = FrameHub()
    hub.publish(_frame(5))
    server = VisionControlServer(
        manager,  # type: ignore[arg-type]
        hub,
        facts_provider=_facts,
        camera_running=lambda: True,
        measured_fps_provider=lambda: 13.75,
        bind_host="127.0.0.1",
        port=0,
        snapshot_timeout=0.5,
    ).start()
    assert server.bound_port is not None
    def publish_fresh() -> None:
        time.sleep(0.05)
        hub.publish(_frame(6))

    publisher = threading.Thread(target=publish_fresh, daemon=True)
    publisher.start()
    try:
        code, payload = _request(
            server.bound_port,
            "/api/v1/vision/snapshot",
            method="POST",
        )
        assert code == 200
        assert payload["capture"]["snapshot_count"] == 1
        assert manager.snapshots == [6]
    finally:
        publisher.join(timeout=1.0)
        server.stop()



def test_recording_start_does_not_depend_on_measured_cadence() -> None:
    manager = _Manager()
    server = VisionControlServer(
        manager,  # type: ignore[arg-type]
        FrameHub(),
        facts_provider=_facts,
        camera_running=lambda: True,
        measured_fps_provider=lambda: None,
        bind_host="127.0.0.1",
        port=0,
    ).start()
    assert server.bound_port is not None
    try:
        code, payload = _request(
            server.bound_port,
            "/api/v1/vision/recording/start",
            method="POST",
        )
        assert code == 200
        assert payload["capture"]["recording"] is True
        assert manager.started == 1
    finally:
        server.stop()


def test_snapshot_rejects_when_camera_is_not_running() -> None:
    manager = _Manager()
    server = VisionControlServer(
        manager,  # type: ignore[arg-type]
        FrameHub(),
        facts_provider=_facts,
        camera_running=lambda: False,
        measured_fps_provider=lambda: None,
        bind_host="127.0.0.1",
        port=0,
    ).start()
    assert server.bound_port is not None
    try:
        code, payload = _request(
            server.bound_port,
            "/api/v1/vision/snapshot",
            method="POST",
        )
        assert code == 409
        assert payload["ok"] is False
        assert "camera is not running" in payload["error"]
    finally:
        server.stop()


class _DelayedVisionControlServer(VisionControlServer):
    def _serve(self, server) -> None:
        time.sleep(0.05)
        super()._serve(server)


def test_immediate_stop_waits_for_delayed_serve_forever_without_deadlock() -> None:
    server = _DelayedVisionControlServer(
        _Manager(),  # type: ignore[arg-type]
        FrameHub(),
        facts_provider=_facts,
        camera_running=lambda: True,
        measured_fps_provider=lambda: None,
        bind_host="127.0.0.1",
        port=0,
    ).start()

    started = time.monotonic()
    server.stop(join_timeout=1.0)
    elapsed = time.monotonic() - started

    assert elapsed < 1.0
    assert server.bound_port is None
    assert server.is_running is False


def test_thread_start_failure_rolls_back_bound_control_server(
    monkeypatch,
) -> None:
    server = VisionControlServer(
        _Manager(),  # type: ignore[arg-type]
        FrameHub(),
        facts_provider=_facts,
        camera_running=lambda: True,
        measured_fps_provider=lambda: None,
        bind_host="127.0.0.1",
        port=0,
    )

    def fail_start(_thread) -> None:
        raise RuntimeError("thread start failed")

    monkeypatch.setattr(threading.Thread, "start", fail_start)

    with pytest.raises(RuntimeError, match="thread start failed"):
        server.start()

    assert server.bound_port is None
    assert server.is_running is False
    assert server._server is None
    assert server._thread is None
