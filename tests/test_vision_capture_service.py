from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

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
