"""Small HTTP/JSON control plane for Vision capture actions on TCP 47011."""

from __future__ import annotations

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Optional

from fomo_servo.vision.camera_owner import CameraFacts
from fomo_servo.vision.frame_hub import FrameHub

from .manager import CaptureError, CaptureManager

LOGGER = logging.getLogger(__name__)
DEFAULT_CONTROL_PORT = 47011
DEFAULT_SNAPSHOT_TIMEOUT_SECONDS = 1.0


class _ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    # Clean shutdown waits for in-flight snapshot/start/stop handlers so they
    # cannot outlive CaptureManager and continue writing after shutdown.
    daemon_threads = False


class VisionControlServer:
    """Expose capture-only actions without touching RBRP or RBVS framing."""

    def __init__(
        self,
        manager: CaptureManager,
        hub: FrameHub,
        *,
        facts_provider: Callable[[], Optional[CameraFacts]],
        camera_running: Callable[[], bool],
        measured_fps_provider: Callable[[], Optional[float]],
        bind_host: str = "0.0.0.0",
        port: int = DEFAULT_CONTROL_PORT,
        snapshot_timeout: float = DEFAULT_SNAPSHOT_TIMEOUT_SECONDS,
    ) -> None:
        self._manager = manager
        self._hub = hub
        self._facts_provider = facts_provider
        self._camera_running = camera_running
        self._measured_fps_provider = measured_fps_provider
        self._bind_host = bind_host
        self._port = port
        self._snapshot_timeout = snapshot_timeout
        self._server: Optional[_ReusableThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self.bound_port: Optional[int] = None
        self.last_error: Optional[BaseException] = None

    @property
    def is_running(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    def start(self) -> "VisionControlServer":
        if self.is_running:
            raise RuntimeError("VisionControlServer is already running")
        if not 0 <= self._port <= 65535:
            raise ValueError("Vision control port is outside 0..65535")
        if self._snapshot_timeout <= 0.0:
            raise ValueError("snapshot_timeout must be positive")

        handler = self._make_handler()
        server = _ReusableThreadingHTTPServer(
            (self._bind_host, self._port),
            handler,
        )
        self._server = server
        self.bound_port = int(server.server_address[1])
        self.last_error = None
        thread = threading.Thread(
            target=self._serve,
            args=(server,),
            name="robobeetle-vision-control",
            daemon=True,
        )
        self._thread = thread
        try:
            thread.start()
        except BaseException:
            self._thread = None
            self._server = None
            self.bound_port = None
            server.server_close()
            raise
        return self

    def stop(self, *, join_timeout: float = 2.0) -> None:
        server = self._server
        thread = self._thread

        # BaseServer.shutdown() must only be called while serve_forever() is
        # running in another live thread. The thread captures its server
        # directly, so clearing self._server cannot make it skip serve_forever.
        if server is not None and thread is not None and thread.is_alive():
            server.shutdown()
        if server is not None:
            server.server_close()
        if thread is not None and thread.is_alive():
            thread.join(timeout=join_timeout)
        if thread is not None and thread.is_alive():
            raise RuntimeError("Vision control server did not stop")

        self._server = None
        self._thread = None
        self.bound_port = None

    def _serve(self, server: _ReusableThreadingHTTPServer) -> None:
        try:
            server.serve_forever(poll_interval=0.1)
        except BaseException as error:
            self.last_error = error
            LOGGER.exception("Vision control server failed")

    def _status_payload(self) -> dict:
        latest = self._hub.snapshot()
        facts = self._facts_provider()
        measured_fps = self._measured_fps_provider()
        return {
            "ok": True,
            "camera": {
                "running": bool(self._camera_running()),
                "latest_frame_id": (
                    None if latest is None else latest.frame_id
                ),
                "capture_timestamp_ns": (
                    None if latest is None else latest.capture_timestamp_ns
                ),
                "observed_width": (
                    None if facts is None else facts.observed_width
                ),
                "observed_height": (
                    None if facts is None else facts.observed_height
                ),
                "observed_fps": (
                    None if facts is None else facts.observed_fps
                ),
                "measured_capture_fps": measured_fps,
            },
            "capture": self._manager.status(),
        }

    def _fresh_snapshot(self) -> dict:
        facts = self._facts_provider()
        if facts is None or not self._camera_running():
            raise CaptureError("camera is not running")
        cached = self._hub.snapshot()
        floor = None if cached is None else cached.frame_id
        frame = self._hub.wait_for_newer(
            floor,
            timeout=self._snapshot_timeout,
        )
        if frame is None:
            raise CaptureError("timed out waiting for a fresh camera frame")
        return self._manager.snapshot(frame, facts)

    def _start_recording(self) -> dict:
        facts = self._facts_provider()
        if facts is None or not self._camera_running():
            raise CaptureError("camera is not running")
        return self._manager.start_recording(facts)

    def _make_handler(self):
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format, *args) -> None:
                LOGGER.debug("Vision control: " + format, *args)

            def do_GET(self) -> None:
                if self.path == "/api/v1/vision/status":
                    self._send_json(200, outer._status_payload())
                    return
                self._send_json(404, {"ok": False, "error": "not_found"})

            def do_POST(self) -> None:
                try:
                    if self.path == "/api/v1/vision/snapshot":
                        result = outer._fresh_snapshot()
                    elif self.path == "/api/v1/vision/recording/start":
                        result = outer._start_recording()
                    elif self.path == "/api/v1/vision/recording/stop":
                        result = outer._manager.stop_recording()
                    else:
                        self._send_json(
                            404,
                            {"ok": False, "error": "not_found"},
                        )
                        return
                    self._send_json(
                        200,
                        {"ok": True, "capture": result},
                    )
                except CaptureError as error:
                    self._send_json(
                        409,
                        {"ok": False, "error": str(error)},
                    )
                except BaseException as error:
                    LOGGER.exception("Vision control request failed")
                    self._send_json(
                        500,
                        {"ok": False, "error": str(error)},
                    )

            def _send_json(self, status: int, payload: dict) -> None:
                body = json.dumps(
                    payload,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
                self.send_response(status)
                self.send_header(
                    "Content-Type",
                    "application/json; charset=utf-8",
                )
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(body)
                self.close_connection = True

        return Handler
