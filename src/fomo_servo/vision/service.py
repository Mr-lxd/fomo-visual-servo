"""Slice 1 Vision service composition without systemd or robot-control coupling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Any

from fomo_servo.capture.control import DEFAULT_CONTROL_PORT, VisionControlServer
from fomo_servo.capture.manager import CaptureConfig, CaptureManager

from .camera_owner import CameraOwner
from .frame_hub import FrameHub
from .detection_streaming import (
    DEFAULT_DETECTION_PORT,
    DETECTION_STREAM_VERSION,
    DetectionTcpServer,
)
from .inference_control import InferenceActionResult, InferenceControl
from .inference_result_hub import InferenceResultHub
from .inference_worker import InferenceWorker
from .mode import VisionMode, VisionModeManager
from .streaming import (
    DEFAULT_JPEG_QUALITY,
    DEFAULT_SEND_BUFFER_BYTES,
    DEFAULT_VISION_PORT,
    DEFAULT_WRITE_TIMEOUT_SECONDS,
    VisionTcpServer,
)


@dataclass(frozen=True)
class VisionServiceConfig:
    source: int | str = "/dev/video0"
    width: int = 640
    height: int = 480
    fps: float = 25.0
    fourcc: str = "YUYV"
    bind_host: str = "0.0.0.0"
    port: int = DEFAULT_VISION_PORT
    jpeg_quality: int = DEFAULT_JPEG_QUALITY
    send_buffer_bytes: int = DEFAULT_SEND_BUFFER_BYTES
    write_timeout: float = DEFAULT_WRITE_TIMEOUT_SECONDS
    control_port: int = DEFAULT_CONTROL_PORT
    detection_port: int = DEFAULT_DETECTION_PORT
    capture_output_root: Path = Path("datasets_raw/robobeetle")
    capture_queue_bytes: int = 64 * 1024 * 1024
    capture_min_free_bytes: int = 512 * 1024 * 1024
    inference_onnx: Path | None = None
    inference_report: Path | None = None


class VisionService:
    """Own the Slice 1 camera/hub/server composition and nothing in RBRP."""

    def __init__(
        self,
        config: VisionServiceConfig,
        *,
        capture_factory: Optional[Callable[[int | str], Any]] = None,
    ) -> None:
        onnx_configured = config.inference_onnx is not None
        report_configured = config.inference_report is not None
        if onnx_configured != report_configured:
            raise ValueError(
                "inference_onnx and inference_report must be supplied together"
            )

        self.config = config
        self._lifecycle_closed = False
        self.hub = FrameHub()
        self.capture_manager = CaptureManager(
            CaptureConfig(
                output_root=config.capture_output_root,
                max_queue_bytes=config.capture_queue_bytes,
                min_free_bytes=config.capture_min_free_bytes,
            )
        )
        self.camera_owner = CameraOwner(
            self.hub,
            source=config.source,
            width=config.width,
            height=config.height,
            fps=config.fps,
            fourcc=config.fourcc,
            capture_factory=capture_factory,
            frame_callback=self.capture_manager.offer_frame,
        )
        self.inference_results = InferenceResultHub()
        self.inference_worker: InferenceWorker | None = None
        if onnx_configured and report_configured:
            self.inference_worker = InferenceWorker(
                self.hub,
                config.inference_onnx,
                config.inference_report,
                result_sink=self.inference_results.publish,
            )
        self.stream_server = VisionTcpServer(
            self.hub,
            bind_host=config.bind_host,
            port=config.port,
            jpeg_quality=config.jpeg_quality,
            send_buffer_bytes=config.send_buffer_bytes,
            write_timeout=config.write_timeout,
        )
        self.detection_server = DetectionTcpServer(
            self.inference_results,
            bind_host=config.bind_host,
            port=config.detection_port,
        )
        self.mode_manager = VisionModeManager(
            self.hub,
            self.camera_owner,
            self.stream_server,
        )
        self.inference_control = InferenceControl(
            self.inference_worker,
            is_live=lambda: self.mode_manager.mode is VisionMode.LIVE,
            camera_running=lambda: self.camera_owner.is_running,
        )
        self.control_server = VisionControlServer(
            self.capture_manager,
            self.hub,
            facts_provider=lambda: self.camera_owner.facts,
            camera_running=lambda: self.camera_owner.is_running,
            measured_fps_provider=(
                lambda: self.camera_owner.measured_capture_fps
            ),
            inference_status_provider=self._inference_status,
            inference_start_callback=self.start_inference,
            inference_stop_callback=self.stop_inference,
            bind_host=config.bind_host,
            port=config.control_port,
        )

    def _inference_status(self) -> dict:
        status = self.inference_control.status()
        status["detection_stream_supported"] = True
        status["detection_stream_port"] = (
            self.detection_server.bound_port
            if self.detection_server.bound_port is not None
            else self.config.detection_port
        )
        status["detection_stream_version"] = DETECTION_STREAM_VERSION
        return status

    def start_inference(self) -> InferenceActionResult:
        """Expose manual inference Start/Retry to the HTTP control boundary."""

        return self.inference_control.start_inference()

    def stop_inference(self) -> InferenceActionResult:
        """Expose manual inference Stop/Clear Error to the HTTP control boundary."""

        return self.inference_control.stop_inference()

    def start_live(self) -> None:
        if self._lifecycle_closed:
            raise RuntimeError(
                "VisionService is closed; construct a new VisionService for another lifecycle"
            )
        if self.mode_manager.mode is VisionMode.LIVE:
            return
        try:
            self.mode_manager.set_mode(VisionMode.LIVE)
            self.detection_server.start()
            self.inference_control.open_actions()
            self.control_server.start()
        except BaseException as startup_error:
            self._lifecycle_closed = True
            cleanup_errors: list[BaseException] = []
            try:
                self.inference_control.begin_shutdown()
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)
            try:
                self.control_server.stop()
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)
            try:
                self.detection_server.stop()
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)
            try:
                self.inference_control.finish_shutdown()
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)
            try:
                self.mode_manager.shutdown()
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)
            if cleanup_errors:
                raise startup_error from cleanup_errors[0]
            raise

    def shutdown(self) -> None:
        self._lifecycle_closed = True
        first_error: BaseException | None = None
        cleanups = [
            self.inference_control.begin_shutdown,
            self.control_server.stop,
            self.detection_server.stop,
            self.capture_manager.shutdown,
            self.inference_control.finish_shutdown,
        ]
        cleanups.append(self.mode_manager.shutdown)
        for cleanup in cleanups:
            try:
                cleanup()
            except BaseException as error:
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error
