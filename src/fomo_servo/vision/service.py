"""Slice 1 Vision service composition without systemd or robot-control coupling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Any

from .camera_owner import CameraOwner
from .frame_hub import FrameHub
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


class VisionService:
    """Own the Slice 1 camera/hub/server composition and nothing in RBRP."""

    def __init__(
        self,
        config: VisionServiceConfig,
        *,
        capture_factory: Optional[Callable[[int | str], Any]] = None,
    ) -> None:
        self.config = config
        self.hub = FrameHub()
        self.camera_owner = CameraOwner(
            self.hub,
            source=config.source,
            width=config.width,
            height=config.height,
            fps=config.fps,
            fourcc=config.fourcc,
            capture_factory=capture_factory,
        )
        self.stream_server = VisionTcpServer(
            self.hub,
            bind_host=config.bind_host,
            port=config.port,
            jpeg_quality=config.jpeg_quality,
            send_buffer_bytes=config.send_buffer_bytes,
            write_timeout=config.write_timeout,
        )
        self.mode_manager = VisionModeManager(
            self.hub,
            self.camera_owner,
            self.stream_server,
        )

    def start_live(self) -> None:
        self.mode_manager.set_mode(VisionMode.LIVE)

    def shutdown(self) -> None:
        self.mode_manager.shutdown()
