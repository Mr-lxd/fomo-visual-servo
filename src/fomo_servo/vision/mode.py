"""Minimal OFF/LIVE Vision mode composition for Slice 1."""

from __future__ import annotations

from enum import Enum

from .camera_owner import CameraOwner
from .frame_hub import FrameHub
from .streaming import VisionTcpServer


class VisionMode(str, Enum):
    OFF = "off"
    LIVE = "live"


class VisionModeManager:
    """Compose CameraOwner and stream server without coupling to robot control."""

    def __init__(
        self,
        hub: FrameHub,
        camera_owner: CameraOwner,
        stream_server: VisionTcpServer,
    ) -> None:
        self._hub = hub
        self._camera_owner = camera_owner
        self._stream_server = stream_server
        self._mode = VisionMode.OFF

    @property
    def mode(self) -> VisionMode:
        return self._mode

    def set_mode(self, mode: VisionMode) -> None:
        if mode is self._mode:
            return
        if mode is VisionMode.LIVE:
            self._start_live()
            return
        if mode is VisionMode.OFF:
            self._stop_live()
            return
        raise ValueError("unsupported Vision mode")

    def shutdown(self) -> None:
        self.set_mode(VisionMode.OFF)

    def _start_live(self) -> None:
        self._camera_owner.start()
        try:
            self._stream_server.start()
        except BaseException as start_error:
            try:
                self._camera_owner.stop()
            except BaseException as cleanup_error:
                self._hub.clear()
                raise start_error from cleanup_error
            self._hub.clear()
            raise
        self._mode = VisionMode.LIVE

    def _stop_live(self) -> None:
        # Closing all Vision TCP connections is the generation boundary.
        first_error: BaseException | None = None
        try:
            self._stream_server.stop()
        except BaseException as error:
            first_error = error

        try:
            self._camera_owner.stop()
        except BaseException as error:
            if first_error is None:
                first_error = error
        finally:
            self._hub.clear()
            self._mode = VisionMode.OFF

        if first_error is not None:
            raise first_error
