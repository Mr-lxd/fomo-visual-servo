"""Thread-safe latest-frame fan-out boundary for realtime Vision consumers."""

from __future__ import annotations

import threading
import time
from typing import Optional

from .frame import VisionFrame


class FrameHub:
    """Bounded single-slot hub: publishing a new frame replaces the old frame."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._latest: Optional[VisionFrame] = None
        self._published_count = 0
        self._replacement_count = 0

    @property
    def published_count(self) -> int:
        with self._condition:
            return self._published_count

    @property
    def replacement_count(self) -> int:
        with self._condition:
            return self._replacement_count

    def snapshot(self) -> Optional[VisionFrame]:
        with self._condition:
            return self._latest

    def publish(self, frame: VisionFrame) -> None:
        """Replace the realtime slot and wake consumers waiting for a newer frame."""

        with self._condition:
            if self._latest is not None and frame.frame_id <= self._latest.frame_id:
                raise ValueError("published frame_id must strictly increase")
            if self._latest is not None:
                self._replacement_count += 1
            self._latest = frame
            self._published_count += 1
            self._condition.notify_all()

    def clear(self) -> None:
        """Drop the cached frame at a CameraOwner generation boundary."""

        with self._condition:
            self._latest = None
            self._condition.notify_all()

    def wait_for_newer(
        self, after_frame_id: Optional[int], timeout: Optional[float] = None
    ) -> Optional[VisionFrame]:
        """Return the newest frame newer than after_frame_id, never a backlog."""

        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while True:
                latest = self._latest
                if latest is not None and (
                    after_frame_id is None or latest.frame_id > after_frame_id
                ):
                    return latest
                if deadline is None:
                    self._condition.wait()
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return None
                self._condition.wait(remaining)
