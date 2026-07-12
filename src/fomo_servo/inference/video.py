"""Latest-frame video buffering for real-time FOMO inference."""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

import cv2


@dataclass(frozen=True)
class FramePacket:
    """A captured frame with source index and seconds timestamp."""

    frame_index: int
    timestamp: float
    frame: Any


class LatestFrameBuffer:
    """Bounded one-frame buffer that replaces stale frames instead of accumulating them."""

    def __init__(self) -> None:
        self._queue: queue.Queue[Optional[FramePacket]] = queue.Queue(maxsize=1)

    def put_latest(self, packet: FramePacket) -> None:
        """Put a packet, dropping the currently buffered older packet if necessary."""

        try:
            self._queue.put_nowait(packet)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            self._queue.put_nowait(packet)

    def close(self) -> None:
        """Signal end of capture."""

        try:
            self._queue.put_nowait(None)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            self._queue.put_nowait(None)

    def get(self, timeout: float = 1.0) -> Optional[FramePacket]:
        """Return the newest available packet or ``None`` on timeout/end signal."""

        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None


class LatestFrameReader:
    """Read OpenCV frames on a worker thread and expose only the latest buffered frame."""

    def __init__(self, capture: cv2.VideoCapture) -> None:
        self.capture = capture
        self.buffer = LatestFrameBuffer()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="fomo-latest-frame-reader", daemon=True)

    def start(self) -> "LatestFrameReader":
        """Start the capture worker."""

        self._thread.start()
        return self

    def stop(self) -> None:
        """Stop the worker and release the capture resource."""

        self._stop.set()
        self._thread.join(timeout=2.0)
        self.capture.release()

    def _run(self) -> None:
        frame_index = 0
        try:
            while not self._stop.is_set():
                success, frame = self.capture.read()
                if not success:
                    break
                timestamp_ms = float(self.capture.get(cv2.CAP_PROP_POS_MSEC))
                timestamp = timestamp_ms / 1000.0 if timestamp_ms > 0 else time.time()
                self.buffer.put_latest(FramePacket(frame_index, timestamp, frame))
                frame_index += 1
        finally:
            self.buffer.close()
