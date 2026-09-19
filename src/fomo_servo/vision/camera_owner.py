"""Single-owner OpenCV camera lifecycle for the RoboBeetle Vision runtime."""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Optional

import cv2

from .frame import PixelFormat, VisionFrame
from .frame_hub import FrameHub
from .protocol import MAX_DECODED_PIXELS, MAX_DIMENSION

_MAX_FRAME_ID = (1 << 64) - 1
_CADENCE_WINDOW_SIZE = 64
_MIN_CADENCE_SAMPLES = 8


class CameraOwnerError(RuntimeError):
    """Raised when the single camera owner cannot satisfy its lifecycle contract."""


class CameraReadError(CameraOwnerError):
    """Raised asynchronously when a running camera stops returning frames."""


@dataclass(frozen=True)
class CameraFacts:
    source: str
    observed_width: Optional[int]
    observed_height: Optional[int]
    observed_fps: Optional[float]
    observed_fourcc: Optional[str]


def _decode_fourcc(value: float) -> Optional[str]:
    packed = int(value)
    if packed <= 0:
        return None
    return "".join(chr((packed >> (8 * index)) & 0xFF) for index in range(4))


def _positive_int(value: float) -> Optional[int]:
    return int(value) if value > 0 else None


def _positive_float(value: float) -> Optional[float]:
    return float(value) if value > 0 else None


class CameraOwner:
    """Own exactly one VideoCapture and publish raw BGR frames into FrameHub."""

    def __init__(
        self,
        hub: FrameHub,
        *,
        source: int | str = 0,
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps: Optional[float] = None,
        fourcc: Optional[str] = None,
        capture_factory: Optional[Callable[[int | str], Any]] = None,
        clock_ns: Callable[[], int] = time.monotonic_ns,
        frame_callback: Optional[Callable[[VisionFrame], None]] = None,
    ) -> None:
        self._hub = hub
        self._source = source
        self._requested_width = width
        self._requested_height = height
        self._requested_fps = fps
        self._requested_fourcc = fourcc
        self._capture_factory = capture_factory or cv2.VideoCapture
        self._clock_ns = clock_ns
        self._frame_callback = frame_callback

        self._capture: Any = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._release_lock = threading.Lock()
        self._released = True
        self._cadence_lock = threading.Lock()
        self._capture_timestamps_ns: deque[int] = deque(
            maxlen=_CADENCE_WINDOW_SIZE
        )

        self.finished = threading.Event()
        self.error: Optional[BaseException] = None
        self.facts: Optional[CameraFacts] = None
        self.frames_captured = 0
        self.frame_callback_errors = 0
        self.last_frame_callback_error: Optional[str] = None

    @property
    def is_running(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive() and not self.finished.is_set()

    @property
    def measured_capture_fps(self) -> Optional[float]:
        """Return recent delivered-frame cadence from Pi monotonic timestamps."""

        with self._cadence_lock:
            timestamps = tuple(self._capture_timestamps_ns)
        if len(timestamps) < _MIN_CADENCE_SAMPLES:
            return None
        elapsed_ns = timestamps[-1] - timestamps[0]
        if elapsed_ns <= 0:
            return None
        fps = (len(timestamps) - 1) * 1_000_000_000.0 / elapsed_ns
        return fps if fps > 0.0 else None

    @property
    def cadence_sample_count(self) -> int:
        with self._cadence_lock:
            return len(self._capture_timestamps_ns)

    def start(self) -> "CameraOwner":
        """Open/configure the camera synchronously, then start the capture worker."""

        if self.is_running:
            raise CameraOwnerError("CameraOwner is already running")
        self._validate_requested_configuration()
        self._hub.clear()
        self._stop.clear()
        self.finished.clear()
        self.error = None
        self.facts = None
        self.frames_captured = 0
        self.frame_callback_errors = 0
        self.last_frame_callback_error = None
        with self._cadence_lock:
            self._capture_timestamps_ns.clear()

        source_value: int | str = self._source
        if isinstance(source_value, str) and source_value.isdigit():
            source_value = int(source_value)

        capture = self._capture_factory(source_value)
        self._capture = capture
        self._released = False

        try:
            if not capture.isOpened():
                raise CameraOwnerError(
                    "unable to open video source: {}".format(self._source)
                )
            if self._requested_fourcc is not None:
                capture.set(
                    cv2.CAP_PROP_FOURCC,
                    float(cv2.VideoWriter_fourcc(*self._requested_fourcc)),
                )
            if self._requested_width is not None:
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(self._requested_width))
            if self._requested_height is not None:
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self._requested_height))
            if self._requested_fps is not None:
                capture.set(cv2.CAP_PROP_FPS, float(self._requested_fps))
            self.facts = CameraFacts(
                source=str(self._source),
                observed_width=_positive_int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                observed_height=_positive_int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                observed_fps=_positive_float(capture.get(cv2.CAP_PROP_FPS)),
                observed_fourcc=_decode_fourcc(capture.get(cv2.CAP_PROP_FOURCC)),
            )
        except BaseException:
            self._release_capture()
            raise

        self._thread = threading.Thread(
            target=self._run,
            name="robobeetle-camera-owner",
            daemon=True,
        )
        self._thread.start()
        return self

    def stop(self, *, join_timeout: float = 2.0) -> None:
        """Stop capture and deterministically release the owned camera handle."""

        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=join_timeout)
            if thread.is_alive():
                self._release_capture()
                thread.join(timeout=join_timeout)
            if thread.is_alive():
                raise CameraOwnerError("camera worker did not stop after release")
        self._release_capture()

    def _run(self) -> None:
        frame_id = 0
        try:
            while not self._stop.is_set():
                success, image = self._capture.read()
                if not success:
                    if not self._stop.is_set():
                        self.error = CameraReadError(
                            "camera read failed: no complete frame returned"
                        )
                    break

                capture_timestamp_ns = self._clock_ns()
                shape = getattr(image, "shape", ())
                if len(shape) != 3 or int(shape[2]) != 3:
                    raise CameraOwnerError(
                        "camera frame must be an HxWx3 BGR image"
                    )
                with self._cadence_lock:
                    self._capture_timestamps_ns.append(
                        capture_timestamp_ns
                    )

                height = int(shape[0])
                width = int(shape[1])
                frame = VisionFrame(
                    frame_id=frame_id,
                    capture_timestamp_ns=capture_timestamp_ns,
                    width=width,
                    height=height,
                    pixel_format=PixelFormat.BGR8,
                    image=image,
                )
                self._hub.publish(frame)
                callback = self._frame_callback
                if callback is not None:
                    try:
                        callback(frame)
                    except BaseException as error:
                        self.frame_callback_errors += 1
                        self.last_frame_callback_error = str(error)
                self.frames_captured += 1

                if frame_id == _MAX_FRAME_ID:
                    raise CameraOwnerError(
                        "frame_id exhausted; a new CameraOwner generation is required"
                    )
                frame_id += 1
        except BaseException as error:
            if self.error is None:
                self.error = error
        finally:
            self._release_capture()
            self.finished.set()

    def _release_capture(self) -> None:
        with self._release_lock:
            if self._released:
                return
            capture = self._capture
            self._released = True
            if capture is not None:
                capture.release()

    def _validate_requested_configuration(self) -> None:
        if self._requested_width is not None:
            if not 1 <= self._requested_width <= MAX_DIMENSION:
                raise CameraOwnerError("requested width is outside the supported range")
        if self._requested_height is not None:
            if not 1 <= self._requested_height <= MAX_DIMENSION:
                raise CameraOwnerError("requested height is outside the supported range")
        if self._requested_width is not None and self._requested_height is not None:
            if self._requested_width * self._requested_height > MAX_DECODED_PIXELS:
                raise CameraOwnerError(
                    "requested decoded pixel count exceeds the supported limit"
                )
        if self._requested_fps is not None and self._requested_fps <= 0:
            raise CameraOwnerError("requested fps must be positive")
        if self._requested_fourcc is not None:
            if len(self._requested_fourcc) != 4 or not self._requested_fourcc.isascii():
                raise CameraOwnerError("requested FourCC must be exactly four ASCII characters")
