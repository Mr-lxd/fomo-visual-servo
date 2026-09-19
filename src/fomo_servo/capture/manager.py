"""Snapshot and bounded sequential recording fed by the single CameraOwner."""

from __future__ import annotations

import csv
import json
import shutil
import threading
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional, TextIO

import cv2

from fomo_servo.vision.camera_owner import CameraFacts
from fomo_servo.vision.frame import VisionFrame

from .avi_timing import rewrite_avi_frame_rate
from .session_layout import SessionPaths, plan_next_session


class CaptureError(RuntimeError):
    """Raised for explicit capture-control failures."""


class CaptureState(str, Enum):
    IDLE = "idle"
    RECORDING = "recording"
    STOPPING = "stopping"
    FAILED = "failed"
@dataclass(frozen=True)
class CaptureConfig:
    output_root: Path = Path("datasets_raw/robobeetle")
    session_prefix: str = "capture"
    max_queue_bytes: int = 64 * 1024 * 1024
    min_free_bytes: int = 512 * 1024 * 1024
    snapshot_jpeg_quality: int = 95


@dataclass
class _QueuedFrame:
    frame_id: int
    capture_timestamp_ns: int
    image: Any
    nbytes: int


def _default_writer_factory(path: Path, fps: float, size: tuple[int, int]):
    return cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        size,
    )


def _default_imwrite(path: Path, image: Any, quality: int) -> bool:
    return bool(
        cv2.imwrite(
            str(path),
            image,
            [cv2.IMWRITE_JPEG_QUALITY, int(quality)],
        )
    )


def _default_avi_frame_count(path: Path) -> int:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise CaptureError(f"unable to open finalized AVI: {path}")
        count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        if count < 0:
            raise CaptureError(f"invalid finalized AVI frame count: {count}")
        return count
    finally:
        capture.release()


class CaptureManager:
    """Own capture files while CameraOwner remains the sole camera handle owner."""

    def __init__(
        self,
        config: CaptureConfig,
        *,
        writer_factory: Optional[Callable[[Path, float, tuple[int, int]], Any]] = None,
        imwrite: Optional[Callable[[Path, Any, int], bool]] = None,
        disk_free_bytes: Optional[Callable[[Path], int]] = None,
        wall_now: Optional[Callable[[], datetime]] = None,
        avi_fps_rewriter: Optional[Callable[[Path, float], float]] = None,
        avi_frame_counter: Optional[Callable[[Path], int]] = None,
    ) -> None:
        self.config = config
        self._writer_factory = writer_factory or _default_writer_factory
        self._imwrite = imwrite or _default_imwrite
        self._disk_free_bytes = disk_free_bytes or (
            lambda path: shutil.disk_usage(str(path)).free
        )
        self._wall_now = wall_now or (lambda: datetime.now(timezone.utc))
        self._avi_fps_rewriter = avi_fps_rewriter or rewrite_avi_frame_rate
        self._avi_frame_counter = avi_frame_counter or _default_avi_frame_count

        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._action_lock = threading.Lock()
        self._metadata_lock = threading.Lock()
        self._queue: deque[_QueuedFrame] = deque()
        self._queued_bytes = 0

        self._session: Optional[SessionPaths] = None
        self._session_started: Optional[datetime] = None
        self._camera_facts: Optional[CameraFacts] = None
        self._segments: list[dict[str, Any]] = []
        self._snapshots: list[dict[str, Any]] = []
        self._state = CaptureState.IDLE
        self._accepting = False
        self._worker: Optional[threading.Thread] = None
        self._writer: Any = None
        self._index_file: Optional[TextIO] = None
        self._index_writer: Any = None
        self._current_segment: Optional[dict[str, Any]] = None
        self._last_error: Optional[str] = None
        self._first_dropped_frame_id: Optional[int] = None
        self._last_free_bytes: Optional[int] = None
        self._closed = False

    def offer_frame(self, frame: VisionFrame) -> None:
        """Non-blocking CameraOwner callback; never performs disk I/O."""

        try:
            with self._lock:
                if not self._accepting:
                    return
                segment = self._current_segment
                if segment is None:
                    return
                if (
                    frame.width != segment["width"]
                    or frame.height != segment["height"]
                ):
                    self._fail_locked(
                        "frame_size_mismatch",
                        f"frame {frame.width}x{frame.height} differs from "
                        f"writer {segment['width']}x{segment['height']}",
                        frame.frame_id,
                    )
                    return

            image = frame.image.copy()
            nbytes = int(getattr(image, "nbytes", 0))
            if nbytes <= 0:
                raise CaptureError("captured image has no byte size")
            with self._condition:
                if not self._accepting:
                    return
                if (
                    nbytes > self.config.max_queue_bytes
                    or self._queued_bytes + nbytes > self.config.max_queue_bytes
                ):
                    self._fail_locked(
                        "queue_overflow",
                        "bounded recording queue exceeded "
                        f"{self.config.max_queue_bytes} bytes",
                        frame.frame_id,
                    )
                    return
                self._queue.append(
                    _QueuedFrame(
                        frame_id=frame.frame_id,
                        capture_timestamp_ns=frame.capture_timestamp_ns,
                        image=image,
                        nbytes=nbytes,
                    )
                )
                self._queued_bytes += nbytes
                self._condition.notify()
        except BaseException as error:
            with self._condition:
                if self._accepting:
                    self._fail_locked(
                        "frame_offer_failure",
                        str(error),
                        frame.frame_id,
                    )

    def start_recording(
        self,
        facts: CameraFacts,
    ) -> dict[str, Any]:
        """Start MJPG/AVI; final playback timing is fixed at segment close."""

        with self._action_lock:
            self._ensure_open()
            self._validate_config()
            self._ensure_disk_space()
            session = self._ensure_session(facts)
            with self._lock:
                if self._accepting or (
                    self._worker is not None and self._worker.is_alive()
                ):
                    raise CaptureError("recording is already active or finalizing")

            width = facts.observed_width or 0
            height = facts.observed_height or 0
            fps = float(facts.observed_fps or 0.0)
            if width <= 0 or height <= 0:
                raise CaptureError("camera observed dimensions are unavailable")
            if not 0.0 < fps <= 1000.0:
                raise CaptureError(
                    "camera reported FPS must be within (0, 1000]"
                )

            segment_number = len(self._segments) + 1
            while True:
                filename = (
                    "raw.avi"
                    if segment_number == 1
                    else f"raw_{segment_number:03d}.avi"
                )
                path = session.session_dir / filename
                if not path.exists():
                    break
                segment_number += 1

            writer: Any = None
            index_file: Optional[TextIO] = None
            index_existed_before = session.frame_index_path.exists()
            try:
                writer = self._writer_factory(path, fps, (width, height))
                if not writer.isOpened():
                    raise CaptureError(f"unable to open MJPG writer: {path}")

                index_file = session.frame_index_path.open(
                    "a", newline="", encoding="utf-8"
                )
                index_writer = csv.writer(index_file)
                if (
                    not index_existed_before
                    or session.frame_index_path.stat().st_size == 0
                ):
                    index_writer.writerow(
                        [
                            "segment",
                            "segment_frame_index",
                            "frame_id",
                            "capture_timestamp_ns",
                        ]
                    )
                    index_file.flush()

                started = self._wall_now()
                segment = {
                    "filename": filename,
                    "width": width,
                    "height": height,
                    "camera_reported_fps": facts.observed_fps,
                    "writer_initial_fps": float(fps),
                    "container_fps": float(fps),
                    "actual_capture_fps": None,
                    "capture_elapsed_seconds": None,
                    "timing_error_ratio": None,
                    "timing_header_rewritten": False,
                    "started_utc": started.isoformat(),
                    "ended_utc": None,
                    "frame_count": 0,
                    "video_frame_count": None,
                    "frame_count_verified": False,
                    "first_frame_id": None,
                    "last_frame_id": None,
                    "first_capture_timestamp_ns": None,
                    "last_capture_timestamp_ns": None,
                    "file_size_bytes": None,
                    "status": "recording",
                    "end_reason": None,
                    "failure_error": None,
                    "first_dropped_frame_id": None,
                }
                worker = threading.Thread(
                    target=self._record_worker,
                    name="robobeetle-capture-recorder",
                    daemon=False,
                )

                with self._condition:
                    self._queue.clear()
                    self._queued_bytes = 0
                    self._writer = writer
                    self._index_file = index_file
                    self._index_writer = index_writer
                    self._current_segment = segment
                    self._state = CaptureState.RECORDING
                    self._accepting = True
                    self._last_error = None
                    self._first_dropped_frame_id = None
                    self._worker = worker

                # Persist the initial state before the worker is allowed to
                # write any video frame. CameraOwner may enqueue into the
                # bounded queue during this short interval.
                self._write_metadata()
                worker.start()
                return self.status()
            except BaseException as error:
                original_error = error
                cleanup_errors: list[str] = []

                with self._condition:
                    self._accepting = False
                    self._queue.clear()
                    self._queued_bytes = 0
                    self._writer = None
                    self._index_file = None
                    self._index_writer = None
                    self._current_segment = None
                    self._worker = None
                    self._state = CaptureState.IDLE
                    self._last_error = (
                        f"recording_start_failure: {original_error}"
                    )
                    self._condition.notify_all()

                if index_file is not None:
                    try:
                        index_file.close()
                    except Exception as cleanup_error:
                        cleanup_errors.append(
                            f"index_close={cleanup_error}"
                        )
                if writer is not None:
                    try:
                        writer.release()
                    except Exception as cleanup_error:
                        cleanup_errors.append(
                            f"writer_release={cleanup_error}"
                        )

                # The selected segment path did not exist before this start
                # attempt, so any file now present belongs to the failed
                # transaction and must not become an orphan segment.
                try:
                    if path.is_file():
                        path.unlink()
                except OSError as cleanup_error:
                    cleanup_errors.append(
                        f"avi_unlink={cleanup_error}"
                    )
                if not index_existed_before:
                    try:
                        if session.frame_index_path.is_file():
                            session.frame_index_path.unlink()
                    except OSError as cleanup_error:
                        cleanup_errors.append(
                            f"index_unlink={cleanup_error}"
                        )

                if cleanup_errors:
                    with self._condition:
                        self._state = CaptureState.FAILED
                        self._last_error = (
                            "recording_start_rollback_failure: "
                            f"original={original_error}; cleanup="
                            + "; ".join(cleanup_errors)
                        )

                # Publish either the clean idle rollback or the explicit
                # rollback-failure state. If this write itself fails, expose
                # that failure rather than pretending the rollback completed.
                try:
                    self._write_metadata()
                except Exception as cleanup_error:
                    cleanup_errors.append(
                        f"metadata_rollback={cleanup_error}"
                    )
                    with self._condition:
                        self._state = CaptureState.FAILED
                        self._last_error = (
                            "recording_start_rollback_failure: "
                            f"original={original_error}; cleanup="
                            + "; ".join(cleanup_errors)
                        )
                    try:
                        self._write_metadata()
                    except Exception as retry_error:
                        cleanup_errors.append(
                            f"metadata_retry={retry_error}"
                        )
                        with self._condition:
                            self._last_error = (
                                "recording_start_rollback_failure: "
                                f"original={original_error}; cleanup="
                                + "; ".join(cleanup_errors)
                            )

                if cleanup_errors:
                    raise CaptureError(self._last_error) from original_error
                if isinstance(original_error, CaptureError):
                    raise original_error
                raise CaptureError(
                    f"unable to start recording: {original_error}"
                ) from original_error

    def stop_recording(
        self,
        *,
        join_timeout: Optional[float] = 10.0,
    ) -> dict[str, Any]:
        """Stop accepting frames, drain the bounded queue, and close the segment."""

        with self._action_lock:
            with self._condition:
                worker = self._worker
                worker_alive = worker is not None and worker.is_alive()
                if worker_alive:
                    self._accepting = False
                    if self._state is CaptureState.RECORDING:
                        self._state = CaptureState.STOPPING
                    self._condition.notify_all()
                elif self._state is CaptureState.RECORDING:
                    self._state = CaptureState.IDLE

            if worker_alive:
                worker.join(timeout=join_timeout)
                if worker.is_alive():
                    raise CaptureError(
                        "recording worker did not stop after queue drain"
                    )

            # Always retry the authoritative metadata write, even when the
            # recorder thread already exited on its own.
            try:
                self._write_metadata()
            except BaseException as error:
                with self._condition:
                    previous = self._last_error
                    message = f"metadata_persist_failure: {error}"
                    if previous:
                        message += f"; previous={previous}"
                    self._last_error = message
                    self._state = CaptureState.FAILED
                raise CaptureError(message) from error
            return self.status()
    def snapshot(self, frame: VisionFrame, facts: CameraFacts) -> dict[str, Any]:
        """Write an untouched source-resolution frame as a JPEG snapshot."""

        with self._action_lock:
            self._ensure_open()
            self._validate_config()
            self._ensure_disk_space()
            session = self._ensure_session(facts)

            with self._lock:
                index = len(self._snapshots) + 1
            filename = (
                f"snapshot_{index:06d}_frame_{frame.frame_id:020d}.jpg"
            )
            path = session.frames_dir / filename
            while path.exists():
                index += 1
                filename = (
                    f"snapshot_{index:06d}_frame_{frame.frame_id:020d}.jpg"
                )
                path = session.frames_dir / filename

            image = frame.image.copy()
            if not self._imwrite(
                path, image, self.config.snapshot_jpeg_quality
            ):
                raise CaptureError(f"unable to write snapshot: {path}")

            record = {
                "filename": f"frames/{filename}",
                "frame_id": frame.frame_id,
                "capture_timestamp_ns": frame.capture_timestamp_ns,
                "width": frame.width,
                "height": frame.height,
                "saved_utc": self._wall_now().isoformat(),
            }
            with self._lock:
                self._snapshots.append(record)

            try:
                self._write_metadata()
            except BaseException as error:
                try:
                    self._write_metadata()
                except BaseException as retry_error:
                    message = (
                        f"snapshot_metadata_persist_failure: {error}; "
                        f"metadata_retry_failure: {retry_error}"
                    )
                    with self._condition:
                        if self._state is CaptureState.RECORDING:
                            self._fail_locked(
                                "snapshot_metadata_persist_failure",
                                (
                                    f"{error}; metadata_retry_failure: "
                                    f"{retry_error}"
                                ),
                                None,
                            )
                        else:
                            self._last_error = message
                            if self._state is not CaptureState.STOPPING:
                                self._state = CaptureState.FAILED
                            self._condition.notify_all()
                    raise CaptureError(message) from retry_error

            result = self.status()
            result["last_snapshot"] = record
            return result

    def shutdown(self) -> None:
        """Finalize recording fully before marking the capture manager closed."""

        # Clean service shutdown prioritizes file integrity over a bounded wait.
        # HTTP Stop remains bounded, but process shutdown must not publish
        # closed=true while a recorder thread still owns AVI/CSV resources.
        self.stop_recording(join_timeout=None)
        with self._lock:
            self._closed = True
        if self._session is not None:
            self._write_metadata()

    def status(self) -> dict[str, Any]:
        with self._lock:
            session = self._session
            segment = self._current_segment
            return {
                "state": self._state.value,
                "recording": self._state
                in (CaptureState.RECORDING, CaptureState.STOPPING),
                "session_id": None if session is None else session.session_id,
                "session_dir": None
                if session is None
                else str(session.session_dir),
                "segment": None
                if segment is None
                else segment["filename"],
                "container_fps": None
                if segment is None
                else segment["container_fps"],
                "recorded_frames": sum(
                    int(item["frame_count"]) for item in self._segments
                )
                + (
                    0
                    if segment is None or segment in self._segments
                    else int(segment["frame_count"])
                ),
                "snapshot_count": len(self._snapshots),
                "queue_frames": len(self._queue),
                "queue_bytes": self._queued_bytes,
                "max_queue_bytes": self.config.max_queue_bytes,
                "last_error": self._last_error,
                "first_dropped_frame_id": self._first_dropped_frame_id,
                "free_disk_bytes": self._last_free_bytes,
            }

    def _record_worker(self) -> None:
        writer = self._writer
        index_file = self._index_file
        index_writer = self._index_writer
        segment = self._current_segment
        worker_error: Optional[str] = None

        try:
            while True:
                with self._condition:
                    while not self._queue and self._accepting:
                        self._condition.wait()
                    if not self._queue and not self._accepting:
                        break
                    item = self._queue.popleft()
                    self._queued_bytes -= item.nbytes

                if int(segment["frame_count"]) % 25 == 0:
                    try:
                        free_bytes = int(
                            self._disk_free_bytes(self._session.session_dir)
                        )
                    except Exception as error:
                        with self._lock:
                            if self._first_dropped_frame_id is None:
                                self._first_dropped_frame_id = item.frame_id
                        raise CaptureError(
                            f"disk_check_failure: {error}"
                        ) from error
                    with self._lock:
                        self._last_free_bytes = free_bytes
                    if free_bytes < self.config.min_free_bytes:
                        with self._lock:
                            if self._first_dropped_frame_id is None:
                                self._first_dropped_frame_id = item.frame_id
                        raise CaptureError(
                            "low_disk_space: free disk space "
                            f"{free_bytes} below required "
                            f"{self.config.min_free_bytes} bytes"
                        )

                writer.write(item.image)
                if not writer.isOpened():
                    raise CaptureError("MJPG writer closed during recording")

                with self._lock:
                    frame_index = int(segment["frame_count"])
                    segment["frame_count"] = frame_index + 1
                    if segment["first_frame_id"] is None:
                        segment["first_frame_id"] = item.frame_id
                        segment["first_capture_timestamp_ns"] = (
                            item.capture_timestamp_ns
                        )
                    segment["last_frame_id"] = item.frame_id
                    segment["last_capture_timestamp_ns"] = (
                        item.capture_timestamp_ns
                    )

                index_writer.writerow(
                    [
                        segment["filename"],
                        frame_index,
                        item.frame_id,
                        item.capture_timestamp_ns,
                    ]
                )
                if int(segment["frame_count"]) % 25 == 0:
                    index_file.flush()
        except BaseException as error:
            worker_error = str(error)
            with self._condition:
                self._accepting = False
                self._queue.clear()
                self._queued_bytes = 0
                self._state = CaptureState.FAILED
                self._last_error = worker_error
                self._condition.notify_all()
        finally:
            release_succeeded = True
            try:
                writer.release()
            except Exception as error:
                release_succeeded = False
                if worker_error is None:
                    worker_error = f"writer release failed: {error}"
            try:
                index_file.flush()
                index_file.close()
            except Exception as error:
                if worker_error is None:
                    worker_error = f"frame index close failed: {error}"

            ended = self._wall_now()
            path = self._session.session_dir / segment["filename"]
            frame_count = int(segment["frame_count"])

            frame_count_verified = False
            if release_succeeded:
                try:
                    video_frame_count = int(self._avi_frame_counter(path))
                    segment["video_frame_count"] = video_frame_count
                    if video_frame_count != frame_count:
                        if worker_error is None:
                            worker_error = (
                                "avi_frame_count_mismatch: "
                                f"expected {frame_count}, got {video_frame_count}"
                            )
                    else:
                        frame_count_verified = True
                        segment["frame_count_verified"] = True
                except BaseException as error:
                    if worker_error is None:
                        worker_error = (
                            f"avi_frame_verification_failure: {error}"
                        )

            first_timestamp = segment["first_capture_timestamp_ns"]
            last_timestamp = segment["last_capture_timestamp_ns"]
            if (
                frame_count >= 2
                and first_timestamp is not None
                and last_timestamp is not None
                and last_timestamp > first_timestamp
            ):
                elapsed_seconds = (
                    last_timestamp - first_timestamp
                ) / 1_000_000_000.0
                actual_fps = (frame_count - 1) / elapsed_seconds
                segment["capture_elapsed_seconds"] = elapsed_seconds
                segment["actual_capture_fps"] = actual_fps

                # Never modify AVI timing after a failed release or when the
                # finalized container does not contain exactly the indexed
                # number of frames.
                if release_succeeded and frame_count_verified:
                    try:
                        finalized_fps = self._avi_fps_rewriter(path, actual_fps)
                        segment["container_fps"] = finalized_fps
                        segment["timing_header_rewritten"] = True
                        segment["timing_error_ratio"] = abs(
                            finalized_fps - actual_fps
                        ) / actual_fps
                    except BaseException as error:
                        if worker_error is None:
                            worker_error = (
                                f"avi_timing_finalize_failure: {error}"
                            )

            file_size: Optional[int] = None
            try:
                if path.is_file():
                    file_size = path.stat().st_size
            except OSError:
                file_size = None

            with self._condition:
                segment["ended_utc"] = ended.isoformat()
                segment["file_size_bytes"] = file_size

                failed = self._last_error is not None or worker_error is not None
                if failed:
                    segment["status"] = "failed"
                    segment["end_reason"] = (
                        "recording_failure"
                        if self._last_error is None
                        else self._failure_reason_from_error()
                    )
                    if self._last_error is None:
                        self._last_error = worker_error
                    segment["failure_error"] = self._last_error
                    segment["first_dropped_frame_id"] = (
                        self._first_dropped_frame_id
                    )
                    self._state = CaptureState.FAILED
                else:
                    segment["status"] = "completed"
                    segment["end_reason"] = "user_stop"
                    self._state = CaptureState.IDLE
                self._segments.append(dict(segment))
                self._current_segment = None
                self._writer = None
                self._index_file = None
                self._index_writer = None
                self._accepting = False
                self._queue.clear()
                self._queued_bytes = 0
                self._condition.notify_all()

            try:
                self._write_metadata()
            except BaseException as error:
                with self._condition:
                    previous = self._last_error
                    first_message = f"metadata_finalize_failure: {error}"
                    if previous:
                        first_message += f"; previous={previous}"
                    self._last_error = first_message
                    self._state = CaptureState.FAILED
                    if (
                        self._segments
                        and self._segments[-1]["filename"]
                        == segment["filename"]
                    ):
                        failed_segment = dict(self._segments[-1])
                        if failed_segment["status"] != "failed":
                            failed_segment["status"] = "failed"
                            failed_segment["end_reason"] = (
                                "metadata_finalize_failure"
                            )
                        failed_segment["failure_error"] = first_message
                        self._segments[-1] = failed_segment

                # Retry once after publishing the failure in memory. A later
                # Stop/shutdown call will also retry because stop_recording()
                # always persists metadata even for an already-dead worker.
                try:
                    self._write_metadata()
                except BaseException as retry_error:
                    with self._condition:
                        combined = (
                            f"{first_message}; metadata_retry_failure: "
                            f"{retry_error}"
                        )
                        self._last_error = combined
                        if (
                            self._segments
                            and self._segments[-1]["filename"]
                            == segment["filename"]
                        ):
                            failed_segment = dict(self._segments[-1])
                            failed_segment["failure_error"] = combined
                            self._segments[-1] = failed_segment

    def _fail_locked(
        self,
        reason: str,
        message: str,
        frame_id: Optional[int],
    ) -> None:
        self._accepting = False
        self._state = CaptureState.STOPPING
        self._last_error = f"{reason}: {message}"
        if frame_id is not None and self._first_dropped_frame_id is None:
            self._first_dropped_frame_id = frame_id
        self._condition.notify_all()

    def _failure_reason_from_error(self) -> str:
        error = self._last_error or ""
        if ":" in error:
            return error.split(":", 1)[0]
        return "recording_failure"

    def _ensure_session(self, facts: CameraFacts) -> SessionPaths:
        with self._lock:
            if self._session is not None:
                return self._session

        root = Path(self.config.output_root)
        root.mkdir(parents=True, exist_ok=True)
        session = plan_next_session(
            root,
            prefix=self.config.session_prefix,
        )
        with self._lock:
            if self._session is None:
                self._session = session
                self._session_started = self._wall_now()
                self._camera_facts = facts
                return session

        # Another control request won the race. Remove only our empty allocation.
        try:
            session.frames_dir.rmdir()
            session.session_dir.rmdir()
        except OSError:
            pass
        return self._session

    def _ensure_disk_space(self) -> None:
        root = Path(self.config.output_root)
        root.mkdir(parents=True, exist_ok=True)
        free = int(self._disk_free_bytes(root))
        with self._lock:
            self._last_free_bytes = free
        if free < self.config.min_free_bytes:
            raise CaptureError(
                f"free disk space {free} below required "
                f"{self.config.min_free_bytes} bytes"
            )

    def _ensure_open(self) -> None:
        with self._lock:
            if self._closed:
                raise CaptureError("capture manager is shut down")

    def _validate_config(self) -> None:
        if self.config.max_queue_bytes <= 0:
            raise CaptureError("max_queue_bytes must be positive")
        if self.config.min_free_bytes < 0:
            raise CaptureError("min_free_bytes must be non-negative")
        if not 1 <= self.config.snapshot_jpeg_quality <= 100:
            raise CaptureError("snapshot_jpeg_quality must be within 1..100")

    def _metadata_snapshot(self) -> Optional[dict[str, Any]]:
        with self._lock:
            if self._session is None:
                return None
            facts = self._camera_facts
            current = (
                None
                if self._current_segment is None
                else dict(self._current_segment)
            )
            return {
                "schema_version": 1,
                "kind": "robobeetle_capture_session",
                "session_id": self._session.session_id,
                "session_started_utc": None
                if self._session_started is None
                else self._session_started.isoformat(),
                "updated_utc": self._wall_now().isoformat(),
                "closed": self._closed,
                "camera": None
                if facts is None
                else {
                    "source": facts.source,
                    "observed_width": facts.observed_width,
                    "observed_height": facts.observed_height,
                    "observed_fps": facts.observed_fps,
                    "observed_fourcc": facts.observed_fourcc,
                },
                "recording_state": self._state.value,
                "queue": {
                    "max_bytes": self.config.max_queue_bytes,
                    "queued_bytes": self._queued_bytes,
                    "queued_frames": len(self._queue),
                },
                "free_disk_bytes": self._last_free_bytes,
                "minimum_free_disk_bytes": self.config.min_free_bytes,
                "last_error": self._last_error,
                "first_dropped_frame_id": self._first_dropped_frame_id,
                "segments": [dict(item) for item in self._segments],
                "current_segment": current,
                "snapshots": [dict(item) for item in self._snapshots],
            }
    def _write_metadata(self) -> None:
        # Serialize snapshot creation and replacement together. Otherwise an
        # older snapshot can pause before this lock and later overwrite a newer
        # finalized state written by another thread.
        with self._metadata_lock:
            metadata = self._metadata_snapshot()
            if metadata is None:
                return
            session = self._session
            if session is None:
                return
            payload = json.dumps(
                metadata,
                ensure_ascii=False,
                indent=2,
            ) + "\n"
            temp_path = session.metadata_path.with_suffix(".json.tmp")
            temp_path.write_text(payload, encoding="utf-8")
            temp_path.replace(session.metadata_path)
