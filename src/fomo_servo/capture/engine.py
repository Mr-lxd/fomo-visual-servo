"""Capture engine: raw-frame recording, snapshots, and preview HUD state.

The engine never runs inference and never touches model code. Recording
segments and snapshots always receive the untouched raw camera frame; the HUD
is drawn only on a preview copy (:mod:`fomo_servo.capture.hud`).

Recording produces MJPG ``.avi`` segments at the camera's negotiated
resolution (no resize, no color conversion, no overlays). The first segment
of a session is ``raw.avi``; later segments are ``raw_002.avi``, ...
"""

from __future__ import annotations

import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import cv2

from .hud import INSTRUCTIONS, WINDOW_NAME, draw_status_overlay
from .metadata import CaptureSessionRecord, write_metadata
from .session_layout import SessionPaths


class CaptureError(RuntimeError):
    """Raised when the camera, writer, disk, or snapshot contract fails."""

    def __init__(self, message: str, *, end_reason: str = "capture_error") -> None:
        super().__init__(message)
        self.end_reason = end_reason


class KeyCommand(Enum):
    """Field keyboard commands."""

    NONE = "none"
    TOGGLE_RECORDING = "toggle_recording"
    NEXT_SESSION = "next_session"
    SNAPSHOT = "snapshot"
    QUIT = "quit"


KEY_COMMANDS: Mapping[int, KeyCommand] = {
    ord(" "): KeyCommand.TOGGLE_RECORDING,
    ord("n"): KeyCommand.NEXT_SESSION,
    ord("N"): KeyCommand.NEXT_SESSION,
    ord("s"): KeyCommand.SNAPSHOT,
    ord("S"): KeyCommand.SNAPSHOT,
    ord("q"): KeyCommand.QUIT,
    ord("Q"): KeyCommand.QUIT,
    27: KeyCommand.QUIT,
}


def _default_writer_factory(path: Path, fps: float, size: tuple[int, int]):
    return cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)


def _default_create_window(name: str, fullscreen: bool) -> None:
    try:
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        if fullscreen:
            cv2.setWindowProperty(
                name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
            )
    except cv2.error as error:
        raise CaptureError(
            "OpenCV could not create preview window '{}'; a GUI-enabled OpenCV "
            "build is required for --display: {}".format(name, error)
        ) from error


def _default_window_visible(name: str) -> bool:
    try:
        return float(cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE)) >= 1.0
    except cv2.error:
        return False


@dataclass(frozen=True)
class CaptureIO:
    """Injectable I/O surface so tests can run without a camera or display."""

    clock: Callable[[], float] = time.monotonic
    wall_now: Callable[[], datetime] = lambda: datetime.now(timezone.utc)
    wait_key: Callable[[int], int] = cv2.waitKey
    imshow: Callable[[str, Any], None] = cv2.imshow
    create_window: Callable[[str, bool], None] = _default_create_window
    window_visible: Callable[[str], bool] = _default_window_visible
    destroy_windows: Callable[[], None] = cv2.destroyAllWindows
    imwrite: Callable[[str, Any], bool] = cv2.imwrite
    disk_free_bytes: Callable[[Path], int] = (
        lambda path: shutil.disk_usage(str(path)).free
    )


class CaptureEngine:
    """Drive one capture run: recording segments, snapshots, sessions, HUD."""

    def __init__(
        self,
        capture: Any,
        *,
        planner: Callable[[], SessionPaths],
        facts: Any,
        scene: str = "",
        target: str = "",
        notes: str = "",
        frame_interval_seconds: Optional[float] = None,
        io: Optional[CaptureIO] = None,
        writer_factory: Optional[Callable[[Path, float, tuple[int, int]], Any]] = None,
        window_name: str = WINDOW_NAME,
    ) -> None:
        self._capture = capture
        self._planner = planner
        self._facts = facts
        self._scene = scene
        self._target = target
        self._notes = notes
        self._frame_interval_seconds = frame_interval_seconds
        self.io = io if io is not None else CaptureIO()
        self._writer_factory = writer_factory or _default_writer_factory
        self.window_name = window_name

        self._record: Optional[CaptureSessionRecord] = None
        self._recording = False
        self._writer = None
        self._writer_path: Optional[Path] = None
        self._segment_frame_count = 0
        self._segment_started = 0.0
        self._frames_seen = 0
        self._loop_started = self.io.clock()
        self._last_interval_snapshot = self._loop_started
        self._free_gb: Optional[float] = None
        self._low_disk = False
        self._next_disk_poll = 0.0
        self.session_summaries: list[dict[str, Any]] = []
        self._begin_session()

    # ---------------------------------------------------------------- state

    @property
    def session_id(self) -> str:
        return self._current_record().session.session_id

    @property
    def session_dir(self) -> Path:
        return self._current_record().session.session_dir

    @property
    def recording(self) -> bool:
        return self._recording

    @property
    def frames_seen(self) -> int:
        return self._frames_seen

    @property
    def snapshot_count(self) -> int:
        return self._current_record().snapshot_count

    @property
    def video_frame_count(self) -> int:
        count = self._current_record().video_frame_count
        if self._recording:
            count += self._segment_frame_count
        return count

    @property
    def low_disk(self) -> bool:
        return self._low_disk

    def _current_record(self) -> CaptureSessionRecord:
        if self._record is None:
            raise CaptureError("no capture session is active")
        return self._record

    # ------------------------------------------------------------ recording

    def handle_key(self, key: int) -> KeyCommand:
        """Map a waitKey value to its field command."""

        return KEY_COMMANDS.get(int(key) & 0xFF, KeyCommand.NONE)

    def toggle_recording(self) -> None:
        """Start or stop the current recording segment."""

        if self._recording:
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self) -> None:
        """Open the next raw MJPG segment at the observed camera size."""

        record = self._current_record()
        index = len(record.video_files) + 1
        used_filenames = {str(item["filename"]) for item in record.video_files}
        while True:
            filename = "raw.avi" if index == 1 else "raw_{:03d}.avi".format(index)
            path = record.session.session_dir / filename
            if filename not in used_filenames and not path.exists():
                break
            index += 1
        size = (
            self._facts.observed_width or 0,
            self._facts.observed_height or 0,
        )
        fps = self._writer_fps()
        writer = self._writer_factory(path, fps, size)
        if not writer.isOpened():
            try:
                writer.release()
            except Exception:
                pass
            raise CaptureError(
                "unable to open video writer for '{}' (MJPG, fps {}, size {})".format(
                    path, fps, size
                ),
                end_reason="video_writer_open_failure",
            )
        self._writer = writer
        self._writer_path = path
        self._recording = True
        self._segment_frame_count = 0
        self._segment_started = self.io.clock()

    def stop_recording(self) -> None:
        """Finalize the current segment so the file is closed and playable."""

        if not self._recording:
            return
        seconds = max(self.io.clock() - self._segment_started, 0.0)
        writer = self._writer
        writer_path = self._writer_path
        release_error: Optional[Exception] = None
        try:
            writer.release()
        except Exception as error:
            release_error = error
        file_size_bytes: Optional[int] = None
        try:
            if writer_path is not None and writer_path.is_file():
                file_size_bytes = writer_path.stat().st_size
        except OSError:
            file_size_bytes = None
        self._current_record().add_video_file(
            filename=writer_path.name,
            frame_count=self._segment_frame_count,
            duration_seconds=seconds,
            container_fps=self._writer_fps(),
            file_size_bytes=file_size_bytes,
        )
        self._recording = False
        self._writer = None
        self._writer_path = None
        if release_error is not None:
            raise CaptureError(
                "unable to release video writer '{}': {}".format(
                    writer_path, release_error
                ),
                end_reason="video_writer_release_failure",
            ) from release_error

    def _writer_fps(self) -> float:
        return (
            self._facts.observed_fps
            or self._facts.requested_fps
            or 30.0
        )

    # ------------------------------------------------------------- snapshot

    def snapshot(self, frame: Any, *, manual: bool = True) -> Path:
        """Write the raw frame as the next sequential session JPEG."""

        record = self._current_record()
        index = record.snapshot_count + 1
        while True:
            path = record.session.frames_dir / "frame_{:06d}.jpg".format(index)
            if not path.exists():
                break
            index += 1
        if not self.io.imwrite(str(path), frame):
            raise CaptureError(
                "unable to write snapshot '{}'".format(path),
                end_reason="snapshot_write_failure",
            )
        record.snapshot_count += 1
        return path

    def _maybe_interval_snapshot(self, frame: Any) -> None:
        if self._frame_interval_seconds is None:
            return
        now = self.io.clock()
        if now - self._last_interval_snapshot >= self._frame_interval_seconds:
            self._last_interval_snapshot = now
            self.snapshot(frame, manual=False)

    # -------------------------------------------------------------- sessions

    def _begin_session(self) -> None:
        session = self._planner()
        self._record = CaptureSessionRecord(
            session=session,
            facts=self._facts,
            scene=self._scene,
            target=self._target,
            notes=self._notes,
            frame_interval_seconds=self._frame_interval_seconds,
            start_time=self.io.wall_now(),
        )
        self._recording = False
        self._writer = None
        self._writer_path = None
        self._segment_frame_count = 0

    def _handle_next_session(self) -> None:
        if self._recording:
            self.stop_recording()
        self.finish_session(status="completed", end_reason="user_new_session")
        self._begin_session()

    def finish_session(
        self, *, status: str, end_reason: str
    ) -> dict[str, Any]:
        """Finalize the current session and write its metadata.json."""

        record = self._current_record()
        metadata = record.finalize(
            status=status,
            end_reason=end_reason,
            end_time=self.io.wall_now(),
            measured_capture_fps=self._measured_capture_fps(),
        )
        write_metadata(record.session.metadata_path, metadata)
        self.session_summaries.append(
            {
                "session_id": record.session.session_id,
                "metadata_path": str(record.session.metadata_path),
                "status": status,
                "end_reason": end_reason,
                "video_frame_count": record.video_frame_count,
                "snapshot_count": record.snapshot_count,
            }
        )
        return metadata

    # ------------------------------------------------------------------- HUD

    def hud_state(self) -> dict[str, Any]:
        """Build the preview HUD state (never written to recorded data)."""

        record = self._current_record()
        if self._recording:
            rec_seconds = record.recording_seconds + (
                self.io.clock() - self._segment_started
            )
        else:
            rec_seconds = record.recording_seconds
        observed = self._facts
        resolution = (
            "{}x{}".format(observed.observed_width, observed.observed_height)
            if observed.observed_width and observed.observed_height
            else "unknown"
        )
        measured = self._measured_capture_fps()
        return {
            "camera": observed.source,
            "resolution": resolution,
            "fps": "{:.1f}".format(measured) if measured is not None else "--",
            "free_space_text": (
                "{:.1f} GB".format(self._free_gb)
                if self._free_gb is not None
                else "unknown"
            ),
            "session_id": record.session.session_id,
            "recording": self._recording,
            "rec_seconds": rec_seconds,
            "video_frames": self.video_frame_count,
            "snapshot_count": record.snapshot_count,
            "low_disk_space": self._low_disk,
        }

    def _measured_capture_fps(self) -> Optional[float]:
        elapsed = self.io.clock() - self._loop_started
        if elapsed <= 0.0 or self._frames_seen == 0:
            return None
        return round(self._frames_seen / elapsed, 2)

    def _poll_disk(self, min_free_gb: float) -> None:
        now = self.io.clock()
        if now < self._next_disk_poll:
            return
        self._next_disk_poll = now + 2.0
        try:
            free_bytes = self.io.disk_free_bytes(self.session_dir)
        except Exception as error:
            raise CaptureError(
                "unable to check free disk space: {}".format(error)
            ) from error
        self._free_gb = free_bytes / 1e9
        self._low_disk = self._free_gb < min_free_gb

    # ------------------------------------------------------------------- run

    def run(
        self,
        *,
        display: bool = False,
        fullscreen: bool = False,
        max_frames: Optional[int] = None,
        duration_seconds: Optional[float] = None,
        min_free_gb: float = 5.0,
        start_recording: bool = False,
    ) -> tuple[str, str]:
        """Run the capture loop and return ``(status, end_reason)``.

        Recording segments and snapshots receive raw frames only. On QUIT,
        window close, limits, camera failure, Ctrl+C, or unexpected exceptions the run
        finalizes the open segment, releases the camera, and writes
        ``metadata.json`` for the active session.
        """

        status, end_reason = "completed", "user_quit"
        self._loop_started = self.io.clock()
        self._last_interval_snapshot = self._loop_started
        self._next_disk_poll = 0.0
        try:
            if display:
                self.io.create_window(self.window_name, fullscreen)
            if start_recording:
                self.start_recording()
            while True:
                self._poll_disk(min_free_gb)
                ok, frame = self._capture.read()
                if not ok:
                    raise CaptureError(
                        "camera read failed: no frame returned",
                        end_reason="camera_read_failure",
                    )
                self._frames_seen += 1
                if self._recording and self._writer is not None:
                    shape = getattr(frame, "shape", ())
                    actual_size = (
                        (int(shape[1]), int(shape[0])) if len(shape) >= 2 else None
                    )
                    expected_size = (
                        self._facts.observed_width,
                        self._facts.observed_height,
                    )
                    if actual_size != expected_size:
                        raise CaptureError(
                            "camera frame size {} does not match video writer size {}".format(
                                actual_size, expected_size
                            ),
                            end_reason="frame_size_mismatch",
                        )
                    self._writer.write(frame)
                    if not self._writer.isOpened():
                        raise CaptureError(
                            "video writer closed while recording '{}'".format(
                                self._writer_path
                            ),
                            end_reason="video_writer_write_failure",
                        )
                    self._segment_frame_count += 1
                self._maybe_interval_snapshot(frame)
                if display:
                    preview = draw_status_overlay(frame, self.hud_state())
                    self.io.imshow(self.window_name, preview)
                    command = self.handle_key(self.io.wait_key(1))
                    if command is KeyCommand.QUIT:
                        break
                    if command is KeyCommand.TOGGLE_RECORDING:
                        self.toggle_recording()
                    elif command is KeyCommand.SNAPSHOT:
                        self.snapshot(frame)
                    elif command is KeyCommand.NEXT_SESSION:
                        self._handle_next_session()
                    if not self.io.window_visible(self.window_name):
                        end_reason = "window_closed"
                        break
                if max_frames is not None and self._frames_seen >= max_frames:
                    end_reason = "max_frames"
                    break
                if (
                    duration_seconds is not None
                    and self.io.clock() - self._loop_started >= duration_seconds
                ):
                    end_reason = "duration_reached"
                    break
        except KeyboardInterrupt:
            status, end_reason = "interrupted", "keyboard_interrupt"
        except CaptureError as error:
            status, end_reason = "failed", error.end_reason
            raise
        except Exception:
            status, end_reason = "failed", "unexpected_exception"
            raise
        finally:
            self._cleanup(display=display, status=status, end_reason=end_reason)
        return status, end_reason

    def _cleanup(self, *, display: bool, status: str, end_reason: str) -> None:
        """Finalize the open segment, release resources, and write metadata.

        Best effort while another exception is propagating (errors are then
        printed, not raised); otherwise cleanup failures raise
        :class:`CaptureError`.
        """

        failure_in_flight = sys.exc_info()[0] is not None
        errors: list[str] = []
        if self._recording:
            try:
                self.stop_recording()
            except Exception as error:
                errors.append("recording cleanup failed: {}".format(error))
                if status != "failed":
                    status = "failed"
                    end_reason = getattr(error, "end_reason", "cleanup_failure")
        try:
            self._capture.release()
        except Exception as error:
            errors.append("camera release failed: {}".format(error))
            if status != "failed":
                status, end_reason = "failed", "camera_release_failure"
        if display:
            try:
                self.io.destroy_windows()
            except Exception as error:
                errors.append("window cleanup failed: {}".format(error))
                if status != "failed":
                    status, end_reason = "failed", "window_cleanup_failure"
        if self._record is not None and not self._record.finalized:
            try:
                self.finish_session(status=status, end_reason=end_reason)
            except Exception as error:
                errors.append("metadata write failed: {}".format(error))
        if errors:
            if failure_in_flight:
                for message in errors:
                    print("Error: {}".format(message), file=sys.stderr)
            else:
                raise CaptureError("; ".join(errors), end_reason=end_reason)


__all__ = [
    "CaptureEngine",
    "CaptureError",
    "CaptureIO",
    "INSTRUCTIONS",
    "KeyCommand",
    "WINDOW_NAME",
]
