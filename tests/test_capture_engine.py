"""Capture engine tests: state machine, data safety, termination, cleanup."""

from __future__ import annotations

import json
from datetime import date

import cv2
import numpy as np
import pytest

from fomo_servo.capture.engine import (
    CaptureEngine,
    CaptureError,
    CaptureIO,
    KeyCommand,
)
from fomo_servo.capture.metadata import CameraFacts
from fomo_servo.capture.session_layout import plan_next_session


FOURCC_MJPG = float(int.from_bytes(b"MJPG", "little"))


def _facts() -> CameraFacts:
    return CameraFacts(
        source="0",
        backend="V4L2",
        requested_width=None,
        requested_height=None,
        requested_fps=None,
        observed_width=16,
        observed_height=8,
        observed_fps=5.0,
        observed_fourcc="MJPG",
        controls={"exposure": None, "gain": None, "brightness": 128.0},
    )


class _FakeCapture:
    def __init__(
        self,
        frames,
        *,
        props=None,
        failure="eof",
        fail_after=None,
    ) -> None:
        self._frames = list(frames)
        self._props = props or {}
        self._failure = failure
        self._fail_after = fail_after
        self._read_count = 0
        self.released = False

    def read(self):
        self._read_count += 1
        if self._fail_after is not None and self._read_count > self._fail_after:
            if self._failure == "error":
                raise RuntimeError("camera read failure")
            if self._failure == "interrupt":
                raise KeyboardInterrupt()
            return False, None
        if not self._frames:
            return False, None
        return True, self._frames.pop(0)

    def get(self, prop):
        return float(self._props.get(prop, 0.0))

    def set(self, prop, value):
        return True

    def isOpened(self):
        return True

    def release(self):
        self.released = True


class _FakeWriter:
    def __init__(self, path, fps, size, *, opened=True, release_error=None) -> None:
        self.path = path
        self.fps = fps
        self.size = size
        self.frames = []
        self.released = False
        self._opened = opened
        self._release_error = release_error

    def write(self, frame):
        self.frames.append(frame.copy())

    def release(self):
        self.released = True
        if self._release_error is not None:
            raise self._release_error

    def isOpened(self):
        return self._opened


class _FakeDisplay:
    def __init__(self, *, keys=None, visible=True, free_bytes=100 << 30) -> None:
        self._keys = list(keys or [])
        self._visible = visible
        self._free_bytes = free_bytes
        self.shown_frames = []
        self.windows_created = []
        self.destroyed = False
        self.written_images = []

    def wait_key(self, timeout):
        if self._keys:
            return self._keys.pop(0)
        return -1

    def imshow(self, name, frame):
        self.shown_frames.append(frame.copy())

    def create_window(self, name, fullscreen):
        self.windows_created.append((name, fullscreen))

    def window_visible(self, name):
        return self._visible

    def destroy_windows(self):
        self.destroyed = True

    def imwrite(self, path, frame):
        self.written_images.append((str(path), frame.copy()))
        return True

    def disk_free_bytes(self, path):
        return self._free_bytes


def _clock_stepper(step=0.0, start=0.0):
    state = {"now": start}

    def clock():
        state["now"] += step
        return state["now"]

    return clock


def _make_engine(
    tmp_path,
    frames,
    *,
    props=None,
    keys=None,
    visible=True,
    free_bytes=100 << 30,
    clock_step=0.01,
    writer_opened=True,
    writer_release_error=None,
    failure="eof",
    fail_after=None,
    frame_interval_seconds=None,
):
    capture = _FakeCapture(
        frames, props=props, failure=failure, fail_after=fail_after
    )
    display = _FakeDisplay(keys=keys, visible=visible, free_bytes=free_bytes)
    writers = []

    def writer_factory(path, fps, size):
        writer = _FakeWriter(
            path,
            fps,
            size,
            opened=writer_opened,
            release_error=writer_release_error,
        )
        writers.append(writer)
        return writer

    output_root = tmp_path / "datasets_raw" / "lab_pool"

    def planner():
        return plan_next_session(output_root, prefix="pool", date=date(2026, 8, 31))

    io = CaptureIO(
        clock=_clock_stepper(step=clock_step),
        wait_key=display.wait_key,
        imshow=display.imshow,
        create_window=display.create_window,
        window_visible=display.window_visible,
        destroy_windows=display.destroy_windows,
        imwrite=display.imwrite,
        disk_free_bytes=display.disk_free_bytes,
    )
    engine = CaptureEngine(
        capture,
        planner=planner,
        facts=_facts(),
        scene="pool-scene",
        target="fish-target",
        notes="note-1",
        frame_interval_seconds=frame_interval_seconds,
        io=io,
        writer_factory=writer_factory,
    )
    return engine, display, writers, capture


def _frames(count, height=8, width=16):
    return [
        np.full((height, width, 3), 40 + (index % 200), dtype=np.uint8)
        for index in range(count)
    ]


def _read_metadata(session_dir):
    return json.loads((session_dir / "metadata.json").read_text(encoding="utf-8"))


def test_key_mapping_matches_field_shortcuts(tmp_path) -> None:
    engine, _display, _writers, _capture = _make_engine(tmp_path, _frames(1))
    assert engine.handle_key(ord(" ")) is KeyCommand.TOGGLE_RECORDING
    assert engine.handle_key(ord("n")) is KeyCommand.NEXT_SESSION
    assert engine.handle_key(ord("N")) is KeyCommand.NEXT_SESSION
    assert engine.handle_key(ord("s")) is KeyCommand.SNAPSHOT
    assert engine.handle_key(ord("S")) is KeyCommand.SNAPSHOT
    assert engine.handle_key(ord("q")) is KeyCommand.QUIT
    assert engine.handle_key(ord("Q")) is KeyCommand.QUIT
    assert engine.handle_key(27) is KeyCommand.QUIT
    assert engine.handle_key(ord("x")) is KeyCommand.NONE


def test_recording_lifecycle_uses_segmented_raw_files(tmp_path) -> None:
    engine, _display, writers, _capture = _make_engine(tmp_path, _frames(2))

    engine.toggle_recording()
    assert engine.recording is True
    assert writers[0].path.name == "raw.avi"
    assert writers[0].fps == pytest.approx(5.0)
    assert writers[0].size == (16, 8)

    engine.toggle_recording()
    assert engine.recording is False
    assert writers[0].released is True
    assert engine.video_frame_count == 0

    engine.toggle_recording()
    assert engine.recording is True
    assert writers[1].path.name == "raw_002.avi"


def test_space_key_records_raw_frames_without_overlay(tmp_path) -> None:
    frames = _frames(5)
    engine, display, writers, _capture = _make_engine(
        tmp_path,
        frames,
        keys=[ord(" "), -1, ord(" "), ord("q")],
    )

    status, end_reason = engine.run(display=True)

    assert (status, end_reason) == ("completed", "user_quit")
    assert engine.video_frame_count == 2
    recorded = writers[0].frames
    assert len(recorded) == 2
    assert np.array_equal(recorded[0], frames[1])
    assert np.array_equal(recorded[1], frames[2])
    assert len(display.shown_frames) == 4
    assert not np.array_equal(display.shown_frames[0], frames[0])
    metadata = _read_metadata(engine.session_dir)
    assert metadata["video_frame_count"] == 2
    assert metadata["video_filename"] == "raw.avi"
    assert metadata["status"] == "completed"


def test_snapshot_saves_raw_frame_with_sequential_names(tmp_path) -> None:
    frames = _frames(3)
    engine, display, _writers, _capture = _make_engine(
        tmp_path, frames, keys=[ord("s"), ord("s"), ord("q")]
    )

    status, _end_reason = engine.run(display=True)

    assert status == "completed"
    assert engine.snapshot_count == 2
    saved = display.written_images
    assert saved[0][0].endswith("frame_000001.jpg")
    assert saved[1][0].endswith("frame_000002.jpg")
    assert np.array_equal(saved[0][1], frames[0])
    assert np.array_equal(saved[1][1], frames[1])
    metadata = _read_metadata(engine.session_dir)
    assert metadata["snapshot_count"] == 2


def test_frame_interval_creates_automatic_snapshots(tmp_path) -> None:
    frames = _frames(20)
    engine, display, _writers, _capture = _make_engine(
        tmp_path,
        frames,
        clock_step=0.2,
        frame_interval_seconds=1.0,
    )

    status, end_reason = engine.run(display=False, max_frames=len(frames))

    assert (status, end_reason) == ("completed", "max_frames")
    assert 2 <= engine.snapshot_count <= 10
    names = [path for path, _frame in display.written_images]
    assert len(set(names)) == len(names)
    assert names[0].endswith("frame_000001.jpg")


def test_max_frames_terminates_and_cleans_up(tmp_path) -> None:
    engine, display, _writers, capture = _make_engine(
        tmp_path, _frames(50), keys=None
    )

    status, end_reason = engine.run(display=False, max_frames=5)

    assert (status, end_reason) == ("completed", "max_frames")
    assert engine.frames_seen == 5
    assert capture.released is True
    assert display.windows_created == []
    assert display.destroyed is False
    metadata = _read_metadata(engine.session_dir)
    assert metadata["status"] == "completed"
    assert metadata["end_reason"] == "max_frames"


def test_duration_terminates_before_source_ends(tmp_path) -> None:
    engine, _display, _writers, capture = _make_engine(
        tmp_path, _frames(500), clock_step=0.5
    )

    status, end_reason = engine.run(display=False, duration_seconds=3.0)

    assert (status, end_reason) == ("completed", "duration_reached")
    assert 1 <= engine.frames_seen < 500
    assert capture.released is True
    metadata = _read_metadata(engine.session_dir)
    assert metadata["end_reason"] == "duration_reached"


def test_camera_read_false_is_reported_as_failure(tmp_path) -> None:
    engine, _display, _writers, capture = _make_engine(tmp_path, _frames(3))

    with pytest.raises(CaptureError, match="camera read failed"):
        engine.run(display=False)

    assert engine.frames_seen == 3
    assert capture.released is True
    metadata = _read_metadata(engine.session_dir)
    assert metadata["status"] == "failed"
    assert metadata["end_reason"] == "camera_read_failure"


def test_keyboard_interrupt_is_a_safe_stop(tmp_path) -> None:
    frames = _frames(5)
    engine, _display, writers, capture = _make_engine(
        tmp_path, frames, keys=[ord(" "), ord(" ")], fail_after=2, failure="interrupt"
    )

    status, end_reason = engine.run(display=True)

    assert (status, end_reason) == ("interrupted", "keyboard_interrupt")
    assert writers[0].released is True
    assert capture.released is True
    metadata = _read_metadata(engine.session_dir)
    assert metadata["status"] == "interrupted"
    assert metadata["end_reason"] == "keyboard_interrupt"
    assert metadata["video_frame_count"] == 1


def test_unexpected_exception_still_finalizes_data(tmp_path) -> None:
    engine, _display, writers, capture = _make_engine(
        tmp_path, _frames(5), keys=[ord(" ")], fail_after=1, failure="error"
    )

    with pytest.raises(RuntimeError):
        engine.run(display=True)

    assert writers[0].released is True
    assert capture.released is True
    metadata = _read_metadata(engine.session_dir)
    assert metadata["status"] == "failed"
    assert metadata["end_reason"] == "unexpected_exception"


def test_writer_failure_fails_loudly_and_cleans_up(tmp_path) -> None:
    engine, _display, writers, capture = _make_engine(
        tmp_path, _frames(3), keys=[ord(" ")], writer_opened=False
    )

    with pytest.raises(CaptureError):
        engine.run(display=True)

    assert capture.released is True
    metadata = _read_metadata(engine.session_dir)
    assert metadata["status"] == "failed"


def test_headless_startup_writer_failure_uses_run_cleanup(tmp_path) -> None:
    engine, _display, _writers, capture = _make_engine(
        tmp_path, _frames(3), writer_opened=False
    )

    with pytest.raises(CaptureError, match="unable to open video writer"):
        engine.run(display=False, max_frames=1, start_recording=True)

    assert capture.released is True
    metadata = _read_metadata(engine.session_dir)
    assert metadata["status"] == "failed"
    assert metadata["end_reason"] == "video_writer_open_failure"


def test_writer_release_failure_keeps_segment_in_failed_metadata(tmp_path) -> None:
    engine, _display, writers, capture = _make_engine(
        tmp_path,
        _frames(3),
        keys=[ord(" "), ord("q")],
        writer_release_error=RuntimeError("flush failed"),
    )

    with pytest.raises(CaptureError, match="release video writer"):
        engine.run(display=True)

    assert writers[0].released is True
    assert capture.released is True
    metadata = _read_metadata(engine.session_dir)
    assert metadata["status"] == "failed"
    assert metadata["end_reason"] == "video_writer_release_failure"
    assert metadata["video_frame_count"] == 1
    assert metadata["video_filename"] == "raw.avi"


def test_existing_raw_video_name_is_never_reused(tmp_path) -> None:
    engine, _display, writers, _capture = _make_engine(tmp_path, _frames(1))
    existing = engine.session_dir / "raw.avi"
    existing.write_bytes(b"existing raw data")

    engine.start_recording()

    assert writers[0].path.name == "raw_002.avi"
    assert existing.read_bytes() == b"existing raw data"


def test_segment_names_remain_monotonic_after_a_collision(tmp_path) -> None:
    engine, _display, writers, _capture = _make_engine(tmp_path, _frames(1))
    (engine.session_dir / "raw.avi").write_bytes(b"existing raw data")

    engine.start_recording()
    engine.stop_recording()
    engine.start_recording()

    assert [writer.path.name for writer in writers] == [
        "raw_002.avi",
        "raw_003.avi",
    ]


def test_existing_snapshot_name_is_never_reused(tmp_path) -> None:
    engine, display, _writers, _capture = _make_engine(tmp_path, _frames(1))
    existing = engine.session_dir / "frames" / "frame_000001.jpg"
    existing.write_bytes(b"existing snapshot")

    saved_path = engine.snapshot(_frames(1)[0])

    assert saved_path.name == "frame_000002.jpg"
    assert display.written_images[0][0].endswith("frame_000002.jpg")
    assert existing.read_bytes() == b"existing snapshot"


def test_recording_rejects_frame_size_mismatch(tmp_path) -> None:
    engine, _display, writers, capture = _make_engine(
        tmp_path, _frames(2, height=8, width=15)
    )

    with pytest.raises(CaptureError, match="frame size"):
        engine.run(display=False, max_frames=1, start_recording=True)

    assert writers[0].frames == []
    assert capture.released is True
    metadata = _read_metadata(engine.session_dir)
    assert metadata["status"] == "failed"
    assert metadata["end_reason"] == "frame_size_mismatch"


def test_hud_video_count_includes_open_segment_frames(tmp_path) -> None:
    engine, _display, _writers, _capture = _make_engine(tmp_path, _frames(1))
    engine.start_recording()
    engine._segment_frame_count = 7

    assert engine.hud_state()["video_frames"] == 7


def test_window_close_ends_the_run(tmp_path) -> None:
    engine, display, _writers, _capture = _make_engine(
        tmp_path, _frames(3), keys=None, visible=False
    )

    status, end_reason = engine.run(display=True)

    assert (status, end_reason) == ("completed", "window_closed")
    assert display.windows_created == [("FOMO Dataset Capture", False)]
    assert display.destroyed is True


@pytest.mark.parametrize("quit_key", [ord("q"), ord("Q"), 27])
def test_gui_quit_keys_finalize_and_release(tmp_path, quit_key) -> None:
    engine, display, _writers, capture = _make_engine(
        tmp_path, _frames(2), keys=[quit_key]
    )

    status, end_reason = engine.run(display=True)

    assert (status, end_reason) == ("completed", "user_quit")
    assert capture.released is True
    assert display.destroyed is True
    assert _read_metadata(engine.session_dir)["status"] == "completed"


def test_fullscreen_is_forwarded_only_to_display_window(tmp_path) -> None:
    engine, display, _writers, _capture = _make_engine(
        tmp_path, _frames(2), keys=[ord("q")]
    )

    engine.run(display=True, fullscreen=True)

    assert display.windows_created == [("FOMO Dataset Capture", True)]


def test_next_session_key_finalizes_and_advances(tmp_path) -> None:
    frames = _frames(4)
    engine, _display, _writers, _capture = _make_engine(
        tmp_path, frames, keys=[ord("n"), ord("n"), ord("q")]
    )

    status, _end_reason = engine.run(display=True)

    assert status == "completed"
    assert engine.session_id == "pool-20260831-003"
    first = _read_metadata(engine.session_dir.parent / "pool-20260831-001")
    second = _read_metadata(engine.session_dir.parent / "pool-20260831-002")
    third = _read_metadata(engine.session_dir)
    assert first["end_reason"] == "user_new_session"
    assert second["end_reason"] == "user_new_session"
    assert third["end_reason"] == "user_quit"
    assert first["scene"] == "pool-scene"
    assert third["session_id"] == "pool-20260831-003"


def test_low_disk_space_sets_warning_state(tmp_path) -> None:
    low_engine, _display, _writers, _capture = _make_engine(
        tmp_path, _frames(2), free_bytes=1 << 30
    )
    high_engine, _display_high, _writers_high, _capture_high = _make_engine(
        tmp_path / "high", _frames(2), free_bytes=100 << 30
    )

    low_engine.run(display=False, max_frames=1, min_free_gb=5.0)
    high_engine.run(display=False, max_frames=1, min_free_gb=5.0)

    assert low_engine.low_disk is True
    assert low_engine.hud_state()["low_disk_space"] is True
    assert high_engine.low_disk is False


def test_metadata_records_counts_scene_and_platform(tmp_path) -> None:
    frames = _frames(4)
    engine, display, _writers, _capture = _make_engine(
        tmp_path, frames, keys=[ord("s"), ord("q")]
    )

    engine.run(display=True)

    metadata = _read_metadata(engine.session_dir)
    assert metadata["scene"] == "pool-scene"
    assert metadata["target"] == "fish-target"
    assert metadata["notes"] == "note-1"
    assert metadata["source"] == "0"
    assert metadata["camera_backend"] == "V4L2"
    assert metadata["observed_width"] == 16
    assert metadata["observed_fps"] == pytest.approx(5.0)
    assert metadata["snapshot_count"] == 1
    assert metadata["measured_capture_fps"] > 0.0
    assert set(metadata["platform"]) == {"platform", "machine", "python", "opencv"}
    assert metadata["camera_controls"]["brightness"] == 128.0
