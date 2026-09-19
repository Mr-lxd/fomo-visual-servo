from __future__ import annotations

import csv
import json
import threading
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pytest

from fomo_servo.capture.manager import CaptureConfig, CaptureError, CaptureManager
from fomo_servo.capture.session_layout import plan_next_session
from fomo_servo.vision.camera_owner import CameraFacts
from fomo_servo.vision.frame import PixelFormat, VisionFrame


def _facts() -> CameraFacts:
    return CameraFacts(
        source="/dev/video0",
        observed_width=4,
        observed_height=3,
        observed_fps=25.0,
        observed_fourcc="YUYV",
    )


def _frame(frame_id: int, value: int = 0) -> VisionFrame:
    image = np.full((3, 4, 3), value, dtype=np.uint8)
    return VisionFrame(
        frame_id=frame_id,
        capture_timestamp_ns=1000 + frame_id,
        width=4,
        height=3,
        pixel_format=PixelFormat.BGR8,
        image=image,
    )


class _Writer:
    def __init__(self) -> None:
        self.opened = True
        self.frames: list[np.ndarray] = []

    def isOpened(self) -> bool:
        return self.opened

    def write(self, image) -> None:
        self.frames.append(image.copy())

    def release(self) -> None:
        self.opened = False


class _WriterFactory:
    def __init__(self, writer=None) -> None:
        self.writer = writer or _Writer()
        self.calls: list[tuple[Path, float, tuple[int, int]]] = []

    def __call__(self, path: Path, fps: float, size: tuple[int, int]):
        self.calls.append((path, fps, size))
        return self.writer


def test_session_planner_never_reuses_existing_directory(tmp_path: Path) -> None:
    first = plan_next_session(tmp_path, prefix="capture")
    second = plan_next_session(tmp_path, prefix="capture")

    assert first.session_id.endswith("-001")
    assert second.session_id.endswith("-002")
    assert first.session_dir != second.session_dir
    assert first.frames_dir.is_dir()
    assert second.frames_dir.is_dir()


def test_snapshot_writes_untouched_frame_and_metadata(tmp_path: Path) -> None:
    saved: list[tuple[Path, np.ndarray, int]] = []

    def imwrite(path: Path, image, quality: int) -> bool:
        saved.append((path, image.copy(), quality))
        path.write_bytes(b"jpeg")
        return True

    manager = CaptureManager(
        CaptureConfig(
            output_root=tmp_path,
            min_free_bytes=0,
        ),
        imwrite=imwrite,
    )
    frame = _frame(7, value=42)

    result = manager.snapshot(frame, _facts())

    assert result["snapshot_count"] == 1
    assert saved[0][2] == 95
    assert np.array_equal(saved[0][1], frame.image)
    assert "frame_00000000000000000007" in saved[0][0].name

    metadata = json.loads(
        (Path(result["session_dir"]) / "metadata.json").read_text(
            encoding="utf-8"
        )
    )
    assert metadata["snapshots"][0]["frame_id"] == 7
    assert metadata["snapshots"][0]["capture_timestamp_ns"] == 1007
    assert metadata["camera"]["observed_width"] == 4


def test_recording_preserves_sequential_ids_and_frame_index(tmp_path: Path) -> None:
    factory = _WriterFactory()
    manager = CaptureManager(
        CaptureConfig(
            output_root=tmp_path,
            min_free_bytes=0,
            max_queue_bytes=1024 * 1024,
        ),
        writer_factory=factory,
        avi_frame_counter=lambda _path: len(factory.writer.frames),
        avi_fps_rewriter=lambda _path, _fps: 13.75,
        wall_now=lambda: datetime(2026, 9, 19, tzinfo=timezone.utc),
    )

    started = manager.start_recording(_facts())
    manager.offer_frame(_frame(10, 10))
    manager.offer_frame(_frame(11, 11))
    manager.offer_frame(_frame(12, 12))
    stopped = manager.stop_recording()

    assert started["state"] == "recording"
    assert stopped["state"] == "idle"
    assert stopped["recorded_frames"] == 3
    assert factory.calls[0][1] == 25.0
    assert len(factory.writer.frames) == 3
    assert [int(image[0, 0, 0]) for image in factory.writer.frames] == [
        10,
        11,
        12,
    ]
    session_dir = Path(stopped["session_dir"])
    with (session_dir / "frame_index.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))

    assert [int(row["frame_id"]) for row in rows] == [10, 11, 12]
    assert [int(row["segment_frame_index"]) for row in rows] == [0, 1, 2]
    assert all(row["segment"] == "raw.avi" for row in rows)

    metadata = json.loads(
        (session_dir / "metadata.json").read_text(encoding="utf-8")
    )
    segment = metadata["segments"][0]
    assert segment["frame_count"] == 3
    assert segment["first_frame_id"] == 10
    assert segment["last_frame_id"] == 12
    assert segment["camera_reported_fps"] == 25.0
    assert segment["container_fps"] == 13.75
    assert segment["actual_capture_fps"] is not None
    assert segment["capture_elapsed_seconds"] is not None
    assert segment["timing_error_ratio"] is not None
    assert segment["status"] == "completed"


class _BlockingWriter(_Writer):
    def __init__(self) -> None:
        super().__init__()
        self.entered = threading.Event()
        self.continue_write = threading.Event()

    def write(self, image) -> None:
        self.entered.set()
        assert self.continue_write.wait(timeout=2.0)
        super().write(image)
def test_queue_overflow_fails_recording_instead_of_silent_drop(
    tmp_path: Path,
) -> None:
    writer = _BlockingWriter()
    factory = _WriterFactory(writer)
    one_frame_bytes = int(_frame(1).image.nbytes)
    manager = CaptureManager(
        CaptureConfig(
            output_root=tmp_path,
            min_free_bytes=0,
            max_queue_bytes=one_frame_bytes,
        ),
        writer_factory=factory,
        avi_frame_counter=lambda _path: len(factory.writer.frames),
        avi_fps_rewriter=lambda _path, fps: fps,
    )

    manager.start_recording(_facts())
    manager.offer_frame(_frame(1))
    assert writer.entered.wait(timeout=1.0)

    manager.offer_frame(_frame(2))
    manager.offer_frame(_frame(3))

    failed = manager.status()
    assert failed["state"] == "stopping"
    assert failed["first_dropped_frame_id"] == 3
    assert "queue_overflow" in failed["last_error"]

    writer.continue_write.set()
    final = manager.stop_recording()

    assert final["state"] == "failed"
    assert final["recorded_frames"] == 2

    metadata = json.loads(
        (Path(final["session_dir"]) / "metadata.json").read_text(
            encoding="utf-8"
        )
    )
    failed_segment = metadata["segments"][0]
    assert failed_segment["status"] == "failed"
    assert failed_segment["first_dropped_frame_id"] == 3
    assert "queue_overflow" in failed_segment["failure_error"]


def test_runtime_low_disk_fails_before_writing_next_frame(tmp_path: Path) -> None:
    factory = _WriterFactory()
    free_values = iter([10_000, 100])

    manager = CaptureManager(
        CaptureConfig(
            output_root=tmp_path,
            min_free_bytes=1_000,
            max_queue_bytes=1024 * 1024,
        ),
        writer_factory=factory,
        avi_frame_counter=lambda _path: len(factory.writer.frames),
        avi_fps_rewriter=lambda _path, fps: fps,
        disk_free_bytes=lambda _path: next(free_values),
    )

    manager.start_recording(_facts())
    manager.offer_frame(_frame(21, 21))

    deadline = threading.Event()
    # Wait until the worker observes low disk and closes the segment.
    for _ in range(100):
        if manager.status()["state"] == "failed":
            break
        deadline.wait(0.005)

    status = manager.status()
    assert status["state"] == "failed"
    assert status["first_dropped_frame_id"] == 21
    assert status["free_disk_bytes"] == 100
    assert "low_disk_space" in status["last_error"]
    assert factory.writer.frames == []


def test_start_failure_releases_writer_and_rolls_back_state(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from fomo_servo.capture.session_layout import SessionPaths

    session_dir = tmp_path / "20260919" / "capture-20260919-001"
    frames_dir = session_dir / "frames"
    frames_dir.mkdir(parents=True)
    bad_index_path = session_dir / "frame_index.csv"
    bad_index_path.mkdir()

    session = SessionPaths(
        session_id=session_dir.name,
        session_dir=session_dir,
        frames_dir=frames_dir,
        metadata_path=session_dir / "metadata.json",
        frame_index_path=bad_index_path,
    )
    monkeypatch.setattr(
        "fomo_servo.capture.manager.plan_next_session",
        lambda *_args, **_kwargs: session,
    )

    factory = _WriterFactory()
    manager = CaptureManager(
        CaptureConfig(
            output_root=tmp_path,
            min_free_bytes=0,
        ),
        writer_factory=factory,
        avi_frame_counter=lambda _path: len(factory.writer.frames),
        avi_fps_rewriter=lambda _path, fps: fps,
    )

    import pytest

    with pytest.raises(Exception):
        manager.start_recording(_facts())

    assert factory.writer.opened is False
    status = manager.status()
    assert status["state"] == "idle"
    assert status["recording"] is False
    assert status["queue_frames"] == 0
    assert "recording_start_failure" in status["last_error"]


def test_recording_metadata_compares_container_fps_with_actual_timestamps(
    tmp_path: Path,
) -> None:
    factory = _WriterFactory()
    manager = CaptureManager(
        CaptureConfig(
            output_root=tmp_path,
            min_free_bytes=0,
            max_queue_bytes=1024 * 1024,
        ),
        writer_factory=factory,
        avi_frame_counter=lambda _path: len(factory.writer.frames),
        avi_fps_rewriter=lambda _path, fps: fps,
    )

    manager.start_recording(_facts())

    for frame_id, timestamp_ns in enumerate(
        [1_000_000_000, 1_100_000_000, 1_200_000_000],
        start=30,
    ):
        image = np.full((3, 4, 3), frame_id, dtype=np.uint8)
        manager.offer_frame(
            VisionFrame(
                frame_id=frame_id,
                capture_timestamp_ns=timestamp_ns,
                width=4,
                height=3,
                pixel_format=PixelFormat.BGR8,
                image=image,
            )
        )

    stopped = manager.stop_recording()
    metadata = json.loads(
        (Path(stopped["session_dir"]) / "metadata.json").read_text(
            encoding="utf-8"
        )
    )
    segment = metadata["segments"][0]

    assert segment["camera_reported_fps"] == 25.0
    assert segment["container_fps"] == 10.0
    assert segment["capture_elapsed_seconds"] == pytest.approx(0.2)
    assert segment["actual_capture_fps"] == pytest.approx(10.0)
    assert segment["timing_error_ratio"] == pytest.approx(0.0)


def test_avi_timing_finalize_failure_marks_segment_failed(
    tmp_path: Path,
) -> None:
    factory = _WriterFactory()

    def fail_rewrite(_path: Path, _fps: float) -> float:
        raise RuntimeError("cannot rewrite AVI header")

    manager = CaptureManager(
        CaptureConfig(
            output_root=tmp_path,
            min_free_bytes=0,
            max_queue_bytes=1024 * 1024,
        ),
        writer_factory=factory,
        avi_frame_counter=lambda _path: len(factory.writer.frames),
        avi_fps_rewriter=fail_rewrite,
    )

    manager.start_recording(_facts())
    manager.offer_frame(_frame(50, 50))
    manager.offer_frame(_frame(51, 51))
    final = manager.stop_recording()

    assert final["state"] == "failed"
    assert "avi_timing_finalize_failure" in final["last_error"]

    metadata = json.loads(
        (Path(final["session_dir"]) / "metadata.json").read_text(
            encoding="utf-8"
        )
    )
    segment = metadata["segments"][0]
    assert segment["status"] == "failed"
    assert segment["timing_header_rewritten"] is False
    assert "avi_timing_finalize_failure" in segment["failure_error"]


def test_real_mjpg_avi_is_finalized_to_segment_timestamp_fps(
    tmp_path: Path,
) -> None:
    facts = CameraFacts(
        source="/dev/video0",
        observed_width=64,
        observed_height=48,
        observed_fps=25.0,
        observed_fourcc="YUYV",
    )
    manager = CaptureManager(
        CaptureConfig(
            output_root=tmp_path,
            min_free_bytes=0,
            max_queue_bytes=8 * 1024 * 1024,
        )
    )

    manager.start_recording(facts)

    for index in range(10):
        manager.offer_frame(
            VisionFrame(
                frame_id=100 + index,
                capture_timestamp_ns=1_000_000_000 + index * 100_000_000,
                width=64,
                height=48,
                pixel_format=PixelFormat.BGR8,
                image=np.full(
                    (48, 64, 3),
                    index * 10,
                    dtype=np.uint8,
                ),
            )
        )

    stopped = manager.stop_recording()
    assert stopped["state"] == "idle"

    root = Path(stopped["session_dir"])
    metadata = json.loads(
        (root / "metadata.json").read_text(encoding="utf-8")
    )
    segment = metadata["segments"][0]

    assert segment["writer_initial_fps"] == pytest.approx(25.0)
    assert segment["actual_capture_fps"] == pytest.approx(10.0)
    assert segment["container_fps"] == pytest.approx(10.0, rel=1e-6)
    assert segment["timing_header_rewritten"] is True
    assert segment["timing_error_ratio"] < 1e-6

    cap = cv2.VideoCapture(str(root / segment["filename"]))
    assert cap.isOpened()
    assert cap.get(cv2.CAP_PROP_FRAME_COUNT) == pytest.approx(10)
    assert cap.get(cv2.CAP_PROP_FPS) == pytest.approx(10.0, rel=1e-3)

    decoded = 0
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        decoded += 1
    cap.release()

    assert decoded == 10


class _ReleaseFailWriter(_Writer):
    def release(self) -> None:
        raise RuntimeError("release failed")


def test_finalized_avi_frame_count_mismatch_marks_segment_failed(
    tmp_path: Path,
) -> None:
    factory = _WriterFactory()
    rewrites: list[float] = []
    manager = CaptureManager(
        CaptureConfig(
            output_root=tmp_path,
            min_free_bytes=0,
            max_queue_bytes=1024 * 1024,
        ),
        writer_factory=factory,
        avi_frame_counter=lambda _path: max(0, len(factory.writer.frames) - 1),
        avi_fps_rewriter=lambda _path, fps: rewrites.append(fps) or fps,
    )

    manager.start_recording(_facts())
    manager.offer_frame(_frame(60, 60))
    manager.offer_frame(_frame(61, 61))
    manager.offer_frame(_frame(62, 62))
    final = manager.stop_recording()

    assert final["state"] == "failed"
    assert "avi_frame_count_mismatch" in final["last_error"]
    assert rewrites == []

    metadata = json.loads(
        (Path(final["session_dir"]) / "metadata.json").read_text(
            encoding="utf-8"
        )
    )
    segment = metadata["segments"][-1]
    assert segment["status"] == "failed"
    assert segment["frame_count"] == 3
    assert segment["video_frame_count"] == 2
    assert segment["frame_count_verified"] is False
    assert segment["timing_header_rewritten"] is False


def test_writer_release_failure_skips_avi_timing_rewrite(
    tmp_path: Path,
) -> None:
    writer = _ReleaseFailWriter()
    factory = _WriterFactory(writer)
    rewrites: list[float] = []
    manager = CaptureManager(
        CaptureConfig(
            output_root=tmp_path,
            min_free_bytes=0,
            max_queue_bytes=1024 * 1024,
        ),
        writer_factory=factory,
        avi_frame_counter=lambda _path: len(writer.frames),
        avi_fps_rewriter=lambda _path, fps: rewrites.append(fps) or fps,
    )

    manager.start_recording(_facts())
    manager.offer_frame(_frame(70, 70))
    manager.offer_frame(_frame(71, 71))
    final = manager.stop_recording()

    assert final["state"] == "failed"
    assert "writer release failed" in final["last_error"]
    assert rewrites == []


def test_start_worker_failure_removes_new_files_and_rolls_back_metadata(
    tmp_path: Path,
    monkeypatch,
) -> None:
    writer = _Writer()

    def writer_factory(path: Path, _fps: float, _size: tuple[int, int]):
        path.write_bytes(b"partial-avi")
        return writer

    manager = CaptureManager(
        CaptureConfig(output_root=tmp_path, min_free_bytes=0),
        writer_factory=writer_factory,
        avi_frame_counter=lambda _path: 0,
    )

    def fail_start(_thread) -> None:
        raise RuntimeError("thread start failed")

    monkeypatch.setattr(threading.Thread, "start", fail_start)

    with pytest.raises(CaptureError):
        manager.start_recording(_facts())

    status = manager.status()
    session_dir = Path(status["session_dir"])
    assert status["state"] == "idle"
    assert status["recording"] is False
    assert not (session_dir / "raw.avi").exists()
    assert not (session_dir / "frame_index.csv").exists()

    metadata = json.loads(
        (session_dir / "metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["recording_state"] == "idle"
    assert metadata["current_segment"] is None
    assert "recording_start_failure" in metadata["last_error"]


def test_shutdown_waits_for_live_writer_before_publishing_closed(
    tmp_path: Path,
) -> None:
    writer = _BlockingWriter()
    factory = _WriterFactory(writer)
    manager = CaptureManager(
        CaptureConfig(
            output_root=tmp_path,
            min_free_bytes=0,
            max_queue_bytes=1024 * 1024,
        ),
        writer_factory=factory,
        avi_frame_counter=lambda _path: len(writer.frames),
        avi_fps_rewriter=lambda _path, fps: fps,
    )

    manager.start_recording(_facts())
    manager.offer_frame(_frame(80, 80))
    assert writer.entered.wait(timeout=1.0)

    with pytest.raises(CaptureError):
        manager.stop_recording(join_timeout=0.01)

    session_dir = Path(manager.status()["session_dir"])
    metadata = json.loads(
        (session_dir / "metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["closed"] is False

    done = threading.Event()
    shutdown_error: list[BaseException] = []

    def run_shutdown() -> None:
        try:
            manager.shutdown()
        except BaseException as error:
            shutdown_error.append(error)
        finally:
            done.set()

    shutdown_thread = threading.Thread(target=run_shutdown)
    shutdown_thread.start()
    assert not done.wait(timeout=0.05)

    writer.continue_write.set()
    shutdown_thread.join(timeout=2.0)

    assert done.is_set()
    assert shutdown_error == []
    metadata = json.loads(
        (session_dir / "metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["closed"] is True


def test_metadata_snapshot_and_replace_are_serialized(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def write_snapshot(path: Path, _image, _quality: int) -> bool:
        path.write_bytes(b"jpeg")
        return True

    manager = CaptureManager(
        CaptureConfig(output_root=tmp_path, min_free_bytes=0),
        imwrite=write_snapshot,
    )
    manager.snapshot(_frame(90, 90), _facts())

    original_snapshot = manager._metadata_snapshot
    first_entered = threading.Event()
    release_first = threading.Event()
    calls = 0
    calls_lock = threading.Lock()

    def blocking_snapshot():
        nonlocal calls
        with calls_lock:
            calls += 1
            call_number = calls
        if call_number == 1:
            first_entered.set()
            assert release_first.wait(timeout=1.0)
        return original_snapshot()

    monkeypatch.setattr(manager, "_metadata_snapshot", blocking_snapshot)

    first = threading.Thread(target=manager._write_metadata)
    second = threading.Thread(target=manager._write_metadata)
    first.start()
    assert first_entered.wait(timeout=1.0)
    second.start()

    threading.Event().wait(0.05)
    with calls_lock:
        assert calls == 1

    release_first.set()
    first.join(timeout=1.0)
    second.join(timeout=1.0)
    with calls_lock:
        assert calls == 2


class _WriteFailWriter(_Writer):
    def write(self, image) -> None:
        raise RuntimeError("simulated writer failure")


def test_dead_worker_metadata_failure_is_explicit_and_stop_retries_persistence(
    tmp_path: Path,
    monkeypatch,
) -> None:
    writer = _WriteFailWriter()
    factory = _WriterFactory(writer)
    manager = CaptureManager(
        CaptureConfig(
            output_root=tmp_path,
            min_free_bytes=0,
            max_queue_bytes=1024 * 1024,
        ),
        writer_factory=factory,
        avi_frame_counter=lambda _path: 0,
        avi_fps_rewriter=lambda _path, fps: fps,
    )

    manager.start_recording(_facts())
    session_dir = Path(manager.status()["session_dir"])

    real_write_metadata = manager._write_metadata
    metadata_attempts = 0

    def fail_worker_metadata_twice() -> None:
        nonlocal metadata_attempts
        metadata_attempts += 1
        if metadata_attempts <= 2:
            raise OSError(f"simulated metadata failure {metadata_attempts}")
        real_write_metadata()

    monkeypatch.setattr(
        manager,
        "_write_metadata",
        fail_worker_metadata_twice,
    )

    manager.offer_frame(_frame(100, 100))
    worker = manager._worker
    assert worker is not None
    worker.join(timeout=1.0)
    assert not worker.is_alive()

    before_retry = manager.status()
    assert before_retry["state"] == "failed"
    assert "metadata_finalize_failure" in before_retry["last_error"]
    assert "metadata_retry_failure" in before_retry["last_error"]

    # The worker is already dead. stop_recording() must still retry the
    # authoritative metadata write rather than returning early.
    final = manager.stop_recording()
    assert metadata_attempts == 3
    assert final["state"] == "failed"

    metadata = json.loads(
        (session_dir / "metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["recording_state"] == "failed"
    assert metadata["current_segment"] is None
    segment = metadata["segments"][-1]
    assert segment["status"] == "failed"
    assert "metadata_finalize_failure" in segment["failure_error"]
    assert "simulated writer failure" in segment["failure_error"]


def test_start_rollback_cleanup_failure_is_reported_and_marks_failed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    writer = _ReleaseFailWriter()

    def writer_factory(path: Path, _fps: float, _size: tuple[int, int]):
        path.write_bytes(b"partial-avi")
        return writer

    manager = CaptureManager(
        CaptureConfig(output_root=tmp_path, min_free_bytes=0),
        writer_factory=writer_factory,
        avi_frame_counter=lambda _path: 0,
    )

    def fail_start(_thread) -> None:
        raise RuntimeError("thread start failed")

    monkeypatch.setattr(threading.Thread, "start", fail_start)

    with pytest.raises(
        CaptureError,
        match="recording_start_rollback_failure",
    ) as caught:
        manager.start_recording(_facts())

    assert "writer_release=release failed" in str(caught.value)
    status = manager.status()
    assert status["state"] == "failed"
    assert "recording_start_rollback_failure" in status["last_error"]

    metadata = json.loads(
        (Path(status["session_dir"]) / "metadata.json").read_text(
            encoding="utf-8"
        )
    )
    assert metadata["recording_state"] == "failed"
    assert "writer_release=release failed" in metadata["last_error"]


def test_start_rollback_metadata_failure_is_not_swallowed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    writer = _Writer()

    def writer_factory(path: Path, _fps: float, _size: tuple[int, int]):
        path.write_bytes(b"partial-avi")
        return writer

    manager = CaptureManager(
        CaptureConfig(output_root=tmp_path, min_free_bytes=0),
        writer_factory=writer_factory,
        avi_frame_counter=lambda _path: 0,
    )

    real_write_metadata = manager._write_metadata
    metadata_attempts = 0

    def fail_rollback_metadata() -> None:
        nonlocal metadata_attempts
        metadata_attempts += 1
        if metadata_attempts >= 2:
            raise OSError("simulated rollback metadata failure")
        real_write_metadata()

    monkeypatch.setattr(
        manager,
        "_write_metadata",
        fail_rollback_metadata,
    )

    def fail_start(_thread) -> None:
        raise RuntimeError("thread start failed")

    monkeypatch.setattr(threading.Thread, "start", fail_start)

    with pytest.raises(
        CaptureError,
        match="recording_start_rollback_failure",
    ) as caught:
        manager.start_recording(_facts())

    message = str(caught.value)
    assert "metadata_rollback=simulated rollback metadata failure" in message
    assert "metadata_retry=simulated rollback metadata failure" in message
    assert metadata_attempts == 3

    status = manager.status()
    assert status["state"] == "failed"
    assert "recording_start_rollback_failure" in status["last_error"]


def test_snapshot_metadata_write_retries_once_and_succeeds(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def imwrite(path: Path, _image, _quality: int) -> bool:
        path.write_bytes(b"jpeg")
        return True

    manager = CaptureManager(
        CaptureConfig(output_root=tmp_path, min_free_bytes=0),
        imwrite=imwrite,
    )

    real_write_metadata = manager._write_metadata
    attempts = 0

    def fail_once() -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("simulated snapshot metadata failure")
        real_write_metadata()

    monkeypatch.setattr(manager, "_write_metadata", fail_once)

    result = manager.snapshot(_frame(110, 110), _facts())

    assert attempts == 2
    assert result["state"] == "idle"
    assert result["last_error"] is None
    assert result["snapshot_count"] == 1

    metadata = json.loads(
        (Path(result["session_dir"]) / "metadata.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(metadata["snapshots"]) == 1
    assert metadata["snapshots"][0]["frame_id"] == 110


def test_snapshot_metadata_double_failure_is_explicit_and_later_stop_retries(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def imwrite(path: Path, _image, _quality: int) -> bool:
        path.write_bytes(b"jpeg")
        return True

    manager = CaptureManager(
        CaptureConfig(output_root=tmp_path, min_free_bytes=0),
        imwrite=imwrite,
    )

    real_write_metadata = manager._write_metadata
    attempts = 0

    def fail_twice() -> None:
        nonlocal attempts
        attempts += 1
        if attempts <= 2:
            raise OSError(f"simulated snapshot metadata failure {attempts}")
        real_write_metadata()

    monkeypatch.setattr(manager, "_write_metadata", fail_twice)

    with pytest.raises(
        CaptureError,
        match="snapshot_metadata_persist_failure",
    ):
        manager.snapshot(_frame(111, 111), _facts())

    failed = manager.status()
    assert failed["state"] == "failed"
    assert failed["snapshot_count"] == 1
    assert "snapshot_metadata_persist_failure" in failed["last_error"]
    assert attempts == 2

    # The in-memory record and JPEG are intentionally retained so a later
    # authoritative metadata write can restore the session index.
    session_dir = Path(failed["session_dir"])
    assert len(list((session_dir / "frames").glob("snapshot_*.jpg"))) == 1

    persisted = manager.stop_recording()
    assert attempts == 3
    assert persisted["state"] == "failed"

    metadata = json.loads(
        (session_dir / "metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["recording_state"] == "failed"
    assert len(metadata["snapshots"]) == 1
    assert metadata["snapshots"][0]["frame_id"] == 111
    assert "snapshot_metadata_persist_failure" in metadata["last_error"]


def test_snapshot_metadata_failure_during_recording_stops_acceptance_and_fails_segment(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def imwrite(path: Path, _image, _quality: int) -> bool:
        path.write_bytes(b"jpeg")
        return True

    factory = _WriterFactory()
    manager = CaptureManager(
        CaptureConfig(
            output_root=tmp_path,
            min_free_bytes=0,
            max_queue_bytes=1024 * 1024,
        ),
        writer_factory=factory,
        avi_frame_counter=lambda _path: len(factory.writer.frames),
        avi_fps_rewriter=lambda _path, fps: fps,
        imwrite=imwrite,
    )
    manager.start_recording(_facts())

    real_write_metadata = manager._write_metadata
    attempts = 0

    def fail_twice() -> None:
        nonlocal attempts
        attempts += 1
        if attempts <= 2:
            raise OSError(f"simulated snapshot metadata failure {attempts}")
        real_write_metadata()

    monkeypatch.setattr(manager, "_write_metadata", fail_twice)

    with pytest.raises(
        CaptureError,
        match="snapshot_metadata_persist_failure",
    ):
        manager.snapshot(_frame(112, 112), _facts())

    stopping = manager.status()
    assert stopping["state"] == "stopping"
    assert stopping["recording"] is True
    assert stopping["first_dropped_frame_id"] is None
    assert "snapshot_metadata_persist_failure" in stopping["last_error"]

    # The worker finalization gets the next metadata attempt and must preserve
    # the snapshot persistence failure as the segment failure reason.
    final = manager.stop_recording()
    assert attempts >= 3
    assert final["state"] == "failed"
    assert final["snapshot_count"] == 1

    metadata = json.loads(
        (Path(final["session_dir"]) / "metadata.json").read_text(
            encoding="utf-8"
        )
    )
    assert metadata["segments"][-1]["status"] == "failed"
    assert "snapshot_metadata_persist_failure" in (
        metadata["segments"][-1]["failure_error"]
    )
    assert metadata["snapshots"][-1]["frame_id"] == 112
