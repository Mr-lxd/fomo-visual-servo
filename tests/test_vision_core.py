from __future__ import annotations

import threading
from dataclasses import dataclass

import cv2
import pytest

from fomo_servo.vision.camera_owner import CameraOwner, CameraOwnerError, CameraReadError
from fomo_servo.vision.frame import PixelFormat, VisionFrame
from fomo_servo.vision.frame_hub import FrameHub


@dataclass
class _Image:
    shape: tuple[int, int, int] = (480, 640, 3)


def _frame(frame_id: int) -> VisionFrame:
    return VisionFrame(
        frame_id=frame_id,
        capture_timestamp_ns=frame_id + 1,
        width=640,
        height=480,
        pixel_format=PixelFormat.BGR8,
        image=_Image(),
    )


def test_frame_hub_is_a_single_replaceable_latest_slot() -> None:
    hub = FrameHub()
    first = _frame(1)
    newest = _frame(4)

    hub.publish(first)
    hub.publish(newest)

    assert hub.snapshot() is newest
    assert hub.published_count == 2
    assert hub.replacement_count == 1
    assert hub.wait_for_newer(1, timeout=0.01) is newest
    assert hub.wait_for_newer(4, timeout=0.01) is None


def test_frame_hub_rejects_nonincreasing_ids_but_clear_starts_new_generation() -> None:
    hub = FrameHub()
    hub.publish(_frame(5))
    with pytest.raises(ValueError, match="strictly increase"):
        hub.publish(_frame(5))
    with pytest.raises(ValueError, match="strictly increase"):
        hub.publish(_frame(4))

    hub.clear()
    restarted = _frame(0)
    hub.publish(restarted)
    assert hub.snapshot() is restarted


class _SequenceCapture:
    def __init__(self, frames: list[_Image]) -> None:
        self.frames = list(frames)
        self.opened = True
        self.release_count = 0
        self.read_count = 0
        self.set_calls: list[tuple[int, float]] = []

    def isOpened(self) -> bool:
        return self.opened

    def read(self):
        self.read_count += 1
        if self.frames:
            return True, self.frames.pop(0)
        return False, None

    def release(self) -> None:
        self.release_count += 1
        self.opened = False

    def set(self, prop: int, value: float) -> bool:
        self.set_calls.append((prop, value))
        return True

    def get(self, prop: int) -> float:
        values = {
            cv2.CAP_PROP_FRAME_WIDTH: 640.0,
            cv2.CAP_PROP_FRAME_HEIGHT: 480.0,
            cv2.CAP_PROP_FPS: 25.0,
            cv2.CAP_PROP_FOURCC: float(
                ord("Y")
                | (ord("U") << 8)
                | (ord("Y") << 16)
                | (ord("V") << 24)
            ),
        }
        return values.get(prop, 0.0)


class _RecordingHub:
    def __init__(self) -> None:
        self.frames: list[VisionFrame] = []
        self.clear_count = 0

    def clear(self) -> None:
        self.clear_count += 1
        self.frames.clear()

    def publish(self, frame: VisionFrame) -> None:
        self.frames.append(frame)


def test_camera_owner_assigns_ids_and_acquisition_return_timestamps() -> None:
    capture = _SequenceCapture([_Image(), _Image()])
    hub = _RecordingHub()
    clock_values = iter([101, 202])

    def clock_ns() -> int:
        assert capture.read_count in (1, 2)
        return next(clock_values)

    owner = CameraOwner(
        hub,  # type: ignore[arg-type]
        source="/dev/video0",
        width=640,
        height=480,
        fps=25.0,
        fourcc="YUYV",
        capture_factory=lambda _source: capture,
        clock_ns=clock_ns,
    ).start()
    assert owner.finished.wait(timeout=1.0)

    assert [frame.frame_id for frame in hub.frames] == [0, 1]
    assert [frame.capture_timestamp_ns for frame in hub.frames] == [101, 202]
    assert all(frame.pixel_format is PixelFormat.BGR8 for frame in hub.frames)
    assert owner.frames_captured == 2
    assert isinstance(owner.error, CameraReadError)
    assert capture.release_count == 1
    assert hub.clear_count == 1
    assert owner.facts is not None
    assert owner.facts.observed_width == 640
    assert owner.facts.observed_height == 480
    assert owner.facts.observed_fps == 25.0
    assert owner.facts.observed_fourcc == "YUYV"
    assert (cv2.CAP_PROP_FRAME_WIDTH, 640.0) in capture.set_calls
    assert (cv2.CAP_PROP_FRAME_HEIGHT, 480.0) in capture.set_calls
    assert (cv2.CAP_PROP_FPS, 25.0) in capture.set_calls
    assert any(prop == cv2.CAP_PROP_FOURCC for prop, _ in capture.set_calls)


class _BlockingCapture(_SequenceCapture):
    def __init__(self) -> None:
        super().__init__([])
        self.entered_read = threading.Event()
        self.released = threading.Event()

    def read(self):
        self.entered_read.set()
        self.released.wait(timeout=1.0)
        return False, None

    def release(self) -> None:
        super().release()
        self.released.set()


def test_camera_owner_stop_releases_a_blocked_capture_once() -> None:
    capture = _BlockingCapture()
    owner = CameraOwner(
        FrameHub(),
        capture_factory=lambda _source: capture,
    ).start()
    assert capture.entered_read.wait(timeout=1.0)

    owner.stop(join_timeout=0.02)

    assert owner.finished.is_set()
    assert capture.release_count == 1


class _BadShapeCapture(_SequenceCapture):
    def __init__(self) -> None:
        super().__init__([_Image(shape=(480, 640, 1))])


def test_camera_owner_surfaces_invalid_frame_shape_and_releases() -> None:
    capture = _BadShapeCapture()
    owner = CameraOwner(
        FrameHub(),
        capture_factory=lambda _source: capture,
        clock_ns=lambda: 1,
    ).start()

    assert owner.finished.wait(timeout=1.0)
    assert owner.error is not None
    assert "HxWx3" in str(owner.error)
    assert capture.release_count == 1


class _ClosedCapture(_SequenceCapture):
    def __init__(self) -> None:
        super().__init__([])
        self.opened = False


def test_camera_owner_clears_previous_facts_before_failed_new_generation() -> None:
    first = _SequenceCapture([_Image()])
    second = _ClosedCapture()
    captures = iter([first, second])
    owner = CameraOwner(
        FrameHub(),
        capture_factory=lambda _source: next(captures),
        clock_ns=lambda: 1,
    )

    owner.start()
    assert owner.finished.wait(timeout=1.0)
    assert owner.facts is not None

    with pytest.raises(CameraOwnerError, match="unable to open"):
        owner.start()

    assert owner.facts is None
    assert second.release_count == 1


def test_camera_owner_rejects_requested_image_above_validation_profile_before_open() -> None:
    factory_called = False

    def factory(_source):
        nonlocal factory_called
        factory_called = True
        return _SequenceCapture([])

    owner = CameraOwner(
        FrameHub(),
        width=8192,
        height=8192,
        capture_factory=factory,
    )

    with pytest.raises(Exception, match="decoded pixel count"):
        owner.start()

    assert not factory_called
