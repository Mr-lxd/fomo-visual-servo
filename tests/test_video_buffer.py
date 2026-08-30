from __future__ import annotations

from fomo_servo.inference import FramePacket, LatestFrameBuffer
from fomo_servo.inference.video import LatestFrameReader


def test_latest_frame_buffer_drops_stale_packet() -> None:
    buffer = LatestFrameBuffer()
    buffer.put_latest(FramePacket(1, 1.0, "old"))
    buffer.put_latest(FramePacket(2, 2.0, "new"))

    packet = buffer.get(timeout=0.1)

    assert packet is not None
    assert packet.frame_index == 2
    assert packet.frame == "new"


class _FinishedCapture:
    def read(self) -> tuple[bool, None]:
        return False, None

    def release(self) -> None:
        pass

    def get(self, _property: int) -> float:
        return 0.0


def test_latest_frame_reader_exposes_finished_signal() -> None:
    reader = LatestFrameReader(_FinishedCapture()).start()  # type: ignore[arg-type]
    try:
        assert reader.finished.wait(timeout=1.0)
    finally:
        reader.stop()


class _SingleFrameCapture(_FinishedCapture):
    def __init__(self) -> None:
        self._read = False

    def read(self) -> tuple[bool, str | None]:
        if not self._read:
            self._read = True
            return True, "final-frame"
        return False, None


def test_latest_frame_reader_preserves_final_frame_when_source_ends() -> None:
    reader = LatestFrameReader(_SingleFrameCapture()).start()  # type: ignore[arg-type]
    try:
        assert reader.finished.wait(timeout=1.0)
        packet = reader.buffer.get(timeout=0.1)
        assert packet is not None
        assert packet.frame == "final-frame"
    finally:
        reader.stop()


class _FailingCapture(_FinishedCapture):
    def read(self):
        raise RuntimeError("decoder exploded")


def test_latest_frame_reader_exposes_worker_exception() -> None:
    reader = LatestFrameReader(_FailingCapture()).start()  # type: ignore[arg-type]
    try:
        assert reader.finished.wait(timeout=1.0)
        assert reader.error is not None
        assert "decoder exploded" in str(reader.error)
        assert reader.decoded_frame_count == 0
    finally:
        reader.stop()


class _TruncatedCapture(_SingleFrameCapture):
    def get(self, property_id: int) -> float:
        import cv2

        if property_id == cv2.CAP_PROP_FRAME_COUNT:
            return 2.0
        return 0.0


def test_latest_frame_reader_reports_early_file_decode_failure() -> None:
    reader = LatestFrameReader(_TruncatedCapture()).start()  # type: ignore[arg-type]
    try:
        assert reader.finished.wait(timeout=1.0)
        assert reader.error is not None
        assert "expected 2" in str(reader.error)
    finally:
        reader.stop()
