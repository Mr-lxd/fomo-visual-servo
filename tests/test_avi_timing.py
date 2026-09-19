from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

import fomo_servo.capture.avi_timing as avi_timing
from fomo_servo.capture.avi_timing import AviTimingError, rewrite_avi_frame_rate


def test_rewrite_avi_frame_rate_changes_timing_without_reencoding(
    tmp_path: Path,
) -> None:
    path = tmp_path / "timing.avi"
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        25.0,
        (64, 48),
    )
    assert writer.isOpened()

    for value in range(20):
        image = np.full((48, 64, 3), value, dtype=np.uint8)
        writer.write(image)
    writer.release()

    before_size = path.stat().st_size
    cap = cv2.VideoCapture(str(path))
    assert cap.isOpened()
    assert cap.get(cv2.CAP_PROP_FRAME_COUNT) == pytest.approx(20)
    assert cap.get(cv2.CAP_PROP_FPS) == pytest.approx(25.0, rel=1e-3)
    cap.release()

    written_fps = rewrite_avi_frame_rate(path, 13.75)

    assert written_fps == pytest.approx(13.75, rel=1e-8)
    assert path.stat().st_size == before_size

    cap = cv2.VideoCapture(str(path))
    assert cap.isOpened()
    assert cap.get(cv2.CAP_PROP_FRAME_COUNT) == pytest.approx(20)
    assert cap.get(cv2.CAP_PROP_FPS) == pytest.approx(13.75, rel=1e-3)

    decoded = 0
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        decoded += 1
    cap.release()

    assert decoded == 20


def test_rewrite_avi_frame_rate_rejects_non_avi(tmp_path: Path) -> None:
    path = tmp_path / "not.avi"
    path.write_bytes(b"not-an-avi")

    with pytest.raises(AviTimingError):
        rewrite_avi_frame_rate(path, 14.0)


@pytest.mark.parametrize("fps", [0.0, -1.0, 1000.1])
def test_rewrite_avi_frame_rate_rejects_invalid_fps(
    tmp_path: Path,
    fps: float,
) -> None:
    path = tmp_path / "dummy.avi"
    path.write_bytes(b"RIFF" + b"\x00" * 8)

    with pytest.raises(AviTimingError):
        rewrite_avi_frame_rate(path, fps)


def test_rewrite_avi_frame_rate_rolls_back_on_fsync_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    path = tmp_path / "rollback.avi"
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        25.0,
        (64, 48),
    )
    assert writer.isOpened()
    for value in range(8):
        writer.write(np.full((48, 64, 3), value, dtype=np.uint8))
    writer.release()

    calls = 0
    real_fsync = avi_timing.os.fsync

    def fail_first_fsync(fd: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("simulated fsync failure")
        real_fsync(fd)

    monkeypatch.setattr(avi_timing.os, "fsync", fail_first_fsync)

    with pytest.raises(AviTimingError, match="rolled back"):
        rewrite_avi_frame_rate(path, 13.5)

    cap = cv2.VideoCapture(str(path))
    assert cap.isOpened()
    assert cap.get(cv2.CAP_PROP_FRAME_COUNT) == pytest.approx(8)
    assert cap.get(cv2.CAP_PROP_FPS) == pytest.approx(25.0, rel=1e-3)
    cap.release()
