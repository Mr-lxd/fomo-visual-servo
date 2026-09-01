"""HUD overlay tests: pure drawing, raw-frame independence, status content."""

from __future__ import annotations

import numpy as np
import pytest

from fomo_servo.capture.hud import INSTRUCTIONS, draw_status_overlay, status_lines


def _state(**overrides) -> dict:
    state = dict(
        camera="0",
        resolution="640x480",
        fps="29.8",
        free_space_text="42.3 GB",
        session_id="pool-20260831-001",
        recording=False,
        rec_seconds=0.0,
        video_frames=0,
        snapshot_count=0,
        low_disk_space=False,
    )
    state.update(overrides)
    return state


def test_status_lines_show_preview_and_counters() -> None:
    lines = status_lines(_state())

    joined = "\n".join(lines)
    assert "CAMERA: 0" in joined
    assert "RES: 640x480" in joined
    assert "FPS: 29.8" in joined
    assert "FREE SPACE: 42.3 GB" in joined
    assert "SESSION: pool-20260831-001" in joined
    assert "STATUS: PREVIEW" in joined
    assert "REC TIME: 00:00:00" in joined
    assert "VIDEO FRAMES: 0" in joined
    assert "SAVED IMAGES: 0" in joined
    assert INSTRUCTIONS not in joined


def test_recording_state_is_clearly_marked() -> None:
    lines = status_lines(_state(recording=True, rec_seconds=83.0, video_frames=1843))

    joined = "\n".join(lines)
    assert "● REC" in joined
    assert "PREVIEW" not in joined
    assert "REC TIME: 00:01:23" in joined
    assert "VIDEO FRAMES: 1843" in joined


def test_low_disk_space_state_adds_warning_line() -> None:
    lines = status_lines(_state(low_disk_space=True))

    assert any("LOW DISK" in line for line in lines)


def test_draw_status_overlay_never_mutates_the_raw_frame() -> None:
    raw = np.full((240, 320, 3), 64, dtype=np.uint8)
    raw_copy = raw.copy()

    preview = draw_status_overlay(raw, _state(recording=True))

    assert np.array_equal(raw, raw_copy), "raw frame must stay untouched"
    assert preview is not raw
    assert not np.array_equal(preview, raw), "preview copy must carry the overlay"
    assert preview.shape == raw.shape
    assert preview.dtype == raw.dtype


def test_draw_status_overlay_is_repeatable(tmp_path) -> None:
    raw = np.zeros((240, 320, 3), dtype=np.uint8)

    first = draw_status_overlay(raw, _state())
    second = draw_status_overlay(raw, _state())

    assert np.array_equal(first, second)
