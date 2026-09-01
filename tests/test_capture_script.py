"""CLI-level tests for scripts/capture_dataset.py and run.py dispatch."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import date
from pathlib import Path

import cv2
import numpy as np
import pytest

from test_capture_engine import FOURCC_MJPG, _FakeCapture
from scripts.capture_dataset import main


def _props() -> dict:
    return {
        cv2.CAP_PROP_FRAME_WIDTH: 16,
        cv2.CAP_PROP_FRAME_HEIGHT: 8,
        cv2.CAP_PROP_FPS: 5.0,
        cv2.CAP_PROP_FOURCC: FOURCC_MJPG,
    }


def _frames(count):
    return [
        np.full((8, 16, 3), 50 + index, dtype=np.uint8) for index in range(count)
    ]


def test_headless_run_creates_session_tree_and_metadata(tmp_path: Path) -> None:
    from fomo_servo.capture.engine import CaptureIO

    tick = {"value": 0.0}

    def clock() -> float:
        tick["value"] += 0.1
        return tick["value"]

    output_root = tmp_path / "datasets_raw" / "lab_pool"
    frames = _frames(6)
    arguments = [
        "--source",
        "0",
        "--output-root",
        str(output_root),
        "--max-frames",
        "4",
    ]

    exit_code = main(
        arguments,
        capture_factory=lambda: _FakeCapture(frames, props=_props()),
        io=CaptureIO(clock=clock),
    )

    assert exit_code == 0
    today = date.today().strftime("%Y%m%d")
    session_dir = output_root / today / f"pool-{today}-001"
    assert (session_dir / "raw.avi").is_file()
    assert (session_dir / "raw.avi").stat().st_size > 0
    assert (session_dir / "frames").is_dir()
    metadata = json.loads((session_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "completed"
    assert metadata["end_reason"] == "max_frames"
    assert metadata["video_frame_count"] == 4
    assert metadata["snapshot_count"] == 0
    assert metadata["observed_width"] == 16
    assert metadata["observed_height"] == 8
    assert metadata["camera_backend"] is None  # fake capture reports no backend
    assert metadata["actual_video_size_bytes"] == (session_dir / "raw.avi").stat().st_size
    assert metadata["actual_storage_gb_per_hour"] > 0.0


def test_headless_writer_open_failure_releases_camera_and_writes_failed_metadata(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    class _ClosedWriter:
        def isOpened(self):
            return False

        def release(self):
            return None

    monkeypatch.setattr(
        "fomo_servo.capture.engine._default_writer_factory",
        lambda path, fps, size: _ClosedWriter(),
    )
    capture = _FakeCapture(_frames(2), props=_props())
    output_root = tmp_path / "datasets_raw" / "lab_pool"

    exit_code = main(
        ["--source", "0", "--output-root", str(output_root)],
        capture_factory=lambda: capture,
    )

    assert exit_code == 1
    assert capture.released is True
    session_dirs = list((output_root / date.today().strftime("%Y%m%d")).iterdir())
    metadata = json.loads(
        (session_dirs[0] / "metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["status"] == "failed"
    assert metadata["end_reason"] == "video_writer_open_failure"
    assert "unable to open video writer" in capsys.readouterr().err


def test_camera_read_failure_returns_nonzero_and_failed_metadata(
    tmp_path: Path, capsys
) -> None:
    output_root = tmp_path / "datasets_raw" / "lab_pool"

    exit_code = main(
        ["--source", "0", "--output-root", str(output_root)],
        capture_factory=lambda: _FakeCapture([], props=_props()),
    )

    assert exit_code == 1
    session_dirs = list((output_root / date.today().strftime("%Y%m%d")).iterdir())
    metadata = json.loads(
        (session_dirs[0] / "metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["status"] == "failed"
    assert metadata["end_reason"] == "camera_read_failure"
    assert "camera read failed" in capsys.readouterr().err


def test_display_false_never_creates_windows_or_previews(tmp_path: Path) -> None:
    calls = {"create_window": 0, "imshow": 0}

    def create_window(name, fullscreen):
        calls["create_window"] += 1

    def imshow(name, frame):
        calls["imshow"] += 1

    from fomo_servo.capture.engine import CaptureIO

    io = CaptureIO(create_window=create_window, imshow=imshow)
    output_root = tmp_path / "datasets_raw" / "lab_pool"

    exit_code = main(
        ["--source", "0", "--output-root", str(output_root), "--max-frames", "2"],
        capture_factory=lambda: _FakeCapture(_frames(4), props=_props()),
        io=io,
    )

    assert exit_code == 0
    assert calls == {"create_window": 0, "imshow": 0}


def test_cli_rejects_non_positive_limits(tmp_path: Path, capsys) -> None:
    output_root = tmp_path / "datasets_raw" / "lab_pool"
    factory = lambda: _FakeCapture(_frames(2), props=_props())

    assert (
        main(
            ["--source", "0", "--output-root", str(output_root), "--max-frames", "0"],
            capture_factory=factory,
        )
        == 1
    )
    assert "--max-frames" in capsys.readouterr().err

    assert (
        main(
            [
                "--source",
                "0",
                "--output-root",
                str(output_root),
                "--duration-seconds",
                "0",
            ],
            capture_factory=factory,
        )
        == 1
    )
    assert "--duration-seconds" in capsys.readouterr().err

    assert (
        main(
            [
                "--source",
                "0",
                "--output-root",
                str(output_root),
                "--frame-interval-seconds",
                "-1",
            ],
            capture_factory=factory,
        )
        == 1
    )
    assert "--frame-interval-seconds" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("option", "value"),
    [("--width", "0"), ("--height", "-1"), ("--fps", "0")],
)
def test_cli_rejects_non_positive_camera_requests(
    tmp_path: Path, capsys, option: str, value: str
) -> None:
    output_root = tmp_path / "datasets_raw" / "lab_pool"

    exit_code = main(
        ["--source", "0", "--output-root", str(output_root), option, value],
        capture_factory=lambda: _FakeCapture(_frames(2), props=_props()),
    )

    assert exit_code == 1
    assert option in capsys.readouterr().err


def test_cli_warns_when_free_space_below_threshold(tmp_path: Path, capsys) -> None:
    from fomo_servo.capture.engine import CaptureIO

    io = CaptureIO(disk_free_bytes=lambda path: 1 << 30)
    output_root = tmp_path / "datasets_raw" / "lab_pool"

    exit_code = main(
        [
            "--source",
            "0",
            "--output-root",
            str(output_root),
            "--max-frames",
            "1",
            "--min-free-gb",
            "5.0",
        ],
        capture_factory=lambda: _FakeCapture(_frames(2), props=_props()),
        io=io,
    )

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "FREE SPACE" in captured.out
    assert "LOW DISK" in captured.out


def test_cli_does_not_warn_above_threshold(tmp_path: Path, capsys) -> None:
    from fomo_servo.capture.engine import CaptureIO

    io = CaptureIO(disk_free_bytes=lambda path: 100 << 30)
    output_root = tmp_path / "datasets_raw" / "lab_pool"

    exit_code = main(
        [
            "--source",
            "0",
            "--output-root",
            str(output_root),
            "--max-frames",
            "1",
            "--min-free-gb",
            "5.0",
        ],
        capture_factory=lambda: _FakeCapture(_frames(2), props=_props()),
        io=io,
    )

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "FREE SPACE" in captured.out
    assert "LOW DISK" not in captured.out


def test_run_py_dispatches_capture_dataset_entry() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)

    result = subprocess.run(
        [sys.executable, str(repo_root / "run.py"), "capture_dataset", "--help"],
        cwd=repo_root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--output-root" in result.stdout
    assert "--display" in result.stdout
    assert "--fullscreen" in result.stdout


def test_fullscreen_implies_display_at_runtime(tmp_path: Path) -> None:
    from fomo_servo.capture.engine import CaptureIO

    windows = []
    io = CaptureIO(
        create_window=lambda name, fullscreen: windows.append((name, fullscreen)),
        imshow=lambda name, frame: None,
        wait_key=lambda timeout: ord("q"),
        window_visible=lambda name: True,
        destroy_windows=lambda: None,
    )
    output_root = tmp_path / "datasets_raw" / "lab_pool"

    exit_code = main(
        ["--source", "0", "--output-root", str(output_root), "--fullscreen"],
        capture_factory=lambda: _FakeCapture(_frames(2), props=_props()),
        io=io,
    )

    assert exit_code == 0
    assert windows == [("FOMO Dataset Capture", True)]
