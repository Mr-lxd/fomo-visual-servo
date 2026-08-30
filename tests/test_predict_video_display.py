"""Opt-in desktop preview tests for ``scripts/predict_video.py``."""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import pytest

from fomo_servo.inference import FramePacket
from test_onnx_runtime_predictor import _write_model_and_report


class _Capture:
    """Minimal camera capture that records release without owning hardware."""

    def __init__(self, *, width: int, height: int, fps: float) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self.released = False

    def get(self, property_id: int) -> float:
        if property_id == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self.width)
        if property_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self.height)
        if property_id == cv2.CAP_PROP_FPS:
            return self.fps
        return 0.0

    def release(self) -> None:
        self.released = True


class _Buffer:
    def __init__(self, packets: list[FramePacket]) -> None:
        self._packets = list(packets)

    def get(self, timeout: float = 1.0):
        del timeout
        if self._packets:
            return self._packets.pop(0)
        return None


class _Reader:
    def __init__(self, capture: _Capture, frames: list[np.ndarray]) -> None:
        self.capture = capture
        self.buffer = _Buffer(
            [
                FramePacket(index, index / capture.fps, frame.copy())
                for index, frame in enumerate(frames)
            ]
        )
        self.error = None
        self.finished = type(
            "_Finished", (), {"is_set": staticmethod(lambda: True)}
        )()

    def start(self):
        return self

    def stop(self) -> None:
        self.capture.release()


class _Writer:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.frames: list[np.ndarray] = []
        self.released = False

    def isOpened(self) -> bool:
        return True

    def write(self, frame: np.ndarray) -> None:
        self.events.append("write")
        self.frames.append(frame.copy())

    def release(self) -> None:
        self.events.append("writer_release")
        self.released = True


def _arguments(
    tmp_path: Path, onnx_path: Path, report_path: Path, *, display: bool
) -> list[str]:
    arguments = [
        "--onnx",
        str(onnx_path),
        "--onnx-report",
        str(report_path),
        "--source",
        "0",
        "--output-video",
        str(tmp_path / "out" / "camera.mp4"),
        "--output-csv",
        str(tmp_path / "out" / "camera.csv"),
        "--output-jsonl",
        str(tmp_path / "out" / "camera.jsonl"),
    ]
    if display:
        arguments.append("--display")
    return arguments


def _install_runtime(
    monkeypatch: pytest.MonkeyPatch,
    *,
    frames: list[np.ndarray],
    events: list[str],
) -> tuple[_Capture, _Writer]:
    from scripts import predict_video

    height, width = frames[0].shape[:2]
    capture = _Capture(width=width, height=height, fps=25.0)
    writer = _Writer(events)
    monkeypatch.setattr(predict_video, "_open_source", lambda _source: capture)
    monkeypatch.setattr(
        predict_video, "LatestFrameReader", lambda opened: _Reader(opened, frames)
    )
    monkeypatch.setattr(predict_video.cv2, "VideoWriter", lambda *args: writer)
    return capture, writer


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_display_defaults_off() -> None:
    from scripts.predict_video import build_parser

    args = build_parser().parse_args(
        [
            "--onnx",
            "model.onnx",
            "--onnx-report",
            "model.onnx.json",
            "--source",
            "0",
            "--output-video",
            "out.mp4",
            "--output-csv",
            "out.csv",
            "--output-jsonl",
            "out.jsonl",
        ]
    )

    assert args.display is False


def test_headless_mode_never_calls_highgui(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import predict_video

    onnx_path, report_path, _model = _write_model_and_report(tmp_path)
    frames = [np.zeros((48, 64, 3), dtype=np.uint8)]
    events: list[str] = []
    capture, writer = _install_runtime(
        monkeypatch, frames=frames, events=events
    )

    def reject_highgui(*_args, **_kwargs):
        raise AssertionError("headless mode must not call OpenCV HighGUI")

    for name in ("namedWindow", "imshow", "waitKey", "destroyWindow"):
        monkeypatch.setattr(predict_video.cv2, name, reject_highgui)

    exit_code = predict_video.main(
        _arguments(tmp_path, onnx_path, report_path, display=False)
    )

    assert exit_code == 0
    assert len(_csv_rows(tmp_path / "out" / "camera.csv")) == 1
    assert capture.released
    assert writer.released


@pytest.mark.parametrize("stop_key", (ord("q"), 27))
def test_display_q_or_escape_stops_after_writing_annotated_frame_and_releases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stop_key: int,
) -> None:
    from scripts import predict_video

    onnx_path, report_path, _model = _write_model_and_report(tmp_path)
    frames = [np.zeros((48, 64, 3), dtype=np.uint8) for _ in range(3)]
    original_first_frame = frames[0].copy()
    events: list[str] = []
    capture, writer = _install_runtime(
        monkeypatch, frames=frames, events=events
    )
    shown_frames: list[np.ndarray] = []
    closed_windows: list[str] = []
    monkeypatch.setattr(
        predict_video.cv2,
        "getBuildInformation",
        lambda: "General configuration\n  GUI: QT5\n",
    )
    monkeypatch.setenv("DISPLAY", ":test")
    monkeypatch.setattr(
        predict_video.cv2,
        "namedWindow",
        lambda name, flags: events.append("open"),
    )

    def show(_name: str, frame: np.ndarray) -> None:
        events.append("show")
        shown_frames.append(frame.copy())

    monkeypatch.setattr(predict_video.cv2, "imshow", show)
    monkeypatch.setattr(predict_video.cv2, "waitKey", lambda _delay: stop_key)
    monkeypatch.setattr(
        predict_video.cv2,
        "destroyWindow",
        lambda name: (events.append("close"), closed_windows.append(name)),
    )

    exit_code = predict_video.main(
        _arguments(tmp_path, onnx_path, report_path, display=True)
    )

    assert exit_code == 0
    assert len(_csv_rows(tmp_path / "out" / "camera.csv")) == 1
    assert len((tmp_path / "out" / "camera.jsonl").read_text().splitlines()) == 1
    assert events.index("write") < events.index("show")
    assert len(writer.frames) == len(shown_frames) == 1
    np.testing.assert_array_equal(writer.frames[0], shown_frames[0])
    assert not np.array_equal(shown_frames[0], original_first_frame)
    np.testing.assert_array_equal(frames[0], original_first_frame)
    assert capture.released and writer.released
    assert closed_windows


def test_display_reports_headless_opencv_before_opening_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from scripts import predict_video

    monkeypatch.setattr(
        predict_video.cv2,
        "getBuildInformation",
        lambda: "General configuration\n  GUI: NONE\n",
    )
    monkeypatch.setattr(
        predict_video,
        "_open_source",
        lambda _source: pytest.fail("camera must not open without GUI capability"),
    )

    exit_code = predict_video.main(
        _arguments(
            tmp_path,
            tmp_path / "missing.onnx",
            tmp_path / "missing.onnx.json",
            display=True,
        )
    )

    assert exit_code == 1
    error = capsys.readouterr().err.lower()
    assert "gui" in error
    assert "preview" in error
    assert "headless" in error


def test_linux_display_requires_desktop_session_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts.predict_video import InferenceError, _validate_preview_environment

    with pytest.raises(InferenceError, match="VNC desktop terminal"):
        _validate_preview_environment(
            gui_backend="QT5", environment={}, platform_name="linux"
        )


def test_highgui_failure_is_diagnostic_and_all_resources_are_released(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from scripts import predict_video

    onnx_path, report_path, _model = _write_model_and_report(tmp_path)
    frames = [np.zeros((48, 64, 3), dtype=np.uint8)]
    events: list[str] = []
    capture, writer = _install_runtime(
        monkeypatch, frames=frames, events=events
    )
    monkeypatch.setattr(
        predict_video.cv2,
        "getBuildInformation",
        lambda: "General configuration\n  GUI: QT5\n",
    )
    monkeypatch.setenv("DISPLAY", ":test")
    monkeypatch.setattr(predict_video.cv2, "namedWindow", lambda *_args: None)
    monkeypatch.setattr(
        predict_video.cv2,
        "imshow",
        lambda *_args: (_ for _ in ()).throw(cv2.error("display disconnected")),
    )
    monkeypatch.setattr(predict_video.cv2, "destroyWindow", lambda name: None)

    exit_code = predict_video.main(
        _arguments(tmp_path, onnx_path, report_path, display=True)
    )

    assert exit_code == 1
    error = capsys.readouterr().err.lower()
    assert "preview" in error
    assert "display disconnected" in error
    assert capture.released and writer.released
    assert len(_csv_rows(tmp_path / "out" / "camera.csv")) == 1
