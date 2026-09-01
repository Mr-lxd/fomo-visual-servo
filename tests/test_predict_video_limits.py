"""Controlled termination tests for scripts/predict_video.py."""

from __future__ import annotations

import csv
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from fomo_servo.inference import FramePacket
from test_onnx_runtime_predictor import _write_model_and_report


def _write_input_video(path: Path, frame_count: int = 6) -> list[np.ndarray]:
    """Write an MJPG AVI and return the identical BGR frames for stub readers."""

    frames = []
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), 5.0, (16, 8)
    )
    assert writer.isOpened()
    try:
        for index in range(frame_count):
            frame = np.zeros((8, 16, 3), dtype=np.uint8)
            frame[:, :, 1] = 32 + index
            writer.write(frame)
            frames.append(frame.copy())
    finally:
        writer.release()
    return frames


def _base_arguments(
    tmp_path: Path, onnx_path: Path, report_path: Path, source: str
) -> list[str]:
    return [
        "--onnx",
        str(onnx_path),
        "--onnx-report",
        str(report_path),
        "--source",
        source,
        "--output-video",
        str(tmp_path / "out" / "prediction.mp4"),
        "--output-csv",
        str(tmp_path / "out" / "prediction.csv"),
        "--output-jsonl",
        str(tmp_path / "out" / "prediction.jsonl"),
    ]


def _csv_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class _StubBuffer:
    """Deliver pre-built packets synchronously; advance a fake clock per frame."""

    def __init__(self, packets: list[FramePacket], clock: dict[str, float]) -> None:
        self._packets = list(packets)
        self._clock = clock

    def get(self, timeout: float = 1.0):
        if self._packets:
            self._clock["now"] = round(self._clock["now"] + 0.6, 6)
            return self._packets.pop(0)
        return None


class _StubReader:
    """Synchronous LatestFrameReader replacement with no frame dropping."""

    def __init__(self, capture) -> None:
        self._capture = capture
        self.buffer = None
        self.finished = threading.Event()
        self.error = None

    def start(self):
        return self

    def stop(self) -> None:
        self._capture.release()


def _install_stub_reader(
    monkeypatch: pytest.MonkeyPatch, frames: list[np.ndarray], clock: dict[str, float]
) -> None:
    from scripts import predict_video

    def factory(capture):
        reader = _StubReader(capture)
        packets = [
            FramePacket(index, 0.1 * index, frame.copy())
            for index, frame in enumerate(frames)
        ]
        reader.buffer = _StubBuffer(packets, clock)
        return reader

    monkeypatch.setattr(predict_video, "LatestFrameReader", factory)


def test_max_frames_stops_processing_and_flushes_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.predict_video import main

    onnx_path, report_path, _model = _write_model_and_report(tmp_path)
    frames = _write_input_video(tmp_path / "input.avi", frame_count=6)
    clock = {"now": 0.0}
    _install_stub_reader(monkeypatch, frames, clock)
    arguments = _base_arguments(
        tmp_path, onnx_path, report_path, str(tmp_path / "input.avi")
    ) + ["--max-frames", "3"]

    exit_code = main(arguments)

    assert exit_code == 0
    rows = _csv_rows(tmp_path / "out" / "prediction.csv")
    assert len(rows) == 3
    jsonl_lines = (tmp_path / "out" / "prediction.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(jsonl_lines) == 3
    assert all(
        json.loads(line)["runtime"] == "onnxruntime" for line in jsonl_lines
    )
    output_video = tmp_path / "out" / "prediction.mp4"
    assert output_video.is_file() and output_video.stat().st_size > 0


def test_duration_seconds_stops_processing_after_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import predict_video

    onnx_path, report_path, _model = _write_model_and_report(tmp_path)
    frames = _write_input_video(tmp_path / "input.avi", frame_count=6)
    clock = {"now": 0.0}
    _install_stub_reader(monkeypatch, frames, clock)
    monkeypatch.setattr(
        predict_video, "time", SimpleNamespace(monotonic=lambda: clock["now"])
    )
    arguments = _base_arguments(
        tmp_path, onnx_path, report_path, str(tmp_path / "input.avi")
    ) + ["--duration-seconds", "1.0"]

    exit_code = predict_video.main(arguments)

    assert exit_code == 0
    rows = _csv_rows(tmp_path / "out" / "prediction.csv")
    assert len(rows) == 2
    jsonl_lines = (tmp_path / "out" / "prediction.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(jsonl_lines) == 2
    output_video = tmp_path / "out" / "prediction.mp4"
    assert output_video.is_file() and output_video.stat().st_size > 0


def test_without_limits_processes_until_source_ends(tmp_path: Path) -> None:
    """No limit arguments must keep the existing run-to-EOF/infinity behavior."""

    from scripts.predict_video import main

    onnx_path, report_path, _model = _write_model_and_report(tmp_path)
    _write_input_video(tmp_path / "input.avi", frame_count=6)
    arguments = _base_arguments(
        tmp_path, onnx_path, report_path, str(tmp_path / "input.avi")
    )

    exit_code = main(arguments)

    assert exit_code == 0
    rows = _csv_rows(tmp_path / "out" / "prediction.csv")
    assert len(rows) >= 1
    jsonl_lines = (tmp_path / "out" / "prediction.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(jsonl_lines) == len(rows)


def test_process_every_frame_preserves_complete_offline_video_sequence(
    tmp_path: Path,
) -> None:
    """Offline validation mode must emit one telemetry row per source frame."""

    from scripts.predict_video import main

    onnx_path, report_path, _model = _write_model_and_report(tmp_path)
    _write_input_video(tmp_path / "input.avi", frame_count=6)
    arguments = _base_arguments(
        tmp_path, onnx_path, report_path, str(tmp_path / "input.avi")
    ) + ["--process-every-frame"]

    exit_code = main(arguments)

    assert exit_code == 0
    rows = _csv_rows(tmp_path / "out" / "prediction.csv")
    assert [int(row["frame_index"]) for row in rows] == list(range(6))
    jsonl_lines = (tmp_path / "out" / "prediction.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(jsonl_lines) == 6


def test_max_frames_must_be_positive(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from scripts.predict_video import main

    arguments = _base_arguments(
        tmp_path, tmp_path / "missing.onnx", tmp_path / "missing.json", "0"
    ) + ["--max-frames", "0"]

    exit_code = main(arguments)

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "--max-frames" in captured.err


def test_duration_seconds_must_be_positive(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from scripts.predict_video import main

    arguments = _base_arguments(
        tmp_path, tmp_path / "missing.onnx", tmp_path / "missing.json", "0"
    ) + ["--duration-seconds", "0"]

    exit_code = main(arguments)

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "--duration-seconds" in captured.err
