"""Capture metadata schema and camera-fact reading tests."""

from __future__ import annotations

import json
from datetime import date, datetime, timezone

import cv2
import pytest

from fomo_servo.capture.metadata import (
    CAMERA_CONTROL_PROPERTIES,
    CameraFacts,
    CaptureSessionRecord,
    read_camera_controls,
    read_camera_facts,
    write_metadata,
)
from fomo_servo.capture.session_layout import plan_next_session


def _facts(**overrides) -> CameraFacts:
    values = dict(
        source="0",
        backend="V4L2",
        requested_width=None,
        requested_height=None,
        requested_fps=None,
        observed_width=640,
        observed_height=480,
        observed_fps=30.0,
        observed_fourcc="MJPG",
        controls={"exposure": -1.0, "gain": None, "brightness": 128.0},
    )
    values.update(overrides)
    return CameraFacts(**values)


REQUIRED_METADATA_KEYS = {
    "schema_version",
    "kind",
    "dataset_role",
    "status",
    "end_reason",
    "session_id",
    "scene",
    "target",
    "notes",
    "start_time_utc",
    "end_time_utc",
    "duration_seconds",
    "source",
    "camera_backend",
    "requested_width",
    "requested_height",
    "requested_fps",
    "observed_width",
    "observed_height",
    "observed_fps",
    "observed_fourcc",
    "measured_capture_fps",
    "video_container",
    "codec",
    "video_filename",
    "video_files",
    "video_frame_count",
    "snapshot_count",
    "recording_duration_seconds",
    "actual_video_size_bytes",
    "actual_storage_gb_per_hour",
    "frame_interval_seconds",
    "frames_directory",
    "platform",
    "camera_controls",
}

REQUIRED_PLATFORM_KEYS = {"platform", "machine", "python", "opencv"}


def _record(tmp_path) -> CaptureSessionRecord:
    session = plan_next_session(
        tmp_path / "datasets_raw" / "lab_pool", prefix="pool", date=date(2026, 8, 31)
    )
    return CaptureSessionRecord(
        session=session,
        facts=_facts(),
        scene="pool-clear-water-front-view",
        target="fish-target",
        notes="first visit",
        frame_interval_seconds=None,
        start_time=datetime(2026, 8, 31, 10, 0, 0, tzinfo=timezone.utc),
    )


def test_metadata_contains_full_schema(tmp_path) -> None:
    record = _record(tmp_path)
    record.add_video_file(
        filename="raw.avi",
        frame_count=912,
        duration_seconds=30.4,
        container_fps=30.0,
        file_size_bytes=50_000_000,
    )
    record.snapshot_count = 3

    metadata = record.finalize(
        status="completed",
        end_reason="user_quit",
        end_time=datetime(2026, 8, 31, 10, 1, 30, tzinfo=timezone.utc),
        measured_capture_fps=29.9,
    )

    assert REQUIRED_METADATA_KEYS.issubset(metadata)
    assert metadata["session_id"] == "pool-20260831-001"
    assert metadata["dataset_role"] == "LAB POOL ENGINEERING VALIDATION DATASET"
    assert metadata["status"] == "completed"
    assert metadata["end_reason"] == "user_quit"
    assert metadata["video_filename"] == "raw.avi"
    assert metadata["video_frame_count"] == 912
    assert metadata["video_files"][0]["frame_count"] == 912
    assert metadata["video_files"][0]["duration_seconds"] == pytest.approx(30.4)
    assert metadata["video_files"][0]["file_size_bytes"] == 50_000_000
    assert metadata["snapshot_count"] == 3
    assert metadata["recording_duration_seconds"] == pytest.approx(30.4)
    assert metadata["actual_video_size_bytes"] == 50_000_000
    assert metadata["actual_storage_gb_per_hour"] == pytest.approx(5.921, abs=0.001)
    assert metadata["codec"] == "MJPG"
    assert metadata["video_container"] == "avi"
    assert metadata["measured_capture_fps"] == pytest.approx(29.9)
    assert REQUIRED_PLATFORM_KEYS.issubset(metadata["platform"])
    assert metadata["camera_controls"]["exposure"] is None
    assert metadata["camera_controls"]["gain"] is None
    assert metadata["camera_controls"]["brightness"] == 128.0
    assert metadata["start_time_utc"].startswith("2026-08-31T10:00:00")
    assert metadata["duration_seconds"] == pytest.approx(90.0, abs=0.001)
    assert record.finalized is True


def test_direct_camera_facts_controls_are_normalized_without_guessing(tmp_path) -> None:
    facts = _facts(
        controls={
            "exposure": "2.5",
            "auto_exposure": True,
            "gain": 12,
            "brightness": 128.0,
            "contrast": float("nan"),
            "saturation": float("inf"),
            "unknown_vendor_control": 99.0,
        }
    )
    session = plan_next_session(
        tmp_path / "datasets_raw" / "lab_pool",
        prefix="pool",
        date=date(2026, 8, 31),
    )
    record = CaptureSessionRecord(
        session=session,
        facts=facts,
        start_time=datetime(2026, 8, 31, 10, 0, 0, tzinfo=timezone.utc),
    )

    controls = record.to_metadata()["camera_controls"]

    assert set(controls) == {name for name, _property in CAMERA_CONTROL_PROPERTIES}
    assert controls["exposure"] is None
    assert controls["auto_exposure"] is None
    assert controls["gain"] == 12.0
    assert controls["brightness"] == 128.0
    assert controls["contrast"] is None
    assert controls["saturation"] is None
    assert controls["focus"] is None


def test_unreliable_camera_controls_are_reported_as_null() -> None:
    class _FakeCapture:
        def __init__(self, props):
            self._props = props

        def get(self, prop):
            return float(self._props.get(prop, 0.0))

    props = {
        cv2.CAP_PROP_EXPOSURE: -1.0,
        cv2.CAP_PROP_GAIN: 0.0,
        cv2.CAP_PROP_BRIGHTNESS: 128.0,
        cv2.CAP_PROP_AUTO_EXPOSURE: 3.0,
    }

    controls = read_camera_controls(_FakeCapture(props))

    assert set(controls) == {
        "exposure",
        "auto_exposure",
        "gain",
        "white_balance",
        "auto_white_balance",
        "focus",
        "autofocus",
        "brightness",
        "contrast",
        "saturation",
    }
    assert controls["exposure"] is None
    assert controls["gain"] is None
    assert controls["brightness"] == 128.0
    assert controls["auto_exposure"] == 3.0
    assert controls["white_balance"] is None
    assert controls["autofocus"] is None


def test_read_camera_facts_reads_observed_properties() -> None:
    class _FakeCapture:
        def __init__(self, props):
            self._props = props

        def get(self, prop):
            return float(self._props.get(prop, 0.0))

    fourcc = float(int.from_bytes(b"MJPG", "little"))
    capture = _FakeCapture(
        {
            cv2.CAP_PROP_FRAME_WIDTH: 1920,
            cv2.CAP_PROP_FRAME_HEIGHT: 1080,
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FOURCC: fourcc,
        }
    )

    facts = read_camera_facts(
        capture,
        source="/dev/video0",
        requested_width=1920,
        requested_height=1080,
        requested_fps=30.0,
    )

    assert facts.source == "/dev/video0"
    assert facts.observed_width == 1920
    assert facts.observed_height == 1080
    assert facts.observed_fps == pytest.approx(30.0)
    assert facts.observed_fourcc == "MJPG"
    assert facts.requested_width == 1920
    assert set(facts.controls) == {
        "exposure",
        "auto_exposure",
        "gain",
        "white_balance",
        "auto_white_balance",
        "focus",
        "autofocus",
        "brightness",
        "contrast",
        "saturation",
    }


def test_metadata_write_is_valid_json_round_trip(tmp_path) -> None:
    record = _record(tmp_path)
    metadata = record.finalize(
        status="completed",
        end_reason="source_ended",
        end_time=datetime(2026, 8, 31, 10, 0, 5, tzinfo=timezone.utc),
        measured_capture_fps=None,
    )
    target = record.session.metadata_path

    write_metadata(target, metadata)
    reloaded = json.loads(target.read_text(encoding="utf-8"))

    assert reloaded == metadata
    assert target.read_text(encoding="utf-8").endswith("\n")


def test_finalize_can_be_called_only_once(tmp_path) -> None:
    record = _record(tmp_path)
    record.finalize(
        status="completed",
        end_reason="user_quit",
        end_time=datetime(2026, 8, 31, 10, 0, 5, tzinfo=timezone.utc),
        measured_capture_fps=None,
    )

    with pytest.raises(RuntimeError):
        record.finalize(
            status="failed",
            end_reason="unexpected_exception",
            end_time=datetime(2026, 8, 31, 10, 0, 6, tzinfo=timezone.utc),
            measured_capture_fps=None,
        )
