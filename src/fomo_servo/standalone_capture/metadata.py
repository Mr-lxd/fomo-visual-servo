"""Capture session metadata schema and camera fact reading.

``camera_controls`` follows a strict no-guessing rule: a property is reported
as ``null`` unless OpenCV returns a finite value other than ``0`` or ``-1``
(the values OpenCV uses to signal "not available"). A genuine zero reading is
therefore also reported as ``null`` rather than guessed.
"""

from __future__ import annotations

import json
import math
from numbers import Real
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Optional

import cv2

from .session_layout import SessionPaths


DATASET_ROLE = "LAB POOL ENGINEERING VALIDATION DATASET"
METADATA_KIND = "fomo_capture_session"
VIDEO_CONTAINER = "avi"
VIDEO_CODEC = "MJPG"

CAMERA_CONTROL_PROPERTIES = (
    ("exposure", "CAP_PROP_EXPOSURE"),
    ("auto_exposure", "CAP_PROP_AUTO_EXPOSURE"),
    ("gain", "CAP_PROP_GAIN"),
    ("white_balance", "CAP_PROP_WB_TEMPERATURE"),
    ("auto_white_balance", "CAP_PROP_AUTO_WB"),
    ("focus", "CAP_PROP_FOCUS"),
    ("autofocus", "CAP_PROP_AUTOFOCUS"),
    ("brightness", "CAP_PROP_BRIGHTNESS"),
    ("contrast", "CAP_PROP_CONTRAST"),
    ("saturation", "CAP_PROP_SATURATION"),
)

REQUIRED_PLATFORM_KEYS = ("platform", "machine", "python", "opencv")


@dataclass(frozen=True)
class CameraFacts:
    """One-time snapshot of requested/observed camera properties and controls."""

    source: str
    backend: Optional[str]
    requested_width: Optional[int]
    requested_height: Optional[int]
    requested_fps: Optional[float]
    observed_width: Optional[int]
    observed_height: Optional[int]
    observed_fps: Optional[float]
    observed_fourcc: Optional[str]
    controls: Mapping[str, Optional[float]]


def _sanitize_control_value(value: Any) -> Optional[float]:
    """Apply the no-guessing rule to one control reading.

    ``None``, non-numeric, non-finite, ``0`` and ``-1`` (OpenCV's
    not-available sentinels) all map to ``None``.
    """

    if value is None or isinstance(value, bool) or not isinstance(value, Real):
        return None
    number = float(value)
    if not math.isfinite(number) or number in (0.0, -1.0):
        return None
    return number


def _read_control(capture: Any, property_name: str) -> Optional[float]:
    prop = getattr(cv2, property_name, None)
    if prop is None:
        return None
    try:
        value = float(capture.get(prop))
    except Exception:
        return None
    return _sanitize_control_value(value)


def read_camera_controls(capture: Any) -> dict[str, Optional[float]]:
    """Read the documented V4L2/OpenCV camera controls, or ``None`` each."""

    return {
        name: _read_control(capture, property_name)
        for name, property_name in CAMERA_CONTROL_PROPERTIES
    }


def _decode_fourcc(value: float) -> Optional[str]:
    packed = int(value)
    if packed <= 0:
        return None
    return "".join(chr((packed >> (8 * index)) & 0xFF) for index in range(4))


def read_camera_facts(
    capture: Any,
    *,
    source: str,
    requested_width: Optional[int],
    requested_height: Optional[int],
    requested_fps: Optional[float],
) -> CameraFacts:
    """Snapshot observed camera properties after opening ``capture``."""

    backend: Optional[str] = None
    backend_property = getattr(cv2, "CAP_PROP_BACKEND", None)
    if backend_property is not None:
        try:
            backend_value = float(capture.get(backend_property))
        except Exception:
            backend_value = 0.0
        if backend_value > 0 and hasattr(cv2, "videoio_registry"):
            try:
                backend = cv2.videoio_registry.getBackendName(int(backend_value))
            except Exception:
                backend = None

    def observed_int(property_name: str) -> Optional[int]:
        value = float(capture.get(getattr(cv2, property_name)))
        return int(value) if value > 0 else None

    def observed_float(property_name: str) -> Optional[float]:
        value = float(capture.get(getattr(cv2, property_name)))
        return value if value > 0 else None

    fourcc_value = float(capture.get(cv2.CAP_PROP_FOURCC))
    return CameraFacts(
        source=str(source),
        backend=backend,
        requested_width=requested_width,
        requested_height=requested_height,
        requested_fps=requested_fps,
        observed_width=observed_int("CAP_PROP_FRAME_WIDTH"),
        observed_height=observed_int("CAP_PROP_FRAME_HEIGHT"),
        observed_fps=observed_float("CAP_PROP_FPS"),
        observed_fourcc=_decode_fourcc(fourcc_value),
        controls=read_camera_controls(capture),
    )


class CaptureSessionRecord:
    """Mutable statistics for one session plus its metadata serialization."""

    def __init__(
        self,
        *,
        session: SessionPaths,
        facts: CameraFacts,
        scene: str = "",
        target: str = "",
        notes: str = "",
        frame_interval_seconds: Optional[float] = None,
        start_time: datetime,
    ) -> None:
        self.session = session
        self.facts = facts
        self.scene = scene
        self.target = target
        self.notes = notes
        self.frame_interval_seconds = frame_interval_seconds
        self.start_time = start_time
        self.video_files: list[dict[str, Any]] = []
        self.snapshot_count = 0
        self.recording_seconds = 0.0
        self._status: Optional[str] = None
        self._end_reason: Optional[str] = None
        self._end_time: Optional[datetime] = None
        self._measured_capture_fps: Optional[float] = None
        self.finalized = False

    @property
    def video_frame_count(self) -> int:
        return sum(int(item["frame_count"]) for item in self.video_files)

    @property
    def actual_video_size_bytes(self) -> Optional[int]:
        """Return summed on-disk video bytes, or ``None`` if any size is unknown."""

        sizes = [item.get("file_size_bytes") for item in self.video_files]
        if not sizes:
            return 0
        if any(size is None for size in sizes):
            return None
        return sum(int(size) for size in sizes)

    def add_video_file(
        self,
        *,
        filename: str,
        frame_count: int,
        duration_seconds: float,
        container_fps: float,
        file_size_bytes: Optional[int] = None,
    ) -> None:
        """Append one finalized recording segment."""

        self.video_files.append(
            {
                "filename": filename,
                "frame_count": int(frame_count),
                "duration_seconds": round(float(duration_seconds), 3),
                "container_fps": round(float(container_fps), 3),
                "file_size_bytes": (
                    int(file_size_bytes) if file_size_bytes is not None else None
                ),
            }
        )
        self.recording_seconds += float(duration_seconds)

    def finalize(
        self,
        *,
        status: str,
        end_reason: str,
        end_time: datetime,
        measured_capture_fps: Optional[float],
    ) -> dict[str, Any]:
        """Freeze the session outcome once and return the metadata mapping."""

        if self.finalized:
            raise RuntimeError(
                "session '{}' is already finalized".format(self.session.session_id)
            )
        self._status = status
        self._end_reason = end_reason
        self._end_time = end_time
        self._measured_capture_fps = measured_capture_fps
        self.finalized = True
        return self.to_metadata()

    def to_metadata(self) -> dict[str, Any]:
        duration = None
        if self._end_time is not None:
            duration = round(
                (self._end_time - self.start_time).total_seconds(), 3
            )
        actual_size = self.actual_video_size_bytes
        actual_gb_per_hour = None
        if actual_size is not None and self.recording_seconds > 0.0:
            actual_gb_per_hour = round(
                (actual_size / self.recording_seconds) * 3600.0 / 1_000_000_000.0,
                3,
            )
        return {
            "schema_version": 1,
            "kind": METADATA_KIND,
            "dataset_role": DATASET_ROLE,
            "status": self._status,
            "end_reason": self._end_reason,
            "session_id": self.session.session_id,
            "scene": self.scene,
            "target": self.target,
            "notes": self.notes,
            "start_time_utc": self.start_time.isoformat(),
            "end_time_utc": (
                self._end_time.isoformat() if self._end_time is not None else None
            ),
            "duration_seconds": duration,
            "source": self.facts.source,
            "camera_backend": self.facts.backend,
            "requested_width": self.facts.requested_width,
            "requested_height": self.facts.requested_height,
            "requested_fps": self.facts.requested_fps,
            "observed_width": self.facts.observed_width,
            "observed_height": self.facts.observed_height,
            "observed_fps": self.facts.observed_fps,
            "observed_fourcc": self.facts.observed_fourcc,
            "measured_capture_fps": self._measured_capture_fps,
            "video_container": VIDEO_CONTAINER,
            "codec": VIDEO_CODEC,
            "video_filename": (
                self.video_files[0]["filename"] if self.video_files else None
            ),
            "video_files": list(self.video_files),
            "video_frame_count": self.video_frame_count,
            "snapshot_count": self.snapshot_count,
            "recording_duration_seconds": round(self.recording_seconds, 3),
            "actual_video_size_bytes": actual_size,
            "actual_storage_gb_per_hour": actual_gb_per_hour,
            "frame_interval_seconds": self.frame_interval_seconds,
            "frames_directory": self.session.frames_dir.name,
            "platform": {
                "platform": _platform_text(),
                "machine": _python_attribute("platform", "machine"),
                "python": _python_attribute("platform", "python_version"),
                "opencv": cv2.__version__,
            },
            "camera_controls": {
                name: _sanitize_control_value(self.facts.controls.get(name))
                for name, _property_name in CAMERA_CONTROL_PROPERTIES
            },
        }


def _platform_text() -> str:
    try:
        import platform as platform_module

        return platform_module.platform()
    except Exception:
        return "unknown"


def _python_attribute(module_name: str, attribute: str) -> str:
    try:
        import platform as platform_module

        return str(getattr(platform_module, attribute)())
    except Exception:
        return "unknown"


def write_metadata(path: Path, metadata: Mapping[str, Any]) -> None:
    """Serialize ``metadata`` as pretty JSON with a trailing newline."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(dict(metadata), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
