"""Standalone lab-pool dataset capture (no inference, no model dependency)."""

from .engine import (
    CaptureEngine,
    CaptureError,
    CaptureIO,
    KeyCommand,
)
from .hud import INSTRUCTIONS, WINDOW_NAME
from .metadata import (
    DATASET_ROLE,
    CameraFacts,
    CaptureSessionRecord,
    read_camera_controls,
    read_camera_facts,
    write_metadata,
)
from .session_layout import (
    SessionPaths,
    derive_session_prefix,
    plan_next_session,
)

__all__ = [
    "DATASET_ROLE",
    "CaptureEngine",
    "CaptureError",
    "CaptureIO",
    "CaptureSessionRecord",
    "CameraFacts",
    "INSTRUCTIONS",
    "KeyCommand",
    "SessionPaths",
    "WINDOW_NAME",
    "derive_session_prefix",
    "plan_next_session",
    "read_camera_controls",
    "read_camera_facts",
    "write_metadata",
]
