"""Public contract for the bounded vision inference worker."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

from fomo_servo.postprocess import Detection

from .frame_hub import FrameHub


class InferenceState(str, Enum):
    """Lifecycle states exposed by :class:`InferenceWorker`."""

    DISABLED = "disabled"
    STARTING = "starting"
    RUNNING = "running"
    FAILED = "failed"


@dataclass(frozen=True)
class InferenceResult:
    """One completed inference result associated with one captured frame."""

    frame_id: int
    capture_timestamp_ns: int
    inference_started_ns: int
    inference_finished_ns: int
    detections: tuple[Detection, ...]


class InferenceWorker:
    """Expose the inference-worker contract before runtime behavior is added."""

    def __init__(
        self,
        hub: FrameHub,
        onnx_path: str | Path,
        report_path: str | Path,
        *,
        predictor_factory: Optional[Callable[..., Any]] = None,
        clock_ns: Optional[Callable[[], int]] = None,
        wait_timeout: float = 0.5,
        fps_window_size: int = 30,
    ) -> None:
        self._hub = hub
        self._onnx_path = Path(onnx_path)
        self._report_path = Path(report_path)
        self._predictor_factory = predictor_factory
        self._clock_ns = clock_ns
        self._wait_timeout = wait_timeout
        self._fps_window_size = fps_window_size
        self._latest_result: Optional[InferenceResult] = None

    def start(self) -> None:
        """Start the worker; runtime processing is implemented in a later task."""

    def stop(self) -> None:
        """Stop the worker; runtime processing is implemented in a later task."""

    def status(self) -> dict[str, Any]:
        """Return the exact disabled-state status snapshot."""

        return {
            "state": InferenceState.DISABLED.value,
            "artifact_name": None,
            "model_sha256": None,
            "confidence_threshold": None,
            "latest_frame_id": None,
            "capture_timestamp_ns": None,
            "inference_fps": None,
            "latency_ms": None,
            "detection_count": None,
            "last_error": None,
            "processed_frames": 0,
            "skipped_frames": 0,
        }

    def latest_result(self) -> Optional[InferenceResult]:
        """Return the latest completed result, if one exists."""

        return self._latest_result
