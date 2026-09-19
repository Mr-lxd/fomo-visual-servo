"""Public contract for the bounded vision inference worker."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from pathlib import Path
import threading
from typing import Any, Callable, Optional

from fomo_servo.inference.ort_predictor import OnnxRuntimePredictor
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


_EXPECTED_CONTRACT = {
    "artifact_name": "d2_mobilenet_v2_fomo_seed42_epoch40",
    "checkpoint_seed": 42,
    "checkpoint_epoch": 40,
    "confidence_threshold": 0.40,
    "onnx_sha256": "3dea74511bf2c44844192e75594fd53d4c4ce941f8b53b15767e020832bf9b08",
    "input_shape": (1, 3, 192, 192),
    "input_dtype": "float32",
    "input_color_order": "RGB",
    "input_value_range": (0.0, 1.0),
    "output_shape": (1, 8, 24, 24),
    "output_dtype": "float32",
    "output_semantic": "raw_logits",
    "opset": 17,
}


class InferenceWorker:
    """Initialize and validate the fixed inference model in a worker thread."""

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
        self._predictor_factory = (
            OnnxRuntimePredictor.from_files
            if predictor_factory is None
            else predictor_factory
        )
        self._clock_ns = clock_ns
        self._wait_timeout = wait_timeout
        self._fps_window_size = fps_window_size
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._state = InferenceState.DISABLED
        self._identity: Optional[dict[str, Any]] = None
        self._last_error: Optional[str] = None
        self._after_frame_id: Optional[int] = None
        self._latest_result: Optional[InferenceResult] = None

    def start(self) -> None:
        """Start model initialization once, rejecting another live start."""

        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("inference worker is already running")
            self._state = InferenceState.STARTING
            self._identity = None
            self._last_error = None
            self._after_frame_id = None
            self._stop_event = threading.Event()
            self._thread = threading.Thread(
                target=self._run,
                name="vision-inference-worker",
                daemon=True,
            )
            self._thread.start()

    def stop(self) -> None:
        """Request shutdown and join the current worker thread."""

        with self._lock:
            thread = self._thread
            stop_event = self._stop_event
        stop_event.set()
        if thread is not None and thread is not threading.current_thread():
            thread.join()
        with self._lock:
            if (
                thread is not None
                and self._thread is thread
                and self._state != InferenceState.FAILED
            ):
                self._state = InferenceState.DISABLED
                self._identity = None
                self._last_error = None
                self._after_frame_id = None

    def status(self) -> dict[str, Any]:
        """Return the lifecycle snapshot and validated model identity."""

        with self._lock:
            identity = self._identity
            return {
                "state": self._state.value,
                "artifact_name": None if identity is None else identity["artifact_name"],
                "model_sha256": None if identity is None else identity["onnx_sha256"],
                "confidence_threshold": (
                    None if identity is None else identity["confidence_threshold"]
                ),
                "latest_frame_id": None,
                "capture_timestamp_ns": None,
                "inference_fps": None,
                "latency_ms": None,
                "detection_count": None,
                "last_error": self._last_error,
                "processed_frames": 0,
                "skipped_frames": 0,
            }

    def latest_result(self) -> Optional[InferenceResult]:
        """Return the latest completed result, if one exists."""

        return self._latest_result

    def _run(self) -> None:
        try:
            predictor = self._predictor_factory(self._onnx_path, self._report_path)
            contract = getattr(predictor, "contract", None)
            self._validate_contract(contract)
            cached = self._hub.snapshot()
            after_frame_id = None if cached is None else cached.frame_id
        except Exception as error:
            with self._lock:
                self._state = InferenceState.FAILED
                self._identity = None
                self._after_frame_id = None
                self._last_error = str(error) or type(error).__name__
            return

        with self._lock:
            if self._stop_event.is_set():
                self._state = InferenceState.DISABLED
                self._identity = None
                self._last_error = None
                self._after_frame_id = None
                return
            self._identity = {
                "artifact_name": contract.artifact_name,
                "onnx_sha256": contract.onnx_sha256,
                "confidence_threshold": contract.confidence_threshold,
            }
            self._after_frame_id = after_frame_id
            self._state = InferenceState.RUNNING

        self._stop_event.wait()

    @staticmethod
    def _validate_contract(contract: Any) -> None:
        if contract is None:
            raise ValueError("contract mismatch: contract is missing")
        for field, expected in _EXPECTED_CONTRACT.items():
            try:
                actual = getattr(contract, field)
            except AttributeError as error:
                raise ValueError(f"contract mismatch: {field} is missing") from error
            if field == "confidence_threshold":
                matches = (
                    isinstance(actual, (int, float))
                    and not isinstance(actual, bool)
                    and math.isclose(
                        actual, expected, rel_tol=0.0, abs_tol=1e-9
                    )
                )
            else:
                matches = InferenceWorker._strict_equal(actual, expected)
            if not matches:
                raise ValueError(
                    f"contract mismatch: {field} expected {expected!r}, got {actual!r}"
                )

    @staticmethod
    def _strict_equal(actual: Any, expected: Any) -> bool:
        if type(actual) is not type(expected):
            return False
        if isinstance(expected, tuple):
            return len(actual) == len(expected) and all(
                InferenceWorker._strict_equal(item, expected_item)
                for item, expected_item in zip(actual, expected)
            )
        return actual == expected
