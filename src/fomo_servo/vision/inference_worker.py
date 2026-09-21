"""Public contract for the bounded vision inference worker."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
import math
from pathlib import Path
import threading
import time
from typing import Any, Callable, Optional

import cv2

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
class ModelIdentity:
    """Immutable identity of the validated model used for inference."""

    artifact_name: str
    onnx_sha256: str
    confidence_threshold: float


@dataclass(frozen=True)
class InferenceResult:
    """One completed inference result associated with one captured frame."""

    frame_id: int
    capture_timestamp_ns: int
    inference_started_ns: int
    inference_finished_ns: int
    latency_ms: float
    frame_width: int
    frame_height: int
    model_identity: ModelIdentity
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
        result_sink: Optional[Callable[[InferenceResult], None]] = None,
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
        self._clock_ns = time.monotonic_ns if clock_ns is None else clock_ns
        self._result_sink = result_sink
        self._wait_timeout = wait_timeout
        self._fps_window_size = fps_window_size
        self._lock = threading.RLock()
        self._state_condition = threading.Condition(self._lock)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._generation = 0
        self._state = InferenceState.DISABLED
        self._identity: Optional[ModelIdentity] = None
        self._last_error: Optional[str] = None
        self._after_frame_id: Optional[int] = None
        self._latest_result: Optional[InferenceResult] = None
        self._processed_frames = 0
        self._skipped_frames = 0
        self._completion_timestamps_ns: deque[int] = deque(maxlen=fps_window_size)

    def start(self) -> None:
        """Start model initialization once, rejecting another live start."""

        with self._lock:
            if self._thread is not None:
                if self._state == InferenceState.FAILED:
                    raise RuntimeError(
                        "failed generation must be stopped before restart"
                    )
                if self._thread.is_alive():
                    raise RuntimeError("inference worker is already running")
            self._generation += 1
            generation = self._generation
            self._set_state_locked(InferenceState.STARTING)
            self._identity = None
            self._last_error = None
            self._after_frame_id = None
            self._latest_result = None
            self._processed_frames = 0
            self._skipped_frames = 0
            self._completion_timestamps_ns.clear()
            self._stop_event = threading.Event()
            self._thread = threading.Thread(
                target=self._run,
                args=(generation,),
                name="vision-inference-worker",
                daemon=True,
            )
            thread = self._thread
            try:
                thread.start()
            except BaseException as error:
                if self._thread is thread:
                    self._thread = None
                    self._set_state_locked(InferenceState.FAILED)
                    self._identity = None
                    self._after_frame_id = None
                    self._last_error = str(error) or type(error).__name__
                raise

    def request_stop(self) -> None:
        """Request a worker stop without joining or clearing its generation."""

        with self._lock:
            self._stop_event.set()

    def stop(self) -> None:
        """Request shutdown and join the current worker thread."""

        self.request_stop()
        with self._lock:
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join()
        with self._lock:
            if (
                thread is not None
                and self._thread is thread
            ):
                self._thread = None
                if self._state != InferenceState.FAILED:
                    self._set_state_locked(InferenceState.DISABLED)
                    self._identity = None
                    self._last_error = None
                    self._after_frame_id = None

    def reset_disabled(self) -> None:
        """Clear completed-generation diagnostics after ``stop()`` released it."""

        with self._lock:
            if self._thread is not None:
                raise RuntimeError(
                    "reset_disabled requires stop() has released worker thread handle"
                )
            self._identity = None
            self._last_error = None
            self._after_frame_id = None
            self._latest_result = None
            self._processed_frames = 0
            self._skipped_frames = 0
            self._completion_timestamps_ns.clear()
            self._set_state_locked(InferenceState.DISABLED)

    def status(self) -> dict[str, Any]:
        """Return the lifecycle snapshot and validated model identity."""

        with self._lock:
            identity = self._identity
            result = self._latest_result
            inference_fps = None
            if len(self._completion_timestamps_ns) >= 2:
                elapsed_ns = (
                    self._completion_timestamps_ns[-1]
                    - self._completion_timestamps_ns[0]
                )
                if elapsed_ns > 0:
                    inference_fps = (len(self._completion_timestamps_ns) - 1) / (
                        elapsed_ns / 1_000_000_000.0
                    )
            return {
                "state": self._state.value,
                "artifact_name": None if identity is None else identity.artifact_name,
                "model_sha256": None if identity is None else identity.onnx_sha256,
                "confidence_threshold": (
                    None if identity is None else identity.confidence_threshold
                ),
                "latest_frame_id": None if result is None else result.frame_id,
                "capture_timestamp_ns": (
                    None if result is None else result.capture_timestamp_ns
                ),
                "inference_fps": inference_fps,
                "latency_ms": None if result is None else result.latency_ms,
                "detection_count": None if result is None else len(result.detections),
                "last_error": self._last_error,
                "processed_frames": self._processed_frames,
                "skipped_frames": self._skipped_frames,
            }

    def latest_result(self) -> Optional[InferenceResult]:
        """Return the latest completed result, if one exists."""

        with self._lock:
            return self._latest_result

    def _set_state_locked(self, state: InferenceState) -> None:
        self._state = state
        self._state_condition.notify_all()

    def _run(self, generation: int) -> None:
        try:
            predictor = self._predictor_factory(self._onnx_path, self._report_path)
            contract = getattr(predictor, "contract", None)
            self._validate_contract(contract)
            cached = self._hub.snapshot()
            after_frame_id = None if cached is None else cached.frame_id
        except Exception as error:
            with self._lock:
                if generation != self._generation:
                    return
                self._set_state_locked(InferenceState.FAILED)
                if generation != self._generation:
                    return
                self._identity = None
                self._after_frame_id = None
                self._last_error = str(error) or type(error).__name__
            return

        model_identity = ModelIdentity(
            artifact_name=contract.artifact_name,
            onnx_sha256=contract.onnx_sha256,
            confidence_threshold=contract.confidence_threshold,
        )
        with self._lock:
            if generation != self._generation:
                return
            if self._stop_event.is_set():
                self._set_state_locked(InferenceState.DISABLED)
                self._identity = None
                self._last_error = None
                self._after_frame_id = None
                return
            self._identity = model_identity
            self._after_frame_id = after_frame_id
            self._set_state_locked(InferenceState.RUNNING)

        previous_successful_frame_id: Optional[int] = None
        while not self._stop_event.is_set():
            try:
                frame = self._hub.wait_for_newer(
                    after_frame_id, timeout=self._wait_timeout
                )
                if frame is None:
                    continue
                started = self._clock_ns()
                rgb = cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB)
                prediction = predictor.predict_rgb_image(rgb)
                finished = self._clock_ns()
                result = InferenceResult(
                    frame_id=frame.frame_id,
                    capture_timestamp_ns=frame.capture_timestamp_ns,
                    inference_started_ns=started,
                    inference_finished_ns=finished,
                    latency_ms=(finished - frame.capture_timestamp_ns) / 1_000_000.0,
                    frame_width=frame.width,
                    frame_height=frame.height,
                    model_identity=model_identity,
                    detections=tuple(prediction.detections),
                )
                with self._lock:
                    if (
                        generation != self._generation
                        or self._stop_event.is_set()
                    ):
                        return
                    skipped_frames = max(
                        0,
                        frame.frame_id - previous_successful_frame_id - 1
                        if previous_successful_frame_id is not None
                        else 0,
                    )
                    self._after_frame_id = frame.frame_id
                    self._latest_result = result
                    self._processed_frames += 1
                    self._skipped_frames += skipped_frames
                    self._completion_timestamps_ns.append(finished)
                    self._state_condition.notify_all()
                if self._result_sink is not None:
                    self._result_sink(result)
                previous_successful_frame_id = frame.frame_id
                after_frame_id = frame.frame_id
            except Exception as error:
                with self._lock:
                    if generation != self._generation:
                        return
                    self._set_state_locked(InferenceState.FAILED)
                    if generation != self._generation:
                        return
                    self._last_error = str(error) or type(error).__name__
                return

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
