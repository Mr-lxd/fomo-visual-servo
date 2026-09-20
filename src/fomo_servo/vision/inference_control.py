"""Service-level manual lifecycle coordination for Vision inference."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import threading
from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from .inference_worker import InferenceWorker


@dataclass(frozen=True)
class InferenceActionResult:
    """HTTP-facing outcome of an admitted inference action."""

    http_status: int
    action: str
    outcome: str


class InferenceControlError(Exception):
    """Stable API error raised when an inference action cannot be admitted."""

    def __init__(self, code: str, http_status: int, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.http_status = http_status
        self.message = message


_KNOWN_STATES = frozenset(("disabled", "starting", "running", "failed"))
LOGGER = logging.getLogger(__name__)


def _disabled_status() -> dict[str, Any]:
    return {
        "state": "disabled",
        "artifact_name": None,
        "model_sha256": None,
        "confidence_threshold": None,
        "latest_frame_id": None,
        "capture_timestamp_ns": None,
        "processed_frames": 0,
        "skipped_frames": 0,
        "inference_fps": None,
        "latency_ms": None,
        "detection_count": None,
        "last_error": None,
    }


class InferenceControl:
    """Coordinate manual inference actions without owning camera or HTTP I/O."""

    def __init__(
        self,
        worker: InferenceWorker | None,
        *,
        is_live: Callable[[], bool],
        camera_running: Callable[[], bool],
    ) -> None:
        self._worker = worker
        self._is_live = is_live
        self._camera_running = camera_running
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._actions_open = False
        self._shutdown = False
        self._operation: str | None = None
        self._transient_error: str | None = None
        self._lifecycle_error: str | None = None
        self._lifecycle_failure_latched = False
        self._coordinator_thread: threading.Thread | None = None
        self._coordinator_busy = False
        self._coordinator_exit = False

    def open_actions(self) -> None:
        """Allow actions once ``VisionService`` has reached LIVE mode."""

        with self._condition:
            if self._shutdown:
                return
            self._actions_open = True
            self._condition.notify_all()

    def start_inference(self) -> InferenceActionResult:
        """Admit a manual Start or failed-generation Retry when permitted."""

        try:
            with self._condition:
                self._raise_if_shutdown_locked()
                self._raise_if_lifecycle_failed_locked()
                if self._operation == "stopping":
                    self._raise("inference_busy", 409, "inference is stopping")
                if self._operation == "retrying":
                    return InferenceActionResult(
                        200,
                        "inference/start",
                        "already_retrying",
                    )
                if self._worker is None:
                    self._raise(
                        "inference_not_configured",
                        409,
                        "inference is not configured",
                    )

                state = self._worker_state_locked()
                if state == "starting":
                    return InferenceActionResult(
                        200,
                        "inference/start",
                        "already_starting",
                    )
                if state == "running":
                    return InferenceActionResult(
                        200,
                        "inference/start",
                        "already_running",
                    )
                self._require_start_prerequisites_locked()
                self._transient_error = None

                if state == "disabled":
                    self._worker.start()
                    return InferenceActionResult(
                        202,
                        "inference/start",
                        "accepted",
                    )

                if state == "failed":
                    self._start_coordinator_locked()
                    self._operation = "retrying"
                    try:
                        self._worker.request_stop()
                    except BaseException:
                        self._operation = None
                        self._condition.notify_all()
                        raise
                    self._condition.notify_all()
                    return InferenceActionResult(
                        202,
                        "inference/start",
                        "accepted",
                    )

                self._raise(
                    "invalid_inference_state",
                    409,
                    "inference worker state is invalid",
                )
        except InferenceControlError:
            raise
        except BaseException as error:
            raise InferenceControlError(
                "internal_error",
                500,
                "inference control failed: {}".format(
                    str(error) or type(error).__name__
                ),
            ) from error

    def stop_inference(self) -> InferenceActionResult:
        """Admit a non-blocking manual Stop or Clear Error action."""

        try:
            with self._condition:
                self._raise_if_shutdown_locked()
                self._raise_if_lifecycle_failed_locked()
                if self._operation == "stopping":
                    return InferenceActionResult(
                        200,
                        "inference/stop",
                        "already_stopping",
                    )
                if self._operation == "retrying":
                    self._raise("inference_busy", 409, "inference is retrying")
                if self._worker is None:
                    return InferenceActionResult(
                        200,
                        "inference/stop",
                        "already_disabled",
                    )

                state = self._worker_state_locked()
                if state == "disabled":
                    return InferenceActionResult(
                        200,
                        "inference/stop",
                        "already_disabled",
                    )
                if state not in ("starting", "running", "failed"):
                    self._raise(
                        "invalid_inference_state",
                        409,
                        "inference worker state is invalid",
                    )

                self._start_coordinator_locked()
                self._operation = "stopping"
                try:
                    self._worker.request_stop()
                except BaseException:
                    self._operation = None
                    self._condition.notify_all()
                    raise
                self._transient_error = None
                self._condition.notify_all()
                return InferenceActionResult(
                    202,
                    "inference/stop",
                    "accepted",
                )
        except InferenceControlError:
            raise
        except BaseException as error:
            raise InferenceControlError(
                "internal_error",
                500,
                "inference control failed: {}".format(
                    str(error) or type(error).__name__
                ),
            ) from error

    def status(self) -> dict[str, Any]:
        """Return the authoritative worker snapshot plus control metadata."""

        with self._condition:
            snapshot = _disabled_status()
            if self._worker is not None:
                worker_status = self._worker.status()
                snapshot.update(worker_status)
            snapshot["configured"] = self._worker is not None
            snapshot["control_supported"] = True
            snapshot["operation"] = self._operation
            if self._lifecycle_error is not None:
                snapshot["last_error"] = self._lifecycle_error
            elif self._transient_error is not None:
                snapshot["last_error"] = self._transient_error
            return snapshot

    def begin_shutdown(self) -> None:
        """Close admission and signal an existing worker without joining it."""

        with self._condition:
            self._actions_open = False
            self._shutdown = True
            if self._worker is not None:
                self._worker.request_stop()
            self._condition.notify_all()

    def finish_shutdown(self) -> None:
        """Join coordination, then reclaim the worker after HTTP is stopped."""

        first_error: BaseException | None = None
        try:
            self.begin_shutdown()
        except BaseException as error:
            first_error = error

        with self._condition:
            self._coordinator_exit = True
            self._condition.notify_all()
            coordinator = self._coordinator_thread
        if coordinator is not None and coordinator is not threading.current_thread():
            try:
                coordinator.join()
            except BaseException as error:
                if first_error is None:
                    first_error = error

        worker = self._worker
        if worker is not None:
            try:
                worker.stop()
            except BaseException as error:
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error

    def _worker_state_locked(self) -> str:
        if self._worker is None:
            return "disabled"
        status = self._worker.status()
        state = status.get("state") if isinstance(status, dict) else None
        if state not in _KNOWN_STATES:
            self._raise(
                "invalid_inference_state",
                409,
                "inference worker state is invalid",
            )
        return state

    def _require_start_prerequisites_locked(self) -> None:
        if not self._actions_open or not self._is_live():
            self._raise("vision_not_live", 409, "Vision service is not LIVE")
        if not self._camera_running():
            self._raise(
                "camera_not_running",
                409,
                "camera is not running",
            )

    def _raise_if_shutdown_locked(self) -> None:
        if self._shutdown:
            self._raise(
                "service_shutting_down",
                503,
                "Vision service is shutting down",
            )

    def _raise_if_lifecycle_failed_locked(self) -> None:
        if self._lifecycle_failure_latched:
            self._raise(
                "inference_control_failed",
                409,
                "inference lifecycle cleanup failed; restart the Vision process",
            )

    @staticmethod
    def _raise(code: str, http_status: int, message: str) -> None:
        raise InferenceControlError(code, http_status, message)

    def _start_coordinator_locked(self) -> None:
        thread = self._coordinator_thread
        if thread is not None:
            if thread.is_alive():
                return
            raise RuntimeError("inference lifecycle coordinator is not running")

        thread = threading.Thread(
            target=self._coordinator_loop,
            name="vision-inference-lifecycle",
            daemon=False,
        )
        self._coordinator_thread = thread
        try:
            thread.start()
        except BaseException:
            self._coordinator_thread = None
            raise

    def _coordinator_loop(self) -> None:
        while True:
            with self._condition:
                self._condition.wait_for(
                    lambda: self._coordinator_exit
                    or (
                        self._operation is not None
                        and not self._coordinator_busy
                    )
                )
                if self._coordinator_exit:
                    return
                operation = self._operation
                if operation not in ("stopping", "retrying"):
                    self._latch_lifecycle_failure_locked(
                        RuntimeError("invalid inference lifecycle operation")
                    )
                    return
                self._coordinator_busy = True

            try:
                if operation == "stopping":
                    self._complete_stop()
                else:
                    self._complete_retry()
            except BaseException as error:
                with self._condition:
                    self._latch_lifecycle_failure_locked(error)
            finally:
                with self._condition:
                    self._coordinator_busy = False
                    self._condition.notify_all()
                    if self._lifecycle_failure_latched:
                        return

    def _complete_stop(self) -> None:
        worker = self._worker
        if worker is None:
            return
        worker.stop()
        worker.reset_disabled()
        with self._condition:
            if self._operation == "stopping":
                self._operation = None
                self._transient_error = None
            self._condition.notify_all()

    def _complete_retry(self) -> None:
        worker = self._worker
        if worker is None:
            return
        worker.stop()
        with self._condition:
            if self._shutdown:
                if self._operation == "retrying":
                    self._operation = None
                self._condition.notify_all()
                return
            if not self._actions_open or not self._is_live():
                self._transient_error = (
                    "retry was not started because Vision service is not LIVE"
                )
                if self._operation == "retrying":
                    self._operation = None
                self._condition.notify_all()
                return
            if not self._camera_running():
                self._transient_error = (
                    "retry was not started because camera is not running"
                )
                if self._operation == "retrying":
                    self._operation = None
                self._condition.notify_all()
                return
            try:
                worker.start()
            except BaseException:
                if self._operation == "retrying":
                    self._operation = None
                self._condition.notify_all()
                return
            if self._operation == "retrying":
                self._operation = None
                self._transient_error = None
            self._condition.notify_all()

    def _latch_lifecycle_failure_locked(self, error: BaseException) -> None:
        message = "inference lifecycle cleanup failed: {}".format(
            str(error) or type(error).__name__
        )
        self._lifecycle_failure_latched = True
        self._lifecycle_error = message
        LOGGER.error("%s", message)
        self._coordinator_exit = True
        self._condition.notify_all()
