"""Thread-safe latest-result handoff for frame-synchronised inference metadata."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .inference_worker import InferenceResult


class InferenceResultHub:
    """Publish monotonic inference results to realtime metadata consumers."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._latest: Optional["InferenceResult"] = None

    def publish(self, result: "InferenceResult") -> None:
        """Publish one completed result, requiring strictly increasing frame IDs."""

        with self._condition:
            latest = self._latest
            if latest is not None and result.frame_id <= latest.frame_id:
                raise ValueError("inference result frame_id must increase")
            self._latest = result
            self._condition.notify_all()

    def snapshot(self) -> Optional["InferenceResult"]:
        """Return the latest immutable result without blocking."""

        with self._condition:
            return self._latest

    def wait_for_newer(
        self,
        after_frame_id: Optional[int],
        *,
        timeout: float,
    ) -> Optional["InferenceResult"]:
        """Wait for a result newer than after_frame_id or return None."""

        if timeout < 0:
            raise ValueError("timeout must be non-negative")

        with self._condition:
            def is_newer() -> bool:
                latest = self._latest
                if latest is None:
                    return False
                return after_frame_id is None or latest.frame_id > after_frame_id

            if not is_newer():
                self._condition.wait_for(is_newer, timeout=timeout)
            latest = self._latest
            if latest is None:
                return None
            if after_frame_id is not None and latest.frame_id <= after_frame_id:
                return None
            return latest
