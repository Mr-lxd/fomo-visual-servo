from __future__ import annotations

import threading
import time

import pytest

from fomo_servo.vision.inference_result_hub import InferenceResultHub
from fomo_servo.vision.inference_worker import InferenceResult, ModelIdentity


IDENTITY = ModelIdentity(
    artifact_name="d2_mobilenet_v2_fomo_seed42_epoch40",
    onnx_sha256="3dea74511bf2c44844192e75594fd53d4c4ce941f8b53b15767e020832bf9b08",
    confidence_threshold=0.40,
)


def _result(frame_id: int) -> InferenceResult:
    return InferenceResult(
        frame_id=frame_id,
        capture_timestamp_ns=1_000 + frame_id,
        inference_started_ns=2_000 + frame_id,
        inference_finished_ns=3_000 + frame_id,
        latency_ms=0.002,
        frame_width=640,
        frame_height=480,
        model_identity=IDENTITY,
        detections=(),
    )


def test_hub_starts_empty_and_times_out_without_a_result() -> None:
    hub = InferenceResultHub()

    assert hub.snapshot() is None
    assert hub.wait_for_newer(None, timeout=0.0) is None
    assert hub.wait_for_newer(5, timeout=0.01) is None


def test_publish_updates_snapshot_and_waiter_observes_newest_result() -> None:
    hub = InferenceResultHub()
    observed: list[InferenceResult | None] = []

    def waiter() -> None:
        observed.append(hub.wait_for_newer(None, timeout=1.0))

    thread = threading.Thread(target=waiter)
    thread.start()
    time.sleep(0.01)

    result = _result(7)
    hub.publish(result)
    thread.join(timeout=1.0)

    assert not thread.is_alive()
    assert observed == [result]
    assert hub.snapshot() is result
    assert hub.wait_for_newer(6, timeout=0.0) is result
    assert hub.wait_for_newer(7, timeout=0.0) is None


def test_publish_requires_strictly_increasing_frame_ids() -> None:
    hub = InferenceResultHub()
    first = _result(10)
    hub.publish(first)

    with pytest.raises(ValueError, match="frame_id must increase"):
        hub.publish(_result(10))
    with pytest.raises(ValueError, match="frame_id must increase"):
        hub.publish(_result(9))

    assert hub.snapshot() is first


def test_negative_wait_timeout_is_rejected_without_mutating_state() -> None:
    hub = InferenceResultHub()

    with pytest.raises(ValueError, match="timeout must be non-negative"):
        hub.wait_for_newer(None, timeout=-0.1)

    assert hub.snapshot() is None
