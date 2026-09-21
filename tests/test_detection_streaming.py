from __future__ import annotations

import json
import socket
import threading
import time
from dataclasses import replace

import pytest

from fomo_servo.postprocess import Detection
from fomo_servo.vision.detection_streaming import (
    DETECTION_COORDINATE_SPACE,
    DETECTION_STREAM_VERSION,
    DetectionStreamConsumer,
    DetectionStreamError,
    DetectionTcpServer,
    encode_inference_result,
)
from fomo_servo.vision.inference_result_hub import InferenceResultHub
from fomo_servo.vision.inference_worker import InferenceResult, ModelIdentity


IDENTITY = ModelIdentity(
    artifact_name="d2_mobilenet_v2_fomo_seed42_epoch40",
    onnx_sha256="3dea74511bf2c44844192e75594fd53d4c4ce941f8b53b15767e020832bf9b08",
    confidence_threshold=0.40,
)


def _detection(**overrides) -> Detection:
    values = {
        "class_id": 0,
        "class_name": "creature",
        "confidence": 0.875,
        "mean_confidence": 0.80,
        "component_area_cells": 3,
        "heatmap_x": 12.0,
        "heatmap_y": 8.0,
        "input_x": 96.0,
        "input_y": 64.0,
        "original_x": 320.5,
        "original_y": 207.25,
    }
    values.update(overrides)
    return Detection(**values)


def _result(frame_id: int, detections=None) -> InferenceResult:
    if detections is None:
        detections = (_detection(),)
    return InferenceResult(
        frame_id=frame_id,
        capture_timestamp_ns=1_000_000 + frame_id,
        inference_started_ns=2_000_000 + frame_id,
        inference_finished_ns=3_000_000 + frame_id,
        latency_ms=2.0,
        frame_width=640,
        frame_height=480,
        model_identity=IDENTITY,
        detections=tuple(detections),
    )


def _wait_until(predicate, timeout: float = 1.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.002)
    return bool(predicate())


def _recv_line(sock: socket.socket) -> bytes:
    chunks: list[bytes] = []
    while True:
        chunk = sock.recv(1)
        if not chunk:
            raise RuntimeError("socket closed before newline")
        chunks.append(chunk)
        if chunk == b"\n":
            return b"".join(chunks)


def test_encoder_emits_minimal_original_frame_centroid_contract() -> None:
    encoded = encode_inference_result(_result(42))
    assert encoded.endswith(b"\n")

    payload = json.loads(encoded)
    assert payload == {
        "type": "detections",
        "version": DETECTION_STREAM_VERSION,
        "frame_id": 42,
        "capture_timestamp_ns": 1_000_042,
        "width": 640,
        "height": 480,
        "coordinate_space": DETECTION_COORDINATE_SPACE,
        "detections": [
            {
                "class_id": 0,
                "class_name": "creature",
                "confidence": 0.875,
                "original_x": 320.5,
                "original_y": 207.25,
            }
        ],
    }
    assert "bbox" not in encoded.decode("utf-8")
    assert "heatmap" not in encoded.decode("utf-8")
    assert "component_area" not in encoded.decode("utf-8")


def test_encoder_emits_empty_detection_list_to_clear_old_overlay() -> None:
    payload = json.loads(encode_inference_result(_result(9, detections=())))

    assert payload["frame_id"] == 9
    assert payload["detections"] == []


def test_encoder_rejects_unbounded_detection_count() -> None:
    with pytest.raises(DetectionStreamError, match="too many detections"):
        encode_inference_result(
            _result(9, detections=tuple(_detection() for _ in range(257)))
        )


@pytest.mark.parametrize(
    "result, message",
    [
        (replace(_result(1), frame_width=0), "frame_width"),
        (replace(_result(1), frame_height=0), "frame_height"),
        (_result(1, [_detection(class_id=-1)]), "class_id"),
        (_result(1, [_detection(confidence=float("nan"))]), "confidence"),
        (_result(1, [_detection(confidence=1.1)]), "confidence"),
        (_result(1, [_detection(original_x=-0.1)]), "original_x"),
        (_result(1, [_detection(original_x=640.0)]), "original_x"),
        (_result(1, [_detection(original_y=480.0)]), "original_y"),
        (_result(1, [_detection(class_name="")]), "class_name"),
        (_result(1, [_detection(class_name="x" * 129)]), "class_name"),
    ],
)
def test_encoder_rejects_invalid_wire_values(
    result: InferenceResult, message: str
) -> None:
    with pytest.raises(DetectionStreamError, match=message):
        encode_inference_result(result)


class _SlowSocket:
    def __init__(self) -> None:
        self.payloads: list[bytes] = []
        self.first_send_entered = threading.Event()
        self.release_first_send = threading.Event()

    def sendall(self, data: bytes) -> None:
        if not self.payloads:
            self.first_send_entered.set()
            assert self.release_first_send.wait(timeout=1.0)
        self.payloads.append(bytes(data))


def test_slow_consumer_skips_intermediate_results_instead_of_queueing() -> None:
    hub = InferenceResultHub()
    hub.publish(_result(1))
    fake_socket = _SlowSocket()
    stop = threading.Event()
    consumer = DetectionStreamConsumer(hub, fake_socket)
    thread = threading.Thread(
        target=consumer.run,
        kwargs={
            "stop_event": stop,
            "accept_frame_id_floor": None,
            "wait_timeout": 0.01,
        },
        daemon=True,
    )
    thread.start()
    assert fake_socket.first_send_entered.wait(timeout=1.0)

    for frame_id in range(2, 101):
        hub.publish(_result(frame_id))
    fake_socket.release_first_send.set()

    assert _wait_until(lambda: len(fake_socket.payloads) >= 2)
    stop.set()
    thread.join(timeout=1.0)

    assert not thread.is_alive()
    first = json.loads(fake_socket.payloads[0])
    second = json.loads(fake_socket.payloads[1])
    assert first["frame_id"] == 1
    assert second["frame_id"] == 100
    assert consumer.metrics.sent_results >= 2
    assert consumer.metrics.last_sent_frame_id == 100
    assert consumer.metrics.skipped_result_ids >= 98


class _FailingSocket:
    def sendall(self, _data: bytes) -> None:
        raise TimeoutError("slow metadata client")


def test_send_timeout_terminates_consumer_without_retry_queue() -> None:
    hub = InferenceResultHub()
    hub.publish(_result(1))
    consumer = DetectionStreamConsumer(hub, _FailingSocket())

    metrics = consumer.run(
        stop_event=threading.Event(),
        accept_frame_id_floor=None,
        wait_timeout=0.01,
    )

    assert metrics.send_errors == 1
    assert metrics.sent_results == 0
    assert "slow metadata client" in (metrics.last_error or "")


def test_tcp_server_never_replays_cached_preconnection_result() -> None:
    hub = InferenceResultHub()
    hub.publish(_result(5))
    server = DetectionTcpServer(
        hub,
        bind_host="127.0.0.1",
        port=0,
        write_timeout=0.2,
    ).start()
    assert server.bound_port is not None

    first = socket.create_connection(
        ("127.0.0.1", server.bound_port), timeout=1.0
    )
    try:
        assert server.client_connected.wait(timeout=1.0)
        first.settimeout(0.05)
        with pytest.raises(socket.timeout):
            first.recv(1)

        hub.publish(_result(9))
        first.settimeout(1.0)
        assert json.loads(_recv_line(first))["frame_id"] == 9
    finally:
        first.close()

    assert _wait_until(lambda: not server.client_connected.is_set())
    hub.publish(_result(10))

    second = socket.create_connection(
        ("127.0.0.1", server.bound_port), timeout=1.0
    )
    try:
        assert server.client_connected.wait(timeout=1.0)
        second.settimeout(0.05)
        with pytest.raises(socket.timeout):
            second.recv(1)

        hub.publish(_result(11))
        second.settimeout(1.0)
        assert json.loads(_recv_line(second))["frame_id"] == 11
    finally:
        second.close()
        server.stop()

    assert server.last_error is None
    assert server.bound_port is None


def test_tcp_server_rejects_unexpected_client_to_server_bytes() -> None:
    hub = InferenceResultHub()
    server = DetectionTcpServer(
        hub,
        bind_host="127.0.0.1",
        port=0,
        write_timeout=0.2,
    ).start()
    assert server.bound_port is not None

    client = socket.create_connection(
        ("127.0.0.1", server.bound_port), timeout=1.0
    )
    try:
        assert server.client_connected.wait(timeout=1.0)
        client.sendall(b"x")
        assert _wait_until(lambda: not server.client_connected.is_set())
    finally:
        client.close()
        server.stop()

    assert server.last_error is None


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"port": -1}, "port"),
        ({"port": 65536}, "port"),
        ({"send_buffer_bytes": 0}, "send_buffer"),
        ({"write_timeout": 0.0}, "write_timeout"),
    ],
)
def test_tcp_server_rejects_invalid_configuration_before_listening(
    kwargs: dict, message: str
) -> None:
    server = DetectionTcpServer(
        InferenceResultHub(),
        bind_host="127.0.0.1",
        **kwargs,
    )

    with pytest.raises(ValueError, match=message):
        server.start()

    assert server.bound_port is None
