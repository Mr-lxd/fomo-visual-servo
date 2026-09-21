from __future__ import annotations

import socket
import threading
import time
from dataclasses import dataclass

import numpy as np

from fomo_servo.vision.frame import PixelFormat, VisionFrame
from fomo_servo.vision.frame_hub import FrameHub
from fomo_servo.vision.protocol import MAX_PAYLOAD_SIZE, decode_header
from fomo_servo.vision.streaming import LiveStreamConsumer, VisionTcpServer


@dataclass
class _Image:
    shape: tuple[int, int, int] = (2, 2, 3)


def _frame(frame_id: int) -> VisionFrame:
    return VisionFrame(
        frame_id=frame_id,
        capture_timestamp_ns=frame_id + 1,
        width=2,
        height=2,
        pixel_format=PixelFormat.BGR8,
        image=_Image(),
    )


def _wait_until(predicate, timeout: float = 1.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.002)
    return bool(predicate())


class _SlowSocket:
    def __init__(self) -> None:
        self.header_ids: list[int] = []
        self.payloads: list[bytes] = []
        self.first_payload_entered = threading.Event()
        self.release_first_payload = threading.Event()

    def sendall(self, data: bytes) -> None:
        if len(data) == 32:
            self.header_ids.append(decode_header(data).frame_id)
            return
        if not self.payloads:
            self.first_payload_entered.set()
            assert self.release_first_payload.wait(timeout=1.0)
        self.payloads.append(bytes(data))


def test_slow_sender_skips_intermediate_frames_instead_of_queueing_them() -> None:
    hub = FrameHub()
    hub.publish(_frame(1))
    fake_socket = _SlowSocket()
    stop = threading.Event()
    consumer = LiveStreamConsumer(
        hub,
        fake_socket,
        jpeg_encoder=lambda _frame: b"jpeg",
    )
    thread = threading.Thread(
        target=consumer.run,
        kwargs={"stop_event": stop, "accept_frame_id_floor": None},
        daemon=True,
    )
    thread.start()
    assert fake_socket.first_payload_entered.wait(timeout=1.0)

    for frame_id in range(2, 101):
        hub.publish(_frame(frame_id))
    fake_socket.release_first_payload.set()

    assert _wait_until(lambda: len(fake_socket.header_ids) >= 2)
    stop.set()
    thread.join(timeout=1.0)

    assert not thread.is_alive()
    assert fake_socket.header_ids[:2] == [1, 100]
    assert 2 not in fake_socket.header_ids
    assert consumer.metrics.skipped_frame_ids >= 98
    assert consumer.metrics.sent_frames >= 2
    assert hub.replacement_count >= 99


class _RecordingSocket:
    def __init__(self) -> None:
        self.header_ids: list[int] = []
        self.payloads: list[bytes] = []

    def sendall(self, data: bytes) -> None:
        if len(data) == 32:
            self.header_ids.append(decode_header(data).frame_id)
        else:
            self.payloads.append(bytes(data))


def test_accept_floor_waits_for_capture_newer_than_cached_preconnection_frame() -> None:
    hub = FrameHub()
    hub.publish(_frame(5))
    fake_socket = _RecordingSocket()
    stop = threading.Event()
    consumer = LiveStreamConsumer(
        hub,
        fake_socket,
        jpeg_encoder=lambda _frame: b"jpeg",
    )
    thread = threading.Thread(
        target=consumer.run,
        kwargs={"stop_event": stop, "accept_frame_id_floor": 5},
        daemon=True,
    )
    thread.start()

    time.sleep(0.03)
    assert fake_socket.header_ids == []

    hub.publish(_frame(9))
    assert _wait_until(lambda: fake_socket.header_ids == [9])
    stop.set()
    thread.join(timeout=1.0)


def test_oversized_encoded_frame_is_dropped_not_truncated() -> None:
    hub = FrameHub()
    hub.publish(_frame(1))
    fake_socket = _RecordingSocket()
    stop = threading.Event()
    consumer = LiveStreamConsumer(
        hub,
        fake_socket,
        jpeg_encoder=lambda _frame: b"x" * (MAX_PAYLOAD_SIZE + 1),
    )
    thread = threading.Thread(
        target=consumer.run,
        kwargs={"stop_event": stop, "accept_frame_id_floor": None, "wait_timeout": 0.01},
        daemon=True,
    )
    thread.start()

    assert _wait_until(lambda: consumer.metrics.oversized_frames == 1)
    stop.set()
    thread.join(timeout=1.0)

    assert fake_socket.header_ids == []
    assert fake_socket.payloads == []
    assert consumer.metrics.sent_frames == 0
    assert consumer.metrics.last_selected_frame_id == 1


class _FailingSocket(_RecordingSocket):
    def sendall(self, data: bytes) -> None:
        raise TimeoutError("slow client")


def test_write_timeout_terminates_consumer_without_retry_queue() -> None:
    hub = FrameHub()
    hub.publish(_frame(1))
    consumer = LiveStreamConsumer(
        hub,
        _FailingSocket(),
        jpeg_encoder=lambda _frame: b"jpeg",
    )

    metrics = consumer.run(
        stop_event=threading.Event(),
        accept_frame_id_floor=None,
        wait_timeout=0.01,
    )

    assert metrics.send_errors == 1
    assert metrics.sent_frames == 0
    assert "slow client" in (metrics.last_error or "")


def _real_frame(frame_id: int) -> VisionFrame:
    return VisionFrame(
        frame_id=frame_id,
        capture_timestamp_ns=frame_id + 1,
        width=2,
        height=2,
        pixel_format=PixelFormat.BGR8,
        image=np.zeros((2, 2, 3), dtype=np.uint8),
    )


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise RuntimeError("socket closed before expected bytes")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _recv_one_frame_id(sock: socket.socket) -> int:
    header_bytes = _recv_exact(sock, 32)
    header = decode_header(header_bytes)
    _recv_exact(sock, header.payload_size)
    return header.frame_id


def test_tcp_server_reconnect_never_replays_cached_preconnection_frame() -> None:
    hub = FrameHub()
    hub.publish(_real_frame(5))
    server = VisionTcpServer(
        hub,
        bind_host="127.0.0.1",
        port=0,
        write_timeout=0.2,
    ).start()
    assert server.bound_port is not None

    first = socket.create_connection(("127.0.0.1", server.bound_port), timeout=1.0)
    try:
        assert server.client_connected.wait(timeout=1.0)
        first.settimeout(0.05)
        try:
            first.recv(1)
        except (TimeoutError, socket.timeout):
            pass
        else:
            raise AssertionError("cached frame 5 must not replay on first accept")

        hub.publish(_real_frame(9))
        first.settimeout(1.0)
        assert _recv_one_frame_id(first) == 9
    finally:
        first.close()

    assert _wait_until(lambda: not server.client_connected.is_set())
    hub.publish(_real_frame(10))

    second = socket.create_connection(("127.0.0.1", server.bound_port), timeout=1.0)
    try:
        assert server.client_connected.wait(timeout=1.0)
        second.settimeout(0.05)
        try:
            second.recv(1)
        except (TimeoutError, socket.timeout):
            pass
        else:
            raise AssertionError("cached frame 10 must not replay after reconnect")

        hub.publish(_real_frame(11))
        second.settimeout(1.0)
        assert _recv_one_frame_id(second) == 11
    finally:
        second.close()
        server.stop()

    assert server.last_error is None


def test_tcp_server_rejects_unexpected_client_to_server_bytes() -> None:
    hub = FrameHub()
    server = VisionTcpServer(
        hub,
        bind_host="127.0.0.1",
        port=0,
        write_timeout=0.2,
    ).start()
    assert server.bound_port is not None

    client = socket.create_connection(("127.0.0.1", server.bound_port), timeout=1.0)
    try:
        assert server.client_connected.wait(timeout=1.0)
        client.sendall(b"x")
        assert _wait_until(lambda: not server.client_connected.is_set())
    finally:
        client.close()
        server.stop()

    assert server.last_error is None


def test_tcp_server_rejects_invalid_jpeg_quality_before_listening() -> None:
    server = VisionTcpServer(
        FrameHub(),
        bind_host="127.0.0.1",
        port=0,
        jpeg_quality=0,
    )

    try:
        server.start()
    except ValueError as error:
        assert "JPEG quality" in str(error)
    else:
        server.stop()
        raise AssertionError("invalid JPEG quality must fail before listening")

    assert server.bound_port is None
