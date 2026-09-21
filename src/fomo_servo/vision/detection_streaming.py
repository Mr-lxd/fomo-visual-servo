"""Latest-only inference detection metadata stream on TCP port 47012.

The stream deliberately does not modify RBVS v1 on port 47010. Each newline-
delimited JSON record is associated with the source camera frame by frame_id
and carries only original-frame centroid coordinates required by the Qt text
overlay. It is not a bounding-box, tracker, or robot-control protocol.
"""

from __future__ import annotations

import json
import logging
import math
import select
import socket
import threading
from dataclasses import dataclass
from typing import Callable, Optional

from .inference_result_hub import InferenceResultHub
from .inference_worker import InferenceResult
from .protocol import MAX_DECODED_PIXELS, MAX_DIMENSION

LOGGER = logging.getLogger(__name__)

DEFAULT_DETECTION_PORT = 47012
DETECTION_STREAM_VERSION = 1
DETECTION_COORDINATE_SPACE = "original_frame_pixels"
DEFAULT_SEND_BUFFER_BYTES = 32 * 1024
DEFAULT_WRITE_TIMEOUT_SECONDS = 0.5
MAX_DETECTION_LINE_BYTES = 64 * 1024
MAX_CLASS_NAME_BYTES = 128
MAX_DETECTIONS_PER_RESULT = 256


class DetectionStreamError(ValueError):
    """Raised when detection metadata violates the Slice 5 wire contract."""


def _finite_number(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DetectionStreamError(f"{name} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise DetectionStreamError(f"{name} must be finite")
    return numeric


def encode_inference_result(result: InferenceResult) -> bytes:
    """Encode one completed inference result as one NDJSON UTF-8 record."""

    if isinstance(result.frame_id, bool) or not isinstance(result.frame_id, int):
        raise DetectionStreamError("frame_id must be an integer")
    if result.frame_id < 0:
        raise DetectionStreamError("frame_id must be non-negative")
    if (
        isinstance(result.capture_timestamp_ns, bool)
        or not isinstance(result.capture_timestamp_ns, int)
        or result.capture_timestamp_ns <= 0
    ):
        raise DetectionStreamError("capture_timestamp_ns must be a positive integer")
    if not 1 <= result.frame_width <= MAX_DIMENSION:
        raise DetectionStreamError("frame_width is outside the supported range")
    if not 1 <= result.frame_height <= MAX_DIMENSION:
        raise DetectionStreamError("frame_height is outside the supported range")
    if result.frame_width * result.frame_height > MAX_DECODED_PIXELS:
        raise DetectionStreamError("frame decoded pixel count exceeds the supported limit")

    if len(result.detections) > MAX_DETECTIONS_PER_RESULT:
        raise DetectionStreamError("too many detections in one result")

    detections: list[dict[str, object]] = []
    for detection in result.detections:
        if (
            isinstance(detection.class_id, bool)
            or not isinstance(detection.class_id, int)
            or detection.class_id < 0
        ):
            raise DetectionStreamError("class_id must be a non-negative integer")
        confidence = _finite_number("confidence", detection.confidence)
        original_x = _finite_number("original_x", detection.original_x)
        original_y = _finite_number("original_y", detection.original_y)
        if not 0.0 <= confidence <= 1.0:
            raise DetectionStreamError("confidence must be within 0..1")
        if not 0.0 <= original_x <= float(result.frame_width - 1):
            raise DetectionStreamError("original_x is outside the frame")
        if not 0.0 <= original_y <= float(result.frame_height - 1):
            raise DetectionStreamError("original_y is outside the frame")

        class_name = str(detection.class_name)
        encoded_name = class_name.encode("utf-8")
        if not encoded_name:
            raise DetectionStreamError("class_name must not be empty")
        if len(encoded_name) > MAX_CLASS_NAME_BYTES:
            raise DetectionStreamError("class_name is too long")

        detections.append(
            {
                "class_id": detection.class_id,
                "class_name": class_name,
                "confidence": confidence,
                "original_x": original_x,
                "original_y": original_y,
            }
        )

    record = {
        "type": "detections",
        "version": DETECTION_STREAM_VERSION,
        "frame_id": result.frame_id,
        "capture_timestamp_ns": result.capture_timestamp_ns,
        "width": result.frame_width,
        "height": result.frame_height,
        "coordinate_space": DETECTION_COORDINATE_SPACE,
        "detections": detections,
    }
    try:
        encoded = (
            json.dumps(
                record,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            ).encode("utf-8")
            + b"\n"
        )
    except (TypeError, ValueError) as error:
        raise DetectionStreamError("failed to encode detection metadata") from error
    if len(encoded) > MAX_DETECTION_LINE_BYTES:
        raise DetectionStreamError("detection metadata exceeds the wire limit")
    return encoded


@dataclass
class DetectionStreamMetrics:
    sent_results: int = 0
    sent_bytes: int = 0
    send_errors: int = 0
    encode_errors: int = 0
    skipped_result_ids: int = 0
    last_selected_frame_id: Optional[int] = None
    last_sent_frame_id: Optional[int] = None
    last_error: Optional[str] = None


class DetectionStreamConsumer:
    """Send newest completed inference results without building a backlog."""

    def __init__(self, hub: InferenceResultHub, client_socket) -> None:
        self._hub = hub
        self._socket = client_socket
        self.metrics = DetectionStreamMetrics()

    def run(
        self,
        *,
        stop_event: threading.Event,
        accept_frame_id_floor: Optional[int],
        wait_timeout: float = 0.1,
        connection_alive: Optional[Callable[[], bool]] = None,
    ) -> DetectionStreamMetrics:
        after_frame_id = accept_frame_id_floor
        while not stop_event.is_set():
            result = self._hub.wait_for_newer(after_frame_id, timeout=wait_timeout)
            if result is None:
                if connection_alive is not None and not connection_alive():
                    break
                continue
            if connection_alive is not None and not connection_alive():
                break

            previous_selected = self.metrics.last_selected_frame_id
            after_frame_id = result.frame_id
            self.metrics.last_selected_frame_id = result.frame_id
            if (
                previous_selected is not None
                and result.frame_id > previous_selected + 1
            ):
                self.metrics.skipped_result_ids += (
                    result.frame_id - previous_selected - 1
                )

            try:
                payload = encode_inference_result(result)
            except Exception as error:
                self.metrics.encode_errors += 1
                self.metrics.last_error = str(error)
                continue
            try:
                self._socket.sendall(payload)
            except (OSError, TimeoutError) as error:
                self.metrics.send_errors += 1
                self.metrics.last_error = str(error)
                break

            self.metrics.sent_results += 1
            self.metrics.sent_bytes += len(payload)
            self.metrics.last_sent_frame_id = result.frame_id
        return self.metrics


class DetectionTcpServer:
    """Dedicated single-client listener for Slice 5 detection metadata."""

    def __init__(
        self,
        hub: InferenceResultHub,
        *,
        bind_host: str = "0.0.0.0",
        port: int = DEFAULT_DETECTION_PORT,
        send_buffer_bytes: int = DEFAULT_SEND_BUFFER_BYTES,
        write_timeout: float = DEFAULT_WRITE_TIMEOUT_SECONDS,
    ) -> None:
        self._hub = hub
        self._bind_host = bind_host
        self._port = port
        self._send_buffer_bytes = send_buffer_bytes
        self._write_timeout = write_timeout
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._listener: Optional[socket.socket] = None
        self._client: Optional[socket.socket] = None
        self._client_lock = threading.Lock()
        self.bound_port: Optional[int] = None
        self.last_metrics: Optional[DetectionStreamMetrics] = None
        self.last_error: Optional[BaseException] = None
        self.client_connected = threading.Event()

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> "DetectionTcpServer":
        if self.is_running:
            raise RuntimeError("DetectionTcpServer is already running")
        if not 0 <= self._port <= 65535:
            raise ValueError("detection TCP port is outside 0..65535")
        if self._send_buffer_bytes <= 0:
            raise ValueError("send_buffer_bytes must be positive")
        if self._write_timeout <= 0:
            raise ValueError("write_timeout must be positive")

        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((self._bind_host, self._port))
        listener.listen(1)
        listener.settimeout(0.2)
        self._listener = listener
        self.bound_port = int(listener.getsockname()[1])
        self.last_error = None
        self.client_connected.clear()
        self._stop.clear()

        thread = threading.Thread(
            target=self._accept_loop,
            name="robobeetle-detection-tcp-server",
            daemon=True,
        )
        self._thread = thread
        try:
            thread.start()
        except BaseException:
            self._thread = None
            self._listener = None
            self.bound_port = None
            listener.close()
            raise
        return self

    def stop(self, *, join_timeout: float = 2.0) -> None:
        self._stop.set()
        self.disconnect_client()
        listener = self._listener
        self._listener = None
        if listener is not None:
            try:
                listener.close()
            except OSError:
                pass
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=join_timeout)
        if thread is not None and thread.is_alive():
            raise RuntimeError("Detection TCP server did not stop")
        self._thread = None
        self.bound_port = None

    def disconnect_client(self) -> None:
        with self._client_lock:
            client = self._client
            self._client = None
        if client is None:
            return
        try:
            client.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            client.close()
        except OSError:
            pass

    def _accept_loop(self) -> None:
        try:
            while not self._stop.is_set():
                listener = self._listener
                if listener is None:
                    break
                try:
                    client, address = listener.accept()
                except socket.timeout:
                    continue
                except OSError:
                    if not self._stop.is_set():
                        raise
                    break

                with self._client_lock:
                    self._client = client
                try:
                    self._configure_client(client)
                    cached = self._hub.snapshot()
                    accept_floor = None if cached is None else cached.frame_id
                    LOGGER.info(
                        "Detection client connected from %s; accept result floor=%s",
                        address,
                        accept_floor,
                    )
                    self.client_connected.set()
                    consumer = DetectionStreamConsumer(self._hub, client)
                    self.last_metrics = consumer.metrics
                    consumer.run(
                        stop_event=self._stop,
                        accept_frame_id_floor=accept_floor,
                        connection_alive=lambda: _socket_peer_alive(client),
                    )
                finally:
                    self.client_connected.clear()
                    self.disconnect_client()
        except BaseException as error:
            self.last_error = error
            LOGGER.exception("Detection metadata server failed")

    def _configure_client(self, client: socket.socket) -> None:
        client.settimeout(self._write_timeout)
        client.setsockopt(
            socket.SOL_SOCKET,
            socket.SO_SNDBUF,
            self._send_buffer_bytes,
        )
        try:
            client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            pass


def _socket_peer_alive(client: socket.socket) -> bool:
    """Return false on peer close or any unexpected client-to-server byte."""

    try:
        readable, _, exceptional = select.select([client], [], [client], 0.0)
    except (OSError, ValueError):
        return False
    if exceptional:
        return False
    if not readable:
        return True
    # The Slice 5 metadata stream is server-to-client only.
    return False
