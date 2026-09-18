"""JPEG realtime consumer and bounded single-client RBVS TCP server."""

from __future__ import annotations

import logging
import select
import socket
import threading
from dataclasses import dataclass
from typing import Callable, Optional

import cv2

from .frame import PixelFormat, VisionFrame
from .frame_hub import FrameHub
from .protocol import MAX_PAYLOAD_SIZE, VisionFrameHeader, encode_header

LOGGER = logging.getLogger(__name__)
DEFAULT_VISION_PORT = 47010
DEFAULT_JPEG_QUALITY = 80
DEFAULT_SEND_BUFFER_BYTES = 64 * 1024
DEFAULT_WRITE_TIMEOUT_SECONDS = 0.5


class FrameEncodingError(RuntimeError):
    """Raised when a raw VisionFrame cannot be represented as a JPEG."""


def encode_jpeg(frame: VisionFrame, *, quality: int = DEFAULT_JPEG_QUALITY) -> bytes:
    """Encode exactly one raw BGR frame without resize, overlay, or queueing."""

    if frame.pixel_format is not PixelFormat.BGR8:
        raise FrameEncodingError("only BGR8 raw frames can be JPEG encoded")
    shape = getattr(frame.image, "shape", ())
    if len(shape) != 3 or int(shape[2]) != 3:
        raise FrameEncodingError("raw frame must be an HxWx3 BGR image")
    if int(shape[1]) != frame.width or int(shape[0]) != frame.height:
        raise FrameEncodingError("raw frame dimensions do not match metadata")
    if not 1 <= quality <= 100:
        raise FrameEncodingError("JPEG quality must be within 1..100")

    ok, encoded = cv2.imencode(
        ".jpg",
        frame.image,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)],
    )
    if not ok:
        raise FrameEncodingError("OpenCV failed to encode JPEG")
    return encoded.tobytes()


@dataclass
class StreamMetrics:
    selected_frames: int = 0
    sent_frames: int = 0
    encoded_bytes: int = 0
    encode_errors: int = 0
    oversized_frames: int = 0
    send_errors: int = 0
    skipped_frame_ids: int = 0
    last_selected_frame_id: Optional[int] = None
    last_sent_frame_id: Optional[int] = None
    last_error: Optional[str] = None


class LiveStreamConsumer:
    """Select newest raw frames and synchronously send at most one unfinished frame."""

    def __init__(
        self,
        hub: FrameHub,
        client_socket,
        *,
        jpeg_quality: int = DEFAULT_JPEG_QUALITY,
        jpeg_encoder: Optional[Callable[[VisionFrame], bytes]] = None,
    ) -> None:
        self._hub = hub
        self._socket = client_socket
        self._jpeg_quality = jpeg_quality
        self._jpeg_encoder = jpeg_encoder
        self.metrics = StreamMetrics()

    def run(
        self,
        *,
        stop_event: threading.Event,
        accept_frame_id_floor: Optional[int],
        wait_timeout: float = 0.1,
        connection_alive: Optional[Callable[[], bool]] = None,
    ) -> StreamMetrics:
        """Run until stopped, peer closes, or one synchronous frame write fails."""

        after_frame_id = accept_frame_id_floor
        while not stop_event.is_set():
            frame = self._hub.wait_for_newer(after_frame_id, timeout=wait_timeout)
            if frame is None:
                if connection_alive is not None and not connection_alive():
                    break
                continue

            if connection_alive is not None and not connection_alive():
                break

            previous_selected = self.metrics.last_selected_frame_id
            after_frame_id = frame.frame_id
            self.metrics.selected_frames += 1
            if previous_selected is not None and frame.frame_id > previous_selected + 1:
                self.metrics.skipped_frame_ids += frame.frame_id - previous_selected - 1
            self.metrics.last_selected_frame_id = frame.frame_id

            try:
                if self._jpeg_encoder is None:
                    payload = encode_jpeg(frame, quality=self._jpeg_quality)
                else:
                    payload = self._jpeg_encoder(frame)
            except Exception as error:
                self.metrics.encode_errors += 1
                self.metrics.last_error = str(error)
                continue

            if len(payload) < 1 or len(payload) > MAX_PAYLOAD_SIZE:
                self.metrics.oversized_frames += 1
                self.metrics.last_error = "encoded JPEG outside RBVS payload limit"
                continue

            header = VisionFrameHeader(
                frame_id=frame.frame_id,
                capture_timestamp_ns=frame.capture_timestamp_ns,
                width=frame.width,
                height=frame.height,
                payload_size=len(payload),
            )
            wire_header = encode_header(header)

            try:
                self._socket.sendall(wire_header)
                self._socket.sendall(payload)
            except (OSError, TimeoutError) as error:
                self.metrics.send_errors += 1
                self.metrics.last_error = str(error)
                break

            self.metrics.sent_frames += 1
            self.metrics.encoded_bytes += len(payload)
            self.metrics.last_sent_frame_id = frame.frame_id

        return self.metrics


class VisionTcpServer:
    """Dedicated RBVS listener with one active realtime viewer."""

    def __init__(
        self,
        hub: FrameHub,
        *,
        bind_host: str = "0.0.0.0",
        port: int = DEFAULT_VISION_PORT,
        jpeg_quality: int = DEFAULT_JPEG_QUALITY,
        send_buffer_bytes: int = DEFAULT_SEND_BUFFER_BYTES,
        write_timeout: float = DEFAULT_WRITE_TIMEOUT_SECONDS,
    ) -> None:
        self._hub = hub
        self._bind_host = bind_host
        self._port = port
        self._jpeg_quality = jpeg_quality
        self._send_buffer_bytes = send_buffer_bytes
        self._write_timeout = write_timeout

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._listener: Optional[socket.socket] = None
        self._client: Optional[socket.socket] = None
        self._client_lock = threading.Lock()
        self.bound_port: Optional[int] = None
        self.last_metrics: Optional[StreamMetrics] = None
        self.last_error: Optional[BaseException] = None
        self.client_connected = threading.Event()

    @property
    def is_running(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    def start(self) -> "VisionTcpServer":
        if self.is_running:
            raise RuntimeError("VisionTcpServer is already running")
        if not 0 <= self._port <= 65535:
            raise ValueError("Vision TCP port is outside 0..65535")
        if not 1 <= self._jpeg_quality <= 100:
            raise ValueError("JPEG quality must be within 1..100")
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

        self._thread = threading.Thread(
            target=self._accept_loop,
            name="robobeetle-vision-tcp-server",
            daemon=True,
        )
        self._thread.start()
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
            raise RuntimeError("Vision TCP server did not stop")

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
                        "Vision client connected from %s; accept frame floor=%s",
                        address,
                        accept_floor,
                    )
                    self.client_connected.set()
                    consumer = LiveStreamConsumer(
                        self._hub,
                        client,
                        jpeg_quality=self._jpeg_quality,
                    )
                    # Publish the mutable metrics object before run() so the
                    # foreground service can report live acceptance diagnostics.
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
            LOGGER.exception("Vision TCP server failed")

    def _configure_client(self, client: socket.socket) -> None:
        client.settimeout(self._write_timeout)
        client.setsockopt(
            socket.SOL_SOCKET,
            socket.SO_SNDBUF,
            int(self._send_buffer_bytes),
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

    # RBVS v1 is server-to-client only. A readable client socket therefore
    # means either orderly/reset peer close or unexpected inbound bytes.
    # Both end the sole active viewer session.
    return False
