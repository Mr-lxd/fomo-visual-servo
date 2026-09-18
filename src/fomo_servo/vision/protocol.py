"""RoboBeetle Vision Stream (RBVS) v1 frame-header codec.

All multi-byte integers use network byte order (big endian). The capture
clock is Raspberry-Pi-local monotonic nanoseconds and is not directly
comparable with an unsynchronised Windows wall clock.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from enum import IntEnum
from typing import Final

MAGIC: Final[bytes] = b"RBVS"
VERSION: Final[int] = 1
HEADER_SIZE: Final[int] = 32
FLAGS: Final[int] = 0
MAX_PAYLOAD_SIZE: Final[int] = 4_194_304  # 4 MiB
MAX_DIMENSION: Final[int] = 8192
MAX_DECODED_PIXELS: Final[int] = 4096 * 2160
_MAX_U64: Final[int] = (1 << 64) - 1
_HEADER: Final[struct.Struct] = struct.Struct("!4sBBBBQQHHI")


class PayloadCodec(IntEnum):
    """RBVS payload codec registry."""

    JPEG = 1


class VisionProtocolError(ValueError):
    """Raised when an RBVS v1 header violates the frozen wire contract."""


@dataclass(frozen=True)
class VisionFrameHeader:
    """Semantic fields carried by one RBVS v1 JPEG frame header."""

    frame_id: int
    capture_timestamp_ns: int
    width: int
    height: int
    payload_size: int
    codec: PayloadCodec = PayloadCodec.JPEG
    flags: int = FLAGS


def _require_uint(name: str, value: int, maximum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise VisionProtocolError(f"{name} must be an integer")
    if value < 0 or value > maximum:
        raise VisionProtocolError(f"{name} is outside 0..{maximum}")


def validate_header(header: VisionFrameHeader) -> VisionFrameHeader:
    """Validate semantic RBVS v1 limits before allocation or serialization."""

    _require_uint("frame_id", header.frame_id, _MAX_U64)
    _require_uint("capture_timestamp_ns", header.capture_timestamp_ns, _MAX_U64)
    if header.capture_timestamp_ns == 0:
        raise VisionProtocolError("capture_timestamp_ns must be positive")
    if not 1 <= header.width <= MAX_DIMENSION:
        raise VisionProtocolError("width is outside the supported range")
    if not 1 <= header.height <= MAX_DIMENSION:
        raise VisionProtocolError("height is outside the supported range")
    if header.width * header.height > MAX_DECODED_PIXELS:
        raise VisionProtocolError("decoded pixel count exceeds the supported limit")
    if not 1 <= header.payload_size <= MAX_PAYLOAD_SIZE:
        raise VisionProtocolError("payload_size is outside the supported range")
    if header.codec != PayloadCodec.JPEG:
        raise VisionProtocolError("unsupported payload codec")
    if header.flags != FLAGS:
        raise VisionProtocolError("RBVS v1 flags must be zero")
    return header


def encode_header(header: VisionFrameHeader) -> bytes:
    """Serialize a validated semantic header to the exact 32-byte RBVS wire form."""

    validate_header(header)
    return _HEADER.pack(
        MAGIC,
        VERSION,
        HEADER_SIZE,
        int(header.codec),
        header.flags,
        header.frame_id,
        header.capture_timestamp_ns,
        header.width,
        header.height,
        header.payload_size,
    )


def encode_frame(
    header: VisionFrameHeader, payload: bytes | bytearray | memoryview
) -> bytes:
    """Serialize one complete RBVS frame without truncating or padding JPEG data."""

    payload_bytes = bytes(payload)
    if len(payload_bytes) != header.payload_size:
        raise VisionProtocolError("JPEG payload length does not match header")
    return encode_header(header) + payload_bytes


def decode_header(data: bytes | bytearray | memoryview) -> VisionFrameHeader:
    """Decode exactly one RBVS v1 header and reject malformed fields."""

    if len(data) != HEADER_SIZE:
        raise VisionProtocolError(f"RBVS header must be exactly {HEADER_SIZE} bytes")
    (
        magic,
        version,
        header_size,
        codec,
        flags,
        frame_id,
        capture_timestamp_ns,
        width,
        height,
        payload_size,
    ) = _HEADER.unpack(data)
    if magic != MAGIC:
        raise VisionProtocolError("bad RBVS magic")
    if version != VERSION:
        raise VisionProtocolError("unsupported RBVS version")
    if header_size != HEADER_SIZE:
        raise VisionProtocolError("unsupported RBVS header size")
    try:
        payload_codec = PayloadCodec(codec)
    except ValueError as error:
        raise VisionProtocolError("unsupported payload codec") from error
    return validate_header(
        VisionFrameHeader(
            frame_id=frame_id,
            capture_timestamp_ns=capture_timestamp_ns,
            width=width,
            height=height,
            payload_size=payload_size,
            codec=payload_codec,
            flags=flags,
        )
    )


if _HEADER.size != HEADER_SIZE:  # pragma: no cover - import-time invariant
    raise RuntimeError("RBVS v1 struct size drifted from the frozen 32-byte contract")
