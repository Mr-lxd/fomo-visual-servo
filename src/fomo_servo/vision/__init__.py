"""Vision runtime domain and transport primitives."""

from .protocol import (
    HEADER_SIZE,
    MAGIC,
    MAX_DECODED_PIXELS,
    MAX_DIMENSION,
    MAX_PAYLOAD_SIZE,
    PayloadCodec,
    VisionFrameHeader,
    VisionProtocolError,
    decode_header,
    encode_frame,
    encode_header,
)

__all__ = [
    "HEADER_SIZE", "MAGIC", "MAX_DECODED_PIXELS", "MAX_DIMENSION", "MAX_PAYLOAD_SIZE",
    "PayloadCodec", "VisionFrameHeader", "VisionProtocolError",
    "decode_header", "encode_frame", "encode_header",
]
