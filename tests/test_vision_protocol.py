from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from fomo_servo.vision.protocol import (
    HEADER_SIZE,
    MAX_DECODED_PIXELS,
    MAX_PAYLOAD_SIZE,
    PayloadCodec,
    VisionFrameHeader,
    VisionProtocolError,
    decode_header,
    encode_frame,
    encode_header,
)

GOLDEN_HEADER = bytes.fromhex(
    "52 42 56 53 01 20 01 00 "
    "01 02 03 04 05 06 07 08 "
    "11 12 13 14 15 16 17 18 "
    "02 80 01 e0 00 01 23 45"
)


def golden_semantic_header() -> VisionFrameHeader:
    return VisionFrameHeader(
        frame_id=0x0102030405060708,
        capture_timestamp_ns=0x1112131415161718,
        width=640,
        height=480,
        payload_size=0x00012345,
    )


def test_v1_header_has_frozen_32_byte_golden_encoding() -> None:
    encoded = encode_header(golden_semantic_header())

    assert HEADER_SIZE == 32
    assert len(encoded) == HEADER_SIZE
    assert encoded == GOLDEN_HEADER


def test_v1_golden_header_decodes_round_trip() -> None:
    decoded = decode_header(GOLDEN_HEADER)

    assert decoded == golden_semantic_header()
    assert decoded.codec is PayloadCodec.JPEG
    assert decoded.flags == 0


@pytest.mark.parametrize(
    ("offset", "value", "message"),
    [
        (0, ord("X"), "magic"),
        (4, 2, "version"),
        (5, 31, "header size"),
        (6, 2, "codec"),
        (7, 1, "flags"),
    ],
)
def test_malformed_registry_fields_are_rejected(
    offset: int, value: int, message: str
) -> None:
    wire = bytearray(GOLDEN_HEADER)
    wire[offset] = value

    with pytest.raises(VisionProtocolError, match=message):
        decode_header(wire)


def test_wrong_header_length_is_rejected() -> None:
    with pytest.raises(VisionProtocolError, match="exactly 32"):
        decode_header(GOLDEN_HEADER[:-1])


@pytest.mark.parametrize(
    "header",
    [
        VisionFrameHeader(1, 1, 0, 480, 100),
        VisionFrameHeader(1, 1, 640, 0, 100),
        VisionFrameHeader(1, 1, 640, 480, 0),
        VisionFrameHeader(1, 0, 640, 480, 100),
        VisionFrameHeader(1, 1, 640, 480, MAX_PAYLOAD_SIZE + 1),
        VisionFrameHeader(1, 1, 8192, 8192, 100),
    ],
)
def test_invalid_semantic_limits_are_rejected(header: VisionFrameHeader) -> None:
    with pytest.raises(VisionProtocolError):
        encode_header(header)


def test_frame_id_zero_is_valid_at_stream_start() -> None:
    header = VisionFrameHeader(
        frame_id=0,
        capture_timestamp_ns=1,
        width=640,
        height=480,
        payload_size=1,
    )

    assert decode_header(encode_header(header)) == header


FULL_FRAME_HEADER = bytes.fromhex(
    "52 42 56 53 01 20 01 00 "
    "00 00 00 00 00 00 00 00 "
    "00 00 00 00 00 00 00 01 "
    "00 02 00 02 00 00 02 ae"
)
FIXTURE_PATH = Path(__file__).parent / "fixtures" / "rbvs_v1_2x2.jpg"
FIXTURE_PAYLOAD_SHA256 = (
    "742b658462184778ddf5d92207786bb7f08182f75e7f0f5444f8112192b50cd6"
)
FIXTURE_FRAME_SHA256 = (
    "635bb1a0371f9c9c4242f2924a4c0ffb0590e3edff16945313a89407520bdfa6"
)


def test_validation_profile_constants_are_frozen() -> None:
    assert MAX_PAYLOAD_SIZE == 4_194_304
    assert MAX_DECODED_PIXELS == 8_847_360


def test_full_frame_fixture_is_frozen_and_serializes_exactly() -> None:
    payload = FIXTURE_PATH.read_bytes()
    assert len(payload) == 686
    assert hashlib.sha256(payload).hexdigest() == FIXTURE_PAYLOAD_SHA256

    header = VisionFrameHeader(
        frame_id=0,
        capture_timestamp_ns=1,
        width=2,
        height=2,
        payload_size=len(payload),
    )
    assert encode_header(header) == FULL_FRAME_HEADER
    wire = encode_frame(header, payload)
    assert hashlib.sha256(wire).hexdigest() == FIXTURE_FRAME_SHA256


def test_complete_frame_encoder_rejects_payload_length_mismatch() -> None:
    header = VisionFrameHeader(0, 1, 2, 2, 686)
    with pytest.raises(VisionProtocolError, match="length"):
        encode_frame(header, b"not-the-fixture")
