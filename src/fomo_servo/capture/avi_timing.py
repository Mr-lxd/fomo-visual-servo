"""Finalize MJPG/AVI constant frame rate from captured monotonic timing."""

from __future__ import annotations

import os
import struct
from fractions import Fraction
from pathlib import Path
from typing import BinaryIO, Iterator


class AviTimingError(RuntimeError):
    """Raised when the expected AVI timing headers cannot be updated safely."""


def _chunks(
    handle: BinaryIO,
    start: int,
    end: int,
) -> Iterator[tuple[bytes, int, int]]:
    position = start
    while position + 8 <= end:
        handle.seek(position)
        header = handle.read(8)
        if len(header) != 8:
            return
        fourcc = header[:4]
        size = struct.unpack("<I", header[4:])[0]
        data_start = position + 8
        data_end = data_start + size
        if data_end > end:
            raise AviTimingError(
                f"AVI chunk {fourcc!r} extends beyond parent bounds"
            )
        yield fourcc, data_start, size
        position = data_end + (size & 1)


def _find_timing_headers(
    handle: BinaryIO,
    file_size: int,
) -> tuple[int, int]:
    handle.seek(0)
    header = handle.read(12)
    if len(header) != 12 or header[:4] != b"RIFF" or header[8:] != b"AVI ":
        raise AviTimingError("file is not a RIFF AVI container")

    riff_size = struct.unpack("<I", header[4:8])[0]
    riff_end = min(file_size, 8 + riff_size)
    avih_offset: int | None = None
    video_strh_offset: int | None = None

    for fourcc, data_start, size in _chunks(handle, 12, riff_end):
        if fourcc != b"LIST" or size < 4:
            continue
        handle.seek(data_start)
        list_type = handle.read(4)
        if list_type != b"hdrl":
            continue

        hdrl_start = data_start + 4
        hdrl_end = data_start + size
        for child, child_start, child_size in _chunks(
            handle, hdrl_start, hdrl_end
        ):
            if child == b"avih":
                if child_size < 4:
                    raise AviTimingError("AVI avih chunk is truncated")
                avih_offset = child_start
                continue
            if child != b"LIST" or child_size < 4:
                continue

            handle.seek(child_start)
            if handle.read(4) != b"strl":
                continue
            strl_start = child_start + 4
            strl_end = child_start + child_size
            for stream_chunk, stream_start, stream_size in _chunks(
                handle, strl_start, strl_end
            ):
                if stream_chunk != b"strh" or stream_size < 28:
                    continue
                handle.seek(stream_start)
                if handle.read(4) == b"vids":
                    video_strh_offset = stream_start
                    break
            if video_strh_offset is not None and avih_offset is not None:
                break

        if video_strh_offset is not None and avih_offset is not None:
            break

    if avih_offset is None:
        raise AviTimingError("AVI main avih timing header was not found")
    if video_strh_offset is None:
        raise AviTimingError("AVI video strh timing header was not found")
    return avih_offset, video_strh_offset


def rewrite_avi_frame_rate(path: Path, fps: float) -> float:
    """Rewrite AVI timing headers after recording without touching frame data.

    Returns the exact rate/scale FPS written into the video stream header.
    """

    if not 0.0 < fps <= 1000.0:
        raise AviTimingError("AVI frame rate must be within (0, 1000]")

    path = Path(path)
    file_size = path.stat().st_size
    fraction = Fraction(fps).limit_denominator(1_000_000)
    rate = int(fraction.numerator)
    scale = int(fraction.denominator)
    if not (0 < rate <= 0xFFFFFFFF and 0 < scale <= 0xFFFFFFFF):
        raise AviTimingError("AVI rate/scale exceeds uint32 range")

    microseconds_per_frame = max(1, int(round(1_000_000.0 / fps)))

    with path.open("r+b") as handle:
        avih_offset, strh_offset = _find_timing_headers(handle, file_size)

        handle.seek(avih_offset)
        original_avih = handle.read(4)
        handle.seek(strh_offset + 20)
        original_strh_timing = handle.read(8)
        if len(original_avih) != 4 or len(original_strh_timing) != 8:
            raise AviTimingError("AVI timing header is truncated")

        try:
            handle.seek(avih_offset)
            handle.write(struct.pack("<I", microseconds_per_frame))

            # AVISTREAMHEADER: dwScale @ +20, dwRate @ +24.
            handle.seek(strh_offset + 20)
            handle.write(struct.pack("<II", scale, rate))

            handle.flush()
            os.fsync(handle.fileno())
        except BaseException as error:
            # Best-effort process-level rollback. This intentionally does not
            # claim power-loss atomicity; it only prevents ordinary I/O errors
            # from leaving mixed old/new AVI timing fields.
            try:
                handle.seek(avih_offset)
                handle.write(original_avih)
                handle.seek(strh_offset + 20)
                handle.write(original_strh_timing)
                handle.flush()
                os.fsync(handle.fileno())
            except BaseException as rollback_error:
                raise AviTimingError(
                    "AVI timing rewrite failed and rollback also failed: "
                    f"{rollback_error}"
                ) from error
            raise AviTimingError(
                f"AVI timing rewrite failed and was rolled back: {error}"
            ) from error

    return rate / scale
