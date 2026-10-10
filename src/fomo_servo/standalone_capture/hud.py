"""Preview-only status overlay for the dataset capture window.

The HUD is drawn onto a *copy* of the raw camera frame and is used only for
the on-screen preview. Recorded video segments and snapshot JPEGs always
receive the untouched raw frame.
"""

from __future__ import annotations

from typing import Any, Mapping

import cv2
import numpy as np


WINDOW_NAME = "FOMO Dataset Capture"

INSTRUCTIONS = "SPACE: REC  |  N: NEW SESSION  |  S: SNAPSHOT  |  Q/ESC: QUIT"

_TEXT_COLOR = (255, 255, 255)
_REC_COLOR = (0, 0, 255)
_WARNING_COLOR = (0, 0, 255)
_INSTRUCTION_COLOR = (0, 255, 255)
_BACKGROUND_COLOR = (0, 0, 0)
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def format_seconds(seconds: float) -> str:
    """Format ``seconds`` as zero-padded ``HH:MM:SS``."""

    total = max(int(round(seconds)), 0)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return "{:02d}:{:02d}:{:02d}".format(hours, minutes, secs)


def status_lines(state: Mapping[str, Any]) -> list[str]:
    """Build the HUD text lines from the engine state mapping."""

    status = "● REC" if state["recording"] else "PREVIEW"
    lines = [
        "CAMERA: {}".format(state["camera"]),
        "RES: {}".format(state["resolution"]),
        "FPS: {}".format(state["fps"]),
        "FREE SPACE: {}".format(state["free_space_text"]),
        "SESSION: {}".format(state["session_id"]),
        "STATUS: {}".format(status),
        "REC TIME: {}".format(format_seconds(state["rec_seconds"])),
        "VIDEO FRAMES: {}".format(state["video_frames"]),
        "SAVED IMAGES: {}".format(state["snapshot_count"]),
    ]
    if state["low_disk_space"]:
        lines.append("WARNING: LOW DISK SPACE")
    return lines


def draw_status_overlay(frame_bgr: np.ndarray, state: Mapping[str, Any]) -> np.ndarray:
    """Return a *copy* of BGR ``[H,W,3]`` ``frame_bgr`` with the status HUD.

    The input array is never modified; recorded data must never use the
    returned preview copy.
    """

    preview = frame_bgr.copy()
    height, width = preview.shape[:2]
    scale = max(width / 640.0, 0.5)
    small_scale = 0.45 * scale
    large_scale = 0.7 * scale
    thickness = max(int(round(scale)), 1)
    line_height = int(26 * scale)
    margin_x = int(12 * scale)
    margin_y = int(28 * scale)

    lines = status_lines(state)
    for index, line in enumerate(lines):
        is_rec_line = "STATUS:" in line
        color = _REC_COLOR if is_rec_line else _TEXT_COLOR
        font_scale = large_scale if is_rec_line else small_scale
        text_thickness = thickness + (1 if is_rec_line else 0)
        origin_y = margin_y + index * line_height
        text_offset_x = margin_x
        if is_rec_line and state["recording"]:
            # Hershey fonts cannot render the U+25CF bullet; draw it instead.
            text = line.replace("● ", "", 1)
            radius = int(8 * scale)
            center = (margin_x + radius, origin_y - int(8 * scale))
            cv2.circle(preview, center, radius, _REC_COLOR, -1)
            text_offset_x = margin_x + 2 * radius + int(6 * scale)
        else:
            text = line
        _draw_text_with_background(
            preview,
            text,
            (text_offset_x, origin_y),
            font_scale,
            color,
            text_thickness,
        )

    instruction_origin = (margin_x, height - int(12 * scale))
    _draw_text_with_background(
        preview,
        INSTRUCTIONS,
        instruction_origin,
        small_scale,
        _INSTRUCTION_COLOR,
        thickness,
    )
    return preview


def _draw_text_with_background(
    frame: np.ndarray,
    text: str,
    origin: tuple[int, int],
    font_scale: float,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    (text_width, text_height), baseline = cv2.getTextSize(
        text, _FONT, font_scale, thickness
    )
    left, bottom = origin
    top = max(bottom - text_height - baseline, 0)
    cv2.rectangle(
        frame,
        (left, top),
        (min(left + text_width, frame.shape[1] - 1), min(bottom + baseline, frame.shape[0] - 1)),
        _BACKGROUND_COLOR,
        thickness=-1,
    )
    cv2.putText(
        frame,
        text,
        (left, bottom),
        _FONT,
        font_scale,
        color,
        thickness,
        lineType=cv2.LINE_AA,
    )
