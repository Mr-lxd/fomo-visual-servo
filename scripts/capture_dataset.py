"""Standalone lab-pool dataset capture CLI (no inference, no model).

Records raw MJPG ``.avi`` video and sequential JPEG snapshots from a camera
into auto-managed session directories, with an optional OpenCV preview HUD
for HDMI field operation. This tool never loads a model and never runs
detection.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Optional, Sequence

import cv2

from fomo_servo.capture import (
    CaptureEngine,
    CaptureError,
    CaptureIO,
    INSTRUCTIONS,
    derive_session_prefix,
    read_camera_facts,
)
from fomo_servo.capture.session_layout import plan_next_session


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture raw lab-pool dataset sessions (no inference)."
    )
    parser.add_argument("--source", required=True, help="Camera index or /dev/video node")
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Dataset root, e.g. datasets_raw/lab_pool",
    )
    parser.add_argument(
        "--session-prefix",
        default=None,
        help="Session ID prefix (default: derived from the output-root leaf name)",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Show the local preview window with the status HUD",
    )
    parser.add_argument(
        "--fullscreen",
        action="store_true",
        help="Fullscreen preview (implies --display) for HDMI field operation",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Start recording immediately (always implied in headless mode)",
    )
    parser.add_argument("--width", type=int, default=None, help="Requested capture width")
    parser.add_argument("--height", type=int, default=None, help="Requested capture height")
    parser.add_argument("--fps", type=float, default=None, help="Requested capture fps")
    parser.add_argument(
        "--frame-interval-seconds",
        type=float,
        default=None,
        help="Also save one raw JPEG every N seconds while running",
    )
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=5.0,
        help="Warn when free disk space drops below this many GB",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Stop after N camera frames (default: run until Q/ESC or camera failure)",
    )
    parser.add_argument(
        "--duration-seconds",
        type=float,
        default=None,
        help="Stop after N seconds of capture time",
    )
    parser.add_argument("--scene", default="", help="Free-text scene description")
    parser.add_argument("--target", default="", help="Free-text target description")
    parser.add_argument("--notes", default="", help="Free-text field notes")
    return parser


def _open_capture(source: str):
    """Open a camera by index or device path using the shared project rule."""

    value = int(source) if source.isdigit() else source
    capture = cv2.VideoCapture(value)
    if not capture.isOpened():
        raise CaptureError("unable to open video source: {}".format(source))
    return capture


def main(
    arguments: Optional[Sequence[str]] = None,
    *,
    capture_factory: Optional[Callable[[], object]] = None,
    io: Optional[CaptureIO] = None,
) -> int:
    args = build_parser().parse_args(arguments)
    io = io if io is not None else CaptureIO()
    try:
        if args.max_frames is not None and args.max_frames < 1:
            raise CaptureError("--max-frames must be at least 1")
        if args.duration_seconds is not None and args.duration_seconds <= 0.0:
            raise CaptureError("--duration-seconds must be positive")
        if (
            args.frame_interval_seconds is not None
            and args.frame_interval_seconds <= 0.0
        ):
            raise CaptureError("--frame-interval-seconds must be positive")
        if args.min_free_gb < 0.0:
            raise CaptureError("--min-free-gb must be non-negative")
        if args.width is not None and args.width <= 0:
            raise CaptureError("--width must be positive")
        if args.height is not None and args.height <= 0:
            raise CaptureError("--height must be positive")
        if args.fps is not None and args.fps <= 0.0:
            raise CaptureError("--fps must be positive")

        output_root = Path(args.output_root)
        output_root.mkdir(parents=True, exist_ok=True)
        display = args.display or args.fullscreen

        free_gb = io.disk_free_bytes(output_root) / 1e9
        print("FREE SPACE: {:.1f} GB".format(free_gb))
        if free_gb < args.min_free_gb:
            print(
                "WARNING: LOW DISK SPACE ({:.1f} GB free < {:.1f} GB threshold); "
                "capture will continue but may fill the disk".format(
                    free_gb, args.min_free_gb
                )
            )

        capture = (
            capture_factory()
            if capture_factory is not None
            else _open_capture(args.source)
        )
        try:
            if args.width is not None:
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(args.width))
            if args.height is not None:
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(args.height))
            if args.fps is not None:
                capture.set(cv2.CAP_PROP_FPS, float(args.fps))
            facts = read_camera_facts(
                capture,
                source=args.source,
                requested_width=args.width,
                requested_height=args.height,
                requested_fps=args.fps,
            )
        except CaptureError:
            capture.release()
            raise
        except Exception as error:
            capture.release()
            raise CaptureError(
                "unable to read camera properties: {}".format(error)
            ) from error

        prefix = args.session_prefix or derive_session_prefix(output_root)

        def planner():
            return plan_next_session(output_root, prefix=prefix)

        try:
            engine = CaptureEngine(
                capture,
                planner=planner,
                facts=facts,
                scene=args.scene,
                target=args.target,
                notes=args.notes,
                frame_interval_seconds=args.frame_interval_seconds,
                io=io,
            )
        except BaseException:
            capture.release()
            raise
        print("SESSION: {}".format(engine.session_id))
        print("SESSION DIR: {}".format(engine.session_dir))
        if display:
            print(INSTRUCTIONS)
        status, end_reason = engine.run(
            display=display,
            fullscreen=args.fullscreen,
            max_frames=args.max_frames,
            duration_seconds=args.duration_seconds,
            min_free_gb=args.min_free_gb,
            # Unattended headless runs have no keyboard: record everything.
            start_recording=args.record or not display,
        )
        if engine.low_disk:
            print("WARNING: LOW DISK SPACE during capture")
        for summary in engine.session_summaries:
            print(
                "session {} {}: {} frames, {} snapshots -> {}".format(
                    summary["session_id"],
                    summary["status"],
                    summary["video_frame_count"],
                    summary["snapshot_count"],
                    summary["metadata_path"],
                )
            )
        print("capture {}: {}".format(status, end_reason))
        return 0
    except (CaptureError, ValueError) as error:
        print("Error: {}".format(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
