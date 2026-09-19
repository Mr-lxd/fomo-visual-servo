#!/usr/bin/env python3
"""Run RoboBeetle LIVE Vision plus Slice 2 capture control on Raspberry Pi."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from fomo_servo.vision.service import VisionService, VisionServiceConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture one UVC camera and stream latest JPEG frames over RBVS v1."
    )
    parser.add_argument("--source", default="/dev/video0")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--fourcc", default="YUYV")
    parser.add_argument("--bind", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=47010)
    parser.add_argument("--jpeg-quality", type=int, default=80)
    parser.add_argument("--send-buffer-bytes", type=int, default=64 * 1024)
    parser.add_argument("--write-timeout", type=float, default=0.5)
    parser.add_argument("--control-port", type=int, default=47011)
    parser.add_argument(
        "--capture-output-root",
        type=Path,
        default=Path("datasets_raw/robobeetle"),
    )
    parser.add_argument("--capture-queue-mib", type=int, default=64)
    parser.add_argument("--capture-min-free-mib", type=int, default=512)
    parser.add_argument("--log-level", default="INFO")
    return parser


def service_config_from_args(args: argparse.Namespace) -> VisionServiceConfig:
    return VisionServiceConfig(
        source=args.source,
        width=args.width,
        height=args.height,
        fps=args.fps,
        fourcc=args.fourcc,
        bind_host=args.bind,
        port=args.port,
        jpeg_quality=args.jpeg_quality,
        send_buffer_bytes=args.send_buffer_bytes,
        write_timeout=args.write_timeout,
        control_port=args.control_port,
        capture_output_root=args.capture_output_root,
        capture_queue_bytes=args.capture_queue_mib * 1024 * 1024,
        capture_min_free_bytes=args.capture_min_free_mib * 1024 * 1024,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    service = VisionService(service_config_from_args(args))
    try:
        service.start_live()
        facts = service.camera_owner.facts
        logging.info(
            "Vision LIVE started source=%s observed=%sx%s fps=%s fourcc=%s "
            "stream=%s:%s control=%s:%s capture_root=%s",
            args.source,
            None if facts is None else facts.observed_width,
            None if facts is None else facts.observed_height,
            None if facts is None else facts.observed_fps,
            None if facts is None else facts.observed_fourcc,
            args.bind,
            service.stream_server.bound_port,
            args.bind,
            service.control_server.bound_port,
            args.capture_output_root,
        )

        next_diagnostics = time.monotonic() + 2.0
        while True:
            if service.camera_owner.finished.wait(timeout=0.5):
                error = service.camera_owner.error
                if error is not None:
                    raise RuntimeError(f"CameraOwner stopped: {error}") from error
                break

            server_error = service.stream_server.last_error
            if server_error is not None:
                raise RuntimeError(f"Vision TCP server stopped: {server_error}") from server_error
            control_error = service.control_server.last_error
            if control_error is not None:
                raise RuntimeError(
                    f"Vision control server stopped: {control_error}"
                ) from control_error

            now = time.monotonic()
            if now < next_diagnostics:
                continue
            next_diagnostics = now + 2.0

            latest = service.hub.snapshot()
            metrics = service.stream_server.last_metrics
            sent = 0 if metrics is None else metrics.sent_frames
            selected = 0 if metrics is None else metrics.selected_frames
            skipped = 0 if metrics is None else metrics.skipped_frame_ids
            encoded_bytes = 0 if metrics is None else metrics.encoded_bytes
            average_jpeg = 0 if sent == 0 else encoded_bytes // sent
            capture = service.capture_manager.status()
            logging.info(
                "Vision diag captured=%s latest_frame_id=%s measured_fps=%s "
                "client=%s selected=%s sent=%s skipped_ids=%s "
                "avg_jpeg_bytes=%s capture_state=%s recorded=%s "
                "snapshots=%s queue_bytes=%s",
                service.camera_owner.frames_captured,
                None if latest is None else latest.frame_id,
                service.camera_owner.measured_capture_fps,
                service.stream_server.client_connected.is_set(),
                selected,
                sent,
                skipped,
                average_jpeg,
                capture["state"],
                capture["recorded_frames"],
                capture["snapshot_count"],
                capture["queue_bytes"],
            )
    except KeyboardInterrupt:
        logging.info("Vision LIVE stop requested")
    except Exception:
        logging.exception("Vision LIVE failed")
        return 1
    finally:
        service.shutdown()

    return 0


if __name__ == "__main__":
    sys.exit(main())
