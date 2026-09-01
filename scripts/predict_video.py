"""Run latest-frame FOMO video inference with CSV/JSONL telemetry."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Mapping, Optional, Sequence

import cv2

from fomo_servo.inference import (
    InferenceError,
    LatestFrameReader,
    SequentialFrameReader,
    OnnxRuntimePredictor,
    OrtPredictorError,
    OutputPathError,
    PreprocessingError,
    load_inference_model,
    predict_rgb_image,
    validate_output_paths,
)
from fomo_servo.metrics import SequenceStatistics, normalize_centroid
from fomo_servo.postprocess import PostprocessError, TargetTracker


CSV_COLUMNS = (
    "runtime", "model_sha256", "frame_index", "timestamp", "status", "class_id", "class_name", "confidence",
    "original_x", "original_y", "normalized_x", "normalized_y", "detection_count", "lost_frames",
)

PREVIEW_WINDOW_NAME = "FOMO Visual Servo Preview"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict FOMO centroids for a video or camera.")
    parser.add_argument("--source", required=True, help="Video path or camera index")
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Stop after processing N frames (default: run until the source ends)",
    )
    parser.add_argument(
        "--duration-seconds",
        type=float,
        default=None,
        help="Stop after N seconds of processing time (default: run until the source ends)",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Show annotated frames in a desktop window; press q or Esc to stop",
    )
    parser.add_argument(
        "--process-every-frame",
        action="store_true",
        help=(
            "Decode an offline video sequentially without dropping frames; "
            "not valid for numeric live-camera sources"
        ),
    )
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--checkpoint", type=Path)
    model_group.add_argument("--onnx", type=Path)
    parser.add_argument("--onnx-report", type=Path)
    parser.add_argument("--device", default=None)
    parser.add_argument("--confidence-threshold", type=float, default=None)
    parser.add_argument("--strategy", choices=("highest_confidence", "largest_component", "nearest_previous"), default=None)
    parser.add_argument("--max-match-distance", type=float, default=None)
    parser.add_argument("--output-video", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    return parser


def _open_source(source: str) -> cv2.VideoCapture:
    value = int(source) if source.isdigit() else source
    capture = cv2.VideoCapture(value)
    if not capture.isOpened():
        raise InferenceError("unable to open video source: {}".format(source))
    return capture


def _opencv_gui_backend(build_information: str) -> str:
    """Return the OpenCV HighGUI backend label, or an empty string if absent."""

    for line in build_information.splitlines():
        stripped = line.strip()
        if stripped.upper().startswith("GUI:"):
            return stripped.split(":", 1)[1].strip()
    return ""


def _validate_preview_environment(
    *,
    gui_backend: str,
    environment: Mapping[str, str],
    platform_name: str,
) -> None:
    """Reject unsupported GUI builds or Linux shells outside a desktop session."""

    if not gui_backend or gui_backend.upper() == "NONE":
        raise InferenceError(
            "desktop preview is unavailable because this OpenCV build is headless; "
            "use the optional preview environment with GUI-enabled opencv-python"
        )
    if platform_name.startswith("linux"):
        display = environment.get("DISPLAY")
        wayland_display = environment.get("WAYLAND_DISPLAY")
        if not display and not wayland_display:
            raise InferenceError(
                "desktop preview has no active GUI session; launch this command "
                "from a Raspberry Pi VNC desktop terminal"
            )
        if "QT" in gui_backend.upper() and not display:
            raise InferenceError(
                "the OpenCV Qt preview backend requires the desktop XWayland "
                "environment; launch this command from a Raspberry Pi VNC desktop terminal"
            )


def _open_preview_window() -> None:
    """Validate HighGUI capability and create the optional preview window."""

    gui_backend = _opencv_gui_backend(cv2.getBuildInformation())
    _validate_preview_environment(
        gui_backend=gui_backend,
        environment=os.environ,
        platform_name=sys.platform,
    )
    try:
        cv2.namedWindow(PREVIEW_WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    except cv2.error as error:
        raise InferenceError(
            "unable to open desktop preview window: {}".format(error)
        ) from error


def _show_preview_frame(frame) -> bool:
    """Display one annotated BGR frame and return true for q/Esc stop keys."""

    try:
        cv2.imshow(PREVIEW_WINDOW_NAME, frame)
        key = int(cv2.waitKey(1)) & 0xFF
    except cv2.error as error:
        raise InferenceError(
            "desktop preview failed while displaying a frame: {}".format(error)
        ) from error
    return key in (ord("q"), ord("Q"), 27)


def _close_preview_window() -> None:
    """Close the named preview without hiding cleanup failures."""

    try:
        cv2.destroyWindow(PREVIEW_WINDOW_NAME)
    except cv2.error as error:
        print(
            "Warning: unable to close desktop preview window cleanly: {}".format(
                error
            ),
            file=sys.stderr,
        )


def main(arguments: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(arguments)
    reader = None
    writer = None
    capture = None
    preview_open = False
    statistics = SequenceStatistics()
    processed_frame_count = 0
    try:
        if args.max_frames is not None and args.max_frames < 1:
            raise InferenceError("--max-frames must be at least 1")
        if args.duration_seconds is not None and args.duration_seconds <= 0.0:
            raise InferenceError("--duration-seconds must be positive")
        if args.process_every_frame and args.source.isdigit():
            raise InferenceError(
                "--process-every-frame applies only to offline video files"
            )
        protected_inputs = {}
        if not args.source.isdigit():
            protected_inputs["source"] = Path(args.source)
        for name in ("config", "checkpoint", "onnx", "onnx_report"):
            value = getattr(args, name)
            if value is not None:
                protected_inputs[name] = value
        validate_output_paths(
            protected_inputs=protected_inputs,
            outputs={
                "output_video": args.output_video,
                "output_csv": args.output_csv,
                "output_jsonl": args.output_jsonl,
            },
        )
        if args.display:
            _open_preview_window()
            preview_open = True
        if args.onnx is not None:
            if args.onnx_report is None:
                raise InferenceError("--onnx-report is required with --onnx")
            if args.config is not None or args.device is not None:
                raise InferenceError("--config and --device apply only to --checkpoint")
            ort_predictor = OnnxRuntimePredictor.from_files(
                args.onnx, args.onnx_report
            )
            predict_frame = lambda image: ort_predictor.predict_rgb_image(
                image, confidence_threshold=args.confidence_threshold
            )
            strategy = args.strategy or ort_predictor.contract.selection_strategy
            max_match_distance = (
                args.max_match_distance
                if args.max_match_distance is not None
                else ort_predictor.contract.max_match_distance_pixels
            )
            max_lost_frames = ort_predictor.contract.max_lost_frames
            allowed_class_ids = ort_predictor.contract.allowed_class_ids
            runtime_name = "onnxruntime"
            model_sha256 = ort_predictor.contract.onnx_sha256
        else:
            if args.config is None:
                raise InferenceError("--config is required with --checkpoint")
            if args.onnx_report is not None:
                raise InferenceError("--onnx-report applies only to --onnx")
            try:
                from fomo_servo.config import ConfigurationError, load_config
                from fomo_servo.models import ModelConfigurationError
                from fomo_servo.runtime import RuntimeDeviceError

                config = load_config(args.config)
                request = (
                    config.training.device if args.device is None else args.device
                )
                model, device = load_inference_model(
                    config, args.checkpoint, request
                )
            except (
                ConfigurationError,
                ModelConfigurationError,
                RuntimeDeviceError,
            ) as error:
                raise InferenceError(
                    "unable to initialize PyTorch backend: {}".format(error)
                ) from error
            predict_frame = lambda image: predict_rgb_image(
                model,
                image,
                config=config,
                device=device,
                confidence_threshold=args.confidence_threshold,
            )
            strategy = args.strategy or config.postprocess.selection_strategy
            max_match_distance = (
                args.max_match_distance
                if args.max_match_distance is not None
                else config.postprocess.max_match_distance_pixels
            )
            max_lost_frames = config.postprocess.max_lost_frames
            allowed_class_ids = config.postprocess.allowed_class_ids
            runtime_name = "pytorch"
            model_sha256 = None
        capture = _open_source(args.source)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
        args.output_video.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(args.output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            raise InferenceError("unable to open output video: {}".format(args.output_video))
        tracker = TargetTracker(
            strategy=strategy,
            max_match_distance=max_match_distance,
            max_lost_frames=max_lost_frames,
            allowed_class_ids=allowed_class_ids,
        )
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="", encoding="utf-8") as csv_file, args.output_jsonl.open("w", encoding="utf-8") as jsonl_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
            csv_writer.writeheader()
            reader_type = (
                SequentialFrameReader if args.process_every_frame else LatestFrameReader
            )
            reader = reader_type(capture).start()
            capture = None
            started_monotonic = time.monotonic()
            while True:
                packet = reader.buffer.get(timeout=0.5)
                if packet is None:
                    if reader.finished.is_set():
                        if reader.error is not None:
                            raise InferenceError(
                                "video reader failed: {}".format(reader.error)
                            ) from reader.error
                        break
                    if (
                        args.duration_seconds is not None
                        and time.monotonic() - started_monotonic
                        >= args.duration_seconds
                    ):
                        break
                    continue
                rgb = cv2.cvtColor(packet.frame, cv2.COLOR_BGR2RGB)
                prediction = predict_frame(rgb)
                tracking = tracker.update(prediction.detections)
                statistics.update(tracking.status, tracking.detection, width, height)
                processed_frame_count += 1
                selected = tracking.detection
                class_id = class_name = confidence = original_x = original_y = normalized_x = normalized_y = None
                if selected is not None:
                    normalized_x, normalized_y = normalize_centroid(selected.original_x, selected.original_y, width, height)
                    class_id, class_name, confidence = selected.class_id, selected.class_name, selected.confidence
                    original_x, original_y = selected.original_x, selected.original_y
                row = {
                    "runtime": runtime_name, "model_sha256": model_sha256,
                    "frame_index": packet.frame_index, "timestamp": packet.timestamp, "status": tracking.status,
                    "class_id": class_id, "class_name": class_name, "confidence": confidence,
                    "original_x": original_x, "original_y": original_y,
                    "normalized_x": normalized_x, "normalized_y": normalized_y,
                    "detection_count": len(prediction.detections), "lost_frames": tracking.lost_frames,
                }
                csv_writer.writerow(row)
                jsonl_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                output_frame = packet.frame.copy()
                for detection in prediction.detections:
                    cv2.circle(output_frame, (round(detection.original_x), round(detection.original_y)), 5, (0, 255, 0), -1)
                if selected is not None:
                    cv2.circle(output_frame, (round(selected.original_x), round(selected.original_y)), 9, (0, 165, 255), 2)
                cv2.drawMarker(output_frame, (width // 2, height // 2), (255, 0, 255), cv2.MARKER_CROSS, 20, 2)
                writer.write(output_frame)
                if args.display and _show_preview_frame(output_frame):
                    break
                if (
                    args.max_frames is not None
                    and processed_frame_count >= args.max_frames
                ):
                    break
                if (
                    args.duration_seconds is not None
                    and time.monotonic() - started_monotonic
                    >= args.duration_seconds
                ):
                    break
            if processed_frame_count == 0:
                raise InferenceError("video source produced no decodable frames")
    except (
        InferenceError,
        OrtPredictorError,
        OutputPathError,
        PreprocessingError,
        PostprocessError,
        ValueError,
    ) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    finally:
        if reader is not None:
            reader.stop()
        elif capture is not None:
            capture.release()
        if writer is not None:
            writer.release()
        if preview_open:
            _close_preview_window()
    print(json.dumps(statistics.summary().__dict__, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
