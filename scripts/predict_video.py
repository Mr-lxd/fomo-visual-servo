"""Run latest-frame FOMO video inference with CSV/JSONL telemetry."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

import cv2

from fomo_servo.config import ConfigurationError, load_config
from fomo_servo.inference import InferenceError, LatestFrameReader, load_inference_model, predict_rgb_image
from fomo_servo.metrics import SequenceStatistics, normalize_centroid
from fomo_servo.models import ModelConfigurationError
from fomo_servo.postprocess import PostprocessError, TargetTracker
from fomo_servo.runtime import RuntimeDeviceError


CSV_COLUMNS = (
    "frame_index", "timestamp", "status", "class_id", "class_name", "confidence",
    "original_x", "original_y", "normalized_x", "normalized_y", "detection_count", "lost_frames",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict FOMO centroids for a video or camera.")
    parser.add_argument("--source", required=True, help="Video path or camera index")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
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


def main(arguments: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(arguments)
    reader = None
    writer = None
    capture = None
    statistics = SequenceStatistics()
    try:
        config = load_config(args.config)
        request = config.training.device if args.device is None else args.device
        model, device = load_inference_model(config, args.checkpoint, request)
        capture = _open_source(args.source)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
        writer = cv2.VideoWriter(str(args.output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            raise InferenceError("unable to open output video: {}".format(args.output_video))
        tracker = TargetTracker(
            strategy=args.strategy or config.postprocess.selection_strategy,
            max_match_distance=(args.max_match_distance if args.max_match_distance is not None else config.postprocess.max_match_distance_pixels),
            max_lost_frames=config.postprocess.max_lost_frames,
            allowed_class_ids=config.postprocess.allowed_class_ids,
        )
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="", encoding="utf-8") as csv_file, args.output_jsonl.open("w", encoding="utf-8") as jsonl_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
            csv_writer.writeheader()
            reader = LatestFrameReader(capture).start()
            capture = None
            while True:
                packet = reader.buffer.get(timeout=0.5)
                if packet is None:
                    if reader.finished.is_set():
                        break
                    continue
                rgb = cv2.cvtColor(packet.frame, cv2.COLOR_BGR2RGB)
                prediction = predict_rgb_image(model, rgb, config=config, device=device, confidence_threshold=args.confidence_threshold)
                tracking = tracker.update(prediction.detections)
                statistics.update(tracking.status, tracking.detection, width, height)
                selected = tracking.detection
                class_id = class_name = confidence = original_x = original_y = normalized_x = normalized_y = None
                if selected is not None:
                    normalized_x, normalized_y = normalize_centroid(selected.original_x, selected.original_y, width, height)
                    class_id, class_name, confidence = selected.class_id, selected.class_name, selected.confidence
                    original_x, original_y = selected.original_x, selected.original_y
                row = {
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
    except (ConfigurationError, ModelConfigurationError, RuntimeDeviceError, InferenceError, PostprocessError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    finally:
        if reader is not None:
            reader.stop()
        elif capture is not None:
            capture.release()
        if writer is not None:
            writer.release()
    print(json.dumps(statistics.summary().__dict__, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
