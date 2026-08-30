"""Run FOMO centroid inference on one image and write visual/JSON output."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

import cv2

from fomo_servo.inference import (
    InferenceError,
    OnnxRuntimePredictor,
    OrtPredictorError,
    OutputPathError,
    PreprocessingError,
    load_inference_model,
    predict_rgb_image,
    read_rgb_image,
    validate_output_paths,
)
from fomo_servo.postprocess import PostprocessError, select_target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict FOMO centroids for one image.")
    parser.add_argument("--config", type=Path)
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--checkpoint", type=Path)
    model_group.add_argument("--onnx", type=Path)
    parser.add_argument("--onnx-report", type=Path)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--confidence-threshold", type=float, default=None)
    parser.add_argument("--strategy", choices=("highest_confidence", "largest_component", "nearest_previous"), default=None)
    parser.add_argument("--max-match-distance", type=float, default=None)
    parser.add_argument("--class-id", type=int, action="append", dest="class_ids")
    parser.add_argument("--output-image", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def main(arguments: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(arguments)
    try:
        protected_inputs = {"image": args.image}
        for name in ("config", "checkpoint", "onnx", "onnx_report"):
            value = getattr(args, name)
            if value is not None:
                protected_inputs[name] = value
        validate_output_paths(
            protected_inputs=protected_inputs,
            outputs={
                "output_image": args.output_image,
                "output_json": args.output_json,
            },
        )
        if args.onnx is not None:
            if args.onnx_report is None:
                raise InferenceError("--onnx-report is required with --onnx")
            if args.config is not None or args.device is not None:
                raise InferenceError("--config and --device apply only to --checkpoint")
            ort_predictor = OnnxRuntimePredictor.from_files(
                args.onnx, args.onnx_report
            )
            prediction = ort_predictor.predict_rgb_image(
                read_rgb_image(args.image),
                confidence_threshold=args.confidence_threshold,
            )
            strategy = args.strategy or ort_predictor.contract.selection_strategy
            max_match_distance = (
                args.max_match_distance
                if args.max_match_distance is not None
                else ort_predictor.contract.max_match_distance_pixels
            )
            allowed_class_ids = (
                tuple(args.class_ids)
                if args.class_ids is not None
                else ort_predictor.contract.allowed_class_ids
            )
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
            prediction = predict_rgb_image(
                model,
                read_rgb_image(args.image),
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
            allowed_class_ids = args.class_ids or config.postprocess.allowed_class_ids
            runtime_name = "pytorch"
            model_sha256 = None
        selected = select_target(
            prediction.detections,
            strategy,
            max_match_distance=max_match_distance,
            allowed_class_ids=allowed_class_ids,
        )
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

    image_bgr = cv2.cvtColor(prediction.original_image, cv2.COLOR_RGB2BGR)
    height, width = prediction.original_image.shape[:2]
    cv2.drawMarker(image_bgr, (width // 2, height // 2), (255, 0, 255), cv2.MARKER_CROSS, 20, 2)
    for detection in prediction.detections:
        point = (round(detection.original_x), round(detection.original_y))
        selected_color = selected == detection
        color = (0, 165, 255) if selected_color else (0, 255, 0)
        cv2.circle(image_bgr, point, 9 if selected_color else 5, color, -1)
        cv2.putText(image_bgr, f"{detection.class_name} {detection.confidence:.2f}", (point[0] + 5, point[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    normalized_x = normalized_y = None
    if selected is not None:
        normalized_x = 2.0 * selected.original_x / width - 1.0
        normalized_y = 2.0 * selected.original_y / height - 1.0
        cv2.putText(
            image_bgr,
            f"target nx={normalized_x:.3f} ny={normalized_y:.3f}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 165, 255),
            2,
            cv2.LINE_AA,
        )
    payload = {
        "runtime": runtime_name,
        "model_sha256": model_sha256,
        "image": str(args.image),
        "image_width": width,
        "image_height": height,
        "detections": [item.as_dict() for item in prediction.detections],
        "selected_target": selected.as_dict() if selected is not None else None,
        "normalized_x": normalized_x,
        "normalized_y": normalized_y,
    }
    args.output_image.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output_image), image_bgr):
        print(f"Error: unable to write output image {args.output_image}", file=sys.stderr)
        return 1
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
