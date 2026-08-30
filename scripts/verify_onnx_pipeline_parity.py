"""Verify formal PyTorch/ONNX Runtime parity for local images and videos."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping, Optional, Sequence

import cv2

from fomo_servo.deployment.onnx_export import (
    OnnxExportError,
    load_checkpoint_model,
    load_export_contract,
    sha256_file,
)
from fomo_servo.inference import (
    InferenceError,
    OnnxRuntimePredictor,
    OrtPredictorError,
    OutputPathError,
    read_rgb_image,
    validate_output_paths,
)
from fomo_servo.inference.parity import PipelineParityError, compare_rgb_image_pipeline
from fomo_servo.postprocess import PostprocessError


def build_parser() -> argparse.ArgumentParser:
    """Build the dataset-independent local image/video parity CLI."""

    parser = argparse.ArgumentParser(
        description="Compare formal PyTorch and ONNX Runtime complete inference pipelines."
    )
    parser.add_argument("--export-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--onnx-report", type=Path, required=True)
    parser.add_argument("--image", type=Path, action="append", default=[])
    parser.add_argument("--video", type=Path, action="append", default=[])
    parser.add_argument("--max-video-frames", type=int, default=32)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def main(arguments: Optional[Sequence[str]] = None) -> int:
    """Run parity for every requested local image and decoded video frame."""

    args = build_parser().parse_args(arguments)
    try:
        if not args.image and not args.video:
            raise PipelineParityError("at least one --image or --video is required")
        if args.max_video_frames <= 0:
            raise PipelineParityError("--max-video-frames must be positive")
        protected_inputs = {
            "export_config": args.export_config,
            "checkpoint": args.checkpoint,
            "onnx": args.onnx,
            "onnx_report": args.onnx_report,
        }
        protected_inputs.update(
            {"image_{}".format(index): path for index, path in enumerate(args.image)}
        )
        protected_inputs.update(
            {"video_{}".format(index): path for index, path in enumerate(args.video)}
        )
        validate_output_paths(
            protected_inputs=protected_inputs,
            outputs={"output_json": args.output_json},
        )
        export_contract = load_export_contract(args.export_config)
        model, checkpoint_provenance = load_checkpoint_model(
            export_contract, args.checkpoint
        )
        predictor = OnnxRuntimePredictor.from_files(args.onnx, args.onnx_report)
        _validate_contract_pair(export_contract, predictor, args.checkpoint)
        records: list[dict[str, object]] = []
        for image_path in args.image:
            records.append(
                {
                    "kind": "image",
                    "source": str(image_path),
                    "frame_index": None,
                    **compare_rgb_image_pipeline(
                        model,
                        predictor,
                        read_rgb_image(image_path),
                        pytorch_contract=export_contract,
                        logits_rtol=export_contract.parity_rtol,
                        logits_atol=export_contract.parity_atol,
                        detection_atol=1e-4,
                    ),
                }
            )
        for video_path in args.video:
            records.extend(
                _compare_video(
                    model,
                    predictor,
                    video_path,
                    max_frames=args.max_video_frames,
                    logits_rtol=export_contract.parity_rtol,
                    logits_atol=export_contract.parity_atol,
                    pytorch_contract=export_contract,
                )
            )
        payload: dict[str, object] = {
            "passed": all(bool(record["passed"]) for record in records),
            "checkpoint": checkpoint_provenance,
            "onnx_sha256": predictor.contract.onnx_sha256,
            "validation_threshold": predictor.contract.validation_threshold,
            "record_count": len(records),
            "records": records,
        }
    except (
        InferenceError,
        OnnxExportError,
        OrtPredictorError,
        OutputPathError,
        PipelineParityError,
        PostprocessError,
        OSError,
        ValueError,
    ) as error:
        print("Error: {}".format(error), file=sys.stderr)
        return 1
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    args.output_json.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if bool(payload["passed"]) else 1


def _compare_video(
    model: object,
    predictor: OnnxRuntimePredictor,
    path: Path,
    *,
    max_frames: int,
    logits_rtol: float,
    logits_atol: float,
    pytorch_contract: object,
) -> list[dict[str, object]]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise PipelineParityError("unable to open parity video: {}".format(path))
    expected_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    records: list[dict[str, object]] = []
    try:
        frame_index = 0
        while frame_index < max_frames:
            success, frame_bgr = capture.read()
            if not success:
                if expected_frames > 0 and frame_index < expected_frames:
                    raise PipelineParityError(
                        "video decode stopped after {} frames; expected {}: {}".format(
                            frame_index, expected_frames, path
                        )
                    )
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            records.append(
                {
                    "kind": "video_frame",
                    "source": str(path),
                    "frame_index": frame_index,
                    **compare_rgb_image_pipeline(
                        model,
                        predictor,
                        frame_rgb,
                        pytorch_contract=pytorch_contract,
                        logits_rtol=logits_rtol,
                        logits_atol=logits_atol,
                        detection_atol=1e-4,
                    ),
                }
            )
            frame_index += 1
    finally:
        capture.release()
    if not records:
        raise PipelineParityError("parity video contains no decodable frames: {}".format(path))
    return records


def _validate_contract_pair(
    export_contract: object,
    predictor: OnnxRuntimePredictor,
    checkpoint_path: Optional[Path] = None,
) -> None:
    config_path = getattr(export_contract, "config_path", None)
    expected = {
        "artifact_name": getattr(export_contract, "artifact_name"),
        "source_experiment_config": getattr(
            export_contract, "source_experiment_config"
        ),
        "source_experiment_config_sha256": getattr(
            export_contract, "source_experiment_config_sha256"
        ),
        "export_config_file": (
            Path(config_path).name
            if config_path is not None
            else getattr(export_contract, "export_config_file")
        ),
        "export_config_sha256": (
            sha256_file(Path(config_path))
            if config_path is not None
            else getattr(export_contract, "export_config_sha256")
        ),
        "checkpoint_file": (
            checkpoint_path.name
            if checkpoint_path is not None
            else getattr(export_contract, "checkpoint_file")
        ),
        "checkpoint_sha256": getattr(export_contract, "checkpoint_sha256"),
        "checkpoint_epoch": getattr(export_contract, "checkpoint_epoch"),
        "checkpoint_seed": getattr(export_contract, "checkpoint_seed"),
        "parameter_count": getattr(
            export_contract,
            "checkpoint_parameter_count",
            getattr(export_contract, "parameter_count", None),
        ),
        "config_fingerprint": getattr(
            export_contract,
            "checkpoint_config_fingerprint",
            getattr(export_contract, "config_fingerprint", None),
        ),
        "validation_threshold": getattr(export_contract, "validation_threshold"),
        "validation_threshold_usage": "provenance_only_raw_logits_export",
        "onnx_checker": "passed",
        "parity_passed": True,
        "parity_seed": getattr(export_contract, "parity_seed"),
        "parity_rtol": getattr(export_contract, "parity_rtol"),
        "parity_atol": getattr(export_contract, "parity_atol"),
        "onnx_opset": getattr(export_contract, "opset"),
        "input_name": getattr(export_contract, "input_name"),
        "input_shape": tuple(getattr(export_contract, "input_shape")),
        "input_dtype": getattr(export_contract, "input_dtype"),
        "input_color_order": getattr(export_contract, "input_color_order"),
        "input_value_range": tuple(
            getattr(export_contract, "input_value_range")
        ),
        "output_name": getattr(export_contract, "output_name"),
        "output_shape": tuple(getattr(export_contract, "output_shape")),
        "output_dtype": getattr(export_contract, "output_dtype"),
        "output_semantic": getattr(export_contract, "output_semantic"),
        "output_stride": getattr(export_contract, "output_stride"),
        "class_names": tuple(getattr(export_contract, "class_names")),
        "confidence_threshold": getattr(export_contract, "confidence_threshold"),
        "class_thresholds": _canonical_class_thresholds(
            getattr(export_contract, "class_thresholds")
        ),
        "component_mode": getattr(export_contract, "component_mode"),
        "confidence_mode": getattr(export_contract, "confidence_mode"),
        "selection_strategy": getattr(export_contract, "selection_strategy"),
        "max_match_distance_pixels": getattr(
            export_contract, "max_match_distance_pixels"
        ),
        "max_lost_frames": getattr(export_contract, "max_lost_frames"),
        "allowed_class_ids": getattr(export_contract, "allowed_class_ids"),
    }
    actual = {
        "artifact_name": predictor.contract.artifact_name,
        "source_experiment_config": predictor.contract.source_experiment_config,
        "source_experiment_config_sha256": (
            predictor.contract.source_experiment_config_sha256
        ),
        "export_config_file": predictor.contract.export_config_file,
        "export_config_sha256": predictor.contract.export_config_sha256,
        "checkpoint_file": predictor.contract.checkpoint_file,
        "checkpoint_sha256": predictor.contract.checkpoint_sha256,
        "checkpoint_epoch": predictor.contract.checkpoint_epoch,
        "checkpoint_seed": predictor.contract.checkpoint_seed,
        "parameter_count": predictor.contract.parameter_count,
        "config_fingerprint": predictor.contract.config_fingerprint,
        "validation_threshold": predictor.contract.validation_threshold,
        "validation_threshold_usage": predictor.contract.validation_threshold_usage,
        "onnx_checker": predictor.contract.onnx_checker,
        "parity_passed": predictor.contract.parity_passed,
        "parity_seed": predictor.contract.parity_seed,
        "parity_rtol": predictor.contract.parity_rtol,
        "parity_atol": predictor.contract.parity_atol,
        "onnx_opset": predictor.contract.opset,
        "input_name": predictor.contract.input_name,
        "input_shape": predictor.contract.input_shape,
        "input_dtype": predictor.contract.input_dtype,
        "input_color_order": predictor.contract.input_color_order,
        "input_value_range": predictor.contract.input_value_range,
        "output_name": predictor.contract.output_name,
        "output_shape": predictor.contract.output_shape,
        "output_dtype": predictor.contract.output_dtype,
        "output_semantic": predictor.contract.output_semantic,
        "output_stride": predictor.contract.output_stride,
        "class_names": predictor.contract.class_names,
        "confidence_threshold": predictor.contract.confidence_threshold,
        "class_thresholds": _canonical_class_thresholds(
            predictor.contract.class_thresholds
        ),
        "component_mode": predictor.contract.component_mode,
        "confidence_mode": predictor.contract.confidence_mode,
        "selection_strategy": predictor.contract.selection_strategy,
        "max_match_distance_pixels": predictor.contract.max_match_distance_pixels,
        "max_lost_frames": predictor.contract.max_lost_frames,
        "allowed_class_ids": predictor.contract.allowed_class_ids,
    }
    if actual != expected:
        raise PipelineParityError(
            "export config and ONNX report contract mismatch: expected {}, got {}".format(
                expected, actual
            )
        )


def _canonical_class_thresholds(value: object) -> object:
    if value is None or isinstance(value, Mapping):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(value)
    return value


if __name__ == "__main__":
    raise SystemExit(main())
