"""Evaluate the locked local epoch-58 candidate on parity-clean-v1 test data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import torch

from fomo_servo.config import ConfigurationError, ProjectConfig, load_config
from fomo_servo.datasets import YOLOv5FOMODataset
from fomo_servo.evaluation.edge_impulse import (
    EdgeImpulseDetection,
    decode_edge_impulse_fomo,
    probabilities_from_logits,
)
from fomo_servo.evaluation.parity_clean import (
    ParityCleanError,
    ParityCleanView,
    verify_parity_clean_view,
)
from fomo_servo.evaluation.parity_reporting import (
    edge_ground_truths_from_local,
    evaluate_local_parity,
    serialize_edge_evaluation,
)
from fomo_servo.inference import InferenceError, load_inference_model
from fomo_servo.metrics import ground_truths_from_boxes
from fomo_servo.postprocess import Detection, postprocess_logits


class LocalParityError(RuntimeError):
    """Raised when the explicit locked local parity protocol cannot be honored."""


def build_parser() -> argparse.ArgumentParser:
    """Build the no-sweep, explicit-cleaning-manifest CLI."""

    parser = argparse.ArgumentParser(
        description="Run one fixed-threshold local checkpoint parity evaluation on test."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--cleaning-manifest", type=Path, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--expected-image-count", type=int, default=63)
    return parser


def prepare_output_dir(path: Path) -> Path:
    """Create one new result directory and refuse to overwrite prior outputs."""

    output = Path(path)
    if output.exists():
        raise LocalParityError("refusing to overwrite existing output directory: {}".format(output))
    try:
        output.mkdir(parents=True, exist_ok=False)
    except OSError as error:
        raise LocalParityError("unable to create output directory '{}': {}".format(output, error)) from error
    return output


def run_local_parity(
    *,
    config: ProjectConfig,
    dataset_root: Path,
    checkpoint: Path,
    expected_checkpoint_sha256: str,
    cleaning_manifest: Mapping[str, object],
    threshold: float,
    device_request: str,
    expected_image_count: int,
) -> dict[str, object]:
    """Run local and EI-compatible reports once on the cleaned physical test view."""

    _validate_probability(threshold, "threshold")
    if expected_image_count <= 0:
        raise LocalParityError("expected_image_count must be positive")
    checkpoint_hash = _sha256_file(checkpoint)
    if checkpoint_hash.lower() != expected_checkpoint_sha256.lower():
        raise LocalParityError("checkpoint SHA-256 does not match --expected-checkpoint-sha256")

    view = ParityCleanView(dataset_root, cleaning_manifest, len(config.dataset.class_names))
    cleaning_hashes = verify_parity_clean_view(
        dataset_root, cleaning_manifest, len(config.dataset.class_names)
    )
    dataset = YOLOv5FOMODataset(
        root=dataset_root,
        split="test",
        input_size=config.model.input_size,
        stride=config.model.output_stride,
        class_mode=config.dataset.class_mode,
        merged_class_name=config.dataset.merged_class_name,
        collision_policy=config.dataset.collision_policy,
        augmentation=None,
        train_split=config.dataset.train_split,
        augmentation_seed=config.training.seed,
        label_loader=view.parse_label_file,
    )
    if len(dataset) != expected_image_count:
        raise LocalParityError(
            "cleaned test image count is {}; expected {}".format(len(dataset), expected_image_count)
        )
    model, device = load_inference_model(config, checkpoint, device_request)
    local_predictions: list[tuple[Detection, ...]] = []
    edge_predictions: list[tuple[EdgeImpulseDetection, ...]] = []
    ground_truths = []
    image_sizes: list[tuple[int, int]] = []
    image_metadata: list[dict[str, object]] = []
    model.eval()
    with torch.inference_mode():
        for sample in dataset:
            image = torch.from_numpy(sample.image).unsqueeze(0).to(device=device, dtype=torch.float32)
            logits = model(image)
            if logits.dtype != torch.float32:
                logits = logits.float()
            logits_cpu = logits.detach().cpu()
            local = postprocess_logits(
                logits_cpu,
                class_names=config.dataset.class_names,
                stride=config.model.output_stride,
                transforms=(sample.transform,),
                confidence_threshold=threshold,
                class_thresholds=config.postprocess.class_thresholds,
                component_mode=config.postprocess.component_mode,
                confidence_mode=config.postprocess.confidence_mode,
            )[0]
            edge = _inverse_edge_detections(
                decode_edge_impulse_fomo(
                    probabilities_from_logits(logits_cpu),
                    class_names=config.dataset.class_names,
                    input_size=(config.model.input_size, config.model.input_size),
                    threshold=threshold,
                ),
                sample.transform,
            )
            local_predictions.append(local)
            edge_predictions.append(edge)
            ground_truths.append(
                tuple(ground_truths_from_boxes(sample.original_boxes, config.dataset.class_names))
            )
            image_sizes.append((sample.transform.original_width, sample.transform.original_height))
            image_metadata.append(
                {
                    "image_path": sample.image_path.relative_to(dataset_root).as_posix(),
                    "label_path": sample.label_path.relative_to(dataset_root).as_posix(),
                    "original_size": [sample.transform.original_width, sample.transform.original_height],
                    "letterbox": {
                        "input_size": sample.transform.input_size,
                        "scale": sample.transform.scale,
                        "pad_left": sample.transform.pad_left,
                        "pad_top": sample.transform.pad_top,
                        "pad_right": sample.transform.pad_right,
                        "pad_bottom": sample.transform.pad_bottom,
                    },
                }
            )

    local_report = evaluate_local_parity(
        predictions=local_predictions,
        ground_truths=ground_truths,
        class_names=config.dataset.class_names,
        matching_mode=config.evaluation.matching_mode,
        max_distance_pixels=config.evaluation.max_distance_pixels,
    )
    edge_ground_truths = tuple(edge_ground_truths_from_local(items) for items in ground_truths)
    legacy_report = serialize_edge_evaluation(
        predictions=edge_predictions,
        ground_truths=edge_ground_truths,
        image_sizes=image_sizes,
        class_names=config.dataset.class_names,
        mode="edge_impulse_legacy",
    )
    strict_report = serialize_edge_evaluation(
        predictions=edge_predictions,
        ground_truths=edge_ground_truths,
        image_sizes=image_sizes,
        class_names=config.dataset.class_names,
        mode="strict_one_to_one",
    )
    for index, item in enumerate(image_metadata):
        item["ground_truths"] = [_ground_truth_dict(value) for value in ground_truths[index]]
        item["local_predictions"] = [value.as_dict() for value in local_predictions[index]]
        item["edge_impulse_predictions"] = [_edge_detection_dict(value) for value in edge_predictions[index]]
        item["local_current"] = local_report["image_results"][index]
        item["edge_impulse_legacy"] = legacy_report["image_results"][index]
        item["strict_one_to_one"] = strict_report["image_results"][index]

    return {
        "protocol": "edge-impulse-parity-local-v1",
        "split": "test",
        "image_count": len(dataset),
        "threshold": threshold,
        "dtype": "float32",
        "device": str(device),
        "checkpoint": {"path": str(checkpoint), "sha256": checkpoint_hash},
        "cleaning_manifest": {
            "protocol": cleaning_manifest.get("protocol"),
            "cleaning_view_hash": cleaning_hashes["cleaning_view_hash"],
            "cleaned_test_view_hash": cleaning_hashes["cleaned_test_view_hash"],
        },
        "preprocessing": {
            "input_size": config.model.input_size,
            "layout": "NCHW RGB float32 divided by 255",
            "resize": "repository letterbox",
            "padding_value": 114,
        },
        "local_current_evaluator": local_report,
        "edge_impulse_legacy_evaluator": legacy_report,
        "strict_one_to_one_evaluator": strict_report,
        "images": image_metadata,
    }


def _inverse_edge_detections(
    detections: Sequence[EdgeImpulseDetection], transform: object
) -> tuple[EdgeImpulseDetection, ...]:
    """Map EI input-pixel centroids through a :class:`LetterboxTransform`."""

    from fomo_servo.geometry import LetterboxTransform

    if not isinstance(transform, LetterboxTransform):
        raise LocalParityError("EI detections require LetterboxTransform metadata")
    output = []
    for detection in detections:
        original_x, original_y = transform.inverse_point(*detection.input_centroid)
        original_x = float(np.clip(original_x, 0.0, transform.original_width - 1.0))
        original_y = float(np.clip(original_y, 0.0, transform.original_height - 1.0))
        output.append(replace(detection, original_centroid=(original_x, original_y)))
    return tuple(output)


def _ground_truth_dict(value: object) -> dict[str, object]:
    from fomo_servo.metrics import GroundTruthCentroid

    if not isinstance(value, GroundTruthCentroid):
        raise LocalParityError("unexpected local ground-truth type")
    return {
        "class_id": value.class_id,
        "class_name": value.class_name,
        "original_centroid": [value.original_x, value.original_y],
        "original_bbox": [value.x_min, value.y_min, value.x_max, value.y_max],
    }


def _edge_detection_dict(value: EdgeImpulseDetection) -> dict[str, object]:
    return {
        "class_id": value.class_id,
        "class_name": value.class_name,
        "confidence": value.confidence,
        "input_bbox": list(value.input_bbox),
        "input_centroid": list(value.input_centroid),
        "original_centroid": list(value.original_centroid),
    }


def _sha256_file(path: Path) -> str:
    if not path.is_file():
        raise LocalParityError("checkpoint does not exist: {}".format(path))
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise LocalParityError("unable to hash checkpoint '{}': {}".format(path, error)) from error
    return digest.hexdigest()


def _load_manifest(path: Path) -> Mapping[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise LocalParityError("unable to read cleaning manifest '{}': {}".format(path, error)) from error
    if not isinstance(payload, Mapping):
        raise LocalParityError("cleaning manifest root must be a JSON object")
    return payload


def _validate_probability(value: float, name: str) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not np.isfinite(value) or not 0.0 <= float(value) <= 1.0:
        raise LocalParityError("{} must be a finite probability in [0,1]".format(name))


def main(arguments: Optional[Sequence[str]] = None) -> int:
    """Run one explicitly specified parity evaluation and save a JSON report."""

    args = build_parser().parse_args(arguments)
    try:
        if not args.dataset_root.is_dir():
            raise LocalParityError("dataset root does not exist: {}".format(args.dataset_root))
        # The YAML retains its portable environment-variable path; the explicit
        # CLI input supplies this machine's location for this command only.
        os.environ["FOMO_DATASET_ROOT"] = str(args.dataset_root)
        config = load_config(args.config)
        manifest = _load_manifest(args.cleaning_manifest)
        payload = run_local_parity(
            config=config,
            dataset_root=args.dataset_root,
            checkpoint=args.checkpoint,
            expected_checkpoint_sha256=args.expected_checkpoint_sha256,
            cleaning_manifest=manifest,
            threshold=args.threshold,
            device_request=args.device,
            expected_image_count=args.expected_image_count,
        )
        output_dir = prepare_output_dir(args.output_dir)
        report_path = output_dir / "parity_report.json"
        report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"report": str(report_path), "summary": {
            "local_f1": payload["local_current_evaluator"]["f1"],
            "edge_impulse_legacy_f1": payload["edge_impulse_legacy_evaluator"]["f1"],
            "strict_one_to_one_f1": payload["strict_one_to_one_evaluator"]["f1"],
        }}, ensure_ascii=False))
    except (ConfigurationError, ParityCleanError, LocalParityError, InferenceError, OSError, RuntimeError, ValueError) as error:
        print("Error: {}".format(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
