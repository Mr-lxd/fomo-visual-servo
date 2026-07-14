"""Tune the fixed focal epoch-58 threshold on validation only.

This script deliberately evaluates one already-selected checkpoint.  It uses
the Edge Impulse-compatible decoder and records both strict one-to-one and
legacy matching reports, while selecting only on strict one-to-one F1.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch

from fomo_servo.config import ConfigurationError, load_config
from fomo_servo.evaluation import collect_split_logits
from fomo_servo.evaluation.edge_impulse import (
    EdgeImpulseDetection,
    decode_edge_impulse_fomo,
    probabilities_from_logits,
)
from fomo_servo.evaluation.parity_reporting import (
    edge_ground_truths_from_local,
    serialize_edge_evaluation,
)
from fomo_servo.experiments import dataset_content_manifest, git_commit_sha
from fomo_servo.geometry import LetterboxTransform
from fomo_servo.inference import InferenceError, load_inference_model
from fomo_servo.models import ModelConfigurationError, build_fomo_model, describe_model
from fomo_servo.runtime import RuntimeDeviceError
from fomo_servo.training.snapshots import (
    SnapshotError,
    _sanitize_config,
    config_fingerprint,
    load_epoch_snapshot,
    sha256_file,
    validate_snapshot_compatibility,
)


PROTOCOL_VERSION = "stage_c1_focal_validation_threshold_v1"
MATCHING_DISTANCE = 0.2
EXPECTED_SOURCE_EPOCH = 58
PRE_STAGE_C_TRAINING_COMMIT = "82ebf19cb5946e820424cfb6077b95e1574a95d9"


class StageC1FocalError(RuntimeError):
    """Raised when the locked focal tuning protocol cannot be honored."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tune the focal epoch-58 threshold on validation only."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--expected-image-count", type=int, default=127)
    return parser


def select_best_threshold(
    threshold_results: Sequence[Mapping[str, object]],
) -> Mapping[str, object]:
    """Select the highest strict F1, breaking exact ties by lower threshold."""

    if not threshold_results:
        raise StageC1FocalError("threshold_results must not be empty")
    normalized = []
    for item in threshold_results:
        try:
            threshold = float(item["threshold"])
            strict_f1 = float(item["strict"]["f1"])  # type: ignore[index]
        except (KeyError, TypeError, ValueError) as error:
            raise StageC1FocalError(
                "each threshold result must contain threshold and strict.f1"
            ) from error
        if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise StageC1FocalError("threshold must be a finite probability")
        if not np.isfinite(strict_f1):
            raise StageC1FocalError("strict F1 must be finite")
        normalized.append((strict_f1, -threshold, item))
    return max(normalized, key=lambda value: (value[0], value[1]))[2]


def run_tuning(
    *,
    config_path: Path,
    dataset_root: Path,
    checkpoint: Path,
    output_dir: Path,
    device_request: str,
    expected_image_count: int,
) -> dict[str, object]:
    """Run the locked focal threshold sweep and write one independent artifact."""

    if output_dir.exists():
        raise StageC1FocalError("refusing to overwrite output directory: {}".format(output_dir))
    if not dataset_root.is_dir():
        raise StageC1FocalError("dataset root does not exist: {}".format(dataset_root))
    if expected_image_count <= 0:
        raise StageC1FocalError("expected_image_count must be positive")
    os.environ["FOMO_DATASET_ROOT"] = str(dataset_root)
    config = load_config(config_path)
    candidate = _load_mapping(checkpoint)
    if int(candidate.get("selected_epoch", -1)) != EXPECTED_SOURCE_EPOCH:
        raise StageC1FocalError("checkpoint is not the locked focal epoch-58 candidate")
    source_snapshot_name = str(candidate.get("source_snapshot", ""))
    source_snapshot = checkpoint.parent / "epoch_snapshots" / source_snapshot_name
    if not source_snapshot.is_file():
        raise StageC1FocalError("candidate source snapshot does not exist: {}".format(source_snapshot))
    if str(candidate.get("source_snapshot_sha256", "")).lower() != sha256_file(source_snapshot).lower():
        raise StageC1FocalError("candidate source snapshot SHA-256 metadata does not match")
    source_payload = load_epoch_snapshot(source_snapshot)
    expected_model_metadata = describe_model(config, build_fomo_model(config))
    expected_config = config_fingerprint(config)
    snapshot_config_fingerprint = str(source_payload["config_fingerprint"])
    compatibility_config_fingerprint = expected_config
    compatibility_mode = "current_config_schema"
    if snapshot_config_fingerprint != expected_config:
        legacy_config = _legacy_pre_stage_c_config_fingerprint(config)
        if snapshot_config_fingerprint != legacy_config:
            raise StageC1FocalError(
                "checkpoint config fingerprint differs from current and pre-Stage-C schemas"
            )
        if source_payload.get("git_commit_sha") != PRE_STAGE_C_TRAINING_COMMIT:
            raise StageC1FocalError(
                "legacy config fingerprint is only accepted for the locked focal training commit"
            )
        if config.loss.name != "focal_cross_entropy" or config.loss.object_weight != 1.0:
            raise StageC1FocalError(
                "pre-Stage-C checkpoint compatibility requires the unchanged focal loss config"
            )
        compatibility_config_fingerprint = legacy_config
        compatibility_mode = "pre_stage_c_schema_without_loss_object_weight"
    content_manifest = dataset_content_manifest(
        config.dataset.root,
        config.dataset.train_split,
        config.dataset.validation_split,
    )
    expected_dataset_hash = str(content_manifest["dataset_content_hash"])
    validate_snapshot_compatibility(
        source_payload,
        expected_model_metadata=expected_model_metadata,
        expected_config_fingerprint=compatibility_config_fingerprint,
        expected_dataset_content_hash=expected_dataset_hash,
    )
    model, device = load_inference_model(config, checkpoint, device_request)
    collection = collect_split_logits(
        config, model, device, config.dataset.validation_split
    )
    if len(collection.logits) != expected_image_count:
        raise StageC1FocalError(
            "validation image count is {}; expected {}".format(
                len(collection.logits), expected_image_count
            )
        )
    thresholds = tuple(config.evaluation.checkpoint_selection.threshold_grid)
    threshold_results = []
    for threshold in thresholds:
        strict, legacy = _evaluate_ei_collection(
            collection,
            class_names=config.dataset.class_names,
            input_size=config.model.input_size,
            threshold=float(threshold),
        )
        threshold_results.append(
            {
                "threshold": float(threshold),
                "strict": strict,
                "edge_impulse_legacy": legacy,
            }
        )
    selected = select_best_threshold(threshold_results)
    artifact = {
        "protocol": PROTOCOL_VERSION,
        "evaluator_protocol": "edge_impulse_centroid_distance_v1",
        "matching_mode": "strict_one_to_one",
        "legacy_matching_mode": "edge_impulse_legacy",
        "normalized_distance_threshold": MATCHING_DISTANCE,
        "objective": "strict centroid F1",
        "tie_break_rule": "maximum strict centroid F1; exact ties choose lower threshold",
        "validation_split": config.dataset.validation_split,
        "image_count": len(collection.logits),
        "dtype": "float32",
        "device": str(device),
        "threshold_grid": [float(value) for value in thresholds],
        "threshold_results": threshold_results,
        "selected_threshold": float(selected["threshold"]),
        "selected_strict": selected["strict"],
        "selected_edge_impulse_legacy": selected["edge_impulse_legacy"],
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": sha256_file(checkpoint),
            "source_epoch": EXPECTED_SOURCE_EPOCH,
            "source_snapshot": str(source_snapshot),
            "source_snapshot_sha256": sha256_file(source_snapshot),
            "checkpoint_metadata_git_commit": source_payload["git_commit_sha"],
            "config_fingerprint": snapshot_config_fingerprint,
        },
        "config_fingerprint": expected_config,
        "checkpoint_config_fingerprint": snapshot_config_fingerprint,
        "config_compatibility": {
            "mode": compatibility_mode,
            "validated_against": compatibility_config_fingerprint,
            "legacy_pre_stage_c_fingerprint": _legacy_pre_stage_c_config_fingerprint(config),
            "reason": "Stage C added LossConfig.object_weight; focal epoch-58 was trained before that schema field existed",
        },
        "dataset_content_hash": expected_dataset_hash,
        "evaluation_code_commit": git_commit_sha(config.source_path.parent),
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    artifact_path = output_dir / "threshold_tuning.json"
    artifact_path.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return artifact


def main(arguments: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(arguments)
    try:
        artifact = run_tuning(
            config_path=args.config,
            dataset_root=args.dataset_root,
            checkpoint=args.checkpoint,
            output_dir=args.output_dir,
            device_request=args.device,
            expected_image_count=args.expected_image_count,
        )
        selected = artifact["selected_strict"]
        print(
            json.dumps(
                {
                    "artifact": str(args.output_dir / "threshold_tuning.json"),
                    "selected_threshold": artifact["selected_threshold"],
                    "strict_precision": selected["precision"],
                    "strict_recall": selected["recall"],
                    "strict_f1": selected["f1"],
                    "device": artifact["device"],
                },
                ensure_ascii=False,
            )
        )
    except (
        ConfigurationError,
        InferenceError,
        ModelConfigurationError,
        RuntimeDeviceError,
        SnapshotError,
        StageC1FocalError,
        OSError,
        RuntimeError,
        ValueError,
    ) as error:
        print("Error: {}".format(error), file=sys.stderr)
        return 1
    return 0


def _load_mapping(path: Path) -> dict[str, object]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, ValueError) as error:
        raise StageC1FocalError("unable to load checkpoint '{}': {}".format(path, error)) from error
    if not isinstance(payload, dict):
        raise StageC1FocalError("checkpoint must contain a mapping")
    return payload


def _legacy_pre_stage_c_config_fingerprint(config: object) -> str:
    """Reproduce the locked pre-Stage-C schema fingerprint explicitly.

    Stage C added ``loss.object_weight`` to the configuration dataclass.  The
    focal baseline predates that field, so removing only this field reproduces
    its historical fingerprint without weakening the normal compatibility
    check for any other config or dataset mismatch.
    """

    canonical = _sanitize_config(asdict(config))
    loss = canonical.get("loss")
    if not isinstance(loss, dict):
        raise StageC1FocalError("configuration loss section is not a mapping")
    loss.pop("object_weight", None)
    encoded = json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _evaluate_ei_collection(
    collection: Any,
    *,
    class_names: Sequence[str],
    input_size: int,
    threshold: float,
) -> tuple[dict[str, object], dict[str, object]]:
    predictions = []
    image_sizes = []
    for logits, transform in zip(collection.logits, collection.transforms):
        decoded = decode_edge_impulse_fomo(
            probabilities_from_logits(logits),
            class_names=class_names,
            input_size=(input_size, input_size),
            threshold=threshold,
        )
        predictions.append(_inverse_detections(decoded, transform))
        if not isinstance(transform, LetterboxTransform):
            raise StageC1FocalError("validation transform is not LetterboxTransform")
        image_sizes.append((transform.original_width, transform.original_height))
    ground_truths = tuple(
        edge_ground_truths_from_local(items) for items in collection.ground_truths
    )
    strict = serialize_edge_evaluation(
        predictions=predictions,
        ground_truths=ground_truths,
        image_sizes=image_sizes,
        class_names=class_names,
        mode="strict_one_to_one",
    )
    legacy = serialize_edge_evaluation(
        predictions=predictions,
        ground_truths=ground_truths,
        image_sizes=image_sizes,
        class_names=class_names,
        mode="edge_impulse_legacy",
    )
    return strict, legacy


def _inverse_detections(
    detections: Sequence[EdgeImpulseDetection], transform: object
) -> tuple[EdgeImpulseDetection, ...]:
    if not isinstance(transform, LetterboxTransform):
        raise StageC1FocalError("EI detections require LetterboxTransform metadata")
    output = []
    for detection in detections:
        x, y = transform.inverse_point(*detection.input_centroid)
        x = float(np.clip(x, 0.0, transform.original_width - 1.0))
        y = float(np.clip(y, 0.0, transform.original_height - 1.0))
        output.append(replace(detection, original_centroid=(x, y)))
    return tuple(output)


__all__ = ["StageC1FocalError", "select_best_threshold", "run_tuning"]


if __name__ == "__main__":
    raise SystemExit(main())
