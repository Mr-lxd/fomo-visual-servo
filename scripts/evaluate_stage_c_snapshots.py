"""Scan Stage C validation snapshots and publish one validation-only summary."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch

from fomo_servo.config import ConfigurationError, load_config
from fomo_servo.evaluation import (
    CheckpointSelectionError,
    collect_split_logits,
    evaluate_collected_logits,
    select_best_epoch_report,
)
from fomo_servo.evaluation.edge_impulse import (
    EdgeImpulseDetection,
    decode_edge_impulse_fomo,
    probabilities_from_logits,
)
from fomo_servo.evaluation.parity_reporting import (
    edge_ground_truths_from_local,
    serialize_edge_evaluation,
)
from fomo_servo.experiments import dataset_content_manifest
from fomo_servo.geometry import LetterboxTransform
from fomo_servo.inference import load_inference_model
from fomo_servo.models import build_fomo_model, describe_model
from fomo_servo.runtime import RuntimeDeviceError
from fomo_servo.training.snapshots import (
    SnapshotError,
    config_fingerprint,
    load_epoch_snapshot,
    snapshot_epoch_from_filename,
    validate_snapshot_compatibility,
)


class StageCValidationError(RuntimeError):
    """Raised when a Stage C validation scan cannot honor its fixed protocol."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Scan Stage C snapshots on validation and save strict/EI metrics."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    return parser


def evaluate_ei_collection(
    collection: Any,
    *,
    class_names: Sequence[str],
    input_size: int,
    threshold: float,
) -> tuple[dict[str, object], dict[str, object]]:
    """Decode one FP32 validation logit collection and return strict/legacy reports."""

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
            raise StageCValidationError("validation transform is not LetterboxTransform")
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


def tune_strict_threshold(
    collection: Any,
    *,
    class_names: Sequence[str],
    input_size: int,
    thresholds: Sequence[float],
) -> tuple[float, dict[str, object], dict[str, object]]:
    """Tune strict one-to-one centroid F1 on validation only, with lower-threshold tie break."""

    candidates = []
    reports = []
    for threshold in thresholds:
        strict, legacy = evaluate_ei_collection(
            collection,
            class_names=class_names,
            input_size=input_size,
            threshold=float(threshold),
        )
        reports.append({"threshold": float(threshold), "strict": strict, "legacy": legacy})
        candidates.append((float(strict["f1"]), -float(threshold), float(threshold), strict, legacy))
    if not candidates:
        raise StageCValidationError("validation threshold grid is empty")
    _, _, threshold, strict, legacy = max(candidates, key=lambda item: (item[0], item[1]))
    return threshold, strict, legacy


def scan_stage_c(
    *, config_path: Path, snapshot_dir: Path, output_dir: Path, device_request: str
) -> dict[str, object]:
    """Run the fixed FP32 validation scan without touching test data."""

    if output_dir.exists():
        raise StageCValidationError("refusing to overwrite existing output directory: {}".format(output_dir))
    snapshots = sorted(
        snapshot_dir.glob("epoch_*_weights.pt"),
        key=lambda path: snapshot_epoch_from_filename(path.name),
    )
    if len(snapshots) != 60:
        raise StageCValidationError(
            "Stage C validation requires exactly 60 snapshots, found {}".format(len(snapshots))
        )
    config = load_config(config_path)
    expected_model_metadata = describe_model(config, build_fomo_model(config))
    expected_config_fingerprint = config_fingerprint(config)
    content_manifest = dataset_content_manifest(
        config.dataset.root,
        config.dataset.train_split,
        config.dataset.validation_split,
    )
    expected_dataset_hash = str(content_manifest["dataset_content_hash"])
    reports = []
    collections = {}
    for snapshot in snapshots:
        payload = load_epoch_snapshot(snapshot)
        validate_snapshot_compatibility(
            payload,
            expected_model_metadata=expected_model_metadata,
            expected_config_fingerprint=expected_config_fingerprint,
            expected_dataset_content_hash=expected_dataset_hash,
        )
        model, device = load_inference_model(config, snapshot, device_request)
        collection = collect_split_logits(
            config, model, device, config.evaluation.checkpoint_selection.split
        )
        report = evaluate_collected_logits(
            config,
            collection,
            epoch=payload["epoch"],
            source_snapshot=snapshot.name,
            checkpoint_path=snapshot,
        ).as_dict()
        reports.append(report)
        collections[int(payload["epoch"])] = collection

    primary = dict(
        select_best_epoch_report(
            reports, metric=config.evaluation.checkpoint_selection.metric
        )
    )
    selected_epoch = int(primary["epoch"])
    threshold, strict, legacy = tune_strict_threshold(
        collections[selected_epoch],
        class_names=config.dataset.class_names,
        input_size=config.model.input_size,
        thresholds=config.evaluation.checkpoint_selection.threshold_grid,
    )
    payload = {
        "protocol": "stage_c_validation_v1",
        "split": config.evaluation.checkpoint_selection.split,
        "dtype": "float32",
        "dataset_content_hash": expected_dataset_hash,
        "config": str(config_path),
        "loss": {
            "loss_type": config.loss.name,
            "background_weight": config.loss.background_weight,
            "object_weight": config.loss.object_weight,
            "class_weight_mode": config.loss.class_weight_mode,
            "per_class_weights_applied": False,
        },
        "selection": {
            "metric": config.evaluation.checkpoint_selection.metric,
            "selected_epoch": selected_epoch,
            "source_snapshot": primary["source_snapshot"],
            "centroid_pr_auc_macro": primary["centroid_pr_auc_macro"],
            "strict_validation_threshold": threshold,
        },
        "selected_strict_one_to_one": strict,
        "selected_edge_impulse_legacy": legacy,
        "epoch_reports": reports,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "stage_c_validation_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def _inverse_detections(
    detections: Sequence[EdgeImpulseDetection], transform: object
) -> tuple[EdgeImpulseDetection, ...]:
    if not isinstance(transform, LetterboxTransform):
        raise StageCValidationError("validation transform is not LetterboxTransform")
    result = []
    for item in detections:
        x, y = transform.inverse_point(*item.input_centroid)
        x = float(np.clip(x, 0.0, transform.original_width - 1.0))
        y = float(np.clip(y, 0.0, transform.original_height - 1.0))
        result.append(replace(item, original_centroid=(x, y)))
    return tuple(result)


def main(arguments: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(arguments)
    try:
        payload = scan_stage_c(
            config_path=args.config,
            snapshot_dir=args.snapshot_dir,
            output_dir=args.output_dir,
            device_request=args.device,
        )
        selection = payload["selection"]
        print(json.dumps(selection, ensure_ascii=False))
    except (
        ConfigurationError,
        CheckpointSelectionError,
        RuntimeDeviceError,
        SnapshotError,
        StageCValidationError,
        OSError,
        RuntimeError,
        ValueError,
    ) as error:
        print("Error: {}".format(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
