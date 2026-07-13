"""FP32 offline evaluation and selection of v2 weights-only epoch snapshots."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

from fomo_servo.config import ConfigurationError, load_config
from fomo_servo.evaluation import (
    CheckpointSelectionError,
    collect_split_logits,
    evaluate_collected_logits,
    select_best_epoch_report,
    validate_calibration_request,
)
from fomo_servo.experiments import ExperimentMetadataError, dataset_content_manifest
from fomo_servo.inference import InferenceError, load_inference_model
from fomo_servo.models import ModelConfigurationError, build_fomo_model, describe_model
from fomo_servo.runtime import RuntimeDeviceError
from fomo_servo.training.snapshots import (
    SnapshotError,
    config_fingerprint,
    load_epoch_snapshot,
    snapshot_epoch_from_filename,
    validate_snapshot_compatibility,
    write_inference_candidate,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate FOMO epoch snapshots in FP32 and publish inference candidates."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--split", default=None, help="Selection split; defaults to YAML value.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--no-write-candidates", action="store_true", help="Report metrics without creating candidates."
    )
    return parser


def main(arguments: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(arguments)
    try:
        config = load_config(args.config)
        selection_split = (
            config.evaluation.checkpoint_selection.split if args.split is None else args.split
        )
        snapshots = sorted(
            args.snapshot_dir.glob("epoch_*_weights.pt"),
            key=lambda path: snapshot_epoch_from_filename(path.name),
        )
        if not snapshots:
            raise CheckpointSelectionError(
                "no epoch snapshots found in '{}'".format(args.snapshot_dir)
            )
        expected_model_metadata = describe_model(config, build_fomo_model(config))
        expected_config_fingerprint = config_fingerprint(config)
        content_manifest = dataset_content_manifest(
            config.dataset.root,
            config.dataset.train_split,
            config.dataset.validation_split,
        )
        expected_dataset_content_hash = str(content_manifest["dataset_content_hash"])
        reports = []
        for snapshot in snapshots:
            payload = load_epoch_snapshot(snapshot)
            validate_snapshot_compatibility(
                payload,
                expected_model_metadata=expected_model_metadata,
                expected_config_fingerprint=expected_config_fingerprint,
                expected_dataset_content_hash=expected_dataset_content_hash,
            )
            model, device = load_inference_model(config, snapshot, args.device)
            collection = collect_split_logits(config, model, device, selection_split)
            reports.append(
                evaluate_collected_logits(
                    config,
                    collection,
                    epoch=payload["epoch"],
                    source_snapshot=snapshot.name,
                    checkpoint_path=snapshot,
                ).as_dict()
            )
        primary = select_best_epoch_report(
            reports, metric=config.evaluation.checkpoint_selection.metric
        )
        sweep = select_best_epoch_report(
            reports, metric="max_centroid_f1_over_thresholds"
        )
        calibration = _calibrate_if_enabled(config, primary, args.snapshot_dir, args.device, selection_split)
        _write_outputs(
            args.output_dir,
            reports,
            primary,
            sweep,
            calibration,
            config,
            selection_split,
        )
        if not args.no_write_candidates:
            _write_candidates(
                args.snapshot_dir,
                args.snapshot_dir.parent,
                primary,
                sweep,
                selection_split,
            )
    except (
        ConfigurationError,
        CheckpointSelectionError,
        InferenceError,
        ModelConfigurationError,
        RuntimeDeviceError,
        SnapshotError,
        ExperimentMetadataError,
        OSError,
        RuntimeError,
        ValueError,
    ) as error:
        print("Error: {}".format(error), file=sys.stderr)
        return 1
    return 0


def _write_candidates(
    snapshot_dir: Path,
    output_dir: Path,
    primary: dict[str, object],
    sweep: dict[str, object],
    selection_split: str,
) -> None:
    for report, filename, metric, value_key in (
        (
            primary,
            "best_centroid_pr_auc_macro.pt",
            "centroid_pr_auc_macro",
            "centroid_pr_auc_macro",
        ),
        (
            sweep,
            "best_sweep_centroid_f1.pt",
            "max_centroid_f1_over_thresholds",
            "sweep_centroid_f1",
        ),
    ):
        write_inference_candidate(
            source_snapshot=snapshot_dir / str(report["source_snapshot"]),
            destination=output_dir / filename,
            selection_metric=metric,
            selection_metric_value=float(report[value_key]),
            selection_split=selection_split,
            selection_details={
                "threshold_grid": report["pr_curves"]["per_class"] and [
                    point["threshold"]
                    for point in next(iter(report["pr_curves"]["per_class"].values()))["points"]
                ],
                "integration": report["pr_curves"]["integration"],
                "macro_effective_class_count": report["macro_effective_class_count"],
            },
        )


def _calibrate_if_enabled(config, primary, snapshot_dir, device_request, selection_split):
    optimistic = validate_calibration_request(config, selection_split=selection_split)
    if not config.evaluation.threshold_calibration.enabled:
        return {
            "enabled": False,
            "calibrated_threshold": config.evaluation.threshold_calibration.fallback_threshold,
            "calibration_is_optimistic": False,
        }
    snapshot = snapshot_dir / str(primary["source_snapshot"])
    model, device = load_inference_model(config, snapshot, device_request)
    collection = collect_split_logits(
        config, model, device, config.evaluation.threshold_calibration.split
    )
    calibration_report = evaluate_collected_logits(
        config,
        collection,
        epoch=int(primary["epoch"]),
        source_snapshot=snapshot.name,
        checkpoint_path=snapshot,
    )
    return {
        "enabled": True,
        "split": config.evaluation.threshold_calibration.split,
        "objective": "centroid_f1",
        "calibrated_threshold": calibration_report.sweep_threshold,
        "calibrated_f1": calibration_report.sweep_centroid_f1,
        "calibration_is_optimistic": optimistic,
    }


def _write_outputs(
    output_dir, reports, primary, sweep, calibration, config, selection_split
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    scalar_keys = [
        "epoch", "source_snapshot", "checkpoint_path", "validation_loss",
        "grid_precision", "grid_recall", "grid_f1", "fixed_centroid_precision",
        "fixed_centroid_recall", "fixed_centroid_f1", "sweep_centroid_precision",
        "sweep_centroid_recall", "sweep_centroid_f1", "sweep_threshold",
        "centroid_pr_auc_macro", "centroid_pr_auc_micro", "macro_effective_class_count",
        "mean_localization_error_pixels", "median_localization_error_pixels",
        "mean_count_bias", "count_mae", "fixed_detection_count",
    ]
    with (output_dir / "epoch_snapshot_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=scalar_keys)
        writer.writeheader()
        for report in reports:
            writer.writerow({key: report[key] for key in scalar_keys})
    details = {"selection_dtype": "float32", "reports": reports}
    (output_dir / "epoch_snapshot_metrics.json").write_text(
        json.dumps(details, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "checkpoint_selection_protocol": "v2",
        "selection_metric": config.evaluation.checkpoint_selection.metric,
        "selection_split": selection_split,
        "selected": primary,
        "best_sweep_centroid_f1": sweep,
        "calibration": calibration,
    }
    (output_dir / "selection_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    raise SystemExit(main())
