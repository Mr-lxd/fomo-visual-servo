"""Tune the locked epoch-58 candidate threshold on validation only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

import torch

from fomo_servo.config import ConfigurationError, load_config
from fomo_servo.evaluation import (
    StageBProtocolError,
    build_locked_test_manifest,
    build_threshold_tuning_artifact,
    collect_split_logits,
    write_json_artifact,
)
from fomo_servo.experiments import dataset_content_manifest, git_commit_sha
from fomo_servo.inference import InferenceError, load_inference_model
from fomo_servo.metrics import sweep_confidence_thresholds
from fomo_servo.models import ModelConfigurationError, build_fomo_model, describe_model
from fomo_servo.runtime import RuntimeDeviceError
from fomo_servo.training.snapshots import (
    SnapshotError,
    config_fingerprint,
    load_epoch_snapshot,
    sha256_file,
    validate_snapshot_compatibility,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tune the Stage B confidence threshold on the validation split."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    return parser


def main(arguments: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(arguments)
    try:
        config = load_config(args.config)
        candidate_path = args.candidate
        candidate = _load_mapping(candidate_path)
        _validate_formal_candidate(config, candidate_path, candidate)
        source_path = candidate_path.parent / "epoch_snapshots" / str(candidate["source_snapshot"])
        source = load_epoch_snapshot(source_path)
        expected_model_metadata = describe_model(config, build_fomo_model(config))
        content_manifest = dataset_content_manifest(
            config.dataset.root,
            config.dataset.train_split,
            config.dataset.validation_split,
        )
        expected_config = config_fingerprint(config)
        expected_dataset = str(content_manifest["dataset_content_hash"])
        validate_snapshot_compatibility(
            source,
            expected_model_metadata=expected_model_metadata,
            expected_config_fingerprint=expected_config,
            expected_dataset_content_hash=expected_dataset,
        )
        model, device = load_inference_model(config, candidate_path, args.device)
        selection_split = config.evaluation.checkpoint_selection.split
        if selection_split != config.dataset.validation_split:
            raise StageBProtocolError(
                "Stage B default requires checkpoint selection split to be validation"
            )
        collection = collect_split_logits(config, model, device, selection_split)
        sweep = sweep_confidence_thresholds(
            logits=collection.logits,
            transforms=collection.transforms,
            ground_truths=collection.ground_truths,
            class_names=config.dataset.class_names,
            stride=config.model.output_stride,
            thresholds=config.evaluation.checkpoint_selection.threshold_grid,
            matching_mode=config.evaluation.matching_mode,
            max_distance_pixels=config.evaluation.max_distance_pixels,
            class_thresholds=config.postprocess.class_thresholds,
            component_mode=config.postprocess.component_mode,
            confidence_mode=config.postprocess.confidence_mode,
        )
        current_git = git_commit_sha(config.source_path.parent)
        artifact = build_threshold_tuning_artifact(
            candidate_path=candidate_path,
            source_snapshot_path=source_path,
            source_epoch=int(candidate["selected_epoch"]),
            selection_metric=str(candidate["selection_metric"]),
            selection_metric_value=float(candidate["selection_metric_value"]),
            selection_split=selection_split,
            tuning_split=selection_split,
            threshold_grid=config.evaluation.checkpoint_selection.threshold_grid,
            sweep=sweep,
            config_fingerprint=expected_config,
            dataset_content_hash=expected_dataset,
            git_commit_sha=current_git,
            device=str(device),
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = write_json_artifact(
            args.output_dir / "threshold_tuning.json", artifact
        )
        test_split = "test"
        test_manifest = dataset_content_manifest(
            config.dataset.root, test_split, test_split
        )
        manifest = build_locked_test_manifest(
            candidate_path=candidate_path,
            source_snapshot_path=source_path,
            selected_epoch=int(candidate["selected_epoch"]),
            selected_threshold=float(artifact["selected_threshold"]),
            selection_metric=str(candidate["selection_metric"]),
            selection_metric_value=float(candidate["selection_metric_value"]),
            selection_split=selection_split,
            threshold_tuning_artifact_path=artifact_path,
            test_split=test_split,
            dataset_content_hash=expected_dataset,
            config_fingerprint=expected_config,
            git_commit_sha=current_git,
            test_split_content_hash=str(test_manifest["dataset_content_hash"]),
        )
        manifest_path = write_json_artifact(
            args.output_dir / "locked_test_protocol.json", manifest
        )
        selected_result = next(
            item
            for item in artifact["threshold_results"]
            if item["threshold"] == artifact["selected_threshold"]
        )
        print(
            json.dumps(
                {
                    "threshold_tuning": str(artifact_path),
                    "locked_manifest": str(manifest_path),
                    "selected_threshold": artifact["selected_threshold"],
                    "precision": selected_result["precision"],
                    "recall": selected_result["recall"],
                    "f1": artifact["selected_objective_value"],
                    "device": str(device),
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
        StageBProtocolError,
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
        raise StageBProtocolError("unable to load candidate '{}': {}".format(path, error)) from error
    if not isinstance(payload, dict):
        raise StageBProtocolError("candidate checkpoint must contain a mapping")
    return payload


def _validate_formal_candidate(
    config, candidate_path: Path, candidate: dict[str, object]
) -> None:
    if candidate_path.name != "best_centroid_pr_auc_macro.pt":
        raise StageBProtocolError(
            "Stage B requires best_centroid_pr_auc_macro.pt as the formal candidate"
        )
    if candidate.get("checkpoint_kind") != "inference_candidate":
        raise StageBProtocolError("formal candidate is not an inference_candidate")
    if candidate.get("weights_only") is not True or candidate.get("resumable") is not False:
        raise StageBProtocolError("formal candidate must be weights-only and non-resumable")
    if candidate.get("selection_metric") != config.evaluation.checkpoint_selection.metric:
        raise StageBProtocolError("candidate selection metric is not the configured primary metric")
    if candidate.get("selected_epoch") != 58:
        raise StageBProtocolError("Stage B is locked to formal candidate source epoch 58")
    source_path = candidate_path.parent / "epoch_snapshots" / str(candidate.get("source_snapshot", ""))
    if candidate.get("source_snapshot_sha256") != sha256_file(source_path):
        raise StageBProtocolError("candidate source snapshot SHA-256 metadata does not match")
    if candidate.get("selection_split") != config.evaluation.checkpoint_selection.split:
        raise StageBProtocolError("candidate selection split does not match the locked validation split")
    summary_path = candidate_path.parent / "checkpoint_selection_summary.json"
    if not summary_path.is_file():
        summary_path = candidate_path.parent / "selection_summary.json"
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise StageBProtocolError("unable to read formal checkpoint selection summary: {}".format(error)) from error
    selected = summary.get("selected")
    if not isinstance(selected, dict) or selected.get("epoch") != 58:
        raise StageBProtocolError("formal selection summary does not select epoch 58")
    if selected.get("centroid_pr_auc_macro") != candidate.get("selection_metric_value"):
        raise StageBProtocolError("candidate metadata does not match selection summary")


if __name__ == "__main__":
    raise SystemExit(main())
