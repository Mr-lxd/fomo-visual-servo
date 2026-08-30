"""Checkpoint-selection v2 Stage B threshold locking and test isolation.

This module deliberately keeps threshold selection and locked test evaluation
separate.  Validation logits may be swept; the locked test path accepts one
checkpoint and one threshold and calls the no-sweep evaluator only.
"""

from __future__ import annotations

import csv
import json
import pickle
from dataclasses import dataclass, replace
from math import isfinite
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import torch

from fomo_servo.config import ProjectConfig
from fomo_servo.evaluation.epoch_snapshots import collect_split_logits
from fomo_servo.evaluation.validation import (
    ValidationReport,
    evaluate_logit_collection_at_threshold,
)
from fomo_servo.experiments import dataset_content_manifest, git_commit_sha
from fomo_servo.inference import load_inference_model
from fomo_servo.metrics import CentroidEvaluation, ThresholdSweepResult
from fomo_servo.training.snapshots import (
    SnapshotError,
    config_fingerprint,
    load_epoch_snapshot,
    sha256_file,
)


PROTOCOL_VERSION = "checkpoint_selection_v2_stage_b"
EVALUATOR_PROTOCOL_VERSION = "centroid_f1_grid_fp32_no_test_sweep_v1"
TIE_BREAKING_RULE = "maximum centroid_f1; exact ties choose lower threshold"


class StageBProtocolError(RuntimeError):
    """Raised when a threshold artifact or locked test manifest is unsafe."""


@dataclass(frozen=True)
class ThresholdTuningResult:
    """Validation threshold sweep reduced to a deterministic locked choice."""

    selected_threshold: float
    selected_metrics: CentroidEvaluation
    threshold_results: tuple[dict[str, object], ...]


@dataclass(frozen=True)
class LockedTestManifest:
    """Validated paths and scalar fields needed by one locked test evaluation."""

    raw: Mapping[str, object]
    manifest_path: Path
    checkpoint_path: Path
    source_snapshot_path: Path
    threshold_tuning_artifact_path: Path
    selected_epoch: int
    selected_threshold: float
    test_split: str


def tune_validation_threshold(
    sweep: ThresholdSweepResult,
    *,
    threshold_grid: Sequence[float],
    objective: str = "centroid_f1",
) -> ThresholdTuningResult:
    """Select a validation threshold using the existing sweep semantics.

    The sweep has one ``CentroidEvaluation`` per threshold.  Selection is
    deterministic: highest ``centroid_f1`` wins and exact ties choose the
    numerically lower threshold.  No test data is accepted by this function.
    """

    if objective != "centroid_f1":
        raise StageBProtocolError(
            "Stage B threshold objective must be 'centroid_f1', got '{}'".format(objective)
        )
    if not threshold_grid:
        raise StageBProtocolError("threshold_grid must contain at least one threshold")
    normalized_grid = tuple(float(value) for value in threshold_grid)
    if any(not isfinite(value) or not 0.0 <= value <= 1.0 for value in normalized_grid):
        raise StageBProtocolError("threshold_grid values must be finite probabilities")
    if len(set(normalized_grid)) != len(normalized_grid):
        raise StageBProtocolError("threshold_grid must not contain duplicate thresholds")
    missing = [value for value in normalized_grid if value not in sweep.results]
    if missing:
        raise StageBProtocolError(
            "threshold sweep is missing configured thresholds: {}".format(missing)
        )
    selected_threshold = min(
        normalized_grid,
        key=lambda value: (-sweep.results[value].centroid_f1, value),
    )
    threshold_results = tuple(
        {
            "threshold": value,
            "precision": sweep.results[value].centroid_precision,
            "recall": sweep.results[value].centroid_recall,
            "f1": sweep.results[value].centroid_f1,
            "true_positives": sweep.results[value].true_positives,
            "false_positives": sweep.results[value].false_positives,
            "false_negatives": sweep.results[value].false_negatives,
            "per_class_precision_recall_f1": {
                name: dict(metrics)
                for name, metrics in sweep.results[value].per_class_precision_recall_f1.items()
            },
        }
        for value in normalized_grid
    )
    return ThresholdTuningResult(
        selected_threshold=selected_threshold,
        selected_metrics=sweep.results[selected_threshold],
        threshold_results=threshold_results,
    )


def build_threshold_tuning_artifact(
    *,
    candidate_path: Path,
    source_snapshot_path: Path,
    source_epoch: int,
    selection_metric: str,
    selection_metric_value: float,
    selection_split: str,
    tuning_split: str,
    threshold_grid: Sequence[float],
    sweep: ThresholdSweepResult,
    config_fingerprint: str,
    dataset_content_hash: str,
    git_commit_sha: str,
    device: str,
    objective: str = "centroid_f1",
) -> dict[str, object]:
    """Create a JSON-safe validation threshold artifact with full provenance."""

    if tuning_split != selection_split:
        raise StageBProtocolError(
            "default Stage B threshold tuning must use the selection split; got '{}' and '{}'".format(
                selection_split, tuning_split
            )
        )
    result = tune_validation_threshold(
        sweep, threshold_grid=threshold_grid, objective=objective
    )
    return {
        "protocol": "checkpoint_selection_v2",
        "protocol_version": PROTOCOL_VERSION,
        "candidate_checkpoint_path": str(Path(candidate_path)),
        "candidate_checkpoint_sha256": sha256_file(Path(candidate_path)),
        "source_snapshot_path": str(Path(source_snapshot_path)),
        "source_snapshot_sha256": sha256_file(Path(source_snapshot_path)),
        "source_epoch": int(source_epoch),
        "checkpoint_selection": {
            "metric": selection_metric,
            "value": float(selection_metric_value),
            "split": selection_split,
        },
        "threshold_tuning_split": tuning_split,
        "objective": objective,
        "threshold_grid": [float(value) for value in threshold_grid],
        "threshold_results": list(result.threshold_results),
        "selected_threshold": result.selected_threshold,
        "selected_objective_value": result.selected_metrics.centroid_f1,
        "tie_breaking_rule": TIE_BREAKING_RULE,
        "device": str(device),
        "dtype": "float32",
        "config_fingerprint": config_fingerprint,
        "dataset_content_hash": dataset_content_hash,
        "git_commit_sha": git_commit_sha,
        "selection_and_threshold_tuning_shared": True,
        "threshold_tuning_independent": False,
        "threshold_sweep_performed": True,
    }


def build_locked_test_manifest(
    *,
    candidate_path: Path,
    source_snapshot_path: Path,
    selected_epoch: int,
    selected_threshold: float,
    selection_metric: str,
    selection_metric_value: float,
    selection_split: str,
    threshold_tuning_artifact_path: Path,
    test_split: str,
    dataset_content_hash: str,
    config_fingerprint: str,
    git_commit_sha: str,
    test_split_content_hash: Optional[str] = None,
) -> dict[str, object]:
    """Create the immutable input contract for one final test evaluation."""

    threshold = float(selected_threshold)
    if not isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise StageBProtocolError("selected_threshold must be a finite probability")
    if _canonical_split(test_split) == _canonical_split(selection_split):
        raise StageBProtocolError("test split must differ from validation selection split")
    candidate = str(Path(candidate_path))
    source = str(Path(source_snapshot_path))
    tuning = str(Path(threshold_tuning_artifact_path))
    manifest = {
        "protocol": "checkpoint_selection_v2",
        "protocol_version": PROTOCOL_VERSION,
        "checkpoint_path": candidate,
        "checkpoint_sha256": sha256_file(Path(candidate_path)),
        "candidate_checkpoint_path": candidate,
        "candidate_checkpoint_sha256": sha256_file(Path(candidate_path)),
        "source_snapshot_path": source,
        "source_snapshot_sha256": sha256_file(Path(source_snapshot_path)),
        "selected_epoch": int(selected_epoch),
        "selected_threshold": threshold,
        "threshold_source": "validation",
        "threshold_tuning_artifact_path": tuning,
        "threshold_tuning_artifact_sha256": sha256_file(Path(threshold_tuning_artifact_path)),
        "threshold_tuning_split": selection_split,
        "test_split": test_split,
        "dataset_content_hash": dataset_content_hash,
        "config_fingerprint": config_fingerprint,
        "git_commit_sha": git_commit_sha,
        "evaluator_protocol_version": EVALUATOR_PROTOCOL_VERSION,
        "final_test_threshold_sweep": False,
        "threshold_sweep_performed": False,
        "checkpoint_selection_source": "validation",
        "selection_metric": selection_metric,
        "selection_metric_value": float(selection_metric_value),
        "selection_split": selection_split,
        "selection_and_threshold_tuning_shared": True,
        "threshold_tuning_independent": False,
    }
    if test_split_content_hash is not None:
        manifest["test_split_content_hash"] = test_split_content_hash
    return manifest


def write_json_artifact(path: Path, payload: Mapping[str, object]) -> Path:
    """Write one deterministic JSON artifact and preserve write errors."""

    destination = Path(path)
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (OSError, TypeError, ValueError) as error:
        raise StageBProtocolError(
            "unable to write Stage B artifact '{}': {}".format(destination, error)
        ) from error
    return destination


def validate_locked_manifest(
    manifest: Path | Mapping[str, object],
    *,
    expected_config_fingerprint: Optional[str] = None,
    expected_dataset_content_hash: Optional[str] = None,
    expected_git_commit_sha: Optional[str] = None,
    expected_test_split_content_hash: Optional[str] = None,
) -> LockedTestManifest:
    """Validate hashes, candidate metadata, and split isolation before test."""

    manifest_path, payload = _read_manifest(manifest)
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        raise StageBProtocolError("unsupported or missing Stage B protocol_version")
    required = (
        "selected_epoch",
        "selected_threshold",
        "threshold_source",
        "threshold_tuning_artifact_path",
        "threshold_tuning_artifact_sha256",
        "test_split",
        "dataset_content_hash",
        "config_fingerprint",
        "git_commit_sha",
        "evaluator_protocol_version",
        "final_test_threshold_sweep",
        "selection_metric",
        "selection_metric_value",
        "selection_split",
        "threshold_tuning_split",
    )
    for field in required:
        if field not in payload:
            raise StageBProtocolError("locked test manifest is missing '{}'".format(field))
    checkpoint_text = _same_string_alias(
        payload, "checkpoint_path", "candidate_checkpoint_path", "checkpoint"
    )
    source_text = _same_string_alias(
        payload, "source_snapshot_path", "source_snapshot"
    )
    if not checkpoint_text:
        raise StageBProtocolError("locked test manifest must identify one checkpoint")
    if not source_text:
        raise StageBProtocolError("locked test manifest must identify one source snapshot")
    _reject_multiple_candidates(payload)
    if payload.get("final_test_threshold_sweep") is not False or payload.get(
        "threshold_sweep_performed", False
    ) is not False:
        raise StageBProtocolError("locked test manifest has threshold sweep enabled")
    if payload.get("threshold_source") != "validation":
        raise StageBProtocolError("locked test threshold_source must be 'validation'")
    if payload.get("checkpoint_selection_source") != "validation":
        raise StageBProtocolError(
            "locked test checkpoint_selection_source must be 'validation'"
        )
    if payload.get("selection_metric") != "centroid_pr_auc_macro":
        raise StageBProtocolError(
            "locked test must use centroid_pr_auc_macro as the primary selection metric"
        )
    selection_split = _required_text(payload["selection_split"], "selection_split")
    tuning_split = _required_text(payload["threshold_tuning_split"], "threshold_tuning_split")
    test_split = _required_text(payload["test_split"], "test_split")
    if _canonical_split(selection_split) != _canonical_split(tuning_split):
        raise StageBProtocolError(
            "selection and threshold tuning must use the same validation split"
        )
    if _canonical_split(test_split) in {
        _canonical_split(selection_split),
        _canonical_split(tuning_split),
    }:
        raise StageBProtocolError("locked test and threshold tuning must use a different split")
    selected_epoch = payload.get("selected_epoch")
    if isinstance(selected_epoch, bool) or not isinstance(selected_epoch, int) or selected_epoch <= 0:
        raise StageBProtocolError("selected_epoch must be a positive integer")
    threshold = payload.get("selected_threshold")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise StageBProtocolError("locked test manifest is missing selected_threshold")
    if not isfinite(float(threshold)) or not 0.0 <= float(threshold) <= 1.0:
        raise StageBProtocolError("selected_threshold must be a finite probability")

    checkpoint_path = _resolve_path(checkpoint_text, manifest_path.parent)
    source_path = _resolve_path(source_text, manifest_path.parent)
    tuning_path = _resolve_path(
        _required_text(payload["threshold_tuning_artifact_path"], "threshold_tuning_artifact_path"),
        manifest_path.parent,
    )
    _verify_hash(checkpoint_path, payload, "checkpoint_sha256", "checkpoint SHA-256")
    if payload.get("candidate_checkpoint_sha256") != payload.get("checkpoint_sha256"):
        raise StageBProtocolError("candidate checkpoint hash aliases disagree")
    _verify_hash(source_path, payload, "source_snapshot_sha256", "source snapshot SHA-256")
    _verify_hash(
        tuning_path,
        payload,
        "threshold_tuning_artifact_sha256",
        "threshold tuning artifact SHA-256",
    )
    candidate = _load_torch_mapping(checkpoint_path, "candidate checkpoint")
    if candidate.get("checkpoint_kind") != "inference_candidate":
        raise StageBProtocolError("locked checkpoint is not an inference_candidate")
    if candidate.get("weights_only") is not True or candidate.get("resumable") is not False:
        raise StageBProtocolError("locked checkpoint must be weights-only and non-resumable")
    if candidate.get("selected_epoch") != selected_epoch:
        raise StageBProtocolError("candidate selected_epoch does not match locked manifest")
    if candidate.get("selection_split") != selection_split:
        raise StageBProtocolError("candidate selection split does not match locked manifest")
    if candidate.get("selection_metric") != payload.get("selection_metric"):
        raise StageBProtocolError("candidate selection metric does not match locked manifest")
    if Path(str(candidate.get("source_snapshot", ""))).name != source_path.name:
        raise StageBProtocolError("candidate source snapshot metadata does not match manifest")
    if candidate.get("source_snapshot_sha256") != payload.get("source_snapshot_sha256"):
        raise StageBProtocolError("candidate source snapshot SHA-256 metadata mismatch")
    if candidate.get("selection_metric_value") != payload.get("selection_metric_value"):
        raise StageBProtocolError("candidate selection metric value does not match manifest")

    source = _load_epoch_snapshot_for_manifest(source_path)
    if source.get("epoch") != selected_epoch:
        raise StageBProtocolError("source snapshot epoch does not match locked manifest")
    for field in (
        "model_metadata",
        "config_fingerprint",
        "dataset_content_hash",
        "git_commit_sha",
        "seed",
        "augmentation_preset",
        "checkpoint_threshold",
    ):
        if candidate.get(field) != source.get(field):
            raise StageBProtocolError(
                "source snapshot metadata mismatch for '{}'".format(field)
            )

    tuning = _load_json_mapping(tuning_path, "threshold tuning artifact")
    if tuning.get("candidate_checkpoint_sha256") != sha256_file(checkpoint_path):
        raise StageBProtocolError("threshold artifact candidate hash does not match checkpoint")
    if tuning.get("source_snapshot_sha256") != sha256_file(source_path):
        raise StageBProtocolError("threshold artifact source snapshot hash does not match")
    if tuning.get("selected_threshold") != float(threshold):
        raise StageBProtocolError("threshold tuning artifact does not match locked threshold")
    if tuning.get("threshold_tuning_split") != tuning_split:
        raise StageBProtocolError("threshold tuning artifact split does not match manifest")
    if tuning.get("dtype") != "float32":
        raise StageBProtocolError("threshold tuning artifact must use float32")
    if tuning.get("config_fingerprint") != payload.get("config_fingerprint"):
        raise StageBProtocolError("threshold artifact config fingerprint does not match manifest")
    if tuning.get("dataset_content_hash") != payload.get("dataset_content_hash"):
        raise StageBProtocolError("threshold artifact dataset hash does not match manifest")
    if tuning.get("git_commit_sha") != payload.get("git_commit_sha"):
        raise StageBProtocolError("threshold artifact Git commit does not match manifest")
    checkpoint_selection = tuning.get("checkpoint_selection")
    if not isinstance(checkpoint_selection, Mapping):
        raise StageBProtocolError("threshold artifact is missing checkpoint selection metadata")
    if checkpoint_selection.get("metric") != payload.get("selection_metric"):
        raise StageBProtocolError("threshold artifact selection metric does not match manifest")
    if checkpoint_selection.get("split") != selection_split:
        raise StageBProtocolError("threshold artifact selection split does not match manifest")

    _compare_expected("config fingerprint", payload.get("config_fingerprint"), expected_config_fingerprint)
    _compare_expected("dataset content hash", payload.get("dataset_content_hash"), expected_dataset_content_hash)
    _compare_expected("Git commit", payload.get("git_commit_sha"), expected_git_commit_sha)
    _compare_expected(
        "test split content hash",
        payload.get("test_split_content_hash"),
        expected_test_split_content_hash,
    )
    return LockedTestManifest(
        raw=payload,
        manifest_path=manifest_path,
        checkpoint_path=checkpoint_path,
        source_snapshot_path=source_path,
        threshold_tuning_artifact_path=tuning_path,
        selected_epoch=selected_epoch,
        selected_threshold=float(threshold),
        test_split=test_split,
    )


def run_locked_test(
    config: Optional[ProjectConfig],
    manifest_path: Path,
    device_request: str = "cpu",
    output_dir: Optional[Path] = None,
    *,
    threshold_sweep_requested: bool = False,
    checkpoint_overrides: Optional[Sequence[Path]] = None,
) -> dict[str, object]:
    """Run exactly one locked test evaluation and write final-test artifacts."""

    if threshold_sweep_requested:
        raise StageBProtocolError("locked test rejects threshold sweep requests")
    if checkpoint_overrides:
        raise StageBProtocolError(
            "locked test accepts exactly one manifest checkpoint; checkpoint overrides are forbidden"
        )
    if config is None:
        raise StageBProtocolError("a validated project config is required for locked test evaluation")
    expected_manifest = dataset_content_manifest(
        config.dataset.root,
        config.dataset.train_split,
        config.dataset.validation_split,
    )
    try:
        raw_manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise StageBProtocolError(
            "unable to read locked test manifest before dataset verification: {}".format(error)
        ) from error
    if not isinstance(raw_manifest, dict) or not isinstance(raw_manifest.get("test_split"), str):
        raise StageBProtocolError("locked test manifest must declare test_split")
    test_manifest = dataset_content_manifest(
        config.dataset.root,
        str(raw_manifest["test_split"]),
        str(raw_manifest["test_split"]),
    )
    manifest = validate_locked_manifest(
        manifest_path,
        expected_config_fingerprint=config_fingerprint(config),
        expected_dataset_content_hash=str(expected_manifest["dataset_content_hash"]),
        expected_git_commit_sha=git_commit_sha(config.source_path.parent),
        expected_test_split_content_hash=str(test_manifest["dataset_content_hash"]),
    )
    if manifest.selected_epoch != 58:
        raise StageBProtocolError("formal locked test is restricted to selected epoch 58")
    if manifest.checkpoint_path.name != "best_centroid_pr_auc_macro.pt":
        raise StageBProtocolError(
            "formal locked test requires best_centroid_pr_auc_macro.pt"
        )
    model, device = load_inference_model(config, manifest.checkpoint_path, device_request)
    collection = collect_split_logits(config, model, device, manifest.test_split)
    fixed_config = replace(
        config.postprocess, inference_threshold=manifest.selected_threshold
    )
    report = evaluate_logit_collection_at_threshold(
        logits=collection.logits,
        targets=collection.targets,
        transforms=collection.transforms,
        ground_truths=collection.ground_truths,
        class_names=config.dataset.class_names,
        stride=config.model.output_stride,
        postprocess_config=fixed_config,
        evaluation_config=config.evaluation,
    )
    payload = _final_test_payload(config, manifest, device, report, collection)
    destination = Path(output_dir) if output_dir is not None else Path(manifest_path).parent
    write_final_test_artifacts(destination, payload)
    return payload


def write_final_test_artifacts(
    output_dir: Path, payload: Mapping[str, object]
) -> tuple[Path, Path]:
    """Write final-test-only files without modifying selection artifacts."""

    destination = Path(output_dir)
    json_path = write_json_artifact(destination / "final_test_metrics.json", payload)
    csv_path = destination / "final_test_metrics.csv"
    _write_final_csv(csv_path, payload)
    return json_path, csv_path


def _final_test_payload(
    config: ProjectConfig,
    manifest: LockedTestManifest,
    device: torch.device,
    report: ValidationReport,
    collection: object,
) -> dict[str, object]:
    metrics = report.centroid_metrics
    per_class = {
        name: dict(values) for name, values in metrics.per_class_precision_recall_f1.items()
    }
    macro_f1 = (
        sum(float(values["f1"]) for values in per_class.values()) / len(per_class)
        if per_class
        else 0.0
    )
    return {
        "protocol": "checkpoint_selection_v2",
        "protocol_version": PROTOCOL_VERSION,
        "evaluator_protocol_version": EVALUATOR_PROTOCOL_VERSION,
        "checkpoint_path": str(manifest.checkpoint_path),
        "checkpoint_sha256": manifest.raw["checkpoint_sha256"],
        "selected_epoch": manifest.selected_epoch,
        "threshold": manifest.selected_threshold,
        "threshold_source": "validation",
        "checkpoint_selection_source": "validation",
        "test_split": manifest.test_split,
        "threshold_sweep_performed": False,
        "final_test_threshold_sweep": False,
        "grid_precision": report.grid_metrics.grid_precision,
        "grid_recall": report.grid_metrics.grid_recall,
        "grid_f1": report.grid_metrics.grid_f1,
        "centroid_precision": metrics.centroid_precision,
        "centroid_recall": metrics.centroid_recall,
        "centroid_f1": metrics.centroid_f1,
        "macro_f1": macro_f1,
        "per_class_precision_recall_f1": per_class,
        "mean_localization_error_pixels": metrics.mean_localization_error_pixels,
        "median_localization_error_pixels": metrics.median_localization_error_pixels,
        "mean_count_bias": metrics.mean_count_bias,
        "count_mae": metrics.mean_absolute_count_error,
        "detection_count": metrics.true_positives + metrics.false_positives,
        "dataset_content_hash": manifest.raw["dataset_content_hash"],
        "test_split_content_hash": manifest.raw.get("test_split_content_hash"),
        "config_fingerprint": manifest.raw["config_fingerprint"],
        "git_commit_sha": manifest.raw["git_commit_sha"],
        "device": str(device),
        "dtype": "float32",
        "selection_and_threshold_tuning_shared": True,
        "threshold_tuning_independent": False,
    }


def _write_final_csv(path: Path, payload: Mapping[str, object]) -> None:
    scalar_keys = (
        "selected_epoch",
        "threshold",
        "grid_precision",
        "grid_recall",
        "grid_f1",
        "centroid_precision",
        "centroid_recall",
        "centroid_f1",
        "macro_f1",
        "mean_localization_error_pixels",
        "median_localization_error_pixels",
        "mean_count_bias",
        "count_mae",
        "detection_count",
    )
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=scalar_keys)
            writer.writeheader()
            writer.writerow({key: payload[key] for key in scalar_keys})
    except (OSError, KeyError, ValueError) as error:
        raise StageBProtocolError("unable to write final test CSV '{}': {}".format(path, error)) from error


def _read_manifest(manifest: Path | Mapping[str, object]) -> tuple[Path, dict[str, object]]:
    if isinstance(manifest, Mapping):
        return Path.cwd() / "locked_test_protocol.json", dict(manifest)
    path = Path(manifest)
    return path, _load_json_mapping(path, "locked test manifest")


def _load_json_mapping(path: Path, label: str) -> dict[str, object]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise StageBProtocolError("unable to read {} '{}': {}".format(label, path, error)) from error
    if not isinstance(payload, dict):
        raise StageBProtocolError("{} must be a JSON object".format(label))
    return payload


def _load_torch_mapping(path: Path, label: str) -> dict[str, object]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, ValueError, pickle.UnpicklingError) as error:
        raise StageBProtocolError("unable to load {} '{}': {}".format(label, path, error)) from error
    if not isinstance(payload, dict):
        raise StageBProtocolError("{} must contain a mapping".format(label))
    return payload


def _load_epoch_snapshot_for_manifest(path: Path) -> dict[str, object]:
    try:
        return load_epoch_snapshot(path)
    except (SnapshotError, OSError, RuntimeError, ValueError) as error:
        raise StageBProtocolError("source snapshot validation failed: {}".format(error)) from error


def _verify_hash(path: Path, payload: Mapping[str, object], field: str, label: str) -> None:
    expected = payload.get(field)
    if not isinstance(expected, str) or len(expected) != 64:
        raise StageBProtocolError("locked manifest is missing {}".format(label))
    try:
        actual = sha256_file(path)
    except SnapshotError as error:
        raise StageBProtocolError("{} file validation failed: {}".format(label, error)) from error
    if actual.lower() != expected.lower():
        raise StageBProtocolError("{} does not match locked manifest".format(label))


def _resolve_path(value: str, base: Path) -> Path:
    raw = Path(value)
    if raw.is_absolute():
        return raw
    candidates = (base / raw, Path.cwd() / raw)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def _same_string_alias(payload: Mapping[str, object], *names: str) -> str:
    values = [payload[name] for name in names if name in payload]
    if any(isinstance(value, (list, tuple)) for value in values):
        raise StageBProtocolError("locked test must identify one checkpoint, not multiple candidates")
    strings = [value for value in values if isinstance(value, str) and value]
    if values and len(strings) != len(values):
        raise StageBProtocolError("locked test path fields must be non-empty strings")
    if strings and any(value != strings[0] for value in strings[1:]):
        raise StageBProtocolError("locked test path aliases disagree")
    return strings[0] if strings else ""


def _reject_multiple_candidates(payload: Mapping[str, object]) -> None:
    for name in ("checkpoints", "candidate_checkpoints"):
        if name not in payload:
            continue
        value = payload[name]
        if not isinstance(value, (list, tuple)) or len(value) != 1:
            raise StageBProtocolError("locked test must accept one checkpoint only")


def _required_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise StageBProtocolError("{} must be a non-empty string".format(field))
    return value


def _canonical_split(value: str) -> str:
    return {"val": "validation", "valid": "validation"}.get(value, value)


def _compare_expected(label: str, actual: object, expected: Optional[str]) -> None:
    if expected is not None and actual != expected:
        raise StageBProtocolError(
            "{} does not match current configuration/provenance".format(label)
        )


__all__ = [
    "EVALUATOR_PROTOCOL_VERSION",
    "LockedTestManifest",
    "PROTOCOL_VERSION",
    "StageBProtocolError",
    "ThresholdTuningResult",
    "build_locked_test_manifest",
    "build_threshold_tuning_artifact",
    "run_locked_test",
    "tune_validation_threshold",
    "validate_locked_manifest",
    "write_final_test_artifacts",
    "write_json_artifact",
]
