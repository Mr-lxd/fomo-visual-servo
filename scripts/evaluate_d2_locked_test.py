"""Run the one-shot Stage D2 cleaned-test evaluation.

The command intentionally has no selection logic.  The protocol JSON names the
single epoch-40 snapshot and validation-selected threshold; this script only
verifies those identities and evaluates the cleaned test view once.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from fomo_servo.config import ConfigurationError, ProjectConfig, load_config
from fomo_servo.evaluation.parity_clean import ParityCleanError, verify_parity_clean_view
from fomo_servo.experiments import dataset_content_manifest, git_commit_sha
from fomo_servo.models import build_fomo_model, describe_model
from fomo_servo.training.snapshots import SnapshotError, config_fingerprint, load_epoch_snapshot


class D2LockedTestError(RuntimeError):
    """Raised when the immutable Stage D2 test protocol cannot be honored."""


_EVALUATORS = (
    ("local_current_evaluator", "local_current"),
    ("edge_impulse_legacy_evaluator", "edge_impulse_legacy"),
    ("strict_one_to_one_evaluator", "strict_one_to_one"),
)
_EXPECTED_EPOCH = 40
_EXPECTED_THRESHOLD = 0.40
_EXPECTED_CLEANING_VIEW_HASH = "35b915d3c926425777afb22b0ec684bfd0885f08bbb289133635bde4d584c41c"
_EXPECTED_CLEANED_TEST_HASH = "d52ee0ffd498a24b5f90e75d6bbecbedd289efab5fdafc3bfeb18ea1518a906c"
_EXPECTED_TEST_IMAGE_COUNT = 63


def build_parser() -> argparse.ArgumentParser:
    """Build an explicit, non-tuning D2 test CLI."""

    parser = argparse.ArgumentParser(
        description="Run the immutable Stage D2 cleaned-test protocol once."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    # These flags exist only to produce an explicit protocol error instead of
    # silently accepting a selection override.
    parser.add_argument("--threshold", dest="threshold_override", type=float, default=None)
    parser.add_argument("--checkpoint", dest="checkpoint_override", type=Path, default=None)
    parser.add_argument("--split", dest="split_override", default=None)
    parser.add_argument("--sweep", action="store_true", default=False)
    return parser


def validate_cli_invariants(
    *,
    threshold_override: Optional[float],
    checkpoint_override: Optional[Path],
    split_override: Optional[str],
    sweep_requested: bool,
) -> None:
    """Reject every command-line option that could retune the locked test."""

    if threshold_override is not None:
        raise D2LockedTestError(
            "threshold override is forbidden; use the validation-selected protocol threshold"
        )
    if checkpoint_override is not None:
        raise D2LockedTestError(
            "checkpoint override is forbidden; use the protocol-selected epoch-40 snapshot"
        )
    if split_override is not None:
        raise D2LockedTestError("split override is forbidden; the protocol split is test")
    if sweep_requested:
        raise D2LockedTestError("threshold sweep is forbidden for the locked D2 test")


def summarize_confidences(values: Sequence[float]) -> dict[str, object]:
    """Summarize fixed-test confidence values without changing predictions."""

    values_array = np.asarray(tuple(float(value) for value in values), dtype=np.float64)
    if values_array.size == 0:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "p50": None,
            "p90": None,
        }
    return {
        "count": int(values_array.size),
        "min": float(values_array.min()),
        "max": float(values_array.max()),
        "mean": float(values_array.mean()),
        "p50": round(float(np.percentile(values_array, 50)), 12),
        "p90": round(float(np.percentile(values_array, 90)), 12),
    }


def metric_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    """Flatten the three fixed evaluator reports for CSV serialization."""

    rows: list[dict[str, object]] = []
    for report_key, evaluator_name in _EVALUATORS:
        raw = report.get(report_key)
        if not isinstance(raw, Mapping):
            raise D2LockedTestError("parity report is missing {}".format(report_key))
        row = {
            "evaluator": evaluator_name,
            "matching_mode": raw.get("matching_mode"),
            "true_positives": raw.get("true_positives"),
            "false_positives": raw.get("false_positives"),
            "false_negatives": raw.get("false_negatives"),
            "precision": raw.get("precision"),
            "recall": raw.get("recall"),
            "f1": raw.get("f1"),
            "macro_f1": raw.get("macro_f1"),
            "prediction_count": raw.get("prediction_count"),
            "ground_truth_count": raw.get("ground_truth_count"),
            "mean_absolute_count_error": raw.get("mean_absolute_count_error"),
            "mean_count_bias": raw.get("mean_count_bias"),
            "mean_localization_error_pixels": raw.get("mean_localization_error_pixels"),
            "median_localization_error_pixels": raw.get("median_localization_error_pixels"),
            "mean_localization_error_normalized": raw.get(
                "mean_localization_error_normalized"
            ),
            "median_localization_error_normalized": raw.get(
                "median_localization_error_normalized"
            ),
        }
        rows.append(row)
    return rows


def _load_local_parity_module() -> Any:
    """Load the existing fixed local parity implementation from ``scripts``."""

    try:
        import evaluate_parity_local as module  # type: ignore[import-not-found]

        return module
    except ModuleNotFoundError:
        path = Path(__file__).with_name("evaluate_parity_local.py")
        spec = importlib.util.spec_from_file_location("evaluate_parity_local", path)
        if spec is None or spec.loader is None:
            raise D2LockedTestError("unable to load evaluate_parity_local.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


def _load_json(path: Path, description: str) -> Mapping[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise D2LockedTestError("unable to read {} '{}': {}".format(description, path, error)) from error
    if not isinstance(payload, Mapping):
        raise D2LockedTestError("{} '{}' must contain a JSON object".format(description, path))
    return payload


def _sha256_file(path: Path) -> str:
    if not path.is_file():
        raise D2LockedTestError("required file does not exist: {}".format(path))
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise D2LockedTestError("unable to hash '{}': {}".format(path, error)) from error
    return digest.hexdigest()


def _resolve_protocol_path(value: object, repository_root: Path, field_name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise D2LockedTestError("protocol field '{}' must be a non-empty path".format(field_name))
    path = Path(value)
    return path if path.is_absolute() else repository_root / path


def _require(protocol: Mapping[str, object], field: str) -> object:
    if field not in protocol:
        raise D2LockedTestError("locked protocol is missing '{}'".format(field))
    return protocol[field]


def _require_equal(protocol: Mapping[str, object], field: str, actual: object) -> None:
    expected = _require(protocol, field)
    if expected != actual:
        raise D2LockedTestError(
            "locked protocol mismatch for '{}': expected {!r}, got {!r}".format(
                field, expected, actual
            )
        )


def validate_threshold_artifact(tuning: Mapping[str, object]) -> None:
    """Validate the fixed validation-only selection metadata schema."""

    if tuning.get("split") != "val":
        raise D2LockedTestError("threshold tuning artifact split is not val")
    selection = tuning.get("selection")
    if not isinstance(selection, Mapping):
        raise D2LockedTestError("threshold tuning artifact is missing selection metadata")
    if selection.get("selected_epoch") != _EXPECTED_EPOCH:
        raise D2LockedTestError("threshold artifact selected epoch is not 40")
    if selection.get("strict_validation_threshold") != _EXPECTED_THRESHOLD:
        raise D2LockedTestError("threshold artifact selected threshold is not 0.40")
    if selection.get("metric") != "centroid_pr_auc_macro":
        raise D2LockedTestError("threshold artifact metric is not centroid_pr_auc_macro")


def validate_locked_protocol(
    protocol: Mapping[str, object],
    *,
    config: ProjectConfig,
    repository_root: Path,
    dataset_root: Path,
) -> dict[str, object]:
    """Validate all immutable D2 identities before model inference starts."""

    _require_equal(protocol, "protocol", "d2_locked_test_v1")
    _require_equal(protocol, "selected_epoch", _EXPECTED_EPOCH)
    _require_equal(protocol, "selected_threshold", _EXPECTED_THRESHOLD)
    _require_equal(protocol, "threshold_source", "validation")
    _require_equal(protocol, "test_split", "test")
    _require_equal(protocol, "dtype", "float32")
    _require_equal(protocol, "candidate_count", 1)
    _require_equal(protocol, "final_test_threshold_sweep", False)
    _require_equal(protocol, "threshold_sweep", False)
    if Path(config.dataset.root).resolve() != Path(dataset_root).resolve():
        raise D2LockedTestError("--dataset-root does not match the configured dataset root")
    if config.model.input_size != 192 or config.model.output_stride != 8:
        raise D2LockedTestError("D2 protocol requires the configured 192px stride-8 model")
    _require_equal(protocol, "config_fingerprint", config_fingerprint(config))

    content_manifest = dataset_content_manifest(
        dataset_root, config.dataset.train_split, config.dataset.validation_split
    )
    _require_equal(protocol, "dataset_content_hash", content_manifest["dataset_content_hash"])

    checkpoint = _resolve_protocol_path(_require(protocol, "checkpoint_path"), repository_root, "checkpoint_path")
    checkpoint_hash = _sha256_file(checkpoint)
    _require_equal(protocol, "checkpoint_sha256", checkpoint_hash)
    source_snapshot = _resolve_protocol_path(
        _require(protocol, "source_snapshot_path"), repository_root, "source_snapshot_path"
    )
    source_snapshot_hash = _sha256_file(source_snapshot)
    _require_equal(protocol, "source_snapshot_sha256", source_snapshot_hash)
    if checkpoint.resolve() != source_snapshot.resolve() or checkpoint_hash != source_snapshot_hash:
        raise D2LockedTestError("D2 protocol checkpoint must be the one selected epoch snapshot")

    try:
        snapshot = load_epoch_snapshot(source_snapshot)
    except SnapshotError as error:
        raise D2LockedTestError("selected D2 snapshot is invalid: {}".format(error)) from error
    if snapshot.get("epoch") != _EXPECTED_EPOCH:
        raise D2LockedTestError("selected snapshot payload is not epoch 40")
    for field in ("config_fingerprint", "dataset_content_hash"):
        if snapshot.get(field) != protocol.get(field):
            raise D2LockedTestError("selected snapshot '{}' does not match protocol".format(field))
    _require_equal(protocol, "training_git_commit", snapshot.get("git_commit_sha"))

    model_identity = _require(protocol, "model_identity")
    if not isinstance(model_identity, Mapping) or dict(model_identity) != dict(snapshot.get("model_metadata", {})):
        raise D2LockedTestError("selected snapshot model identity does not match protocol")
    try:
        model = build_fomo_model(config)
        actual_model_identity = describe_model(config, model)
    except (RuntimeError, ValueError, OSError) as error:
        raise D2LockedTestError("unable to reconstruct D2 model identity: {}".format(error)) from error
    if actual_model_identity != dict(model_identity):
        raise D2LockedTestError("configured model identity does not match selected snapshot")

    pretrained_source = _resolve_protocol_path(
        _require(protocol, "pretrained_source"), repository_root, "pretrained_source"
    )
    pretrained_hash = _sha256_file(pretrained_source)
    _require_equal(protocol, "pretrained_sha256", pretrained_hash)
    if config.model.pretrained_source is None or config.model.pretrained_source.resolve() != pretrained_source.resolve():
        raise D2LockedTestError("configured pretrained H5 path does not match protocol")
    if config.model.pretrained_sha256 != pretrained_hash:
        raise D2LockedTestError("configured pretrained H5 SHA-256 does not match protocol")
    load_report = model_identity.get("pretrained_load_report")
    if not isinstance(load_report, Mapping) or load_report.get("loaded_tensor_count") != 95:
        raise D2LockedTestError("protocol does not prove the expected 95 pretrained tensors were loaded")
    if load_report.get("missing_keys") != [] or load_report.get("unexpected_keys") != []:
        raise D2LockedTestError("pretrained load report contains missing or unexpected tensors")

    cleaning_manifest = _resolve_protocol_path(
        _require(protocol, "cleaning_manifest"), repository_root, "cleaning_manifest"
    )
    _require_equal(protocol, "cleaning_manifest_sha256", _sha256_file(cleaning_manifest))
    cleaning_payload = _load_json(cleaning_manifest, "cleaning manifest")
    try:
        cleaning_hashes = verify_parity_clean_view(
            dataset_root, cleaning_payload, len(config.dataset.class_names)
        )
    except ParityCleanError as error:
        raise D2LockedTestError("parity-clean manifest verification failed: {}".format(error)) from error
    _require_equal(protocol, "cleaning_view_hash", cleaning_hashes["cleaning_view_hash"])
    _require_equal(protocol, "cleaned_test_view_hash", cleaning_hashes["cleaned_test_view_hash"])
    if cleaning_hashes["cleaning_view_hash"] != _EXPECTED_CLEANING_VIEW_HASH:
        raise D2LockedTestError("cleaning view hash is not the approved parity-clean-v1 hash")
    if cleaning_hashes["cleaned_test_view_hash"] != _EXPECTED_CLEANED_TEST_HASH:
        raise D2LockedTestError("cleaned test hash is not the approved parity-clean-v1 hash")
    _require_equal(protocol, "test_split_content_hash", cleaning_hashes["cleaned_test_view_hash"])

    tuning_artifact = _resolve_protocol_path(
        _require(protocol, "threshold_tuning_artifact"), repository_root, "threshold_tuning_artifact"
    )
    _require_equal(protocol, "threshold_tuning_artifact_sha256", _sha256_file(tuning_artifact))
    tuning = _load_json(tuning_artifact, "threshold tuning artifact")
    validate_threshold_artifact(tuning)

    current_commit = git_commit_sha(repository_root)
    _require_equal(protocol, "evaluator_code_commit", current_commit)
    return {
        "checkpoint": checkpoint,
        "source_snapshot": source_snapshot,
        "pretrained_source": pretrained_source,
        "cleaning_manifest": cleaning_manifest,
        "threshold_tuning_artifact": tuning_artifact,
        "cleaning_hashes": cleaning_hashes,
        "snapshot": snapshot,
        "model_identity": model_identity,
        "config_fingerprint": config_fingerprint(config),
        "dataset_content_hash": content_manifest["dataset_content_hash"],
        "evaluator_code_commit": current_commit,
    }


def _confidence_values(report: Mapping[str, object], key: str) -> list[float]:
    values: list[float] = []
    images = report.get("images")
    if not isinstance(images, Sequence):
        raise D2LockedTestError("parity report images must be a sequence")
    for image in images:
        if not isinstance(image, Mapping):
            raise D2LockedTestError("parity report image record must be an object")
        predictions = image.get(key)
        if not isinstance(predictions, Sequence):
            raise D2LockedTestError("parity report prediction list '{}' is invalid".format(key))
        for prediction in predictions:
            if not isinstance(prediction, Mapping) or "confidence" not in prediction:
                raise D2LockedTestError("prediction confidence is missing from '{}'".format(key))
            values.append(float(prediction["confidence"]))
    return values


def _build_confidence_distributions(report: Mapping[str, object]) -> dict[str, object]:
    local = _confidence_values(report, "local_predictions")
    edge = _confidence_values(report, "edge_impulse_predictions")
    return {
        "local_current": summarize_confidences(local),
        "edge_impulse_legacy_and_strict": summarize_confidences(edge),
    }


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, object]]) -> None:
    try:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    except (OSError, csv.Error) as error:
        raise D2LockedTestError("unable to write CSV '{}': {}".format(path, error)) from error


def _per_class_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for report_key, evaluator_name in _EVALUATORS:
        raw = report.get(report_key)
        if not isinstance(raw, Mapping) or not isinstance(raw.get("per_class"), Mapping):
            raise D2LockedTestError("per-class report is missing {}".format(report_key))
        for class_name, values in raw["per_class"].items():
            if not isinstance(values, Mapping):
                raise D2LockedTestError("per-class metrics for '{}' are invalid".format(class_name))
            rows.append(
                {
                    "evaluator": evaluator_name,
                    "class_name": class_name,
                    "true_positives": values.get("true_positives"),
                    "false_positives": values.get("false_positives"),
                    "false_negatives": values.get("false_negatives"),
                    "precision": values.get("precision"),
                    "recall": values.get("recall"),
                    "f1": values.get("f1"),
                }
            )
    return rows


def _markdown_report(
    *,
    protocol: Mapping[str, object],
    provenance: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
    class_rows: Sequence[Mapping[str, object]],
    confidence_distributions: Mapping[str, object],
) -> str:
    lines = [
        "# Stage D2 locked cleaned-test report",
        "",
        "This is a single fixed FP32 evaluation. The test set was not used for checkpoint or threshold selection.",
        "",
        "## Locked protocol",
        "",
        "| field | value |",
        "|---|---|",
        "| protocol | `{}` |".format(protocol["protocol"]),
        "| selected epoch | `{}` |".format(protocol["selected_epoch"]),
        "| selected threshold | `{:.2f}` |".format(float(protocol["selected_threshold"])),
        "| threshold source | `{}` |".format(protocol["threshold_source"]),
        "| split | `{}` |".format(protocol["test_split"]),
        "| dtype | `{}` |".format(protocol["dtype"]),
        "| candidate count | `{}` |".format(protocol["candidate_count"]),
        "| cleaning view hash | `{}` |".format(provenance["cleaning_hashes"]["cleaning_view_hash"]),
        "| cleaned test hash | `{}` |".format(provenance["cleaning_hashes"]["cleaned_test_view_hash"]),
        "| evaluator commit | `{}` |".format(provenance["evaluator_code_commit"]),
        "",
        "## Aggregate metrics",
        "",
        "| evaluator | TP | FP | FN | precision | recall | F1 | macro F1 | predictions | GT | count MAE | count bias | mean localization |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        localization = row["mean_localization_error_pixels"]
        if localization is None:
            localization = "n/a; normalized=" + str(row["mean_localization_error_normalized"])
            lines.append(
                "| {evaluator} | {true_positives} | {false_positives} | {false_negatives} | {precision:.6f} | {recall:.6f} | {f1:.6f} | {macro_f1:.6f} | {prediction_count} | {ground_truth_count} | {mean_absolute_count_error:.6f} | {mean_count_bias:.6f} | {localization} |".format(
                    localization=localization,
                    **row
                )
            )
    lines.extend(["", "## Per-class F1", "", "| evaluator | class | precision | recall | F1 |", "|---|---|---:|---:|---:|"])
    for row in class_rows:
        lines.append(
            "| {evaluator} | {class_name} | {precision:.6f} | {recall:.6f} | {f1:.6f} |".format(
                **row
            )
        )
    lines.extend(["", "## Confidence distributions", "", "```json", json.dumps(confidence_distributions, indent=2), "```", ""])
    return "\n".join(lines)


def prepare_output_dir(path: Path) -> Path:
    """Create a new D2 result directory and refuse to overwrite prior results."""

    output = Path(path)
    if output.exists():
        raise D2LockedTestError("refusing to overwrite existing output directory: {}".format(output))
    try:
        output.mkdir(parents=True, exist_ok=False)
    except OSError as error:
        raise D2LockedTestError("unable to create output directory '{}': {}".format(output, error)) from error
    return output


def run_locked_test(
    *,
    config: ProjectConfig,
    protocol: Mapping[str, object],
    repository_root: Path,
    dataset_root: Path,
    output_dir: Path,
    device_request: str,
) -> dict[str, object]:
    """Validate provenance and execute exactly one fixed-threshold test pass."""

    provenance = validate_locked_protocol(
        protocol,
        config=config,
        repository_root=repository_root,
        dataset_root=dataset_root,
    )
    output = prepare_output_dir(output_dir)
    local_parity = _load_local_parity_module()
    cleaning_payload = _load_json(Path(provenance["cleaning_manifest"]), "cleaning manifest")
    report = local_parity.run_local_parity(
        config=config,
        dataset_root=dataset_root,
        checkpoint=Path(provenance["checkpoint"]),
        expected_checkpoint_sha256=str(protocol["checkpoint_sha256"]),
        cleaning_manifest=cleaning_payload,
        threshold=_EXPECTED_THRESHOLD,
        device_request=device_request,
        expected_image_count=_EXPECTED_TEST_IMAGE_COUNT,
    )
    confidence_distributions = _build_confidence_distributions(report)
    rows = metric_rows(report)
    class_rows = _per_class_rows(report)
    payload: dict[str, object] = {
        "protocol": dict(protocol),
        "provenance": {
            "checkpoint": str(provenance["checkpoint"]),
            "checkpoint_sha256": protocol["checkpoint_sha256"],
            "source_snapshot": str(provenance["source_snapshot"]),
            "source_snapshot_sha256": protocol["source_snapshot_sha256"],
            "pretrained_source": str(provenance["pretrained_source"]),
            "pretrained_sha256": protocol["pretrained_sha256"],
            "config_fingerprint": provenance["config_fingerprint"],
            "dataset_content_hash": provenance["dataset_content_hash"],
            "cleaning_manifest": str(provenance["cleaning_manifest"]),
            "cleaning_manifest_sha256": protocol["cleaning_manifest_sha256"],
            "cleaning_view_hash": provenance["cleaning_hashes"]["cleaning_view_hash"],
            "cleaned_test_view_hash": provenance["cleaning_hashes"]["cleaned_test_view_hash"],
            "threshold_tuning_artifact": str(provenance["threshold_tuning_artifact"]),
            "threshold_tuning_artifact_sha256": protocol["threshold_tuning_artifact_sha256"],
            "evaluator_code_commit": provenance["evaluator_code_commit"],
            "dtype": "float32",
            "threshold": _EXPECTED_THRESHOLD,
            "threshold_sweep": False,
            "candidate_count": 1,
            "model_identity": provenance["model_identity"],
        },
        "metrics": report,
        "confidence_distributions": confidence_distributions,
        "metric_rows": rows,
        "per_class_rows": class_rows,
    }
    try:
        (output / "final_test_metrics_d2.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        _write_csv(
            output / "final_test_metrics_d2.csv",
            tuple(rows[0].keys()) if rows else (),
            rows,
        )
        _write_csv(
            output / "per_class_test_d2.csv",
            tuple(class_rows[0].keys()) if class_rows else (),
            class_rows,
        )
        (output / "d2_locked_test_report.md").write_text(
            _markdown_report(
                protocol=protocol,
                provenance=provenance,
                rows=rows,
                class_rows=class_rows,
                confidence_distributions=confidence_distributions,
            ),
            encoding="utf-8",
        )
    except (OSError, TypeError, ValueError) as error:
        raise D2LockedTestError("unable to write locked D2 result artifacts: {}".format(error)) from error
    return payload


def main(arguments: Optional[Sequence[str]] = None) -> int:
    """Run the D2 protocol and report only the fixed result paths."""

    args = build_parser().parse_args(arguments)
    try:
        validate_cli_invariants(
            threshold_override=args.threshold_override,
            checkpoint_override=args.checkpoint_override,
            split_override=args.split_override,
            sweep_requested=args.sweep,
        )
        protocol = _load_json(args.protocol, "locked protocol")
        pretrained_source = protocol.get("pretrained_source")
        if isinstance(pretrained_source, str) and not os.environ.get("FOMO_PRETRAINED_WEIGHTS"):
            os.environ["FOMO_PRETRAINED_WEIGHTS"] = pretrained_source
        os.environ["FOMO_DATASET_ROOT"] = str(args.dataset_root)
        config = load_config(args.config)
        repository_root = Path(__file__).resolve().parents[1]
        payload = run_locked_test(
            config=config,
            protocol=protocol,
            repository_root=repository_root,
            dataset_root=args.dataset_root,
            output_dir=args.output_dir,
            device_request=args.device,
        )
        print(
            json.dumps(
                {
                    "output_dir": str(args.output_dir),
                    "selected_epoch": 40,
                    "threshold": 0.40,
                    "strict_f1": payload["metrics"]["strict_one_to_one_evaluator"]["f1"],
                    "edge_impulse_legacy_f1": payload["metrics"]["edge_impulse_legacy_evaluator"]["f1"],
                },
                ensure_ascii=False,
            )
        )
    except (
        ConfigurationError,
        D2LockedTestError,
        OSError,
        RuntimeError,
        SnapshotError,
        ValueError,
    ) as error:
        print("Error: {}".format(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
