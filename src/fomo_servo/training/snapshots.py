"""Atomic portable weights-only epoch snapshots and inference candidates."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import torch
from torch import nn


class SnapshotError(RuntimeError):
    """Raised when a v2 snapshot or inference candidate is unsafe or invalid."""


def config_fingerprint(config: Any) -> str:
    """Hash a canonical configuration without embedding machine-local paths.

    Dataset contents are separately identified by their content hash.  Roots,
    output paths, resume paths, and source paths are replaced with stable
    markers so a snapshot contains no user-directory or machine-specific path.
    """

    if not is_dataclass(config):
        raise SnapshotError("config_fingerprint requires a dataclass configuration")
    canonical = _sanitize_config(asdict(config))
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def snapshot_filename(epoch: int) -> str:
    """Return the deterministic weights-only epoch snapshot basename."""

    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch <= 0:
        raise SnapshotError("epoch must be a positive integer")
    return "epoch_{:03d}_weights.pt".format(epoch)


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest for an already-published regular file."""

    source = Path(path)
    if not source.is_file():
        raise SnapshotError("snapshot file does not exist: {}".format(source))
    digest = hashlib.sha256()
    try:
        with source.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise SnapshotError("unable to hash snapshot '{}': {}".format(source, error)) from error
    return digest.hexdigest()


def write_epoch_snapshot(
    *,
    model: nn.Module,
    epoch: int,
    output_dir: Path,
    model_metadata: Mapping[str, Any],
    config_fingerprint: str,
    dataset_content_hash: str,
    git_commit_sha: str,
    seed: int,
    augmentation_preset: Optional[str],
    checkpoint_threshold: float,
    keep_last: Optional[int] = None,
) -> Path:
    """Write one CPU weights-only non-resumable snapshot using atomic publication."""

    if keep_last is not None and (
        isinstance(keep_last, bool) or not isinstance(keep_last, int) or keep_last <= 0
    ):
        raise SnapshotError("keep_last must be null or a positive integer")
    destination_directory = Path(output_dir) / "epoch_snapshots"
    destination = destination_directory / snapshot_filename(epoch)
    payload = {
        "checkpoint_kind": "epoch_snapshot",
        "weights_only": True,
        "resumable": False,
        "format": "weights_only",
        "model_state": _cpu_state_dict(model),
        "epoch": epoch,
        "model_metadata": dict(model_metadata),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "config_fingerprint": _required_text(config_fingerprint, "config_fingerprint"),
        "dataset_content_hash": _required_text(dataset_content_hash, "dataset_content_hash"),
        "git_commit_sha": _required_text(git_commit_sha, "git_commit_sha"),
        "seed": _positive_or_zero_integer(seed, "seed"),
        "augmentation_preset": augmentation_preset,
        "checkpoint_threshold": _probability(checkpoint_threshold, "checkpoint_threshold"),
    }
    atomic_torch_save(payload, destination)
    if keep_last is not None:
        _prune_snapshots(destination_directory, keep_last)
    return destination


def write_inference_candidate(
    *,
    source_snapshot: Path,
    destination: Path,
    selection_metric: str,
    selection_metric_value: float,
    selection_split: str,
    selection_details: Mapping[str, Any],
) -> Path:
    """Build an inference-only candidate from a validated published snapshot.

    The source file is read and its model state retained verbatim; it is never
    copied byte-for-byte because selection provenance must be added safely.
    """

    source = Path(source_snapshot)
    payload = load_epoch_snapshot(source)
    if not isinstance(selection_metric_value, (int, float)) or isinstance(
        selection_metric_value, bool
    ):
        raise SnapshotError("selection_metric_value must be numeric")
    candidate = dict(payload)
    required_selection_details = {
        "threshold_grid",
        "integration",
        "macro_effective_class_count",
    }
    missing_selection_details = sorted(
        required_selection_details.difference(selection_details)
    )
    if missing_selection_details:
        raise SnapshotError(
            "selection_details is missing {}".format(missing_selection_details)
        )
    candidate.update(
        {
            "checkpoint_kind": "inference_candidate",
            "weights_only": True,
            "resumable": False,
            "source_snapshot": source.name,
            "source_snapshot_sha256": sha256_file(source),
            "selected_epoch": payload["epoch"],
            "selection_metric": _required_text(selection_metric, "selection_metric"),
            "selection_metric_value": float(selection_metric_value),
            "selection_split": _required_text(selection_split, "selection_split"),
            "selection_dtype": "float32",
            "selection_details": dict(selection_details),
            "pr_auc_threshold_grid": list(selection_details["threshold_grid"]),
            "pr_auc_integration": selection_details["integration"],
            "pr_auc_macro_effective_class_count": selection_details[
                "macro_effective_class_count"
            ],
        }
    )
    atomic_torch_save(candidate, Path(destination))
    return Path(destination)


def load_epoch_snapshot(path: Path) -> dict[str, Any]:
    """Load and validate a v2 epoch snapshot without treating it as resumable."""

    source = Path(path)
    if not source.is_file():
        raise SnapshotError("snapshot file does not exist: {}".format(source))
    try:
        payload = torch.load(source, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, ValueError) as error:
        raise SnapshotError("unable to load snapshot '{}': {}".format(source, error)) from error
    if not isinstance(payload, dict):
        raise SnapshotError("snapshot payload must be a mapping")
    if payload.get("checkpoint_kind") != "epoch_snapshot":
        raise SnapshotError("source checkpoint is not an epoch_snapshot")
    if payload.get("weights_only") is not True or payload.get("resumable") is not False:
        raise SnapshotError("epoch snapshot must be weights-only and non-resumable")
    if not isinstance(payload.get("model_state"), Mapping):
        raise SnapshotError("epoch snapshot is missing model_state mapping")
    forbidden = {"optimizer_state", "scheduler_state", "scaler_state", "rng_state", "history"}
    present_forbidden = sorted(forbidden.intersection(payload))
    if present_forbidden:
        raise SnapshotError(
            "weights-only epoch snapshot contains forbidden training state: {}".format(
                present_forbidden
            )
        )
    epoch = payload.get("epoch")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch <= 0:
        raise SnapshotError("epoch snapshot epoch must be a positive integer")
    if source.name != snapshot_filename(epoch):
        raise SnapshotError(
            "snapshot filename '{}' does not match payload epoch {}".format(
                source.name, epoch
            )
        )
    for key in (
        "model_metadata",
        "config_fingerprint",
        "dataset_content_hash",
        "git_commit_sha",
        "seed",
        "augmentation_preset",
        "checkpoint_threshold",
    ):
        if key not in payload:
            raise SnapshotError("epoch snapshot is missing '{}'".format(key))
    return payload


def validate_snapshot_compatibility(
    payload: Mapping[str, Any],
    *,
    expected_model_metadata: Mapping[str, Any],
    expected_config_fingerprint: str,
    expected_dataset_content_hash: str,
) -> None:
    """Reject a snapshot whose identity or provenance differs from the run.

    Args:
        payload: Validated epoch snapshot mapping.
        expected_model_metadata: Model topology/parameter metadata from the YAML model.
        expected_config_fingerprint: Sanitized fingerprint of the evaluation configuration.
        expected_dataset_content_hash: Content hash of the configured train/validation files.

    Raises:
        SnapshotError: If model identity, configuration, or dataset provenance differs.
    """

    if not isinstance(payload, Mapping):
        raise SnapshotError("snapshot payload must be a mapping")
    actual_model_metadata = payload.get("model_metadata")
    if not isinstance(actual_model_metadata, Mapping):
        raise SnapshotError("snapshot is missing model identity metadata")
    if dict(actual_model_metadata) != dict(expected_model_metadata):
        raise SnapshotError(
            "snapshot model identity mismatch: expected configured model metadata, "
            "got a different model identity"
        )
    actual_config_fingerprint = payload.get("config_fingerprint")
    if actual_config_fingerprint != expected_config_fingerprint:
        raise SnapshotError(
            "snapshot config fingerprint mismatch: snapshot and evaluation YAML differ"
        )
    actual_dataset_hash = payload.get("dataset_content_hash")
    if actual_dataset_hash != expected_dataset_content_hash:
        raise SnapshotError(
            "snapshot dataset content hash mismatch: snapshot and configured dataset differ"
        )


def atomic_torch_save(payload: Mapping[str, Any], destination: Path) -> None:
    """Flush and fsync a temp payload before atomically replacing destination."""

    target = Path(destination)
    temporary_path: Optional[Path] = None
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=".tmp", prefix=target.name + ".", dir=target.parent, delete=False
        ) as handle:
            temporary_path = Path(handle.name)
            torch.save(dict(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, target)
        temporary_path = None
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise SnapshotError("unable to atomically write '{}': {}".format(target, error)) from error
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass


def _cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Detach every state entry into CPU storage for portable inference/evaluation."""

    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def _sanitize_config(value: Any, path: tuple[str, ...] = ()) -> Any:
    """Convert dataclass output to JSON-safe data while hiding local path values."""

    hidden_paths = {
        ("dataset", "root"),
        ("training", "output_dir"),
        ("training", "resume"),
        ("experiment", "summary_csv"),
        ("source_path",),
    }
    if path in hidden_paths:
        return "<{}>".format(".".join(path))
    if isinstance(value, Path):
        return value.as_posix() if not value.is_absolute() else "<absolute-path>"
    if isinstance(value, Mapping):
        return {str(key): _sanitize_config(item, path + (str(key),)) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_sanitize_config(item, path) for item in value]
    return value


def _prune_snapshots(directory: Path, keep_last: int) -> None:
    """Remove only older published v2 epoch snapshots after a successful write."""

    snapshots = sorted(
        directory.glob("epoch_*_weights.pt"),
        key=lambda path: _snapshot_epoch_from_name(path.name),
        reverse=True,
    )
    for stale in snapshots[keep_last:]:
        try:
            stale.unlink()
        except OSError as error:
            raise SnapshotError("unable to prune snapshot '{}': {}".format(stale, error)) from error


def _snapshot_epoch_from_name(name: str) -> int:
    try:
        return int(name.removeprefix("epoch_").removesuffix("_weights.pt"))
    except ValueError as error:
        raise SnapshotError("invalid snapshot filename: {}".format(name)) from error


def snapshot_epoch_from_filename(name: str) -> int:
    """Return and validate the numeric epoch encoded in a snapshot filename."""

    epoch = _snapshot_epoch_from_name(name)
    if snapshot_filename(epoch) != name:
        raise SnapshotError("invalid snapshot filename: {}".format(name))
    return epoch


def _required_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise SnapshotError("{} must be a non-empty string".format(field_name))
    return value


def _positive_or_zero_integer(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SnapshotError("{} must be a non-negative integer".format(field_name))
    return value


def _probability(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0:
        raise SnapshotError("{} must be a probability".format(field_name))
    return float(value)


__all__ = [
    "SnapshotError",
    "atomic_torch_save",
    "config_fingerprint",
    "load_epoch_snapshot",
    "sha256_file",
    "snapshot_epoch_from_filename",
    "snapshot_filename",
    "validate_snapshot_compatibility",
    "write_epoch_snapshot",
    "write_inference_candidate",
]
