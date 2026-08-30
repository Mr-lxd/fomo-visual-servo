"""Reproducible experiment artifact and append-only summary helpers."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Mapping


class ExperimentMetadataError(ValueError):
    """Raised when reproducibility metadata cannot be collected or persisted."""


EXPERIMENT_SUMMARY_COLUMNS = (
    "experiment_name",
    "output_dir",
    "config_copy",
    "git_commit_sha",
    "dataset_file_list_hash",
    "random_seed",
    "best_centroid_f1",
    "best_grid_f1",
    "precision",
    "recall",
    "best_threshold",
    "mean_localization_error_pixels",
    "mean_count_bias",
    "count_mae",
    "best_epoch",
    "total_training_time_seconds",
)


def dataset_file_list_hash(dataset_root: Path) -> str:
    """Hash sorted relative file names below a dataset root.

    The digest intentionally covers the file list, not image contents, so it is
    cheap and stable for an experiment manifest while still detecting added or
    removed dataset files.
    """

    root = Path(dataset_root)
    if not root.is_dir():
        raise ExperimentMetadataError(
            "dataset root does not exist or is not a directory: {}".format(root)
        )
    relative_files = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    )
    if not relative_files:
        raise ExperimentMetadataError(
            "dataset root contains no files: {}".format(root)
        )
    digest = hashlib.sha256()
    for relative_file in relative_files:
        digest.update(relative_file.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def dataset_content_manifest(
    dataset_root: Path, train_split: str, validation_split: str
) -> dict[str, object]:
    """Build a relative-path, size, and content SHA256 manifest for used data files."""

    root = Path(dataset_root)
    if not root.is_dir():
        raise ExperimentMetadataError(
            "dataset root does not exist or is not a directory: {}".format(root)
        )
    relative_paths = {Path("data.yaml")}
    for split in (train_split, validation_split):
        if not isinstance(split, str) or not split.strip():
            raise ExperimentMetadataError("dataset split names must be non-empty strings")
        split_names = [split]
        if split == "val":
            split_names.append("valid")
        elif split == "valid":
            split_names.append("val")
        directories = []
        for split_name in split_names:
            for image_directory, label_directory in (
                (root / split_name / "images", root / split_name / "labels"),
                (root / "images" / split_name, root / "labels" / split_name),
            ):
                if image_directory.is_dir() and label_directory.is_dir():
                    directories = [image_directory, label_directory]
                    break
            if directories:
                break
        if not directories:
            raise ExperimentMetadataError(
                "dataset image/label directories do not exist for split '{}': {}".format(
                    split, root
                )
            )
        for directory in directories:
            relative_paths.update(
                path.relative_to(root)
                for path in directory.rglob("*")
                if path.is_file()
            )
    entries = []
    digest = hashlib.sha256()
    for relative_path in sorted(relative_paths, key=lambda item: item.as_posix()):
        absolute_path = root / relative_path
        if not absolute_path.is_file():
            raise ExperimentMetadataError(
                "dataset manifest file does not exist: {}".format(relative_path)
            )
        content = absolute_path.read_bytes()
        content_sha256 = hashlib.sha256(content).hexdigest()
        relative_text = relative_path.as_posix()
        size = len(content)
        entries.append(
            {
                "relative_path": relative_text,
                "size": size,
                "content_sha256": content_sha256,
            }
        )
        digest.update(relative_text.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        digest.update(content_sha256.encode("ascii"))
        digest.update(b"\n")
    return {
        "train_split": train_split,
        "validation_split": validation_split,
        "files": entries,
        "dataset_content_hash": digest.hexdigest(),
    }


def write_dataset_manifest(output_dir: Path, manifest: Mapping[str, object]) -> Path:
    """Write a JSON dataset manifest containing no absolute paths."""

    destination_directory = Path(output_dir)
    destination = destination_directory / "dataset_manifest.json"
    try:
        destination_directory.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(dict(manifest), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except (OSError, TypeError, ValueError) as error:
        raise ExperimentMetadataError(
            "unable to write dataset manifest '{}': {}".format(destination, error)
        ) from error
    return destination


def git_worktree_fingerprint(repository_hint: Path) -> tuple[bool, str]:
    """Return dirty state and a SHA256 over tracked diff plus untracked file bytes."""

    hint = Path(repository_hint).resolve()
    working_directory = hint if hint.is_dir() else hint.parent
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=str(working_directory),
            check=False,
            capture_output=True,
        )
        diff = subprocess.run(
            ["git", "diff", "HEAD", "--no-ext-diff", "--binary"],
            cwd=str(working_directory),
            check=False,
            capture_output=True,
        )
        untracked = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard", "-z"],
            cwd=str(working_directory),
            check=False,
            capture_output=True,
        )
    except OSError as error:
        raise ExperimentMetadataError(
            "unable to inspect Git worktree '{}': {}".format(working_directory, error)
        ) from error
    if status.returncode or diff.returncode or untracked.returncode:
        detail = (
            status.stderr.decode(errors="replace").strip()
            or diff.stderr.decode(errors="replace").strip()
            or untracked.stderr.decode(errors="replace").strip()
            or "Git command failed"
        )
        raise ExperimentMetadataError(
            "unable to inspect Git worktree '{}': {}".format(working_directory, detail)
        )
    digest = hashlib.sha256()
    digest.update(b"git-diff-head\0")
    digest.update(diff.stdout)
    untracked_paths = [item for item in untracked.stdout.split(b"\0") if item]
    for raw_path in sorted(untracked_paths):
        relative_path = raw_path.decode("utf-8")
        absolute_path = working_directory / Path(relative_path)
        if not absolute_path.is_file():
            raise ExperimentMetadataError(
                "untracked path is not a regular file: {}".format(relative_path)
            )
        digest.update(b"untracked\0")
        digest.update(raw_path)
        digest.update(b"\0")
        digest.update(absolute_path.read_bytes())
        digest.update(b"\n")
    return bool(status.stdout.strip()), digest.hexdigest()


def git_commit_sha(repository_hint: Path) -> str:
    """Return the full Git commit SHA discovered from a file or directory hint."""

    hint = Path(repository_hint).resolve()
    working_directory = hint if hint.is_dir() else hint.parent
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(working_directory),
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as error:
        raise ExperimentMetadataError(
            "unable to execute git while collecting commit SHA: {}".format(error)
        ) from error
    sha = result.stdout.strip()
    if result.returncode != 0 or not re.fullmatch(r"[0-9a-fA-F]{40}", sha):
        detail = result.stderr.strip() or "git rev-parse returned no valid commit SHA"
        raise ExperimentMetadataError(
            "unable to determine Git commit SHA from '{}': {}".format(
                working_directory, detail
            )
        )
    return sha.lower()


def copy_experiment_config(source_path: Path, output_dir: Path) -> Path:
    """Copy the exact source YAML bytes into an experiment output directory."""

    source = Path(source_path)
    destination_directory = Path(output_dir)
    if not source.is_file():
        raise ExperimentMetadataError(
            "experiment config does not exist: {}".format(source)
        )
    try:
        destination_directory.mkdir(parents=True, exist_ok=True)
        destination = destination_directory / "config.yaml"
        if source.resolve() != destination.resolve():
            shutil.copyfile(source, destination)
    except OSError as error:
        raise ExperimentMetadataError(
            "unable to copy experiment config to '{}': {}".format(
                destination_directory, error
            )
        ) from error
    return destination


def write_experiment_metadata(
    output_dir: Path, metadata: Mapping[str, object]
) -> Path:
    """Write JSON metadata for one experiment run."""

    destination_directory = Path(output_dir)
    destination = destination_directory / "experiment_metadata.json"
    try:
        destination_directory.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(dict(metadata), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except (OSError, TypeError, ValueError) as error:
        raise ExperimentMetadataError(
            "unable to write experiment metadata '{}': {}".format(destination, error)
        ) from error
    return destination


def append_experiment_summary(
    summary_path: Path, row: Mapping[str, object]
) -> None:
    """Append one experiment row without rewriting or replacing prior records."""

    missing = [column for column in EXPERIMENT_SUMMARY_COLUMNS if column not in row]
    if missing:
        raise ExperimentMetadataError(
            "experiment summary row is missing columns: {}".format(missing)
        )
    destination = Path(summary_path)
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        exists_with_content = destination.is_file() and destination.stat().st_size > 0
        if exists_with_content:
            with destination.open("r", newline="", encoding="utf-8") as summary_file:
                existing_header = tuple(csv.reader(summary_file).__next__())
            if existing_header != EXPERIMENT_SUMMARY_COLUMNS:
                raise ExperimentMetadataError(
                    "experiment summary CSV header is incompatible with the current schema"
                )
        with destination.open("a", newline="", encoding="utf-8") as summary_file:
            writer = csv.DictWriter(summary_file, fieldnames=EXPERIMENT_SUMMARY_COLUMNS)
            if not exists_with_content:
                writer.writeheader()
            writer.writerow({column: row[column] for column in EXPERIMENT_SUMMARY_COLUMNS})
    except ExperimentMetadataError:
        raise
    except (OSError, csv.Error) as error:
        raise ExperimentMetadataError(
            "unable to append experiment summary '{}': {}".format(destination, error)
        ) from error


__all__ = [
    "EXPERIMENT_SUMMARY_COLUMNS",
    "ExperimentMetadataError",
    "append_experiment_summary",
    "copy_experiment_config",
    "dataset_content_manifest",
    "dataset_file_list_hash",
    "git_commit_sha",
    "git_worktree_fingerprint",
    "write_dataset_manifest",
    "write_experiment_metadata",
]
