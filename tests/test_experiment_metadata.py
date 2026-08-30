"""Tests for append-only, reproducible experiment metadata artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from fomo_servo.experiments import (
    EXPERIMENT_SUMMARY_COLUMNS,
    append_experiment_summary,
    copy_experiment_config,
    dataset_content_manifest,
    dataset_file_list_hash,
    write_dataset_manifest,
    write_experiment_metadata,
)


def test_dataset_file_list_hash_is_stable_and_tracks_file_names(tmp_path: Path) -> None:
    (tmp_path / "images" / "train").mkdir(parents=True)
    (tmp_path / "images" / "train" / "a.jpg").write_bytes(b"a")
    (tmp_path / "labels").mkdir()
    (tmp_path / "labels" / "a.txt").write_text("0 0.5 0.5 1 1\n", encoding="utf-8")

    first = dataset_file_list_hash(tmp_path)
    second = dataset_file_list_hash(tmp_path)
    assert first == second

    (tmp_path / "images" / "train" / "b.jpg").write_bytes(b"b")
    assert dataset_file_list_hash(tmp_path) != first


def test_copy_and_write_metadata_preserve_complete_config_and_values(tmp_path: Path) -> None:
    source = tmp_path / "source.yaml"
    source.write_text("dataset:\n  root: data\n", encoding="utf-8")
    output_dir = tmp_path / "output"

    copied = copy_experiment_config(source, output_dir)
    assert copied.read_text(encoding="utf-8") == source.read_text(encoding="utf-8")

    metadata = {"experiment_name": "aug00_none", "random_seed": 42}
    metadata_path = write_experiment_metadata(output_dir, metadata)
    assert json.loads(metadata_path.read_text(encoding="utf-8")) == metadata


def test_dataset_content_manifest_is_relative_and_content_sensitive(tmp_path: Path) -> None:
    (tmp_path / "data.yaml").write_text("names: [creature]\n", encoding="utf-8")
    for split in ("train", "val"):
        (tmp_path / split / "images").mkdir(parents=True)
        (tmp_path / split / "labels").mkdir()
        (tmp_path / split / "images" / "image.jpg").write_bytes(b"image")
        (tmp_path / split / "labels" / "image.txt").write_text(
            "0 0.5 0.5 1 1\n", encoding="utf-8"
        )

    manifest = dataset_content_manifest(tmp_path, "train", "val")
    assert manifest["dataset_content_hash"]
    assert all("/" in item["relative_path"] or item["relative_path"] == "data.yaml" for item in manifest["files"])
    assert all(str(tmp_path) not in item["relative_path"] for item in manifest["files"])
    manifest_path = write_dataset_manifest(tmp_path / "output", manifest)
    assert manifest_path.is_file()

    first_hash = manifest["dataset_content_hash"]
    (tmp_path / "train" / "labels" / "image.txt").write_text(
        "0 0.4 0.5 1 1\n", encoding="utf-8"
    )
    assert dataset_content_manifest(tmp_path, "train", "val")["dataset_content_hash"] != first_hash


def test_append_experiment_summary_is_append_only(tmp_path: Path) -> None:
    summary_path = tmp_path / "experiments_summary.csv"
    row = {column: str(index) for index, column in enumerate(EXPERIMENT_SUMMARY_COLUMNS)}

    append_experiment_summary(summary_path, row)
    row["experiment_name"] = "second"
    append_experiment_summary(summary_path, row)

    with summary_path.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 2
    assert rows[0]["experiment_name"] == "0"
    assert rows[1]["experiment_name"] == "second"
    assert tuple(rows[0]) == EXPERIMENT_SUMMARY_COLUMNS


def test_append_experiment_summary_rejects_incompatible_existing_header(tmp_path: Path) -> None:
    summary_path = tmp_path / "experiments_summary.csv"
    summary_path.write_text("wrong\nvalue\n", encoding="utf-8")

    with pytest.raises(ValueError, match="header"):
        append_experiment_summary(summary_path, {column: "x" for column in EXPERIMENT_SUMMARY_COLUMNS})
