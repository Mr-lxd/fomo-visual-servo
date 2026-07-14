"""Tests for the explicit, manifest-only Edge Impulse parity-clean view."""

from __future__ import annotations

from pathlib import Path

import pytest

from fomo_servo.evaluation.parity_clean import (
    ParityCleanError,
    ParityCleanView,
    audit_yolo_dataset,
    build_parity_clean_manifest,
    verify_parity_clean_view,
)


def _write_dataset(root: Path) -> Path:
    root.mkdir(parents=True)
    (root / "data.yaml").write_text("names: [fish, shark]\n", encoding="utf-8")
    for split in ("train", "valid", "test"):
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir(parents=True)
    (root / "train" / "images" / "empty.jpg").write_bytes(b"image")
    (root / "train" / "labels" / "empty.txt").write_text("", encoding="utf-8")
    (root / "valid" / "images" / "valid.jpg").write_bytes(b"image")
    (root / "valid" / "labels" / "valid.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
    )
    (root / "test" / "images" / "sample.jpg").write_bytes(b"image")
    (root / "test" / "labels" / "sample.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n1 0.25 0.75 0 0\n", encoding="utf-8"
    )
    return root


def test_manifest_removes_only_declared_zero_area_rows(tmp_path: Path) -> None:
    root = _write_dataset(tmp_path / "dataset")
    audit = audit_yolo_dataset(root, class_count=2)
    manifest = build_parity_clean_manifest(audit)

    view = ParityCleanView(root, manifest, class_count=2)

    assert view.read_label_lines("test/labels/sample.txt") == ("0 0.5 0.5 0.2 0.2",)
    record = manifest["removed_rows"][0]
    assert record["reason"] == "width_lte_zero_and_height_lte_zero"
    assert record["line_number"] == 2
    assert record["raw_content"] == "1 0.25 0.75 0 0"
    assert manifest["source_dataset_read_only"] is True


def test_manifest_rejects_changed_source_hash(tmp_path: Path) -> None:
    root = _write_dataset(tmp_path / "dataset")
    manifest = build_parity_clean_manifest(audit_yolo_dataset(root, class_count=2))
    label_path = root / "test" / "labels" / "sample.txt"
    label_path.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    with pytest.raises(ParityCleanError, match="SHA-256"):
        ParityCleanView(root, manifest, class_count=2).read_label_lines(
            "test/labels/sample.txt"
        )


def test_manifest_rejects_undeclared_invalid_row(tmp_path: Path) -> None:
    root = _write_dataset(tmp_path / "dataset")
    audit = audit_yolo_dataset(root, class_count=2)
    manifest = build_parity_clean_manifest(audit)
    label_path = root / "test" / "labels" / "sample.txt"
    label_path.write_text(
        "0 0.5 0.5 0.2 0.2\n1 0.25 0.75 0 0\n9 0.5 0.5 0.1 0.1\n",
        encoding="utf-8",
    )
    manifest["source_files"]["test/labels/sample.txt"]["sha256"] = (
        __import__("hashlib").sha256(label_path.read_bytes()).hexdigest()
    )

    with pytest.raises(ParityCleanError, match="outside the manifest"):
        ParityCleanView(root, manifest, class_count=2).read_label_lines(
            "test/labels/sample.txt"
        )


def test_audit_reports_empty_label_without_marking_it_invalid(tmp_path: Path) -> None:
    root = _write_dataset(tmp_path / "dataset")

    audit = audit_yolo_dataset(root, class_count=2)

    assert audit.summary_by_split["train"]["empty_label"] == 1
    assert audit.summary_by_split["test"]["width_lte_zero"] == 1
    assert audit.summary_by_split["test"]["height_lte_zero"] == 1
    assert audit.physical_invalid_row_count == 1
    assert audit.undeclared_invalid_rows == ()


def test_cleaning_view_hash_excludes_unconsumed_dataset_metadata(tmp_path: Path) -> None:
    root = _write_dataset(tmp_path / "dataset")
    (root / "README.dataset.txt").write_text("not consumed\n", encoding="utf-8")
    (root / "train" / "data.yaml").write_text("not consumed\n", encoding="utf-8")

    manifest = build_parity_clean_manifest(audit_yolo_dataset(root, class_count=2))

    assert "README.dataset.txt" not in manifest["source_files"]
    assert "train/data.yaml" not in manifest["source_files"]
    assert "data.yaml" in manifest["source_files"]


def test_verify_cleaning_view_checks_every_hashed_input_before_evaluation(
    tmp_path: Path,
) -> None:
    root = _write_dataset(tmp_path / "dataset")
    manifest = build_parity_clean_manifest(audit_yolo_dataset(root, class_count=2))

    verified = verify_parity_clean_view(root, manifest, class_count=2)

    assert verified["cleaning_view_hash"] == manifest["cleaning_view_hash"]
    assert verified["cleaned_test_view_hash"] == manifest["cleaned_test_view_hash"]
    (root / "valid" / "images" / "valid.jpg").write_bytes(b"changed")
    with pytest.raises(ParityCleanError, match="SHA-256"):
        verify_parity_clean_view(root, manifest, class_count=2)
