"""CLI tests for the read-only YOLO invalid-label audit."""

from __future__ import annotations

from pathlib import Path

from scripts.audit_yolo_labels import main


def test_audit_cli_writes_requested_artifacts_without_changing_source(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    (root / "data.yaml").parent.mkdir(parents=True)
    (root / "data.yaml").write_text("names: [fish]\n", encoding="utf-8")
    for split in ("train", "valid", "test"):
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir(parents=True)
    label_path = root / "test" / "labels" / "example.txt"
    (root / "test" / "images" / "example.jpg").write_bytes(b"image")
    label_path.write_text("0 0.5 0.5 0 0\n", encoding="utf-8")
    before = label_path.read_bytes()
    output = tmp_path / "audit"

    assert main(["--dataset-root", str(root), "--class-count", "1", "--output-dir", str(output)]) == 0

    assert label_path.read_bytes() == before
    assert (output / "invalid_label_audit.json").is_file()
    assert (output / "invalid_label_audit.csv").is_file()
    assert (output / "parity-clean-v1.json").is_file()


def test_audit_cli_refuses_to_overwrite_existing_audit_artifact(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    (root / "data.yaml").parent.mkdir(parents=True)
    (root / "data.yaml").write_text("names: [fish]\n", encoding="utf-8")
    for split in ("train", "valid", "test"):
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir(parents=True)
    output = tmp_path / "audit"
    output.mkdir()
    (output / "parity-clean-v1.json").write_text("existing\n", encoding="utf-8")

    assert main(["--dataset-root", str(root), "--class-count", "1", "--output-dir", str(output)]) == 1
