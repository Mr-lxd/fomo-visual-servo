"""Tests for deterministic lab-pool to D2 train-only view conversion."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from fomo_servo.datasets.lab_pool_view import (
    LabPoolConversionError,
    build_lab_pool_training_view,
)
from fomo_servo.datasets.yolo import parse_yolo_label_file


D2_CLASSES = (
    "fish",
    "jellyfish",
    "penguin",
    "puffin",
    "shark",
    "starfish",
    "stingray",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_source(root: Path, shapes: list[dict[str, object]]) -> Path:
    train = root / "images" / "train"
    train.mkdir(parents=True)
    image = train / "frame.jpg"
    image.write_bytes(b"deterministic-image-bytes")
    annotation = train / "frame.json"
    annotation.write_text(
        json.dumps(
            {
                "imagePath": image.name,
                "imageWidth": 640,
                "imageHeight": 480,
                "shapes": shapes,
            }
        ),
        encoding="utf-8",
    )
    return image


def _shape(label: str, x1: float, y1: float, x2: float, y2: float) -> dict[str, object]:
    return {
        "label": label,
        "shape_type": "rectangle",
        "points": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
    }


def test_build_view_applies_approved_mapping_and_records_hashes(tmp_path: Path) -> None:
    source = tmp_path / "source"
    image = _write_source(
        source,
        [
            _shape("jellyfish", 10, 20, 110, 120),
            _shape("fish", 200, 30, 260, 90),
            _shape("tuna", 300, 40, 380, 140),
            _shape("reflection tuna", 400, 50, 450, 100),
            _shape("reflection jellyfish", 500, 60, 570, 130),
        ],
    )
    destination = tmp_path / "view"

    manifest = build_lab_pool_training_view(source, destination)

    data = yaml.safe_load((destination / "data.yaml").read_text(encoding="utf-8"))
    assert tuple(data["names"]) == D2_CLASSES
    boxes = parse_yolo_label_file(destination / "labels/train/frame.txt", len(D2_CLASSES))
    assert [box.source_class_id for box in boxes] == [1, 0, 0]
    assert manifest["counts"] == {
        "train_images": 1,
        "foreground_targets": 3,
        "background_annotations": 2,
        "empty_label_images": 0,
    }
    record = manifest["images"][0]
    assert record["source_image"] == "images/train/frame.jpg"
    assert record["source_annotation"] == "images/train/frame.json"
    assert record["source_image_sha256"] == _sha256(image)
    assert record["source_annotation_sha256"] == _sha256(source / "images/train/frame.json")
    assert record["generated_label_sha256"] == _sha256(destination / "labels/train/frame.txt")
    assert [entry["mapped_d2_class_id"] for entry in record["annotations"]] == [
        1,
        0,
        0,
        None,
        None,
    ]
    assert [entry["background"] for entry in record["annotations"]] == [
        False,
        False,
        False,
        True,
        True,
    ]


def test_build_view_writes_empty_label_for_unannotated_image(tmp_path: Path) -> None:
    source = tmp_path / "source"
    train = source / "images/train"
    train.mkdir(parents=True)
    (train / "negative.jpg").write_bytes(b"negative")

    manifest = build_lab_pool_training_view(source, tmp_path / "view")

    assert (tmp_path / "view/labels/train/negative.txt").read_bytes() == b""
    assert manifest["counts"]["empty_label_images"] == 1


def test_build_view_clamps_only_numerical_boundary_excess(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_source(source, [_shape("jellyfish", 10, -0.00024, 110, 120)])

    manifest = build_lab_pool_training_view(source, tmp_path / "view", clamp_epsilon=1e-6)

    boxes = parse_yolo_label_file(tmp_path / "view/labels/train/frame.txt", len(D2_CLASSES))
    assert len(boxes) == 1
    assert boxes[0].y_center - boxes[0].height / 2.0 >= 0.0
    annotation = manifest["images"][0]["annotations"][0]
    assert annotation["numerical_clamp"] is True
    assert annotation["bbox_normalized"]["y_min"] == 0.0


def test_build_view_rejects_real_geometry_error_without_publishing(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_source(source, [_shape("fish", 10, -1.0, 110, 120)])
    destination = tmp_path / "view"

    with pytest.raises(LabPoolConversionError, match="numerical clamp epsilon"):
        build_lab_pool_training_view(source, destination, clamp_epsilon=1e-6)

    assert not destination.exists()


def test_build_view_is_byte_deterministic_across_destinations(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_source(source, [_shape("fish", 20, 30, 100, 130)])

    first = tmp_path / "first"
    second = tmp_path / "second"
    build_lab_pool_training_view(source, first)
    build_lab_pool_training_view(source, second)

    assert (first / "conversion_manifest.json").read_bytes() == (
        second / "conversion_manifest.json"
    ).read_bytes()
    assert (first / "labels/train/frame.txt").read_bytes() == (
        second / "labels/train/frame.txt"
    ).read_bytes()


def test_training_view_cli_builds_view_and_reports_counts(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_source(source, [_shape("fish", 20, 30, 100, 130)])
    destination = tmp_path / "view"
    root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join((str(root), str(root / "src")))

    result = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/build_lab_pool_training_view.py"),
            "--source",
            str(source),
            "--destination",
            str(destination),
        ],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert '"foreground_targets": 1' in result.stdout
    assert (destination / "conversion_manifest.json").is_file()
