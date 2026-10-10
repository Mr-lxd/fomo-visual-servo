"""Deterministic lab-pool LabelMe to D2 YOLO training-view conversion."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


D2_CLASS_NAMES = (
    "fish",
    "jellyfish",
    "penguin",
    "puffin",
    "shark",
    "starfish",
    "stingray",
)
CLASS_MAPPING: Mapping[str, tuple[int, str] | None] = {
    "jellyfish": (1, "jellyfish"),
    "fish": (0, "fish"),
    "tuna": (0, "fish"),
    "reflection tuna": None,
    "reflection jellyfish": None,
}
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp"})


class LabPoolConversionError(ValueError):
    """Raised when the immutable annotation asset cannot be converted safely."""


def build_lab_pool_training_view(
    source_root: Path | str,
    destination_root: Path | str,
    *,
    clamp_epsilon: float = 1e-6,
) -> dict[str, Any]:
    """Build a D2-compatible train-only view from LabelMe annotations.

    Source rectangles use original-image pixel coordinates. Generated YOLO
    labels use normalized ``class x_center y_center width height`` values in the
    D2 seven-class coordinate contract. The source directory is never modified.
    """

    source = Path(source_root).resolve()
    destination = Path(destination_root).resolve()
    if clamp_epsilon <= 0.0:
        raise LabPoolConversionError("clamp_epsilon must be positive")
    if destination.exists():
        raise LabPoolConversionError("destination already exists: {}".format(destination))
    source_images = source / "images" / "train"
    if not source_images.is_dir():
        raise LabPoolConversionError(
            "source train image directory does not exist: {}".format(source_images)
        )
    images = sorted(
        path for path in source_images.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise LabPoolConversionError("source contains no train images")

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=".{}.tmp-".format(destination.name), dir=destination.parent)
    )
    try:
        manifest = _populate_view(source, images, staging, clamp_epsilon)
        staging.replace(destination)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _populate_view(
    source: Path,
    images: Sequence[Path],
    staging: Path,
    clamp_epsilon: float,
) -> dict[str, Any]:
    output_images = staging / "images" / "train"
    output_labels = staging / "labels" / "train"
    output_images.mkdir(parents=True)
    output_labels.mkdir(parents=True)
    data_yaml = {
        "path": ".",
        "train": "images/train",
        "val": None,
        "names": list(D2_CLASS_NAMES),
    }
    (staging / "data.yaml").write_text(
        yaml.safe_dump(data_yaml, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )

    records: list[dict[str, Any]] = []
    foreground_targets = 0
    background_annotations = 0
    empty_label_images = 0
    for image in images:
        annotation_path = image.with_suffix(".json")
        annotations: list[dict[str, Any]] = []
        label_lines: list[str] = []
        if annotation_path.is_file():
            payload = _load_annotation(annotation_path, image.name)
            width = _positive_dimension(payload, "imageWidth", annotation_path)
            height = _positive_dimension(payload, "imageHeight", annotation_path)
            shapes = payload.get("shapes")
            if not isinstance(shapes, list):
                raise LabPoolConversionError("{}: shapes must be a list".format(annotation_path))
            for index, shape in enumerate(shapes, start=1):
                converted, line = _convert_shape(
                    annotation_path, index, shape, width, height, clamp_epsilon
                )
                annotations.append(converted)
                if line is None:
                    background_annotations += 1
                else:
                    foreground_targets += 1
                    label_lines.append(line)

        label_path = output_labels / "{}.txt".format(image.stem)
        label_bytes = ("\n".join(label_lines) + ("\n" if label_lines else "")).encode("utf-8")
        label_path.write_bytes(label_bytes)
        if not label_lines:
            empty_label_images += 1
        _link_or_copy(image, output_images / image.name)
        records.append(
            {
                "source_image": image.relative_to(source).as_posix(),
                "source_annotation": (
                    annotation_path.relative_to(source).as_posix()
                    if annotation_path.is_file()
                    else None
                ),
                "source_image_sha256": _sha256_file(image),
                "source_annotation_sha256": (
                    _sha256_file(annotation_path) if annotation_path.is_file() else None
                ),
                "generated_label": label_path.relative_to(staging).as_posix(),
                "generated_label_sha256": hashlib.sha256(label_bytes).hexdigest(),
                "annotations": annotations,
            }
        )

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "dataset_identity": "lab_pool_v1_d2_trainonly",
        "source_dataset": source.name,
        "d2_class_names": list(D2_CLASS_NAMES),
        "class_mapping": {
            name: (
                {"class_id": mapped[0], "class_name": mapped[1]}
                if mapped is not None
                else {"background": True}
            )
            for name, mapped in CLASS_MAPPING.items()
        },
        "clamp_epsilon": clamp_epsilon,
        "counts": {
            "train_images": len(images),
            "foreground_targets": foreground_targets,
            "background_annotations": background_annotations,
            "empty_label_images": empty_label_images,
        },
        "images": records,
    }
    (staging / "conversion_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def _load_annotation(path: Path, expected_image_name: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise LabPoolConversionError("unable to read annotation '{}': {}".format(path, error)) from error
    if not isinstance(payload, Mapping):
        raise LabPoolConversionError("{}: annotation root must be a mapping".format(path))
    image_path = payload.get("imagePath")
    if image_path is not None and Path(str(image_path)).name != expected_image_name:
        raise LabPoolConversionError(
            "{}: imagePath '{}' does not match '{}'".format(path, image_path, expected_image_name)
        )
    return payload


def _positive_dimension(payload: Mapping[str, Any], key: str, path: Path) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) <= 0.0:
        raise LabPoolConversionError("{}: {} must be positive".format(path, key))
    return float(value)


def _convert_shape(
    path: Path,
    index: int,
    raw_shape: object,
    image_width: float,
    image_height: float,
    epsilon: float,
) -> tuple[dict[str, Any], str | None]:
    if not isinstance(raw_shape, Mapping):
        raise LabPoolConversionError("{} shape {}: must be a mapping".format(path, index))
    label = raw_shape.get("label")
    if label not in CLASS_MAPPING:
        raise LabPoolConversionError("{} shape {}: unsupported class '{}'".format(path, index, label))
    if raw_shape.get("shape_type") != "rectangle":
        raise LabPoolConversionError("{} shape {}: shape_type must be rectangle".format(path, index))
    points = raw_shape.get("points")
    if not isinstance(points, list) or len(points) not in {2, 4}:
        raise LabPoolConversionError("{} shape {}: rectangle must have two or four points".format(path, index))
    try:
        xs = [float(point[0]) / image_width for point in points]
        ys = [float(point[1]) / image_height for point in points]
    except (TypeError, ValueError, IndexError) as error:
        raise LabPoolConversionError("{} shape {}: invalid rectangle points".format(path, index)) from error
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    values = (x_min, y_min, x_max, y_max)
    if not all(float("-inf") < value < float("inf") for value in values):
        raise LabPoolConversionError("{} shape {}: coordinates must be finite".format(path, index))
    if x_max <= x_min or y_max <= y_min:
        raise LabPoolConversionError("{} shape {}: rectangle area must be positive".format(path, index))
    if any(value < -epsilon or value > 1.0 + epsilon for value in values):
        raise LabPoolConversionError(
            "{} shape {}: geometry exceeds numerical clamp epsilon {}".format(path, index, epsilon)
        )
    clamped = tuple(min(1.0, max(0.0, value)) for value in values)
    numerical_clamp = clamped != values
    x_min, y_min, x_max, y_max = clamped
    mapped = CLASS_MAPPING[str(label)]
    generated: dict[str, float | int] | None = None
    line: str | None = None
    if mapped is not None:
        class_id, _ = mapped
        cx, cy, width, height, rounding_clamp = _stable_yolo_values(
            x_min, y_min, x_max, y_max
        )
        numerical_clamp = numerical_clamp or rounding_clamp
        generated = {
            "class_id": class_id,
            "x_center": cx,
            "y_center": cy,
            "width": width,
            "height": height,
        }
        line = "{} {:.9f} {:.9f} {:.9f} {:.9f}".format(
            class_id, cx, cy, width, height
        )
    return (
        {
            "original_class": label,
            "mapped_d2_class_id": mapped[0] if mapped is not None else None,
            "mapped_d2_class_name": mapped[1] if mapped is not None else None,
            "background": mapped is None,
            "bbox_pixels": {
                "x_min": min(float(point[0]) for point in points),
                "y_min": min(float(point[1]) for point in points),
                "x_max": max(float(point[0]) for point in points),
                "y_max": max(float(point[1]) for point in points),
            },
            "bbox_normalized": {
                "x_min": x_min,
                "y_min": y_min,
                "x_max": x_max,
                "y_max": y_max,
            },
            "generated_yolo": generated,
            "numerical_clamp": numerical_clamp,
        },
        line,
    )


def _stable_yolo_values(
    x_min: float, y_min: float, x_max: float, y_max: float
) -> tuple[float, float, float, float, bool]:
    cx = float("{:.9f}".format((x_min + x_max) / 2.0))
    cy = float("{:.9f}".format((y_min + y_max) / 2.0))
    width = float("{:.9f}".format(x_max - x_min))
    height = float("{:.9f}".format(y_max - y_min))
    safe_width = min(width, 2.0 * min(cx, 1.0 - cx))
    safe_height = min(height, 2.0 * min(cy, 1.0 - cy))
    changed = safe_width != width or safe_height != height
    if safe_width <= 0.0 or safe_height <= 0.0:
        raise LabPoolConversionError("rounded YOLO rectangle has non-positive area")
    return cx, cy, safe_width, safe_height, changed


def _link_or_copy(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "CLASS_MAPPING",
    "D2_CLASS_NAMES",
    "LabPoolConversionError",
    "build_lab_pool_training_view",
]
