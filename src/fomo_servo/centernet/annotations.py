"""Read lab-pool LabelMe annotations with per-box visibility, grouped by session."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from fomo_servo.datasets.lab_pool_view import CLASS_MAPPING, D2_CLASS_NAMES

from .targets import VISIBILITIES

FROZEN_TEST_SESSION = "pool-20260831-005"
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")


class VisibilityError(ValueError):
    """Raised for an unrecognised visibility value."""


class AnnotationError(ValueError):
    """Raised for an unusable annotation file."""


@dataclass(frozen=True)
class GroundTruthBox:
    """Original-image pixel box. ``class_id`` is the D2 index (tuna maps to fish)."""

    class_id: int
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    visibility: str = "full"


@dataclass(frozen=True)
class PoolSample:
    image_path: Path
    session: str
    boxes: tuple[GroundTruthBox, ...]
    reflection_boxes: int = 0


def read_visibility(shape: Mapping[str, Any], where: str) -> str:
    """``attributes.visibility`` first, then ``description``, else ``full``."""

    attributes = shape.get("attributes")
    if isinstance(attributes, Mapping) and "visibility" in attributes:
        value = attributes["visibility"]
        if value not in VISIBILITIES:
            raise VisibilityError("{}: invalid visibility {!r}".format(where, value))
        return value
    description = shape.get("description")
    if isinstance(description, str) and description.strip():
        if description not in VISIBILITIES[1:]:
            raise VisibilityError("{}: invalid description {!r}".format(where, description))
        return description
    return "full"


def session_of(image_path: Path) -> str:
    """Session id is the filename prefix before ``__`` (``pool-20260831-001``)."""

    return image_path.name.split("__", 1)[0]


def load_pool_samples(root: Path | str, *, use_visibility: bool) -> list[PoolSample]:
    """Load every ``images/train`` image of a lab-pool dataset.

    ``use_visibility=False`` reads every box as ``full`` (round 1). Reflection
    labels are background and are dropped (counted in ``reflection_boxes``).
    ``images/test`` is never read.
    """

    train_dir = Path(root) / "images" / "train"
    if not train_dir.is_dir():
        raise AnnotationError("missing train directory: {}".format(train_dir))
    samples = []
    for image in sorted(p for p in train_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES):
        session = session_of(image)
        if session == FROZEN_TEST_SESSION:
            raise AnnotationError("frozen test session found in train: {}".format(image))
        boxes: list[GroundTruthBox] = []
        reflections = 0
        annotation = image.with_suffix(".json")
        if annotation.is_file():
            payload = json.loads(annotation.read_text(encoding="utf-8"))
            for index, shape in enumerate(payload.get("shapes", []), start=1):
                where = "{} shape {}".format(annotation.name, index)
                label = shape.get("label")
                if label not in CLASS_MAPPING:
                    raise AnnotationError("{}: unsupported class {!r}".format(where, label))
                if shape.get("shape_type") != "rectangle":
                    raise AnnotationError("{}: shape_type must be rectangle".format(where))
                mapped = CLASS_MAPPING[label]
                if mapped is None:
                    reflections += 1
                    continue
                visibility = read_visibility(shape, where) if use_visibility else "full"
                xs = [float(p[0]) for p in shape["points"]]
                ys = [float(p[1]) for p in shape["points"]]
                width, height = float(payload["imageWidth"]), float(payload["imageHeight"])
                x0, x1 = max(0.0, min(xs)), min(width, max(xs))
                y0, y1 = max(0.0, min(ys)), min(height, max(ys))
                if x1 <= x0 or y1 <= y0:
                    raise AnnotationError("{}: empty rectangle".format(where))
                boxes.append(GroundTruthBox(mapped[0], x0, y0, x1, y1, visibility))
        samples.append(PoolSample(image, session, tuple(boxes), reflections))
    if not samples:
        raise AnnotationError("no train images in {}".format(train_dir))
    return samples


def class_names() -> tuple[str, ...]:
    return D2_CLASS_NAMES


def sessions(samples: list[PoolSample]) -> list[str]:
    return sorted({s.session for s in samples})


def split_fold(samples: list[PoolSample], held_out: Optional[str]) -> tuple[list[PoolSample], list[PoolSample]]:
    """Return ``(train, test)`` for leave-one-session-out; ``None`` trains on all."""

    if held_out is None:
        return list(samples), []
    return (
        [s for s in samples if s.session != held_out],
        [s for s in samples if s.session == held_out],
    )
