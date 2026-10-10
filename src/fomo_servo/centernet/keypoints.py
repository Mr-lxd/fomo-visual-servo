"""Train-only LabelMe keypoints and signed log1p offsets in stride-grid units."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from fomo_servo.datasets.lab_pool_view import CLASS_MAPPING
from .annotations import (
    AnnotationError, FROZEN_TEST_SESSION, GroundTruthBox, IMAGE_SUFFIXES,
    read_visibility, session_of,
)
from .targets import Box


@dataclass(frozen=True)
class KeypointAnnotation:
    """Optional original-image pixel endpoints; states are visible/occluded."""
    source_class: str
    orientation: str | None
    head: tuple[float, float] | None = None
    tail: tuple[float, float] | None = None
    head_state: str | None = None
    tail_state: str | None = None


@dataclass(frozen=True)
class KPSample:
    """Original-image boxes and keypoints aligned by tuple index."""
    image_path: Path
    session: str
    boxes: tuple[GroundTruthBox, ...]
    keypoints: tuple[KeypointAnnotation, ...]
    reflection_boxes: int = 0


def _description_tokens(shape: dict) -> list[str]:
    return str(shape.get("description") or "").replace(",", " ").replace(";", " ").split()


def load_keypoint_samples(root: Path | str, *, issues: list[str] | None = None) -> list[KPSample]:
    """Read images/train only; report ambiguous/unused points without editing JSON.

    Association uses non-null group_id first, otherwise unique fish/tuna box
    containment with 2 pixel tolerance. Missing orientation, axial and ignore
    suppress both endpoints. Attributes take precedence over description.
    """
    issues = issues if issues is not None else []
    train_dir = Path(root) / "images" / "train"
    if not train_dir.is_dir():
        raise AnnotationError("missing train directory: {}".format(train_dir))
    samples = []
    for image in sorted(p for p in train_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES):
        session = session_of(image)
        if session == FROZEN_TEST_SESSION:
            raise AnnotationError("frozen test session in train: {}".format(image))
        annotation = image.with_suffix(".json")
        payload = json.loads(annotation.read_text(encoding="utf-8")) if annotation.is_file() else {}
        boxes, keypoints, groups = [], [], []
        reflections = 0
        shapes = payload.get("shapes", [])
        for index, shape in enumerate(shapes):
            if shape.get("shape_type") == "point" and shape.get("label") in {"head", "tail"}:
                continue
            where = "{} shape {}".format(annotation.name, index + 1)
            label = shape.get("label")
            if label not in CLASS_MAPPING or shape.get("shape_type") != "rectangle":
                raise AnnotationError("{}: unsupported shape/class".format(where))
            mapped = CLASS_MAPPING[label]
            if mapped is None:
                reflections += 1
                continue
            attrs = shape.get("attributes") or {}
            tokens = _description_tokens(shape)
            orientation = attrs.get("orientation", next((v for v in tokens if v in {"side", "oblique", "axial"}), None))
            if orientation not in {None, "side", "oblique", "axial"}:
                issues.append("{}: invalid orientation {!r}; unused".format(where, orientation))
                orientation = None
            local_shape = dict(shape)
            # Original visibility parser remains unchanged; locally strip only
            # the known orientation token to support 07b description fallback.
            local_shape["description"] = " ".join(v for v in tokens if v not in {"side", "oblique", "axial"})
            visibility = read_visibility(local_shape, where)
            xs, ys = zip(*shape["points"])
            x0, x1 = max(0., min(xs)), min(float(payload["imageWidth"]), max(xs))
            y0, y1 = max(0., min(ys)), min(float(payload["imageHeight"]), max(ys))
            if x1 <= x0 or y1 <= y0:
                raise AnnotationError("{}: empty rectangle".format(where))
            boxes.append(GroundTruthBox(mapped[0], x0, y0, x1, y1, visibility))
            keypoints.append(KeypointAnnotation(label, orientation))
            groups.append(shape.get("group_id"))
        endpoints: dict[tuple[int, str], list[tuple[tuple[float, float], str]]] = {}
        for index, shape in enumerate(shapes):
            if shape.get("shape_type") != "point" or shape.get("label") not in {"head", "tail"}:
                continue
            where = "{} shape {}".format(annotation.name, index + 1)
            x, y = map(float, shape["points"][0])
            group = shape.get("group_id")
            candidates = [i for i, kp in enumerate(keypoints) if kp.source_class in {"fish", "tuna"} and (
                groups[i] == group if group is not None else
                boxes[i].x_min - 2 <= x <= boxes[i].x_max + 2 and boxes[i].y_min - 2 <= y <= boxes[i].y_max + 2
            )]
            if len(candidates) != 1:
                issues.append("{}: {} candidate boxes; point unused".format(where, len(candidates)))
                continue
            i = candidates[0]
            kp, box = keypoints[i], boxes[i]
            if box.visibility == "ignore" or kp.orientation not in {"side", "oblique"}:
                issues.append("{}: ignore/axial/missing orientation; point unused".format(where))
                continue
            state = (shape.get("attributes") or {}).get("point_state", shape.get("description") or "visible")
            if state not in {"visible", "occluded"} or not (0 <= x < payload["imageWidth"] and 0 <= y < payload["imageHeight"]):
                issues.append("{}: invalid state or out-of-frame point; unused".format(where))
                continue
            endpoints.setdefault((i, shape["label"]), []).append(((x, y), state))
        for i, kp in enumerate(keypoints):
            values = {}
            for name in ("head", "tail"):
                entries = endpoints.get((i, name), [])
                if len(entries) == 1:
                    values[name], values[name + "_state"] = entries[0]
                elif len(entries) > 1:
                    issues.append("{} box {}: duplicate {}; unused".format(annotation.name, i, name))
            keypoints[i] = KeypointAnnotation(kp.source_class, kp.orientation, **values)
        samples.append(KPSample(image, session, tuple(boxes), tuple(keypoints), reflections))
    if not samples:
        raise AnnotationError("no train images in {}".format(train_dir))
    return samples


def encode_offsets(head, tail, *, cell_x: int, cell_y: int, stride: int):
    """Pixel endpoints → signed-log1p float32 [hx,hy,tx,ty] and [4] mask.

    Reference is the cell's geometric centre, offsets measured in grid cells.
    Missing endpoints contribute zero target and zero mask; no clipping.
    """
    target, mask = np.zeros(4, np.float32), np.zeros(4, np.float32)
    centre = np.asarray([cell_x + .5, cell_y + .5])
    for index, point in enumerate((head, tail)):
        if point is not None:
            delta = np.asarray(point) / stride - centre
            target[index * 2:index * 2 + 2] = np.sign(delta) * np.log1p(np.abs(delta))
            mask[index * 2:index * 2 + 2] = 1
    return target, mask


def decode_offsets(values, *, cell_x: int, cell_y: int, stride: int) -> np.ndarray:
    """Signed-log1p [4] → float64 [2,2] head/tail letterbox pixel coordinates."""
    encoded = np.asarray(values, dtype=np.float64).reshape(2, 2)
    return (np.asarray([cell_x + .5, cell_y + .5]) + np.sign(encoded) * np.expm1(np.abs(encoded))) * stride


def build_keypoint_targets(boxes: Sequence[Box], keypoints: Sequence[KeypointAnnotation], *, grid_size: int, stride: int):
    """Letterbox pixel annotations → float32 targets/masks [4,G,G].

    Every centre collision is masked, including collisions with unlabelled
    objects. Ignore-region cells suppress keypoint regression independently
    of detection's positive-over-ignore policy.
    """
    target = np.zeros((4, grid_size, grid_size), np.float32)
    mask = np.zeros_like(target)
    counts = np.zeros((grid_size, grid_size), np.int64)
    ignored = np.zeros((grid_size, grid_size), bool)
    cells = []
    for b in boxes:
        x, y = int((b.x_min + b.x_max) / (2 * stride)), int((b.y_min + b.y_max) / (2 * stride))
        cells.append((x, y))
        if 0 <= x < grid_size and 0 <= y < grid_size:
            counts[y, x] += 1
        if b.visibility == "ignore":
            x0, x1 = max(0, int(np.floor(b.x_min / stride))), min(grid_size, int(np.ceil(b.x_max / stride)))
            y0, y1 = max(0, int(np.floor(b.y_min / stride))), min(grid_size, int(np.ceil(b.y_max / stride)))
            ignored[y0:y1, x0:x1] = True
    for b, kp, (x, y) in zip(boxes, keypoints, cells):
        if not (0 <= x < grid_size and 0 <= y < grid_size) or counts[y, x] != 1 or ignored[y, x]:
            continue
        if kp.source_class not in {"fish", "tuna"} or kp.orientation not in {"side", "oblique"}:
            continue
        target[:, y, x], mask[:, y, x] = encode_offsets(kp.head, kp.tail, cell_x=x, cell_y=y, stride=stride)
    return {"keypoint_target": target, "keypoint_mask": mask}
