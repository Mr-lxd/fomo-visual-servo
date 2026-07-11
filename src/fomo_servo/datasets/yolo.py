"""YOLOv5-format dataset loading for FOMO centroid heatmap supervision."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any, Mapping, Sequence, Tuple, Union

import cv2
import numpy as np
import yaml

from fomo_servo.datasets.heatmap import HeatmapTarget, generate_fomo_heatmap
from fomo_servo.geometry.letterbox import LetterboxTransform, letterbox_rgb


class DatasetError(ValueError):
    """Raised when a YOLOv5 dataset layout or data.yaml is invalid."""


class YoloLabelError(ValueError):
    """Raised when a YOLO label line is malformed or outside normalized bounds."""


@dataclass(frozen=True)
class NormalizedYoloBox:
    """A validated YOLO label in normalized original-image coordinates."""

    source_class_id: int
    x_center: float
    y_center: float
    width: float
    height: float


@dataclass(frozen=True)
class AbsoluteBox:
    """An ``(x_min,y_min,x_max,y_max)`` bbox in a named pixel coordinate space."""

    foreground_class_id: int
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @property
    def center(self) -> Tuple[float, float]:
        """Return the bbox centre in the same pixel coordinate space."""

        return (self.x_min + self.x_max) / 2.0, (self.y_min + self.y_max) / 2.0


@dataclass(frozen=True)
class FOMOSample:
    """One FOMO data sample without a batch dimension.

    ``image`` is normalized RGB ``float32 [3,S,S]``. ``original_image`` and
    ``letterbox_image`` are RGB ``uint8 [H,W,3]`` and ``uint8 [S,S,3]``.
    Heatmap shapes are documented by ``HeatmapTarget``.
    """

    image: np.ndarray
    original_image: np.ndarray
    letterbox_image: np.ndarray
    original_boxes: Tuple[AbsoluteBox, ...]
    letterbox_boxes: Tuple[AbsoluteBox, ...]
    transform: LetterboxTransform
    heatmap: HeatmapTarget
    image_path: Path
    label_path: Path


class YOLOv5FOMODataset:
    """Read one YOLOv5 split and produce FOMO stride-space labels.

    The input root must contain ``data.yaml`` and either project-layout
    ``images/<split>/`` / ``labels/<split>/`` directories or Roboflow-layout
    ``<split>/images`` / ``<split>/labels`` directories. Logical ``val`` also
    resolves Roboflow's common physical ``valid`` folder. Missing label files are
    treated as no-target images; malformed existing labels raise ``YoloLabelError``.
    """

    IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
    CLASS_MODES = {"merge_single", "preserve"}

    def __init__(
        self,
        root: Union[str, Path],
        split: str,
        input_size: int,
        stride: int,
        class_mode: str,
        merged_class_name: str = "creature",
        collision_policy: str = "error",
    ) -> None:
        self.root = Path(root)
        if class_mode not in self.CLASS_MODES:
            raise DatasetError("class_mode must be one of {}".format(sorted(self.CLASS_MODES)))
        if not isinstance(split, str) or not split.strip():
            raise DatasetError("split must be a non-empty string")
        if not isinstance(input_size, int) or input_size <= 0:
            raise DatasetError("input_size must be a positive integer")
        if not isinstance(stride, int) or stride <= 0 or input_size % stride != 0:
            raise DatasetError("input_size must be divisible by positive stride")
        if not isinstance(merged_class_name, str) or not merged_class_name.strip():
            raise DatasetError("merged_class_name must be a non-empty string")
        if collision_policy not in {"error", "keep_first"}:
            raise DatasetError(
                "collision_policy must be 'error' or 'keep_first'"
            )

        self.split = split
        self.input_size = input_size
        self.stride = stride
        self.class_mode = class_mode
        self.collision_policy = collision_policy
        self.source_class_names = _load_yolo_class_names(self.root / "data.yaml")
        if class_mode == "merge_single":
            self.class_names = (merged_class_name,)
        else:
            self.class_names = self.source_class_names

        self.images_dir, self.labels_dir = _resolve_split_directories(self.root, split)
        self.image_paths = tuple(
            sorted(
                path
                for path in self.images_dir.iterdir()
                if path.is_file() and path.suffix.lower() in self.IMAGE_SUFFIXES
            )
        )
        if not self.image_paths:
            raise DatasetError("image split contains no supported image files")

    @property
    def num_foreground_classes(self) -> int:
        """Return the foreground class count N used by this dataset."""

        return len(self.class_names)

    @property
    def output_channels(self) -> int:
        """Return background plus foreground output channels, ``1+N``."""

        return 1 + self.num_foreground_classes

    def __len__(self) -> int:
        """Return image count in the requested split."""

        return len(self.image_paths)

    def __getitem__(self, index: int) -> FOMOSample:
        """Load one image and produce normalized image, boxes, metadata, and labels."""

        image_path = self.image_paths[index]
        label_path = self.labels_dir / "{}.txt".format(image_path.stem)
        original_image = _load_rgb_image(image_path)
        original_height, original_width = original_image.shape[:2]
        source_boxes = parse_yolo_label_file(
            label_path, num_source_classes=len(self.source_class_names)
        )

        letterbox_image, transform = letterbox_rgb(original_image, self.input_size)
        original_boxes = []
        letterbox_boxes = []
        centroids = []
        for source_box in source_boxes:
            foreground_class_id = self._map_class_id(source_box.source_class_id)
            original_box = _normalized_to_absolute(
                source_box,
                original_width,
                original_height,
                foreground_class_id,
            )
            letterbox_box = AbsoluteBox(
                foreground_class_id=foreground_class_id,
                x_min=transform.forward_box(
                    original_box.x_min,
                    original_box.y_min,
                    original_box.x_max,
                    original_box.y_max,
                )[0],
                y_min=transform.forward_box(
                    original_box.x_min,
                    original_box.y_min,
                    original_box.x_max,
                    original_box.y_max,
                )[1],
                x_max=transform.forward_box(
                    original_box.x_min,
                    original_box.y_min,
                    original_box.x_max,
                    original_box.y_max,
                )[2],
                y_max=transform.forward_box(
                    original_box.x_min,
                    original_box.y_min,
                    original_box.x_max,
                    original_box.y_max,
                )[3],
            )
            original_boxes.append(original_box)
            letterbox_boxes.append(letterbox_box)
            centroid_x, centroid_y = transform.forward_point(*original_box.center)
            centroids.append((centroid_x, centroid_y, foreground_class_id))

        heatmap = generate_fomo_heatmap(
            centroids=centroids,
            input_size=self.input_size,
            stride=self.stride,
            num_foreground_classes=self.num_foreground_classes,
            collision_policy=self.collision_policy,
        )
        normalized_image = np.ascontiguousarray(
            letterbox_image.transpose(2, 0, 1), dtype=np.float32
        ) / 255.0
        return FOMOSample(
            image=normalized_image,
            original_image=original_image,
            letterbox_image=letterbox_image,
            original_boxes=tuple(original_boxes),
            letterbox_boxes=tuple(letterbox_boxes),
            transform=transform,
            heatmap=heatmap,
            image_path=image_path,
            label_path=label_path,
        )

    def _map_class_id(self, source_class_id: int) -> int:
        if self.class_mode == "merge_single":
            return 0
        return source_class_id


def parse_yolo_label_file(
    label_path: Union[str, Path], num_source_classes: int
) -> Tuple[NormalizedYoloBox, ...]:
    """Parse YOLO rows ``class_id x_center y_center width height``.

    Coordinates must be finite, normalized, positive-sized boxes fully inside
    the original image. A missing label file represents an image without
    targets; an existing malformed file raises ``YoloLabelError`` with line
    context.
    """

    path = Path(label_path)
    if not isinstance(num_source_classes, int) or num_source_classes <= 0:
        raise YoloLabelError("num_source_classes must be a positive integer")
    if not path.exists():
        return ()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise YoloLabelError("unable to read label file '{}': {}".format(path, error)) from error

    boxes = []
    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        fields = stripped.split()
        if len(fields) != 5:
            raise _label_error(path, line_number, "expected five values")
        try:
            source_class_id = int(fields[0])
        except ValueError as error:
            raise _label_error(path, line_number, "class_id must be an integer") from error
        if not 0 <= source_class_id < num_source_classes:
            raise _label_error(path, line_number, "class_id is outside data.yaml names")

        names = ("x_center", "y_center", "width", "height")
        values = []
        for name, raw_value in zip(names, fields[1:]):
            try:
                value = float(raw_value)
            except ValueError as error:
                raise _label_error(path, line_number, "{} must be numeric".format(name)) from error
            if not isfinite(value):
                raise _label_error(path, line_number, "{} must be finite".format(name))
            values.append(value)
        x_center, y_center, width, height = values
        if not 0.0 <= x_center <= 1.0:
            raise _label_error(path, line_number, "x_center must be in [0,1]")
        if not 0.0 <= y_center <= 1.0:
            raise _label_error(path, line_number, "y_center must be in [0,1]")
        if not 0.0 < width <= 1.0:
            raise _label_error(path, line_number, "width must be in (0,1]")
        if not 0.0 < height <= 1.0:
            raise _label_error(path, line_number, "height must be in (0,1]")
        if x_center - width / 2.0 < 0.0 or x_center + width / 2.0 > 1.0:
            raise _label_error(path, line_number, "x_center and width extend outside [0,1]")
        if y_center - height / 2.0 < 0.0 or y_center + height / 2.0 > 1.0:
            raise _label_error(path, line_number, "y_center and height extend outside [0,1]")
        boxes.append(
            NormalizedYoloBox(
                source_class_id=source_class_id,
                x_center=x_center,
                y_center=y_center,
                width=width,
                height=height,
            )
        )
    return tuple(boxes)


def _load_yolo_class_names(data_yaml_path: Path) -> Tuple[str, ...]:
    if not data_yaml_path.is_file():
        raise DatasetError("data.yaml does not exist: {}".format(data_yaml_path))
    try:
        payload = yaml.safe_load(data_yaml_path.read_text(encoding="utf-8"))
    except OSError as error:
        raise DatasetError("unable to read data.yaml: {}".format(error)) from error
    except yaml.YAMLError as error:
        raise DatasetError("unable to parse data.yaml: {}".format(error)) from error
    if not isinstance(payload, Mapping):
        raise DatasetError("data.yaml root must be a mapping")
    return _parse_class_names(payload.get("names"))


def _resolve_split_directories(root: Path, split: str) -> Tuple[Path, Path]:
    """Resolve project or Roboflow image/label directories for a logical split.

    Search order preserves the original project convention before accepting
    Roboflow's split-first convention. ``val`` and ``valid`` are aliases only
    after the exact requested name has been attempted.
    """

    split_names = [split]
    if split == "val":
        split_names.append("valid")
    elif split == "valid":
        split_names.append("val")

    attempted_images = []
    for split_name in split_names:
        project_images = root / "images" / split_name
        attempted_images.append(project_images)
        if project_images.is_dir():
            return project_images, root / "labels" / split_name
        roboflow_images = root / split_name / "images"
        attempted_images.append(roboflow_images)
        if roboflow_images.is_dir():
            return roboflow_images, root / split_name / "labels"

    raise DatasetError(
        "image split directory does not exist; attempted: {}".format(
            ", ".join(str(path) for path in attempted_images)
        )
    )


def _parse_class_names(raw_names: Any) -> Tuple[str, ...]:
    if isinstance(raw_names, Mapping):
        try:
            items = sorted((int(key), value) for key, value in raw_names.items())
        except (TypeError, ValueError) as error:
            raise DatasetError("data.yaml names mapping keys must be integers") from error
        if [index for index, _ in items] != list(range(len(items))):
            raise DatasetError("data.yaml names mapping keys must be contiguous from zero")
        raw_names = [value for _, value in items]
    if not isinstance(raw_names, Sequence) or isinstance(raw_names, (str, bytes)):
        raise DatasetError("data.yaml names must be a list or index mapping")
    if not raw_names:
        raise DatasetError("data.yaml names must contain at least one class")
    names = tuple(raw_names)
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise DatasetError("data.yaml names must contain non-empty strings")
    if len(set(names)) != len(names):
        raise DatasetError("data.yaml names must not contain duplicates")
    return names


def _load_rgb_image(image_path: Path) -> np.ndarray:
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise DatasetError("unable to read image: {}".format(image_path))
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _normalized_to_absolute(
    box: NormalizedYoloBox,
    image_width: int,
    image_height: int,
    foreground_class_id: int,
) -> AbsoluteBox:
    center_x = box.x_center * image_width
    center_y = box.y_center * image_height
    half_width = box.width * image_width / 2.0
    half_height = box.height * image_height / 2.0
    return AbsoluteBox(
        foreground_class_id=foreground_class_id,
        x_min=center_x - half_width,
        y_min=center_y - half_height,
        x_max=center_x + half_width,
        y_max=center_y + half_height,
    )


def _label_error(path: Path, line_number: int, message: str) -> YoloLabelError:
    return YoloLabelError("{}: line {}: {}".format(path, line_number, message))
