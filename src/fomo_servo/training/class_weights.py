"""Training-split heatmap statistics and manual/automatic loss weight resolution."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt
from statistics import median
from typing import TYPE_CHECKING, Sequence, Tuple

import numpy as np

from fomo_servo.datasets import YOLOv5FOMODataset

if TYPE_CHECKING:
    from fomo_servo.config import LossConfig


class ClassWeightError(ValueError):
    """Raised when configured or dataset-derived class weights are invalid."""


@dataclass(frozen=True)
class AutoClassWeightSettings:
    """YAML-derived automatic foreground balancing controls."""

    background_weight: float
    foreground_base_weight: float
    class_balance: str
    min_foreground_weight: float
    max_foreground_weight: float

    def __post_init__(self) -> None:
        values = (
            self.background_weight,
            self.foreground_base_weight,
            self.min_foreground_weight,
            self.max_foreground_weight,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(float(value))
            or value <= 0
            for value in values
        ):
            raise ClassWeightError("auto class-weight settings must be finite positive numbers")
        if self.min_foreground_weight > self.max_foreground_weight:
            raise ClassWeightError("min_foreground_weight must not exceed max_foreground_weight")
        if self.class_balance != "sqrt_inverse_frequency":
            raise ClassWeightError("class_balance must be 'sqrt_inverse_frequency'")


@dataclass(frozen=True)
class ClassTrainingStatistics:
    """Per-foreground-class counts from existing training labels and heatmaps."""

    class_id: int
    class_name: str
    image_count: int
    bbox_count: int
    encoded_centroid_cell_count: int
    same_class_collision_count: int
    different_class_collision_count: int

    def as_dict(self) -> dict[str, int | str]:
        """Return a JSON-compatible per-class statistics record."""

        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "image_count": self.image_count,
            "bbox_count": self.bbox_count,
            "encoded_centroid_cell_count": self.encoded_centroid_cell_count,
            "same_class_collision_count": self.same_class_collision_count,
            "different_class_collision_count": self.different_class_collision_count,
        }


@dataclass(frozen=True)
class ResolvedClassWeights:
    """Final background-plus-foreground weights and evidence used to derive them."""

    mode: str
    weights: Tuple[float, ...]
    statistics: Tuple[ClassTrainingStatistics, ...]

    def as_dict(self) -> dict[str, object]:
        """Return JSON-compatible resolved weights and per-class counts."""

        return {
            "mode": self.mode,
            "class_weights": list(self.weights),
            "class_statistics": [item.as_dict() for item in self.statistics],
        }


def collect_training_heatmap_statistics(
    dataset: YOLOv5FOMODataset,
) -> Tuple[ClassTrainingStatistics, ...]:
    """Count labels and encoded heatmap cells without modifying dataset target generation.

    ``same_class_collision_count`` and ``different_class_collision_count`` mirror
    the existing heatmap generator's encounter order. Different-class collisions
    are attributed to the incoming class that could not replace the first encoded
    class in its grid cell.
    """

    if not isinstance(dataset, YOLOv5FOMODataset):
        raise ClassWeightError("dataset must be a YOLOv5FOMODataset")
    class_count = dataset.num_foreground_classes
    image_counts = np.zeros(class_count, dtype=np.int64)
    bbox_counts = np.zeros(class_count, dtype=np.int64)
    encoded_counts = np.zeros(class_count, dtype=np.int64)
    same_collisions = np.zeros(class_count, dtype=np.int64)
    different_collisions = np.zeros(class_count, dtype=np.int64)

    for sample in dataset:
        present_classes = set()
        first_class_by_cell: dict[tuple[int, int], int] = {}
        for box in sample.letterbox_boxes:
            class_id = box.foreground_class_id
            present_classes.add(class_id)
            bbox_counts[class_id] += 1
            center_x, center_y = box.center
            grid_x = min(max(int(center_x // dataset.stride), 0), dataset.input_size // dataset.stride - 1)
            grid_y = min(max(int(center_y // dataset.stride), 0), dataset.input_size // dataset.stride - 1)
            cell = (grid_x, grid_y)
            existing = first_class_by_cell.get(cell)
            if existing is None:
                first_class_by_cell[cell] = class_id
            elif existing == class_id:
                same_collisions[class_id] += 1
            else:
                different_collisions[class_id] += 1
        for class_id in present_classes:
            image_counts[class_id] += 1
        encoded = np.bincount(
            sample.heatmap.class_index.ravel(), minlength=class_count + 1
        )[1 : class_count + 1]
        encoded_counts += encoded.astype(np.int64)

    return tuple(
        ClassTrainingStatistics(
            class_id=class_id,
            class_name=dataset.class_names[class_id],
            image_count=int(image_counts[class_id]),
            bbox_count=int(bbox_counts[class_id]),
            encoded_centroid_cell_count=int(encoded_counts[class_id]),
            same_class_collision_count=int(same_collisions[class_id]),
            different_class_collision_count=int(different_collisions[class_id]),
        )
        for class_id in range(class_count)
    )


def resolve_auto_class_weights(
    statistics: Sequence[ClassTrainingStatistics], settings: AutoClassWeightSettings
) -> Tuple[float, ...]:
    """Return ``[background, foreground...]`` weights from encoded centroid cells."""

    if not statistics:
        raise ClassWeightError("automatic class weights require at least one foreground class")
    ordered = tuple(sorted(statistics, key=lambda item: item.class_id))
    if [item.class_id for item in ordered] != list(range(len(ordered))):
        raise ClassWeightError("class statistics IDs must be contiguous from zero")
    counts = [item.encoded_centroid_cell_count for item in ordered]
    for item, count in zip(ordered, counts):
        if count <= 0:
            raise ClassWeightError(
                "encoded centroid cell count is zero for class '{}'".format(item.class_name)
            )
    median_count = float(median(counts))
    foreground = []
    for count in counts:
        raw_weight = settings.foreground_base_weight * sqrt(median_count / count)
        foreground.append(
            min(settings.max_foreground_weight, max(settings.min_foreground_weight, raw_weight))
        )
    return (float(settings.background_weight), *tuple(float(value) for value in foreground))


def resolve_training_class_weights(
    loss_config: "LossConfig", dataset: YOLOv5FOMODataset
) -> ResolvedClassWeights:
    """Collect evidence and resolve legacy weights without stacking object weights."""

    statistics = collect_training_heatmap_statistics(dataset)
    loss_name = getattr(loss_config, "name", "")
    if loss_name in {"weighted_softmax_ce", "ei_weighted_xent_legacy"}:
        # These losses own the background-vs-object weighting.  Unit values are
        # retained only as explicit metadata for consumers of old checkpoints;
        # the loss implementation does not read them.
        return ResolvedClassWeights(
            "disabled",
            tuple(1.0 for _ in range(dataset.num_foreground_classes + 1)),
            statistics,
        )
    mode = getattr(loss_config, "class_weight_mode", "manual")
    if mode == "manual":
        weights = getattr(loss_config, "class_weights", None)
        if weights is None:
            raise ClassWeightError("manual class-weight mode requires class_weights")
        if len(weights) != dataset.num_foreground_classes + 1:
            raise ClassWeightError("manual class_weights length does not match dataset classes")
        return ResolvedClassWeights("manual", tuple(float(value) for value in weights), statistics)
    if mode != "auto":
        raise ClassWeightError("class_weight_mode must be 'manual' or 'auto'")
    settings = AutoClassWeightSettings(
        background_weight=getattr(loss_config, "background_weight"),
        foreground_base_weight=getattr(loss_config, "foreground_base_weight"),
        class_balance=getattr(loss_config, "class_balance"),
        min_foreground_weight=getattr(loss_config, "min_foreground_weight"),
        max_foreground_weight=getattr(loss_config, "max_foreground_weight"),
    )
    return ResolvedClassWeights(
        "auto", resolve_auto_class_weights(statistics, settings), statistics
    )
