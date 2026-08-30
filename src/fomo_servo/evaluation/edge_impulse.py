"""Edge Impulse FOMO-compatible decoding and centroid evaluation.

This module is deliberately separate from the repository's regular evaluator.
``edge_impulse_legacy`` reproduces the cited public matching behavior, including
multiple same-class predictions being allowed to match one ground truth.
``strict_one_to_one`` is retained as the engineering/scientific alternative.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot, isfinite
from statistics import mean, median
from typing import Literal, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor


class EdgeImpulseMetricError(ValueError):
    """Raised for incompatible tensors or invalid centroid evaluation inputs."""


MatchingMode = Literal["edge_impulse_legacy", "strict_one_to_one"]


@dataclass(frozen=True)
class EdgeImpulseDetection:
    """One FOMO detection in input and original-image pixel coordinates.

    ``input_bbox`` is ``(x, y, width, height)`` in the model input image.
    ``input_centroid`` and ``original_centroid`` are ``(x, y)`` pixels.
    """

    class_id: int
    class_name: str
    confidence: float
    input_bbox: tuple[float, float, float, float]
    input_centroid: tuple[float, float]
    original_centroid: tuple[float, float]


@dataclass(frozen=True)
class EdgeImpulseGroundTruth:
    """One original-image centroid target with foreground class metadata."""

    class_id: int
    class_name: str
    original_centroid: tuple[float, float]


@dataclass(frozen=True)
class EdgeImpulseMatch:
    """A prediction/ground-truth assignment and normalized centroid distance."""

    prediction_index: int
    ground_truth_index: int
    normalized_distance: float


@dataclass(frozen=True)
class EdgeImpulseImageEvaluation:
    """One image's assignments and unmatched object indices."""

    matches: tuple[EdgeImpulseMatch, ...]
    unmatched_prediction_indices: tuple[int, ...]
    unmatched_ground_truth_indices: tuple[int, ...]


@dataclass(frozen=True)
class EdgeImpulseEvaluation:
    """Dataset metrics with foreground aggregation and image-level provenance."""

    mode: MatchingMode
    precision: float
    recall: float
    f1: float
    macro_f1: float
    per_class_precision_recall_f1: Mapping[str, Mapping[str, float]]
    true_positives: int
    false_positives: int
    false_negatives: int
    prediction_count: int
    ground_truth_count: int
    mean_localization_error_normalized: float
    median_localization_error_normalized: float
    count_error_per_image: tuple[int, ...]
    mean_count_bias: float
    mean_absolute_count_error: float
    non_background_confusion: np.ndarray
    image_results: tuple[EdgeImpulseImageEvaluation, ...]


@dataclass
class _Cube:
    """Mutable raster-space unit-cell union used by the public EI algorithm."""

    x: int
    y: int
    width: int
    height: int
    confidence: float
    class_id: int


def probabilities_from_logits(logits: Tensor) -> Tensor:
    """Return HWC probabilities from local FOMO logits.

    ``logits`` must be ``float [1,C,H,W]`` or ``float [C,H,W]`` where channel
    zero is background. The result is ``float32 [H,W,C]`` and is softmaxed once.
    """

    if not isinstance(logits, Tensor):
        raise EdgeImpulseMetricError("logits must be a torch Tensor")
    value = logits.detach().to(dtype=torch.float32)
    if value.ndim == 4:
        if value.shape[0] != 1:
            raise EdgeImpulseMetricError("batched logits must have batch size one")
        value = value[0]
    if value.ndim != 3 or value.shape[0] < 2:
        raise EdgeImpulseMetricError("logits must have shape [C,H,W] with background plus foreground")
    return value.softmax(dim=0).permute(1, 2, 0).contiguous().cpu()


def decode_edge_impulse_fomo(
    probabilities: Tensor | np.ndarray,
    *,
    class_names: Sequence[str],
    input_size: tuple[int, int],
    threshold: float,
) -> tuple[EdgeImpulseDetection, ...]:
    """Decode EI FOMO output probabilities with the public cube merge semantics.

    ``probabilities`` is ``float [H,W,1+N]`` or ``[1,H,W,1+N]`` with channel
    zero as background. This function never softmaxes, so callers with local
    logits must call :func:`probabilities_from_logits` first. Returned input and
    original centroids initially match; callers apply letterbox inversion later.
    """

    names = _validate_class_names(class_names)
    width, height = _validate_input_size(input_size)
    normalized_threshold = _validate_threshold(threshold)
    values = _as_hwc_probability_array(probabilities)
    grid_height, grid_width, channels = values.shape
    if channels != len(names) + 1:
        raise EdgeImpulseMetricError("probability channels must equal background plus class_names")
    if width % grid_width != 0 or height % grid_height != 0:
        raise EdgeImpulseMetricError("input_size must be divisible by FOMO output grid size")
    if not np.isfinite(values).all():
        raise EdgeImpulseMetricError("probabilities must be finite")
    if (values < 0.0).any():
        raise EdgeImpulseMetricError("probabilities must be non-negative")

    cubes: list[_Cube] = []
    # EI scans y first, x second and ignores background channel zero.
    for grid_y in range(grid_height):
        for grid_x in range(grid_width):
            for channel in range(1, channels):
                _ei_handle_cube(
                    cubes,
                    grid_x,
                    grid_y,
                    float(values[grid_y, grid_x, channel]),
                    channel - 1,
                    normalized_threshold,
                )
    boxes: list[_Cube] = []
    detections: list[EdgeImpulseDetection] = []
    x_factor = width // grid_width
    y_factor = height // grid_height
    for cube in cubes:
        has_overlapping = False
        for existing in boxes:
            if existing.class_id != cube.class_id:
                continue
            if _ei_cube_check_overlap(
                existing, cube.x, cube.y, cube.width, cube.height, cube.confidence
            ):
                has_overlapping = True
                break
        if has_overlapping:
            continue
        boxes.append(cube)
        bbox = (
            float(cube.x * x_factor),
            float(cube.y * y_factor),
            float(cube.width * x_factor),
            float(cube.height * y_factor),
        )
        centroid = (bbox[0] + bbox[2] / 2.0, bbox[1] + bbox[3] / 2.0)
        detections.append(
            EdgeImpulseDetection(
                class_id=cube.class_id,
                class_name=names[cube.class_id],
                confidence=cube.confidence,
                input_bbox=bbox,
                input_centroid=centroid,
                original_centroid=centroid,
            )
        )
    return tuple(detections)


def normalized_centroid_distance(
    first: tuple[float, float], second: tuple[float, float], image_width: float, image_height: float
) -> float:
    """Return ``sqrt((dx/W)^2 + (dy/H)^2)`` for original-image centroids."""

    if image_width <= 0.0 or image_height <= 0.0:
        raise EdgeImpulseMetricError("image dimensions must be positive")
    return hypot((first[0] - second[0]) / image_width, (first[1] - second[1]) / image_height)


class EdgeImpulseCentroidEvaluator:
    """Evaluate foreground centroids under legacy or strict assignment semantics."""

    def __init__(self, class_names: Sequence[str], *, mode: MatchingMode, distance_threshold: float = 0.2) -> None:
        self.class_names = _validate_class_names(class_names)
        if mode not in {"edge_impulse_legacy", "strict_one_to_one"}:
            raise EdgeImpulseMetricError("unsupported matching mode: {}".format(mode))
        self.mode = mode
        self.distance_threshold = _validate_threshold(distance_threshold)

    def evaluate_dataset(
        self,
        predictions: Sequence[Sequence[EdgeImpulseDetection]],
        ground_truths: Sequence[Sequence[EdgeImpulseGroundTruth]],
        *,
        image_sizes: Sequence[tuple[int, int]],
    ) -> EdgeImpulseEvaluation:
        """Evaluate aligned image lists; image sizes are original ``(W,H)`` pixels."""

        if not (len(predictions) == len(ground_truths) == len(image_sizes)):
            raise EdgeImpulseMetricError("predictions, ground_truths, and image_sizes must align")
        class_counts = {name: [0, 0, 0] for name in self.class_names}
        confusion = np.zeros((len(self.class_names), len(self.class_names)), dtype=np.int64)
        image_results = []
        distances = []
        count_errors = []
        tp = fp = fn = prediction_count = ground_truth_count = 0
        for image_predictions, image_targets, image_size in zip(predictions, ground_truths, image_sizes):
            _validate_items(image_predictions, image_targets, self.class_names)
            width, height = _validate_input_size(image_size)
            result = self._evaluate_image(image_predictions, image_targets, width, height)
            image_results.append(result)
            prediction_count += len(image_predictions)
            ground_truth_count += len(image_targets)
            count_errors.append(len(image_predictions) - len(image_targets))
            tp += len(result.matches)
            fp += len(result.unmatched_prediction_indices)
            fn += len(result.unmatched_ground_truth_indices)
            for match in result.matches:
                prediction = image_predictions[match.prediction_index]
                target = image_targets[match.ground_truth_index]
                class_counts[target.class_name][0] += 1
                confusion[target.class_id, prediction.class_id] += 1
                distances.append(match.normalized_distance)
            for index in result.unmatched_prediction_indices:
                class_counts[image_predictions[index].class_name][1] += 1
            for index in result.unmatched_ground_truth_indices:
                class_counts[image_targets[index].class_name][2] += 1
        per_class = {name: _metric_dict(*values) for name, values in class_counts.items()}
        precision = _precision(tp, fp)
        recall = _recall(tp, fn)
        return EdgeImpulseEvaluation(
            mode=self.mode,
            precision=precision,
            recall=recall,
            f1=_f1(precision, recall),
            macro_f1=mean(values["f1"] for values in per_class.values()),
            per_class_precision_recall_f1=per_class,
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            prediction_count=prediction_count,
            ground_truth_count=ground_truth_count,
            mean_localization_error_normalized=mean(distances) if distances else 0.0,
            median_localization_error_normalized=median(distances) if distances else 0.0,
            count_error_per_image=tuple(count_errors),
            mean_count_bias=mean(count_errors) if count_errors else 0.0,
            mean_absolute_count_error=mean(abs(value) for value in count_errors) if count_errors else 0.0,
            non_background_confusion=confusion,
            image_results=tuple(image_results),
        )

    def _evaluate_image(
        self,
        predictions: Sequence[EdgeImpulseDetection],
        ground_truths: Sequence[EdgeImpulseGroundTruth],
        width: int,
        height: int,
    ) -> EdgeImpulseImageEvaluation:
        if self.mode == "edge_impulse_legacy":
            return _legacy_matches(predictions, ground_truths, width, height, self.distance_threshold)
        return _strict_matches(predictions, ground_truths, width, height, self.distance_threshold)


def _legacy_matches(
    predictions: Sequence[EdgeImpulseDetection], ground_truths: Sequence[EdgeImpulseGroundTruth], width: int, height: int, threshold: float
) -> EdgeImpulseImageEvaluation:
    matches = []
    matched_targets = set()
    unmatched_predictions = []
    for prediction_index, prediction in enumerate(predictions):
        candidates = []
        for target_index, target in enumerate(ground_truths):
            if target.class_id != prediction.class_id:
                continue
            distance = normalized_centroid_distance(prediction.original_centroid, target.original_centroid, width, height)
            if distance <= threshold:
                candidates.append((distance, target_index))
        if not candidates:
            unmatched_predictions.append(prediction_index)
            continue
        distance, target_index = min(candidates, key=lambda item: (item[0], item[1]))
        matches.append(EdgeImpulseMatch(prediction_index, target_index, distance))
        matched_targets.add(target_index)
    return EdgeImpulseImageEvaluation(
        matches=tuple(matches),
        unmatched_prediction_indices=tuple(unmatched_predictions),
        unmatched_ground_truth_indices=tuple(index for index in range(len(ground_truths)) if index not in matched_targets),
    )


def _strict_matches(
    predictions: Sequence[EdgeImpulseDetection], ground_truths: Sequence[EdgeImpulseGroundTruth], width: int, height: int, threshold: float
) -> EdgeImpulseImageEvaluation:
    pairs = []
    for prediction_index, prediction in enumerate(predictions):
        for target_index, target in enumerate(ground_truths):
            if target.class_id != prediction.class_id:
                continue
            distance = normalized_centroid_distance(prediction.original_centroid, target.original_centroid, width, height)
            if distance <= threshold:
                pairs.append((distance, prediction_index, target_index))
    used_predictions = set()
    used_targets = set()
    matches = []
    for distance, prediction_index, target_index in sorted(pairs):
        if prediction_index in used_predictions or target_index in used_targets:
            continue
        used_predictions.add(prediction_index)
        used_targets.add(target_index)
        matches.append(EdgeImpulseMatch(prediction_index, target_index, distance))
    return EdgeImpulseImageEvaluation(
        matches=tuple(matches),
        unmatched_prediction_indices=tuple(index for index in range(len(predictions)) if index not in used_predictions),
        unmatched_ground_truth_indices=tuple(index for index in range(len(ground_truths)) if index not in used_targets),
    )


def _ei_handle_cube(cubes: list[_Cube], x: int, y: int, confidence: float, class_id: int, threshold: float) -> None:
    # Exact EI comparison: confidence equal to threshold is active.
    if confidence < threshold:
        return
    for cube in cubes:
        if cube.class_id != class_id:
            continue
        if _ei_cube_check_overlap(cube, x, y, 1, 1, confidence):
            return
    cubes.append(_Cube(x=x, y=y, width=1, height=1, confidence=confidence, class_id=class_id))


def _ei_cube_check_overlap(cube: _Cube, x: int, y: int, width: int, height: int, confidence: float) -> bool:
    # Mirrors `ei_cube_check_overlap` from the cited inference SDK, including
    # its assignment order in the x/y lower-bound branches.
    is_overlapping = not (cube.x + cube.width < x or cube.y + cube.height < y or cube.x > x + width or cube.y > y + height)
    if not is_overlapping:
        return False
    if x < cube.x:
        cube.x = x
        cube.width += cube.x - x
    if y < cube.y:
        cube.y = y
        cube.height += cube.y - y
    if x + width > cube.x + cube.width:
        cube.width += x + width - (cube.x + cube.width)
    if y + height > cube.y + cube.height:
        cube.height += y + height - (cube.y + cube.height)
    if confidence > cube.confidence:
        cube.confidence = confidence
    return True


def _as_hwc_probability_array(probabilities: Tensor | np.ndarray) -> np.ndarray:
    if isinstance(probabilities, Tensor):
        values = probabilities.detach().to(dtype=torch.float32, device="cpu").numpy()
    elif isinstance(probabilities, np.ndarray):
        values = probabilities.astype(np.float32, copy=False)
    else:
        raise EdgeImpulseMetricError("probabilities must be a torch Tensor or numpy array")
    if values.ndim == 4:
        if values.shape[0] != 1:
            raise EdgeImpulseMetricError("batched probabilities must have batch size one")
        values = values[0]
    if values.ndim != 3 or values.shape[2] < 2:
        raise EdgeImpulseMetricError("probabilities must have shape [H,W,C] with background plus foreground")
    return values


def _validate_class_names(class_names: Sequence[str]) -> tuple[str, ...]:
    names = tuple(class_names)
    if not names or any(not isinstance(name, str) or not name.strip() for name in names) or len(set(names)) != len(names):
        raise EdgeImpulseMetricError("class_names must contain unique non-empty strings")
    return names


def _validate_input_size(input_size: tuple[int, int]) -> tuple[int, int]:
    if len(input_size) != 2:
        raise EdgeImpulseMetricError("input_size must be (width, height)")
    width, height = input_size
    if isinstance(width, bool) or isinstance(height, bool) or not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
        raise EdgeImpulseMetricError("input_size values must be positive integers")
    return width, height


def _validate_threshold(threshold: float) -> float:
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not isfinite(float(threshold)) or not 0.0 <= float(threshold) <= 1.0:
        raise EdgeImpulseMetricError("threshold must be a finite probability in [0,1]")
    return float(threshold)


def _validate_items(predictions: Sequence[EdgeImpulseDetection], ground_truths: Sequence[EdgeImpulseGroundTruth], class_names: tuple[str, ...]) -> None:
    for item in predictions:
        if not 0 <= item.class_id < len(class_names) or item.class_name != class_names[item.class_id]:
            raise EdgeImpulseMetricError("prediction class metadata is invalid")
    for item in ground_truths:
        if not 0 <= item.class_id < len(class_names) or item.class_name != class_names[item.class_id]:
            raise EdgeImpulseMetricError("ground-truth class metadata is invalid")


def _metric_dict(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = _precision(tp, fp)
    recall = _recall(tp, fn)
    return {"precision": precision, "recall": recall, "f1": _f1(precision, recall), "true_positives": float(tp), "false_positives": float(fp), "false_negatives": float(fn)}


def _precision(tp: int, fp: int) -> float:
    return tp / (tp + fp) if tp + fp else 0.0


def _recall(tp: int, fn: int) -> float:
    return tp / (tp + fn) if tp + fn else 0.0


def _f1(precision: float, recall: float) -> float:
    return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0


__all__ = [
    "EdgeImpulseCentroidEvaluator",
    "EdgeImpulseDetection",
    "EdgeImpulseEvaluation",
    "EdgeImpulseGroundTruth",
    "EdgeImpulseImageEvaluation",
    "EdgeImpulseMatch",
    "EdgeImpulseMetricError",
    "decode_edge_impulse_fomo",
    "normalized_centroid_distance",
    "probabilities_from_logits",
]
