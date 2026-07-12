"""Validation collection using the same logits postprocessor as inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence

import torch
from torch import Tensor, nn

from fomo_servo.config import EvaluationConfig, PostprocessConfig, ProjectConfig
from fomo_servo.datasets import YOLOv5FOMODataset
from fomo_servo.metrics import (
    CentroidEvaluation,
    CentroidEvaluator,
    GridMetrics,
    ground_truths_from_boxes,
    foreground_micro_metrics,
    sweep_confidence_thresholds,
)
from fomo_servo.postprocess import postprocess_logits


@dataclass(frozen=True)
class ValidationReport:
    """Grid result, configured-threshold centroid result, and best sweep result."""

    grid_metrics: GridMetrics
    centroid_metrics: CentroidEvaluation
    best_threshold: float
    best_centroid_metrics: CentroidEvaluation

    def as_dict(
        self,
        *,
        class_weights: Optional[Sequence[float]] = None,
        class_weight_mode: Optional[str] = None,
        class_statistics: Optional[Sequence[Mapping[str, object]]] = None,
    ) -> dict[str, object]:
        """Return a JSON-compatible validation report and optional training evidence."""

        result = self.best_centroid_metrics
        return {
            "grid_precision": self.grid_metrics.grid_precision,
            "grid_recall": self.grid_metrics.grid_recall,
            "grid_f1": self.grid_metrics.grid_f1,
            "centroid_precision": result.centroid_precision,
            "centroid_recall": result.centroid_recall,
            "centroid_f1": result.centroid_f1,
            "configured_threshold_centroid_precision": self.centroid_metrics.centroid_precision,
            "configured_threshold_centroid_recall": self.centroid_metrics.centroid_recall,
            "configured_threshold_centroid_f1": self.centroid_metrics.centroid_f1,
            "best_confidence_threshold": self.best_threshold,
            "per_class_precision_recall_f1": {
                name: dict(values)
                for name, values in result.per_class_precision_recall_f1.items()
            },
            "confusion_matrix": result.confusion_matrix.tolist(),
            "mean_localization_error_pixels": result.mean_localization_error_pixels,
            "median_localization_error_pixels": result.median_localization_error_pixels,
            "mean_count_bias": result.mean_count_bias,
            "mean_absolute_count_error": result.mean_absolute_count_error,
            # Kept for consumers of reports generated before the shorter name.
            "average_absolute_count_error_per_image": result.mean_absolute_count_error,
            "count_error_per_image": list(result.count_error_per_image),
            "class_weight_mode": class_weight_mode,
            "class_weights": list(class_weights) if class_weights is not None else None,
            "class_statistics": (
                [dict(item) for item in class_statistics]
                if class_statistics is not None
                else None
            ),
        }


def evaluate_logit_collection(
    *,
    logits: Sequence[Tensor],
    targets: Sequence[Tensor],
    transforms: Sequence[object],
    ground_truths: Sequence[Sequence[object]],
    class_names: Sequence[str],
    stride: int,
    postprocess_config: PostprocessConfig,
    evaluation_config: EvaluationConfig,
) -> ValidationReport:
    """Evaluate already-collected per-image logits and metadata on CPU.

    ``logits`` items are ``[C,G,G]``; ``targets`` items are class-index ``[G,G]``.
    The transform and ground-truth sequences retain one item per image.
    """

    if not (len(logits) == len(targets) == len(transforms) == len(ground_truths)):
        raise ValueError("validation collections must have equal lengths")
    normalized_logits = []
    for item in logits:
        if item.ndim == 4 and item.shape[0] == 1:
            item = item[0]
        if item.ndim != 3:
            raise ValueError("each logits item must have shape [C,G,G] or [1,C,G,G]")
        normalized_logits.append(item)
    prediction_tensors = torch.stack([target.to(dtype=torch.int64) for target in targets], dim=0)
    predicted_tensors = torch.stack(
        [item.argmax(dim=0).to(dtype=torch.int64) for item in normalized_logits], dim=0
    )
    grid_metrics = foreground_micro_metrics(predicted_tensors, prediction_tensors)
    evaluator = CentroidEvaluator(
        class_names,
        matching_mode=evaluation_config.matching_mode,
        max_distance_pixels=evaluation_config.max_distance_pixels,
    )
    configured_predictions = []
    for image_logits, transform in zip(normalized_logits, transforms):
        configured_predictions.append(
            postprocess_logits(
                image_logits.unsqueeze(0),
                class_names=class_names,
                stride=stride,
                transforms=(transform,),
                confidence_threshold=postprocess_config.inference_threshold,
                class_thresholds=postprocess_config.class_thresholds,
                component_mode=postprocess_config.component_mode,
                confidence_mode=postprocess_config.confidence_mode,
            )[0]
        )
    centroid_metrics = evaluator.evaluate_dataset(configured_predictions, ground_truths)
    sweep = sweep_confidence_thresholds(
        logits=tuple(normalized_logits),
        transforms=transforms,
        ground_truths=ground_truths,
        class_names=class_names,
        stride=stride,
        thresholds=evaluation_config.threshold_sweep,
        matching_mode=evaluation_config.matching_mode,
        max_distance_pixels=evaluation_config.max_distance_pixels,
    )
    return ValidationReport(
        grid_metrics=grid_metrics,
        centroid_metrics=centroid_metrics,
        best_threshold=sweep.best_threshold,
        best_centroid_metrics=sweep.best_result,
    )


def evaluate_validation_dataset(
    config: ProjectConfig, model: nn.Module, device: torch.device
) -> ValidationReport:
    """Run the full configured validation split with a loaded model."""

    dataset = YOLOv5FOMODataset(
        root=config.dataset.root,
        split=config.dataset.validation_split,
        input_size=config.model.input_size,
        stride=config.model.output_stride,
        class_mode=config.dataset.class_mode,
        merged_class_name=config.dataset.merged_class_name,
        collision_policy=config.dataset.collision_policy,
    )
    logits = []
    targets = []
    transforms = []
    ground_truths = []
    model.eval()
    with torch.no_grad():
        for sample in dataset:
            image = torch.from_numpy(sample.image).unsqueeze(0).to(device)
            logits.append(model(image)[0].detach().cpu())
            targets.append(torch.from_numpy(sample.heatmap.class_index))
            transforms.append(sample.transform)
            ground_truths.append(
                ground_truths_from_boxes(sample.original_boxes, config.dataset.class_names)
            )
    return evaluate_logit_collection(
        logits=tuple(logits),
        targets=tuple(targets),
        transforms=tuple(transforms),
        ground_truths=tuple(ground_truths),
        class_names=config.dataset.class_names,
        stride=config.model.output_stride,
        postprocess_config=config.postprocess,
        evaluation_config=config.evaluation,
    )
