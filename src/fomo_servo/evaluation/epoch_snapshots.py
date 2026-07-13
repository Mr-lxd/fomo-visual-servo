"""FP32 offline evaluation and deterministic selection of FOMO epoch snapshots."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from statistics import mean, median
from typing import Any, Mapping, Optional, Sequence

import torch
from torch import Tensor, nn

from fomo_servo.config import ProjectConfig
from fomo_servo.datasets import YOLOv5FOMODataset
from fomo_servo.datasets.yolo import DatasetError
from fomo_servo.evaluation.validation import evaluate_logit_collection
from fomo_servo.losses import build_classification_loss
from fomo_servo.metrics import centroid_pr_auc, ground_truths_from_boxes, sweep_confidence_thresholds
from fomo_servo.postprocess import postprocess_logits


class CheckpointSelectionError(RuntimeError):
    """Raised when offline checkpoint selection or calibration is invalid."""


@dataclass(frozen=True)
class LogitCollection:
    """CPU FP32 validation tensors and geometry retained for threshold evaluation."""

    logits: tuple[Tensor, ...]
    targets: tuple[Tensor, ...]
    transforms: tuple[object, ...]
    ground_truths: tuple[tuple[object, ...], ...]
    validation_loss: float
    confidence_summary: Mapping[str, Mapping[str, float | int | None]]


@dataclass(frozen=True)
class EpochSelectionReport:
    """JSON/CSV-safe metrics for one fully evaluated snapshot or legacy checkpoint."""

    epoch: int
    source_snapshot: str
    checkpoint_path: str
    validation_loss: float
    grid_precision: float
    grid_recall: float
    grid_f1: float
    fixed_centroid_precision: float
    fixed_centroid_recall: float
    fixed_centroid_f1: float
    sweep_centroid_precision: float
    sweep_centroid_recall: float
    sweep_centroid_f1: float
    sweep_threshold: float
    centroid_pr_auc_macro: Optional[float]
    centroid_pr_auc_micro: Optional[float]
    macro_effective_class_count: int
    mean_localization_error_pixels: float
    median_localization_error_pixels: float
    mean_count_bias: float
    count_mae: float
    fixed_detection_count: int
    confidence_summary: Mapping[str, Mapping[str, float | int | None]]
    fixed_per_class: Mapping[str, Mapping[str, float]]
    sweep_per_class: Mapping[str, Mapping[str, float]]
    pr_curves: Mapping[str, object]

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-compatible report with raw curves and per-class metrics."""

        return {
            "epoch": self.epoch,
            "source_snapshot": self.source_snapshot,
            "checkpoint_path": self.checkpoint_path,
            "validation_loss": self.validation_loss,
            "grid_precision": self.grid_precision,
            "grid_recall": self.grid_recall,
            "grid_f1": self.grid_f1,
            "fixed_centroid_precision": self.fixed_centroid_precision,
            "fixed_centroid_recall": self.fixed_centroid_recall,
            "fixed_centroid_f1": self.fixed_centroid_f1,
            "sweep_centroid_precision": self.sweep_centroid_precision,
            "sweep_centroid_recall": self.sweep_centroid_recall,
            "sweep_centroid_f1": self.sweep_centroid_f1,
            "sweep_threshold": self.sweep_threshold,
            "centroid_pr_auc_macro": self.centroid_pr_auc_macro,
            "centroid_pr_auc_micro": self.centroid_pr_auc_micro,
            "macro_effective_class_count": self.macro_effective_class_count,
            "mean_localization_error_pixels": self.mean_localization_error_pixels,
            "median_localization_error_pixels": self.median_localization_error_pixels,
            "mean_count_bias": self.mean_count_bias,
            "count_mae": self.count_mae,
            "fixed_detection_count": self.fixed_detection_count,
            "confidence_summary": {key: dict(value) for key, value in self.confidence_summary.items()},
            "fixed_per_class": {key: dict(value) for key, value in self.fixed_per_class.items()},
            "sweep_per_class": {key: dict(value) for key, value in self.sweep_per_class.items()},
            "pr_curves": dict(self.pr_curves),
            "selection_dtype": "float32",
        }


def collect_split_logits(
    config: ProjectConfig, model: nn.Module, device: torch.device, split: str
) -> LogitCollection:
    """Evaluate one split in strict FP32 inference mode with augmentation disabled."""

    if not isinstance(split, str) or not split.strip():
        raise CheckpointSelectionError("evaluation split must be a non-empty string")
    try:
        dataset = YOLOv5FOMODataset(
            root=config.dataset.root,
            split=split,
            input_size=config.model.input_size,
            stride=config.model.output_stride,
            class_mode=config.dataset.class_mode,
            merged_class_name=config.dataset.merged_class_name,
            collision_policy=config.dataset.collision_policy,
            augmentation=None,
            train_split=config.dataset.train_split,
            augmentation_seed=config.training.seed,
        )
    except DatasetError as error:
        raise CheckpointSelectionError(
            "requested evaluation split '{}' is unavailable: {}".format(split, error)
        ) from error
    if dataset.is_train:
        raise CheckpointSelectionError(
            "offline selection split '{}' resolves to the train split; validation/test "
            "selection must not use random augmentation".format(split)
        )
    if config.loss.class_weights is None:
        raise CheckpointSelectionError(
            "offline validation loss requires resolved class_weights; automatic training "
            "weights must be read from the source checkpoint before evaluation"
        )
    criterion = build_classification_loss(config.loss).to(device=device, dtype=torch.float32)
    logits: list[Tensor] = []
    targets: list[Tensor] = []
    transforms: list[object] = []
    ground_truths: list[tuple[object, ...]] = []
    losses: list[float] = []
    foreground_confidences: list[float] = []
    background_confidences: list[float] = []
    model.eval()
    # No autocast context is entered: selection values are always float32.
    with torch.inference_mode():
        for sample in dataset:
            image = torch.from_numpy(sample.image).unsqueeze(0).to(
                device=device, dtype=torch.float32
            )
            target = torch.from_numpy(sample.heatmap.class_index).unsqueeze(0).to(
                device=device, dtype=torch.int64
            )
            output = model(image)
            if output.dtype != torch.float32:
                output = output.float()
            losses.append(float(criterion(output, target).item()))
            probabilities = output.softmax(dim=1)[:, 1:].amax(dim=1)[0].detach().cpu()
            target_cpu = target[0].detach().cpu()
            foreground_confidences.extend(probabilities[target_cpu > 0].tolist())
            background_confidences.extend(probabilities[target_cpu == 0].tolist())
            logits.append(output[0].detach().float().cpu())
            targets.append(target_cpu.to(dtype=torch.int64))
            transforms.append(sample.transform)
            ground_truths.append(
                tuple(ground_truths_from_boxes(sample.original_boxes, config.dataset.class_names))
            )
    if not logits:
        raise CheckpointSelectionError("requested evaluation split '{}' contains no images".format(split))
    return LogitCollection(
        logits=tuple(logits),
        targets=tuple(targets),
        transforms=tuple(transforms),
        ground_truths=tuple(ground_truths),
        validation_loss=sum(losses) / len(losses),
        confidence_summary={
            "positive": _confidence_summary(foreground_confidences),
            "negative": _confidence_summary(background_confidences),
        },
    )


def evaluate_collected_logits(
    config: ProjectConfig,
    collection: LogitCollection,
    *,
    epoch: int,
    source_snapshot: str,
    checkpoint_path: Path,
) -> EpochSelectionReport:
    """Compute every v2 selection measure from one cached FP32 logit collection."""

    fixed_postprocess = replace(
        config.postprocess, inference_threshold=config.evaluation.checkpoint_threshold
    )
    selection_evaluation = replace(
        config.evaluation,
        threshold_sweep=config.evaluation.checkpoint_selection.threshold_grid,
    )
    validation = evaluate_logit_collection(
        logits=collection.logits,
        targets=collection.targets,
        transforms=collection.transforms,
        ground_truths=collection.ground_truths,
        class_names=config.dataset.class_names,
        stride=config.model.output_stride,
        postprocess_config=fixed_postprocess,
        evaluation_config=selection_evaluation,
    )
    sweep = sweep_confidence_thresholds(
        logits=collection.logits,
        transforms=collection.transforms,
        ground_truths=collection.ground_truths,
        class_names=config.dataset.class_names,
        stride=config.model.output_stride,
        thresholds=config.evaluation.checkpoint_selection.threshold_grid,
        matching_mode=config.evaluation.matching_mode,
        max_distance_pixels=config.evaluation.max_distance_pixels,
        class_thresholds=config.postprocess.class_thresholds,
        component_mode=config.postprocess.component_mode,
        confidence_mode=config.postprocess.confidence_mode,
    )
    pr_auc = centroid_pr_auc(sweep.results, config.dataset.class_names)
    fixed_detection_count = sum(
        len(
            postprocess_logits(
                logits.unsqueeze(0),
                class_names=config.dataset.class_names,
                stride=config.model.output_stride,
                transforms=(transform,),
                confidence_threshold=config.evaluation.checkpoint_threshold,
                class_thresholds=config.postprocess.class_thresholds,
                component_mode=config.postprocess.component_mode,
                confidence_mode=config.postprocess.confidence_mode,
            )[0]
        )
        for logits, transform in zip(collection.logits, collection.transforms)
    )
    return EpochSelectionReport(
        epoch=epoch,
        source_snapshot=source_snapshot,
        checkpoint_path=str(checkpoint_path),
        validation_loss=collection.validation_loss,
        grid_precision=validation.grid_metrics.grid_precision,
        grid_recall=validation.grid_metrics.grid_recall,
        grid_f1=validation.grid_metrics.grid_f1,
        fixed_centroid_precision=validation.centroid_metrics.centroid_precision,
        fixed_centroid_recall=validation.centroid_metrics.centroid_recall,
        fixed_centroid_f1=validation.centroid_metrics.centroid_f1,
        sweep_centroid_precision=sweep.best_result.centroid_precision,
        sweep_centroid_recall=sweep.best_result.centroid_recall,
        sweep_centroid_f1=sweep.best_result.centroid_f1,
        sweep_threshold=sweep.best_threshold,
        centroid_pr_auc_macro=pr_auc.macro_auc,
        centroid_pr_auc_micro=pr_auc.micro_auc,
        macro_effective_class_count=pr_auc.macro_effective_class_count,
        mean_localization_error_pixels=validation.centroid_metrics.mean_localization_error_pixels,
        median_localization_error_pixels=validation.centroid_metrics.median_localization_error_pixels,
        mean_count_bias=validation.centroid_metrics.mean_count_bias,
        count_mae=validation.centroid_metrics.mean_absolute_count_error,
        fixed_detection_count=fixed_detection_count,
        confidence_summary=collection.confidence_summary,
        fixed_per_class=validation.centroid_metrics.per_class_precision_recall_f1,
        sweep_per_class=sweep.best_result.per_class_precision_recall_f1,
        pr_curves={
            "integration": pr_auc.integration,
            "macro_effective_class_count": pr_auc.macro_effective_class_count,
            "per_class": {
                name: {
                    "ground_truth_count": item.ground_truth_count,
                    "auc": item.auc,
                    "points": [point.__dict__ for point in item.raw_points],
                }
                for name, item in pr_auc.per_class.items()
            },
            "micro_points": [point.__dict__ for point in pr_auc.micro_points],
        },
    )


def select_best_epoch_report(
    reports: Sequence[Mapping[str, object]], *, metric: str
) -> Mapping[str, object]:
    """Choose highest finite metric, then earlier epoch, then source filename."""

    if metric not in {"centroid_pr_auc_macro", "max_centroid_f1_over_thresholds"}:
        raise CheckpointSelectionError("unsupported selection metric: {}".format(metric))
    metric_key = "sweep_centroid_f1" if metric == "max_centroid_f1_over_thresholds" else metric
    candidates = []
    for report in reports:
        epoch = report.get("epoch")
        value = report.get(metric_key)
        source = report.get("source_snapshot", "")
        if isinstance(epoch, bool) or not isinstance(epoch, int):
            raise CheckpointSelectionError("selection report epoch must be an integer")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not torch.isfinite(torch.tensor(float(value))):
            continue
        candidates.append((float(value), epoch, str(source), report))
    if not candidates:
        raise CheckpointSelectionError("no finite '{}' value is available for selection".format(metric))
    return min(candidates, key=lambda item: (-item[0], item[1], item[2]))[3]


def validate_calibration_request(
    config: ProjectConfig, *, selection_split: str
) -> bool:
    """Return optimism marker or reject disabled/missing/implicit split reuse."""

    calibration = config.evaluation.threshold_calibration
    if not calibration.enabled:
        return False
    if calibration.split == selection_split:
        if not calibration.allow_selection_split:
            raise CheckpointSelectionError(
                "calibration split equals selection split; set "
                "evaluation.threshold_calibration.allow_selection_split: true to opt in"
            )
        return True
    return False


def _confidence_summary(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "minimum": None, "maximum": None}
    return {
        "count": len(values),
        "mean": float(mean(values)),
        "median": float(median(values)),
        "minimum": float(min(values)),
        "maximum": float(max(values)),
    }


__all__ = [
    "CheckpointSelectionError",
    "EpochSelectionReport",
    "LogitCollection",
    "collect_split_logits",
    "evaluate_collected_logits",
    "select_best_epoch_report",
    "validate_calibration_request",
]
