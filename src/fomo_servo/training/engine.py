"""Deterministic FOMO training, validation, checkpointing, and resume orchestration."""

from __future__ import annotations

import csv
import json
import random
from time import perf_counter
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LRScheduler, StepLR
from torch.utils.data import DataLoader

from fomo_servo.config import (
    EvaluationConfig,
    PostprocessConfig,
    ProjectConfig,
    SchedulerConfig,
    resolved_augmentation_dict,
)
from fomo_servo.datasets import (
    FOMOBatch,
    YOLOv5FOMODataset,
    collate_fomo_samples,
)
from fomo_servo.losses import FOMOClassificationLoss, build_classification_loss
from fomo_servo.evaluation import evaluate_validation_dataset
from fomo_servo.experiments import (
    ExperimentMetadataError,
    append_experiment_summary,
    copy_experiment_config,
    dataset_file_list_hash,
    git_commit_sha,
    git_worktree_fingerprint,
    dataset_content_manifest,
    write_dataset_manifest,
    write_experiment_metadata,
)
from fomo_servo.metrics import (
    CentroidEvaluator,
    ForegroundMetrics,
    ground_truths_from_boxes,
    foreground_micro_metrics,
)
from fomo_servo.postprocess import postprocess_logits
from fomo_servo.models import build_fomo_model, describe_model
from fomo_servo.runtime import DeviceRequest
from fomo_servo.training.class_weights import (
    ClassTrainingStatistics,
    ClassWeightError,
    ResolvedClassWeights,
    resolve_training_class_weights,
)
from fomo_servo.training.runtime import (
    TrainingRuntime,
    autocast_context,
    create_training_runtime,
    move_training_batch,
    prepare_model,
)
from fomo_servo.training.snapshots import config_fingerprint, write_epoch_snapshot


class TrainingError(RuntimeError):
    """Raised when a FOMO training run cannot safely continue."""


@dataclass(frozen=True)
class EpochMetrics:
    """One validation epoch's grid and centroid metrics."""

    loss: float
    grid_precision: float
    grid_recall: float
    grid_f1: float
    centroid_precision: float = 0.0
    centroid_recall: float = 0.0
    centroid_f1: float = 0.0
    mean_localization_error_pixels: float = 0.0
    median_localization_error_pixels: float = 0.0
    count_error_per_image: tuple[int, ...] = ()
    mean_count_bias: float = 0.0
    mean_absolute_count_error: float = 0.0
    per_class_precision_recall_f1: dict[str, dict[str, float]] | None = None

    @property
    def precision(self) -> float:
        """Backward-compatible alias for ``grid_precision``."""

        return self.grid_precision

    @property
    def recall(self) -> float:
        """Backward-compatible alias for ``grid_recall``."""

        return self.grid_recall

    @property
    def f1(self) -> float:
        """Backward-compatible alias for ``grid_f1``."""

        return self.grid_f1


@dataclass(frozen=True)
class TrainingSummary:
    """Final immutable outcome of one complete or early-stopped training invocation."""

    start_epoch: int
    completed_epochs: int
    best_val_f1: float
    stopped_early: bool
    output_dir: Path
    device: torch.device
    amp_enabled: bool
    best_metric_name: str = "grid_f1"
    best_grid_f1: float = 0.0
    best_centroid_f1: float = 0.0
    class_weight_mode: str = "manual"
    class_weights: tuple[float, ...] = ()
    class_statistics: tuple[ClassTrainingStatistics, ...] = ()
    best_epoch: int = 0
    best_grid_epoch: int = 0
    best_centroid_epoch: int = 0
    checkpoint_threshold: float = 0.5
    final_sweep_best_threshold: float | None = None
    best_val_f1_alias_target: str = "best_grid_f1.pt"
    total_training_time_seconds: float = 0.0
    augmentation_preset: Optional[str] = None
    resolved_augmentation: dict[str, Any] = field(default_factory=dict)
    augmentation_epoch_stats: tuple[dict[str, Any], ...] = ()
    model_metadata: dict[str, Any] = field(default_factory=dict)
    checkpoint_selection_protocol: str = "v2"
    legacy_best_fixed_centroid_epoch: int = 0
    best_pr_auc_macro_epoch: Optional[int] = None
    best_sweep_f1_epoch: Optional[int] = None
    checkpoint_selection_metric: str = "centroid_pr_auc_macro"
    selection_split: str = "validation"
    calibration_split: Optional[str] = None
    calibrated_threshold: Optional[float] = None
    calibration_is_optimistic: bool = False


@dataclass
class AugmentationEpochStats:
    """Mutable train-split augmentation counters for one epoch."""

    total_samples: int = 0
    color_jitter_applied_count: int = 0
    horizontal_flip_applied_count: int = 0
    gaussian_blur_applied_count: int = 0
    gaussian_noise_applied_count: int = 0
    affine_applied_count: int = 0
    clipped_bbox_count: int = 0
    dropped_bbox_count: int = 0
    pre_augmentation_object_count: int = 0
    post_augmentation_object_count: int = 0
    same_class_collision_count: int = 0
    different_class_collision_count: int = 0

    def update(self, metadata: Sequence[Any]) -> None:
        """Aggregate one collated batch of sample metadata."""

        for item in metadata:
            self.total_samples += 1
            self.color_jitter_applied_count += int(bool(item.color_jitter_applied))
            self.horizontal_flip_applied_count += int(bool(item.horizontal_flip_applied))
            self.gaussian_blur_applied_count += int(bool(item.gaussian_blur_applied))
            self.gaussian_noise_applied_count += int(bool(item.gaussian_noise_applied))
            self.affine_applied_count += int(bool(item.affine_applied))
            self.clipped_bbox_count += int(item.clipped_bbox_count)
            self.dropped_bbox_count += int(item.dropped_bbox_count)
            self.pre_augmentation_object_count += int(item.pre_augmentation_object_count)
            self.post_augmentation_object_count += int(item.post_augmentation_object_count)
            self.same_class_collision_count += int(item.same_class_collision_count)
            self.different_class_collision_count += int(item.different_class_collision_count)

    def as_dict(self) -> dict[str, Any]:
        denominator = float(self.total_samples) if self.total_samples else 1.0
        return {
            "total_samples": self.total_samples,
            "color_jitter_applied_count": self.color_jitter_applied_count,
            "color_jitter_applied_rate": self.color_jitter_applied_count / denominator,
            "horizontal_flip_applied_count": self.horizontal_flip_applied_count,
            "horizontal_flip_applied_rate": self.horizontal_flip_applied_count / denominator,
            "gaussian_blur_applied_count": self.gaussian_blur_applied_count,
            "gaussian_blur_applied_rate": self.gaussian_blur_applied_count / denominator,
            "gaussian_noise_applied_count": self.gaussian_noise_applied_count,
            "gaussian_noise_applied_rate": self.gaussian_noise_applied_count / denominator,
            "affine_applied_count": self.affine_applied_count,
            "affine_applied_rate": self.affine_applied_count / denominator,
            "clipped_bbox_count": self.clipped_bbox_count,
            "dropped_bbox_count": self.dropped_bbox_count,
            "pre_augmentation_object_count": self.pre_augmentation_object_count,
            "post_augmentation_object_count": self.post_augmentation_object_count,
            "same_class_collision_count": self.same_class_collision_count,
            "different_class_collision_count": self.different_class_collision_count,
        }


@dataclass(frozen=True)
class TrainEpochResult:
    """Loss and online augmentation statistics for one train epoch."""

    loss: float
    augmentation_stats: dict[str, Any]


@dataclass(frozen=True)
class _ResumeState:
    """Private restored checkpoint metadata used to continue an existing run."""

    start_epoch: int
    best_val_f1: float
    early_stopping_bad_epochs: int
    best_grid_f1: float
    best_centroid_f1: float
    best_epoch: int
    best_grid_epoch: int
    best_centroid_epoch: int


_HISTORY_COLUMNS = (
    "epoch",
    "train_loss",
    "val_loss",
    "grid_precision",
    "grid_recall",
    "grid_f1",
    "centroid_precision",
    "centroid_recall",
    "centroid_f1",
    "mean_localization_error_pixels",
    "median_localization_error_pixels",
    "mean_count_bias",
    "mean_absolute_count_error",
    "count_error_per_image",
    "centroid_per_class_f1",
    "precision",
    "recall",
    "f1",
    "learning_rate",
    "total_samples",
    "color_jitter_applied_count",
    "color_jitter_applied_rate",
    "horizontal_flip_applied_count",
    "horizontal_flip_applied_rate",
    "gaussian_blur_applied_count",
    "gaussian_blur_applied_rate",
    "gaussian_noise_applied_count",
    "gaussian_noise_applied_rate",
    "affine_applied_count",
    "affine_applied_rate",
    "clipped_bbox_count",
    "dropped_bbox_count",
    "pre_augmentation_object_count",
    "post_augmentation_object_count",
    "same_class_collision_count",
    "different_class_collision_count",
    "augmentation_stats",
)


def set_random_seed(seed: int) -> None:
    """Set Python, NumPy, Torch CPU, and available CUDA RNG states deterministically."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise TrainingError("seed must be a non-negative integer")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def ensure_finite_gradients(model: nn.Module) -> None:
    """Raise before optimizer step when any existing parameter gradient is NaN or Inf."""

    if not isinstance(model, nn.Module):
        raise TrainingError("model must be a torch.nn.Module")
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise TrainingError("non-finite gradient detected for parameter '{}'".format(name))


def run_training(
    config: ProjectConfig,
    *,
    device_override: Optional[DeviceRequest] = None,
    resume_override: Optional[Path] = None,
) -> TrainingSummary:
    """Train and validate a YAML-configured FOMO model, then persist resumable state.

    Args:
        config: Validated project YAML, including data, model, loss, and training fields.
        device_override: Optional CLI device overriding ``config.training.device``.
        resume_override: Optional CLI checkpoint overriding ``config.training.resume``.

    Returns:
        A summary of completed epochs and the best validation F1 checkpoint criterion.
    """

    if not isinstance(config, ProjectConfig):
        raise TrainingError("config must be a ProjectConfig instance")
    training_started = perf_counter()
    git_dirty = False
    git_diff_sha256 = ""
    if config.experiment.name is not None:
        try:
            git_dirty, git_diff_sha256 = git_worktree_fingerprint(
                config.source_path.parent
            )
        except ExperimentMetadataError as error:
            raise TrainingError(
                "unable to inspect Git state before experiment training: {}".format(error)
            ) from error
        if git_dirty:
            print("!!! WARNING: Git worktree is dirty; this run is not commit-clean !!!")
            print("!!! git_diff_sha256: {} !!!".format(git_diff_sha256))
    set_random_seed(config.training.seed)
    runtime = create_training_runtime(config.training, device_override)
    model = prepare_model(build_fomo_model(config), runtime)
    model_metadata = describe_model(config, model)
    train_loader, validation_loader = _build_data_loaders(config, runtime)
    train_dataset = train_loader.dataset
    if not isinstance(train_dataset, YOLOv5FOMODataset):
        raise TrainingError("training DataLoader must retain a YOLOv5FOMODataset")
    try:
        resolved_weights = resolve_training_class_weights(config.loss, train_dataset)
    except ClassWeightError as error:
        raise TrainingError("unable to resolve training class weights: {}".format(error)) from error
    _print_class_weight_summary(resolved_weights)
    effective_loss_config = replace(config.loss, class_weights=resolved_weights.weights)
    criterion = build_classification_loss(effective_loss_config).to(runtime.device)
    optimizer = AdamW(
        model.parameters(),
        lr=config.training.optimizer.learning_rate,
        weight_decay=config.training.optimizer.weight_decay,
    )
    scheduler = build_scheduler(optimizer, config.training.scheduler)
    scaler = torch.amp.GradScaler(
        "cuda",
        init_scale=config.training.amp_initial_scale,
        enabled=runtime.amp_enabled,
    )

    output_dir = config.training.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_metadata: dict[str, str] | None = None
    if config.training.epoch_snapshots.enabled:
        try:
            content_manifest = dataset_content_manifest(
                config.dataset.root,
                config.dataset.train_split,
                config.dataset.validation_split,
            )
            snapshot_metadata = {
                "config_fingerprint": config_fingerprint(config),
                "dataset_content_hash": str(content_manifest["dataset_content_hash"]),
                "git_commit_sha": git_commit_sha(Path(__file__).resolve()),
            }
        except (ExperimentMetadataError, ValueError, TypeError) as error:
            raise TrainingError(
                "unable to collect epoch snapshot provenance: {}".format(error)
            ) from error
    history_path = output_dir / "history.csv"
    resume_path = resume_override if resume_override is not None else config.training.resume
    augmentation_epoch_stats: list[dict[str, Any]] = []
    if resume_path is not None:
        previous_summary_path = output_dir / "training_summary.json"
        if previous_summary_path.is_file():
            try:
                previous_summary = json.loads(
                    previous_summary_path.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError) as error:
                raise TrainingError(
                    "unable to read previous training summary '{}': {}".format(
                        previous_summary_path, error
                    )
                ) from error
            previous_stats = previous_summary.get("augmentation_epoch_stats", [])
            if not isinstance(previous_stats, list) or any(
                not isinstance(item, dict) for item in previous_stats
            ):
                raise TrainingError(
                    "previous training summary augmentation_epoch_stats must be a list of mappings"
                )
            augmentation_epoch_stats.extend(previous_stats)
    if resume_path is None:
        _initialize_history(history_path, append=False)
        resume_state = _ResumeState(
            start_epoch=1,
            best_val_f1=float("-inf"),
            early_stopping_bad_epochs=0,
            best_grid_f1=float("-inf"),
            best_centroid_f1=float("-inf"),
            best_epoch=0,
            best_grid_epoch=0,
            best_centroid_epoch=0,
        )
    else:
        resume_state = _restore_checkpoint(
            Path(resume_path),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=runtime.device,
            checkpoint_criterion=config.training.checkpoint_criterion,
        )
        _initialize_history(history_path, append=True)

    best_val_f1 = resume_state.best_val_f1
    best_grid_f1 = resume_state.best_grid_f1
    best_centroid_f1 = resume_state.best_centroid_f1
    best_epoch = resume_state.best_epoch
    best_grid_epoch = resume_state.best_grid_epoch
    best_centroid_epoch = resume_state.best_centroid_epoch
    bad_epochs = resume_state.early_stopping_bad_epochs
    completed_epochs = resume_state.start_epoch - 1
    stopped_early = False
    for epoch in range(resume_state.start_epoch, config.training.epochs + 1):
        train_dataset.set_epoch(epoch)
        train_result = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            runtime=runtime,
        )
        train_loss = train_result.loss
        train_augmentation_stats = dict(train_result.augmentation_stats)
        train_augmentation_stats["epoch"] = epoch
        augmentation_epoch_stats.append(train_augmentation_stats)
        validation = validate_one_epoch(
            model=model,
            loader=validation_loader,
            criterion=criterion,
            runtime=runtime,
            class_names=config.dataset.class_names,
            stride=config.model.output_stride,
            postprocess_config=config.postprocess,
            evaluation_config=config.evaluation,
            checkpoint_threshold=config.evaluation.checkpoint_threshold,
        )
        learning_rate = float(optimizer.param_groups[0]["lr"])
        if scheduler is not None:
            scheduler.step()

        validation_metric = (
            validation.grid_f1
            if config.training.checkpoint_criterion == "grid_f1"
            else validation.centroid_f1
        )
        improved = validation_metric > (
            best_val_f1 + config.training.early_stopping_min_delta
        )
        if improved:
            best_val_f1 = validation_metric
            bad_epochs = 0
            best_epoch = epoch
        else:
            bad_epochs += 1
        grid_improved = validation.grid_f1 > best_grid_f1
        if grid_improved:
            best_grid_f1 = validation.grid_f1
            best_grid_epoch = epoch
        centroid_improved = validation.centroid_f1 > best_centroid_f1
        if centroid_improved:
            best_centroid_f1 = validation.centroid_f1
            best_centroid_epoch = epoch
        checkpoint_arguments = dict(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            best_val_f1=best_val_f1,
            best_grid_f1=best_grid_f1,
            best_centroid_f1=best_centroid_f1,
            best_epoch=best_epoch,
            best_grid_epoch=best_grid_epoch,
            best_centroid_epoch=best_centroid_epoch,
            early_stopping_bad_epochs=bad_epochs,
            checkpoint_criterion=config.training.checkpoint_criterion,
            resolved_weights=resolved_weights,
            validation=validation,
            selection_threshold=config.evaluation.checkpoint_threshold,
            best_val_f1_alias_target=(
                "best_grid_f1.pt"
                if config.training.checkpoint_criterion == "grid_f1"
                else "best_centroid_f1.pt"
            ),
            augmentation_preset=config.augmentation.preset,
            resolved_augmentation=resolved_augmentation_dict(config.augmentation),
            augmentation_stats=train_augmentation_stats,
            model_metadata=model_metadata,
        )
        torch.save(
            _checkpoint_payload(
                **checkpoint_arguments,
                checkpoint_type="last",
                selection_metric="last",
            ),
            output_dir / "last.pt",
        )
        if grid_improved:
            torch.save(
                _checkpoint_payload(
                    **checkpoint_arguments,
                    checkpoint_type="best_grid_f1",
                    selection_metric="grid_f1",
                ),
                output_dir / "best_grid_f1.pt",
            )
        if centroid_improved:
            torch.save(
                _checkpoint_payload(
                    **checkpoint_arguments,
                    checkpoint_type="best_centroid_f1",
                    selection_metric="fixed_centroid_f1",
                ),
                output_dir / "best_centroid_f1.pt",
            )
        if improved:
            torch.save(
                _checkpoint_payload(
                    **checkpoint_arguments,
                    checkpoint_type="best_val_f1_alias",
                    selection_metric=config.training.checkpoint_criterion,
                ),
                output_dir / "best_val_f1.pt",
            )
        if (
            snapshot_metadata is not None
            and epoch % config.training.epoch_snapshots.interval == 0
        ):
            try:
                write_epoch_snapshot(
                    model=model,
                    epoch=epoch,
                    output_dir=output_dir,
                    model_metadata=model_metadata,
                    config_fingerprint=snapshot_metadata["config_fingerprint"],
                    dataset_content_hash=snapshot_metadata["dataset_content_hash"],
                    git_commit_sha=snapshot_metadata["git_commit_sha"],
                    seed=config.training.seed,
                    augmentation_preset=config.augmentation.preset,
                    checkpoint_threshold=config.evaluation.checkpoint_threshold,
                    keep_last=config.training.epoch_snapshots.keep_last,
                )
            except RuntimeError as error:
                raise TrainingError("unable to write epoch snapshot: {}".format(error)) from error
        _append_history(
            history_path,
            epoch=epoch,
            train_loss=train_loss,
            validation=validation,
            learning_rate=learning_rate,
            augmentation_stats=train_augmentation_stats,
        )
        completed_epochs = epoch
        if (
            config.training.early_stopping_patience > 0
            and bad_epochs >= config.training.early_stopping_patience
        ):
            stopped_early = True
            break

    summary = TrainingSummary(
        start_epoch=resume_state.start_epoch,
        completed_epochs=completed_epochs,
        best_val_f1=max(best_val_f1, 0.0),
        stopped_early=stopped_early,
        output_dir=output_dir,
        device=runtime.device,
        amp_enabled=runtime.amp_enabled,
        best_metric_name=config.training.checkpoint_criterion,
        best_grid_f1=max(best_grid_f1, 0.0),
        best_centroid_f1=max(best_centroid_f1, 0.0),
        class_weight_mode=resolved_weights.mode,
        class_weights=resolved_weights.weights,
        class_statistics=resolved_weights.statistics,
        best_epoch=best_epoch,
        best_grid_epoch=best_grid_epoch,
        best_centroid_epoch=best_centroid_epoch,
        checkpoint_threshold=config.evaluation.checkpoint_threshold,
        best_val_f1_alias_target=(
            "best_grid_f1.pt"
            if config.training.checkpoint_criterion == "grid_f1"
            else "best_centroid_f1.pt"
        ),
        total_training_time_seconds=perf_counter() - training_started,
        augmentation_preset=config.augmentation.preset,
        resolved_augmentation=resolved_augmentation_dict(config.augmentation),
        augmentation_epoch_stats=tuple(augmentation_epoch_stats),
        model_metadata=model_metadata,
        legacy_best_fixed_centroid_epoch=best_centroid_epoch,
        checkpoint_selection_metric=config.evaluation.checkpoint_selection.metric,
        selection_split=config.evaluation.checkpoint_selection.split,
        calibration_split=(
            config.evaluation.threshold_calibration.split
            if config.evaluation.threshold_calibration.enabled
            else None
        ),
        calibration_is_optimistic=(
            config.evaluation.threshold_calibration.enabled
            and config.evaluation.threshold_calibration.split
            == config.evaluation.checkpoint_selection.split
            and config.evaluation.threshold_calibration.allow_selection_split
        ),
    )
    if config.experiment.name is not None:
        final_sweep_best_threshold = _record_experiment(
            config=config,
            model=model,
            device=runtime.device,
            output_dir=output_dir,
            summary=summary,
            git_dirty=git_dirty,
            git_diff_sha256=git_diff_sha256,
        )
        summary = replace(
            summary, final_sweep_best_threshold=final_sweep_best_threshold
        )
    _write_training_summary(output_dir / "training_summary.json", summary)
    return summary


def train_one_epoch(
    *,
    model: nn.Module,
    loader: DataLoader[FOMOBatch],
    criterion: FOMOClassificationLoss,
    optimizer: Optimizer,
    scaler: torch.amp.GradScaler,
    runtime: TrainingRuntime,
) -> TrainEpochResult:
    """Run one train epoch and return loss plus augmentation counters."""

    model.train()
    total_loss = 0.0
    total_samples = 0
    augmentation_stats = AugmentationEpochStats()
    for batch in loader:
        augmentation_stats.update(batch.augmentation_metadata)
        images, targets = move_training_batch(batch.images, batch.targets, runtime)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(runtime):
            logits = model(images)
            loss = criterion(logits, targets)
        if not torch.isfinite(loss):
            raise TrainingError("non-finite training loss encountered before backward")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        ensure_finite_gradients(model)
        scaler.step(optimizer)
        scaler.update()
        batch_size = int(images.shape[0])
        total_loss += float(loss.detach().item()) * batch_size
        total_samples += batch_size
    if total_samples == 0:
        raise TrainingError("training DataLoader produced no batches")
    return TrainEpochResult(
        loss=total_loss / total_samples,
        augmentation_stats=augmentation_stats.as_dict(),
    )


def validate_one_epoch(
    *,
    model: nn.Module,
    loader: DataLoader[FOMOBatch],
    criterion: FOMOClassificationLoss,
    runtime: TrainingRuntime,
    class_names: Sequence[str],
    stride: int,
    postprocess_config: PostprocessConfig,
    evaluation_config: EvaluationConfig,
    checkpoint_threshold: float,
) -> EpochMetrics:
    """Run validation using the fixed checkpoint threshold for epoch metrics."""

    model.eval()
    total_loss = 0.0
    total_samples = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    all_predictions = []
    all_ground_truths = []
    centroid_evaluator = CentroidEvaluator(
        class_names,
        matching_mode=evaluation_config.matching_mode,
        max_distance_pixels=evaluation_config.max_distance_pixels,
    )
    with torch.no_grad():
        for batch in loader:
            images, targets = move_training_batch(batch.images, batch.targets, runtime)
            with autocast_context(runtime):
                logits = model(images)
                loss = criterion(logits, targets)
            if not torch.isfinite(loss):
                raise TrainingError("non-finite validation loss encountered")
            predictions = logits.argmax(dim=1).to(dtype=torch.int64)
            metrics = foreground_micro_metrics(predictions, targets)
            true_positives += metrics.true_positives
            false_positives += metrics.false_positives
            false_negatives += metrics.false_negatives
            batch_detections = postprocess_logits(
                logits,
                class_names=class_names,
                stride=stride,
                transforms=batch.transforms,
                confidence_threshold=checkpoint_threshold,
                class_thresholds=postprocess_config.class_thresholds,
                component_mode=postprocess_config.component_mode,
                confidence_mode=postprocess_config.confidence_mode,
            )
            all_predictions.extend(batch_detections)
            all_ground_truths.extend(
                ground_truths_from_boxes(boxes, class_names)
                for boxes in batch.original_boxes
            )
            batch_size = int(images.shape[0])
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size
    if total_samples == 0:
        raise TrainingError("validation DataLoader produced no batches")
    metrics = _metrics_from_counts(true_positives, false_positives, false_negatives)
    centroid_metrics = centroid_evaluator.evaluate_dataset(
        tuple(all_predictions), tuple(all_ground_truths)
    )
    return EpochMetrics(
        loss=total_loss / total_samples,
        grid_precision=metrics.grid_precision,
        grid_recall=metrics.grid_recall,
        grid_f1=metrics.grid_f1,
        centroid_precision=centroid_metrics.centroid_precision,
        centroid_recall=centroid_metrics.centroid_recall,
        centroid_f1=centroid_metrics.centroid_f1,
        mean_localization_error_pixels=centroid_metrics.mean_localization_error_pixels,
        median_localization_error_pixels=centroid_metrics.median_localization_error_pixels,
        count_error_per_image=centroid_metrics.count_error_per_image,
        mean_count_bias=centroid_metrics.mean_count_bias,
        mean_absolute_count_error=centroid_metrics.mean_absolute_count_error,
        per_class_precision_recall_f1={
            name: dict(values)
            for name, values in centroid_metrics.per_class_precision_recall_f1.items()
        },
    )


def build_scheduler(
    optimizer: Optimizer, config: SchedulerConfig
) -> Optional[LRScheduler]:
    """Build the configured no-op or StepLR schedule without CUDA-specific behavior."""

    if config.name == "none":
        return None
    if config.name == "step_lr":
        return StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)
    raise TrainingError("unsupported scheduler name '{}'".format(config.name))


def _build_data_loaders(
    config: ProjectConfig, runtime: TrainingRuntime
) -> tuple[DataLoader[FOMOBatch], DataLoader[FOMOBatch]]:
    """Build deterministic train/validation loaders and validate dataset class semantics."""

    common_arguments = {
        "root": config.dataset.root,
        "input_size": config.model.input_size,
        "stride": config.model.output_stride,
        "class_mode": config.dataset.class_mode,
        "merged_class_name": config.dataset.merged_class_name,
        "collision_policy": config.dataset.collision_policy,
        "train_split": config.dataset.train_split,
        "augmentation_seed": config.training.seed,
    }
    train_dataset = YOLOv5FOMODataset(
        split=config.dataset.train_split,
        augmentation=config.augmentation,
        **common_arguments,
    )
    validation_dataset = YOLOv5FOMODataset(
        split=config.dataset.validation_split,
        augmentation=config.augmentation,
        **common_arguments,
    )
    if train_dataset.class_names != config.dataset.class_names:
        raise TrainingError(
            "train dataset classes {} do not match YAML classes {}".format(
                train_dataset.class_names, config.dataset.class_names
            )
        )
    if validation_dataset.class_names != config.dataset.class_names:
        raise TrainingError(
            "validation dataset classes {} do not match YAML classes {}".format(
                validation_dataset.class_names, config.dataset.class_names
            )
        )
    generator = torch.Generator().manual_seed(config.training.seed)
    validation_generator = torch.Generator().manual_seed(config.training.seed + 1)
    loader_arguments = {
        "batch_size": config.training.batch_size,
        "num_workers": runtime.num_workers,
        "pin_memory": runtime.pin_memory,
        "collate_fn": collate_fomo_samples,
        "worker_init_fn": _seed_data_loader_worker,
        "persistent_workers": False,
    }
    return (
        DataLoader(
            train_dataset,
            shuffle=True,
            generator=generator,
            **loader_arguments,
        ),
        DataLoader(
            validation_dataset,
            shuffle=False,
            generator=validation_generator,
            **loader_arguments,
        ),
    )


def _seed_data_loader_worker(worker_id: int) -> None:
    """Seed Python and NumPy in each DataLoader worker from Torch's worker seed."""

    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _metrics_from_counts(
    true_positives: int, false_positives: int, false_negatives: int
) -> ForegroundMetrics:
    """Derive one foreground micro result from validation batch count totals."""

    precision_denominator = true_positives + false_positives
    recall_denominator = true_positives + false_negatives
    precision = (
        true_positives / precision_denominator if precision_denominator else 0.0
    )
    recall = true_positives / recall_denominator if recall_denominator else 0.0
    denominator = precision + recall
    f1 = 2.0 * precision * recall / denominator if denominator else 0.0
    return ForegroundMetrics(
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        grid_precision=precision,
        grid_recall=recall,
        grid_f1=f1,
    )


def _print_class_weight_summary(resolved_weights: ResolvedClassWeights) -> None:
    """Print final training loss weights plus the counts that produced them."""

    rendered_weights = ", ".join(
        "{:.6g}".format(weight) for weight in resolved_weights.weights
    )
    print("Class weights [{}]: [{}]".format(resolved_weights.mode, rendered_weights))
    print(
        "Class statistics: class_id class_name images bboxes encoded_cells "
        "same_class_collisions different_class_collisions"
    )
    for item in resolved_weights.statistics:
        print(
            "  {} {} {} {} {} {} {}".format(
                item.class_id,
                item.class_name,
                item.image_count,
                item.bbox_count,
                item.encoded_centroid_cell_count,
                item.same_class_collision_count,
                item.different_class_collision_count,
            )
        )


def _record_experiment(
    *,
    config: ProjectConfig,
    model: nn.Module,
    device: torch.device,
    output_dir: Path,
    summary: TrainingSummary,
    git_dirty: bool,
    git_diff_sha256: str,
) -> float:
    """Evaluate the selected checkpoint and append one reproducible experiment row."""

    if config.experiment.name is None:
        raise TrainingError("experiment name is required for experiment recording")
    checkpoint_name = (
        "best_grid_f1.pt"
        if config.training.checkpoint_criterion == "grid_f1"
        else "best_centroid_f1.pt"
    )
    checkpoint_path = output_dir / checkpoint_name
    if not checkpoint_path.is_file():
        raise TrainingError(
            "selected experiment checkpoint does not exist: {}".format(checkpoint_path)
        )
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
            raise TrainingError(
                "selected experiment checkpoint has no model_state: {}".format(
                    checkpoint_path
                )
            )
        model.load_state_dict(checkpoint["model_state"])
        validation_report = evaluate_validation_dataset(config, model, device)
        config_copy = copy_experiment_config(config.source_path, output_dir)
        git_sha = git_commit_sha(config.source_path.parent)
        file_hash = dataset_file_list_hash(config.dataset.root)
        content_manifest = dataset_content_manifest(
            config.dataset.root,
            config.dataset.train_split,
            config.dataset.validation_split,
        )
        write_dataset_manifest(output_dir, content_manifest)
        result = validation_report.best_centroid_metrics
        validation_payload = validation_report.as_dict(
            class_weight_mode=summary.class_weight_mode,
            class_weights=summary.class_weights,
            class_statistics=tuple(item.as_dict() for item in summary.class_statistics),
        )
        metadata = {
            "experiment_name": config.experiment.name,
            "output_dir": str(output_dir),
            "config_copy": str(config_copy),
            "git_commit_sha": git_sha,
            "git_dirty": git_dirty,
            "git_diff_sha256": git_diff_sha256,
            "dataset_file_list_hash": file_hash,
            "dataset_content_hash": content_manifest["dataset_content_hash"],
            "random_seed": config.training.seed,
            "augmentation_preset": config.augmentation.preset,
            "resolved_augmentation": resolved_augmentation_dict(config.augmentation),
            "best_centroid_f1": result.centroid_f1,
            "best_grid_f1": summary.best_grid_f1,
            "precision": result.centroid_precision,
            "recall": result.centroid_recall,
            "best_threshold": validation_report.best_threshold,
            "mean_localization_error_pixels": result.mean_localization_error_pixels,
            "mean_count_bias": result.mean_count_bias,
            "count_mae": result.mean_absolute_count_error,
            "best_epoch": summary.best_epoch,
            "best_grid_epoch": summary.best_grid_epoch,
            "best_centroid_epoch": summary.best_centroid_epoch,
            "checkpoint_threshold": config.evaluation.checkpoint_threshold,
            "final_sweep_best_threshold": validation_report.best_threshold,
            "best_val_f1_alias_target": summary.best_val_f1_alias_target,
            "checkpoint_type": checkpoint.get("checkpoint_type", "legacy"),
            "selection_metric": checkpoint.get(
                "selection_metric", config.training.checkpoint_criterion
            ),
            "selection_threshold": checkpoint.get(
                "selection_threshold", config.evaluation.checkpoint_threshold
            ),
            "checkpoint_epoch": checkpoint.get("epoch"),
            "checkpoint_grid_f1": checkpoint.get("grid_f1"),
            "checkpoint_centroid_precision": checkpoint.get("centroid_precision"),
            "checkpoint_centroid_recall": checkpoint.get("centroid_recall"),
            "checkpoint_centroid_f1": checkpoint.get("centroid_f1"),
            "total_training_time_seconds": summary.total_training_time_seconds,
            "model_metadata": summary.model_metadata,
            "validation_report": validation_payload,
        }
        write_experiment_metadata(output_dir, metadata)
        append_experiment_summary(
            config.experiment.summary_csv,
            {
                "experiment_name": config.experiment.name,
                "output_dir": str(output_dir),
                "config_copy": str(config_copy),
                "git_commit_sha": git_sha,
                "dataset_file_list_hash": file_hash,
                "random_seed": config.training.seed,
                "best_centroid_f1": result.centroid_f1,
                "best_grid_f1": summary.best_grid_f1,
                "precision": result.centroid_precision,
                "recall": result.centroid_recall,
                "best_threshold": validation_report.best_threshold,
                "mean_localization_error_pixels": result.mean_localization_error_pixels,
                "mean_count_bias": result.mean_count_bias,
                "count_mae": result.mean_absolute_count_error,
                "best_epoch": summary.best_epoch,
                "total_training_time_seconds": summary.total_training_time_seconds,
            },
        )
        return validation_report.best_threshold
    except ExperimentMetadataError as error:
        raise TrainingError("unable to record experiment metadata: {}".format(error)) from error
    except (OSError, RuntimeError, ValueError, TypeError) as error:
        raise TrainingError("unable to finalize experiment record: {}".format(error)) from error


def _write_training_summary(summary_path: Path, summary: TrainingSummary) -> None:
    """Persist final run metrics and resolved loss weighting in JSON form."""

    payload = {
        "start_epoch": summary.start_epoch,
        "completed_epochs": summary.completed_epochs,
        "best_metric_name": summary.best_metric_name,
        "best_val_f1": summary.best_val_f1,
        "best_grid_f1": summary.best_grid_f1,
        "best_centroid_f1": summary.best_centroid_f1,
        "best_epoch": summary.best_epoch,
        "best_grid_epoch": summary.best_grid_epoch,
        "best_centroid_epoch": summary.best_centroid_epoch,
        "checkpoint_threshold": summary.checkpoint_threshold,
        "final_sweep_best_threshold": summary.final_sweep_best_threshold,
        "best_val_f1_alias_target": summary.best_val_f1_alias_target,
        "total_training_time_seconds": summary.total_training_time_seconds,
        "stopped_early": summary.stopped_early,
        "device": str(summary.device),
        "amp_enabled": summary.amp_enabled,
        "class_weight_mode": summary.class_weight_mode,
        "class_weights": list(summary.class_weights),
        "class_statistics": [item.as_dict() for item in summary.class_statistics],
        "augmentation_preset": summary.augmentation_preset,
        "resolved_augmentation": summary.resolved_augmentation,
        "augmentation_epoch_stats": list(summary.augmentation_epoch_stats),
        "model_metadata": summary.model_metadata,
        "checkpoint_selection_protocol": summary.checkpoint_selection_protocol,
        "legacy_best_fixed_centroid_epoch": summary.legacy_best_fixed_centroid_epoch,
        "best_pr_auc_macro_epoch": summary.best_pr_auc_macro_epoch,
        "best_sweep_f1_epoch": summary.best_sweep_f1_epoch,
        "checkpoint_selection_metric": summary.checkpoint_selection_metric,
        "selection_split": summary.selection_split,
        "calibration_split": summary.calibration_split,
        "calibrated_threshold": summary.calibrated_threshold,
        "calibration_is_optimistic": summary.calibration_is_optimistic,
    }
    try:
        summary_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    except OSError as error:
        raise TrainingError(
            "unable to write training summary '{}': {}".format(summary_path, error)
        ) from error


def _initialize_history(history_path: Path, *, append: bool) -> None:
    """Create a CSV header for a new run or retain an existing resume history."""

    if append and history_path.is_file():
        try:
            with history_path.open("r", newline="", encoding="utf-8") as history_file:
                reader = csv.DictReader(history_file)
                rows = list(reader)
                existing_columns = tuple(reader.fieldnames or ())
        except OSError as error:
            raise TrainingError(
                "unable to read existing history '{}': {}".format(history_path, error)
            ) from error
        if existing_columns == _HISTORY_COLUMNS:
            return
        try:
            with history_path.open("w", newline="", encoding="utf-8") as history_file:
                writer = csv.DictWriter(history_file, fieldnames=_HISTORY_COLUMNS)
                writer.writeheader()
                for row in rows:
                    writer.writerow({column: row.get(column, "") for column in _HISTORY_COLUMNS})
        except OSError as error:
            raise TrainingError(
                "unable to migrate existing history '{}': {}".format(history_path, error)
            ) from error
        return
    with history_path.open("w", newline="", encoding="utf-8") as history_file:
        csv.DictWriter(history_file, fieldnames=_HISTORY_COLUMNS).writeheader()


def _append_history(
    history_path: Path,
    *,
    epoch: int,
    train_loss: float,
    validation: EpochMetrics,
    learning_rate: float,
    augmentation_stats: Mapping[str, Any],
) -> None:
    """Append one train/validation metrics row to a configured history CSV."""

    with history_path.open("a", newline="", encoding="utf-8") as history_file:
        writer = csv.DictWriter(history_file, fieldnames=_HISTORY_COLUMNS)
        writer.writerow(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": validation.loss,
                "grid_precision": validation.grid_precision,
                "grid_recall": validation.grid_recall,
                "grid_f1": validation.grid_f1,
                "centroid_precision": validation.centroid_precision,
                "centroid_recall": validation.centroid_recall,
                "centroid_f1": validation.centroid_f1,
                "mean_localization_error_pixels": validation.mean_localization_error_pixels,
                "median_localization_error_pixels": validation.median_localization_error_pixels,
                "mean_count_bias": validation.mean_count_bias,
                "mean_absolute_count_error": validation.mean_absolute_count_error,
                "count_error_per_image": json.dumps(validation.count_error_per_image),
                "centroid_per_class_f1": json.dumps(
                    {
                        name: values.get("f1", 0.0)
                        for name, values in (validation.per_class_precision_recall_f1 or {}).items()
                    },
                    sort_keys=True,
                ),
                "precision": validation.grid_precision,
                "recall": validation.grid_recall,
                "f1": validation.grid_f1,
                "learning_rate": learning_rate,
                **{
                    name: augmentation_stats.get(name, 0)
                    for name in _HISTORY_COLUMNS
                    if name in augmentation_stats
                },
                "augmentation_stats": json.dumps(augmentation_stats, sort_keys=True),
            }
        )


def _checkpoint_payload(
    *,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: Optional[LRScheduler],
    scaler: torch.amp.GradScaler,
    epoch: int,
    best_val_f1: float,
    best_grid_f1: float,
    best_centroid_f1: float,
    best_epoch: int,
    best_grid_epoch: int,
    best_centroid_epoch: int,
    early_stopping_bad_epochs: int,
    checkpoint_criterion: str,
    resolved_weights: ResolvedClassWeights,
    validation: EpochMetrics,
    checkpoint_type: str,
    selection_metric: str,
    selection_threshold: float,
    best_val_f1_alias_target: str,
    augmentation_preset: Optional[str],
    resolved_augmentation: Mapping[str, Any],
    augmentation_stats: Mapping[str, Any],
    model_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Capture resume state and explicit fixed-threshold selection metadata."""

    return {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state": scaler.state_dict(),
        "epoch": epoch,
        "checkpoint_type": checkpoint_type,
        "selection_metric": selection_metric,
        "selection_threshold": selection_threshold,
        "grid_f1": validation.grid_f1,
        "centroid_precision": validation.centroid_precision,
        "centroid_recall": validation.centroid_recall,
        "centroid_f1": validation.centroid_f1,
        "best_val_f1_alias_target": best_val_f1_alias_target,
        "best_val_f1": best_val_f1,
        "best_grid_f1": best_grid_f1,
        "best_centroid_f1": best_centroid_f1,
        "best_epoch": best_epoch,
        "best_grid_epoch": best_grid_epoch,
        "best_centroid_epoch": best_centroid_epoch,
        "checkpoint_criterion": checkpoint_criterion,
        "early_stopping_bad_epochs": early_stopping_bad_epochs,
        "class_weight_mode": resolved_weights.mode,
        "class_weights": list(resolved_weights.weights),
        "class_statistics": [item.as_dict() for item in resolved_weights.statistics],
        "rng_state": _capture_rng_state(),
        "augmentation_preset": augmentation_preset,
        "resolved_augmentation": dict(resolved_augmentation),
        "augmentation_stats": dict(augmentation_stats),
        "model_metadata": dict(model_metadata),
    }


def _capture_rng_state() -> dict[str, Any]:
    """Capture all available RNG states so resumed runs remain reproducible."""

    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_checkpoint(
    checkpoint_path: Path,
    *,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: Optional[LRScheduler],
    scaler: torch.amp.GradScaler,
    device: torch.device,
    checkpoint_criterion: str,
) -> _ResumeState:
    """Restore a checkpoint or raise a diagnostic error before any new epoch starts."""

    if not checkpoint_path.is_file():
        raise TrainingError("resume checkpoint does not exist: {}".format(checkpoint_path))
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
    except (OSError, RuntimeError, ValueError) as error:
        raise TrainingError(
            "unable to load resume checkpoint '{}': {}".format(checkpoint_path, error)
        ) from error
    if not isinstance(checkpoint, dict):
        raise TrainingError("resume checkpoint must contain a mapping")
    if (
        checkpoint.get("checkpoint_kind") == "inference_candidate"
        or checkpoint.get("resumable") is False
    ):
        raise TrainingError(
            "resume checkpoint is an inference/evaluation candidate or weights-only "
            "snapshot and cannot resume training: it has no optimizer, scheduler, "
            "scaler, or complete training state; resume from last.pt instead"
        )
    required_keys = {
        "model_state",
        "optimizer_state",
        "scheduler_state",
        "scaler_state",
        "epoch",
        "best_val_f1",
        "early_stopping_bad_epochs",
        "rng_state",
    }
    missing_keys = sorted(required_keys.difference(checkpoint))
    if missing_keys:
        raise TrainingError("resume checkpoint is missing keys: {}".format(missing_keys))
    try:
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        if scheduler is not None:
            if checkpoint["scheduler_state"] is None:
                raise TrainingError("resume checkpoint has no scheduler state")
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        scaler.load_state_dict(checkpoint["scaler_state"])
        _restore_rng_state(checkpoint["rng_state"])
    except (RuntimeError, ValueError, TypeError) as error:
        raise TrainingError(
            "resume checkpoint is incompatible with the configured training run: {}".format(
                error
            )
        ) from error
    epoch = checkpoint["epoch"]
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise TrainingError("resume checkpoint epoch must be a non-negative integer")
    best_val_f1 = checkpoint["best_val_f1"]
    if not isinstance(best_val_f1, (int, float)):
        raise TrainingError("resume checkpoint best_val_f1 must be numeric")
    bad_epochs = checkpoint["early_stopping_bad_epochs"]
    if isinstance(bad_epochs, bool) or not isinstance(bad_epochs, int) or bad_epochs < 0:
        raise TrainingError(
            "resume checkpoint early_stopping_bad_epochs must be non-negative"
        )
    saved_criterion = checkpoint.get("checkpoint_criterion", checkpoint_criterion)
    if saved_criterion not in {"grid_f1", "centroid_f1"}:
        raise TrainingError("resume checkpoint criterion is invalid")
    best_grid_f1 = checkpoint.get(
        "best_grid_f1",
        best_val_f1 if saved_criterion == "grid_f1" else float("-inf"),
    )
    best_centroid_f1 = checkpoint.get(
        "best_centroid_f1",
        best_val_f1 if saved_criterion == "centroid_f1" else float("-inf"),
    )
    best_epoch = checkpoint.get("best_epoch", 0)
    best_grid_epoch = checkpoint.get("best_grid_epoch", 0)
    best_centroid_epoch = checkpoint.get("best_centroid_epoch", 0)
    if (
        isinstance(best_grid_f1, bool)
        or not isinstance(best_grid_f1, (int, float))
        or isinstance(best_centroid_f1, bool)
        or not isinstance(best_centroid_f1, (int, float))
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (best_epoch, best_grid_epoch, best_centroid_epoch)
        )
    ):
        raise TrainingError("resume checkpoint best F1 values or epochs are invalid")
    return _ResumeState(
        start_epoch=epoch + 1,
        best_val_f1=float(best_val_f1),
        early_stopping_bad_epochs=bad_epochs,
        best_grid_f1=float(best_grid_f1),
        best_centroid_f1=float(best_centroid_f1),
        best_epoch=best_epoch,
        best_grid_epoch=best_grid_epoch,
        best_centroid_epoch=best_centroid_epoch,
    )


def _restore_rng_state(state: Any) -> None:
    """Restore Python, NumPy, Torch CPU, and available CUDA RNG states from checkpoint."""

    if not isinstance(state, dict):
        raise TrainingError("resume checkpoint rng_state must be a mapping")
    required_keys = {"python", "numpy", "torch"}
    if not required_keys.issubset(state):
        raise TrainingError("resume checkpoint rng_state is incomplete")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])
