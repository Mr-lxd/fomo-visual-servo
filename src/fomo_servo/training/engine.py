"""Deterministic FOMO training, validation, checkpointing, and resume orchestration."""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LRScheduler, StepLR
from torch.utils.data import DataLoader

from fomo_servo.config import ProjectConfig, SchedulerConfig
from fomo_servo.datasets import (
    FOMOBatch,
    YOLOv5FOMODataset,
    collate_fomo_samples,
)
from fomo_servo.losses import FOMOClassificationLoss, build_classification_loss
from fomo_servo.metrics import ForegroundMetrics, foreground_micro_metrics
from fomo_servo.models import build_fomo_model
from fomo_servo.runtime import DeviceRequest
from fomo_servo.training.runtime import (
    TrainingRuntime,
    autocast_context,
    create_training_runtime,
    move_training_batch,
    prepare_model,
)


class TrainingError(RuntimeError):
    """Raised when a FOMO training run cannot safely continue."""


@dataclass(frozen=True)
class EpochMetrics:
    """One validation epoch's mean loss and foreground micro metrics."""

    loss: float
    precision: float
    recall: float
    f1: float


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


@dataclass(frozen=True)
class _ResumeState:
    """Private restored checkpoint metadata used to continue an existing run."""

    start_epoch: int
    best_val_f1: float
    early_stopping_bad_epochs: int


_HISTORY_COLUMNS = (
    "epoch",
    "train_loss",
    "val_loss",
    "precision",
    "recall",
    "f1",
    "learning_rate",
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
    set_random_seed(config.training.seed)
    runtime = create_training_runtime(config.training, device_override)
    model = prepare_model(build_fomo_model(config), runtime)
    criterion = build_classification_loss(config.loss).to(runtime.device)
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
    train_loader, validation_loader = _build_data_loaders(config, runtime)

    output_dir = config.training.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "history.csv"
    resume_path = resume_override if resume_override is not None else config.training.resume
    if resume_path is None:
        _initialize_history(history_path, append=False)
        resume_state = _ResumeState(
            start_epoch=1,
            best_val_f1=float("-inf"),
            early_stopping_bad_epochs=0,
        )
    else:
        resume_state = _restore_checkpoint(
            Path(resume_path),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=runtime.device,
        )
        _initialize_history(history_path, append=True)

    best_val_f1 = resume_state.best_val_f1
    bad_epochs = resume_state.early_stopping_bad_epochs
    completed_epochs = resume_state.start_epoch - 1
    stopped_early = False
    for epoch in range(resume_state.start_epoch, config.training.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            runtime=runtime,
        )
        validation = validate_one_epoch(
            model=model,
            loader=validation_loader,
            criterion=criterion,
            runtime=runtime,
        )
        learning_rate = float(optimizer.param_groups[0]["lr"])
        if scheduler is not None:
            scheduler.step()

        improved = validation.f1 > (
            best_val_f1 + config.training.early_stopping_min_delta
        )
        if improved:
            best_val_f1 = validation.f1
            bad_epochs = 0
        else:
            bad_epochs += 1
        checkpoint = _checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            best_val_f1=best_val_f1,
            early_stopping_bad_epochs=bad_epochs,
        )
        torch.save(checkpoint, output_dir / "last.pt")
        if improved:
            torch.save(checkpoint, output_dir / "best_val_f1.pt")
        _append_history(
            history_path,
            epoch=epoch,
            train_loss=train_loss,
            validation=validation,
            learning_rate=learning_rate,
        )
        completed_epochs = epoch
        if (
            config.training.early_stopping_patience > 0
            and bad_epochs >= config.training.early_stopping_patience
        ):
            stopped_early = True
            break

    return TrainingSummary(
        start_epoch=resume_state.start_epoch,
        completed_epochs=completed_epochs,
        best_val_f1=max(best_val_f1, 0.0),
        stopped_early=stopped_early,
        output_dir=output_dir,
        device=runtime.device,
        amp_enabled=runtime.amp_enabled,
    )


def train_one_epoch(
    *,
    model: nn.Module,
    loader: DataLoader[FOMOBatch],
    criterion: FOMOClassificationLoss,
    optimizer: Optimizer,
    scaler: torch.amp.GradScaler,
    runtime: TrainingRuntime,
) -> float:
    """Run one train epoch and return sample-weighted scalar loss."""

    model.train()
    total_loss = 0.0
    total_samples = 0
    for batch in loader:
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
    return total_loss / total_samples


def validate_one_epoch(
    *,
    model: nn.Module,
    loader: DataLoader[FOMOBatch],
    criterion: FOMOClassificationLoss,
    runtime: TrainingRuntime,
) -> EpochMetrics:
    """Run one eval epoch and return validation loss plus foreground micro metrics."""

    model.eval()
    total_loss = 0.0
    total_samples = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0
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
            batch_size = int(images.shape[0])
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size
    if total_samples == 0:
        raise TrainingError("validation DataLoader produced no batches")
    metrics = _metrics_from_counts(true_positives, false_positives, false_negatives)
    return EpochMetrics(
        loss=total_loss / total_samples,
        precision=metrics.precision,
        recall=metrics.recall,
        f1=metrics.f1,
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
    }
    train_dataset = YOLOv5FOMODataset(
        split=config.dataset.train_split, **common_arguments
    )
    validation_dataset = YOLOv5FOMODataset(
        split=config.dataset.validation_split, **common_arguments
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
        precision=precision,
        recall=recall,
        f1=f1,
    )


def _initialize_history(history_path: Path, *, append: bool) -> None:
    """Create a CSV header for a new run or retain an existing resume history."""

    if append and history_path.is_file():
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
) -> None:
    """Append one train/validation metrics row to a configured history CSV."""

    with history_path.open("a", newline="", encoding="utf-8") as history_file:
        writer = csv.DictWriter(history_file, fieldnames=_HISTORY_COLUMNS)
        writer.writerow(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": validation.loss,
                "precision": validation.precision,
                "recall": validation.recall,
                "f1": validation.f1,
                "learning_rate": learning_rate,
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
    early_stopping_bad_epochs: int,
) -> dict[str, Any]:
    """Capture state needed to resume the same model, optimizer, schedule, and RNGs."""

    return {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state": scaler.state_dict(),
        "epoch": epoch,
        "best_val_f1": best_val_f1,
        "early_stopping_bad_epochs": early_stopping_bad_epochs,
        "rng_state": _capture_rng_state(),
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
    return _ResumeState(
        start_epoch=epoch + 1,
        best_val_f1=float(best_val_f1),
        early_stopping_bad_epochs=bad_epochs,
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
