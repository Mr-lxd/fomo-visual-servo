"""YAML configuration loading and validation for the project skeleton."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

import yaml


ConfigPath = Union[str, Path]
_UNRESOLVED_ENVIRONMENT_VARIABLE = re.compile(
    r"\$\{[^}]+\}|\$[A-Za-z_][A-Za-z0-9_]*|%[^%]+%"
)


class ConfigurationError(ValueError):
    """Raised when a project YAML configuration is unreadable or invalid."""


@dataclass(frozen=True)
class DatasetConfig:
    """Dataset fields shared by YOLO loading, class mapping, training, and validation."""

    root: Path
    train_split: str
    validation_split: str
    class_names: Tuple[str, ...]
    class_mode: str
    merged_class_name: str
    collision_policy: str


@dataclass(frozen=True)
class ModelConfig:
    """Model fields shared by training, inference, and fixed-shape ONNX export."""

    input_size: int
    output_stride: int
    backbone: str
    width_multiplier: float
    head_channels: int


@dataclass(frozen=True)
class TrainingConfig:
    """YAML-controlled runtime fields for future training and evaluation entry points."""

    device: str = "auto"
    amp: bool = False
    amp_initial_scale: float = 256.0
    num_workers: int = 0
    pin_memory: bool = False
    batch_size: int = 1
    epochs: int = 1
    seed: int = 42
    output_dir: Path = Path("outputs/fomo")
    resume: Optional[Path] = None
    early_stopping_patience: int = 0
    early_stopping_min_delta: float = 0.0
    optimizer: "OptimizerConfig" = field(
        default_factory=lambda: OptimizerConfig(
            name="adamw", learning_rate=0.001, weight_decay=0.0
        )
    )
    scheduler: "SchedulerConfig" = field(
        default_factory=lambda: SchedulerConfig(name="none", step_size=1, gamma=1.0)
    )


@dataclass(frozen=True)
class LossConfig:
    """Loss settings for logits [B,1+N,G,G] and class-index targets [B,G,G]."""

    name: str
    gamma: float
    class_weights: Tuple[float, ...]


@dataclass(frozen=True)
class OptimizerConfig:
    """AdamW hyperparameters sourced exclusively from YAML."""

    name: str
    learning_rate: float
    weight_decay: float


@dataclass(frozen=True)
class SchedulerConfig:
    """Learning-rate schedule fields sourced exclusively from YAML."""

    name: str
    step_size: int
    gamma: float


@dataclass(frozen=True)
class ProjectConfig:
    """Validated project configuration with derived FOMO output dimensions."""

    dataset: DatasetConfig
    model: ModelConfig
    loss: LossConfig
    training: TrainingConfig
    source_path: Path

    @property
    def grid_size(self) -> int:
        """Return the square heatmap size S / stride."""

        return self.model.input_size // self.model.output_stride

    @property
    def output_channels(self) -> int:
        """Return background plus the configured foreground class count."""

        return 1 + len(self.dataset.class_names)


def load_config(path: ConfigPath) -> ProjectConfig:
    """Load a YAML config and validate the minimal FOMO shape contract.

    Args:
        path: YAML file path supplied by a caller, not a hard-coded dataset path.

    Returns:
        A validated configuration with ``grid_size`` and ``output_channels``.

    Raises:
        ConfigurationError: If the file cannot be read, parsed, or validated.
    """

    source_path = Path(path)
    try:
        with source_path.open("r", encoding="utf-8") as config_file:
            payload = yaml.safe_load(config_file)
    except OSError as error:
        raise ConfigurationError(
            "Unable to read configuration '{}': {}".format(source_path, error)
        ) from error
    except yaml.YAMLError as error:
        raise ConfigurationError(
            "Unable to parse YAML configuration '{}': {}".format(source_path, error)
        ) from error

    if not isinstance(payload, Mapping):
        raise ConfigurationError("Configuration root must be a YAML mapping")

    dataset_mapping = _required_mapping(payload, "dataset")
    model_mapping = _required_mapping(payload, "model")

    root = _required_path(dataset_mapping, "root", "dataset")
    train_split = _optional_text(dataset_mapping, "train_split", "train", "dataset")
    validation_split = _optional_text(
        dataset_mapping, "validation_split", "val", "dataset"
    )
    class_names = _required_class_names(dataset_mapping)
    class_mode = _optional_text(
        dataset_mapping,
        "class_mode",
        "merge_single" if len(class_names) == 1 else "preserve",
        "dataset",
    )
    if class_mode not in {"merge_single", "preserve"}:
        raise ConfigurationError(
            "dataset.class_mode must be 'merge_single' or 'preserve'"
        )
    collision_policy = _optional_text(
        dataset_mapping, "collision_policy", "error", "dataset"
    )
    if collision_policy not in {"error", "keep_first"}:
        raise ConfigurationError(
            "dataset.collision_policy must be 'error' or 'keep_first'"
        )
    merged_class_name = _optional_text(
        dataset_mapping, "merged_class_name", class_names[0], "dataset"
    )
    if class_mode == "merge_single" and class_names != (merged_class_name,):
        raise ConfigurationError(
            "dataset.classes must contain only dataset.merged_class_name in merge_single mode"
        )

    input_size = _required_positive_integer(model_mapping, "input_size", "model")
    output_stride = _required_positive_integer(
        model_mapping, "output_stride", "model"
    )
    if input_size % output_stride != 0:
        raise ConfigurationError(
            "model.input_size must be divisible by model.output_stride"
        )
    backbone = _optional_text(
        model_mapping, "backbone", "mobilenet_v2_lite", "model"
    )
    width_multiplier = _optional_positive_float(
        model_mapping, "width_multiplier", 0.35, "model"
    )
    head_channels = _optional_positive_integer(
        model_mapping, "head_channels", 32, "model"
    )

    loss_mapping = _optional_mapping(payload, "loss")
    loss_name = _optional_text(
        loss_mapping, "name", "focal_cross_entropy", "loss"
    )
    if loss_name not in {"weighted_cross_entropy", "focal_cross_entropy"}:
        raise ConfigurationError(
            "loss.name must be 'weighted_cross_entropy' or 'focal_cross_entropy'"
        )
    loss_gamma = _optional_nonnegative_float(loss_mapping, "gamma", 2.0, "loss")
    class_weights = _optional_class_weights(
        loss_mapping,
        expected_count=1 + len(class_names),
    )

    training_mapping = _training_mapping(payload)
    device = _optional_text(training_mapping, "device", "auto", "training")
    amp = _optional_boolean(training_mapping, "amp", False, "training")
    amp_initial_scale = _optional_positive_float(
        training_mapping, "amp_initial_scale", 256.0, "training"
    )
    num_workers = _optional_nonnegative_integer(
        training_mapping, "num_workers", 0, "training"
    )
    pin_memory = _optional_boolean(
        training_mapping, "pin_memory", False, "training"
    )
    batch_size = _optional_positive_integer(
        training_mapping, "batch_size", 1, "training"
    )
    epochs = _optional_positive_integer(training_mapping, "epochs", 1, "training")
    seed = _optional_nonnegative_integer(training_mapping, "seed", 42, "training")
    output_dir = _optional_path(
        training_mapping, "output_dir", "outputs/fomo", "training"
    )
    resume = _optional_nullable_path(training_mapping, "resume", "training")
    early_stopping_patience = _optional_nonnegative_integer(
        training_mapping, "early_stopping_patience", 0, "training"
    )
    early_stopping_min_delta = _optional_nonnegative_float(
        training_mapping, "early_stopping_min_delta", 0.0, "training"
    )
    optimizer_mapping = _optional_mapping(training_mapping, "optimizer")
    optimizer_name = _optional_text(
        optimizer_mapping, "name", "adamw", "training.optimizer"
    )
    if optimizer_name != "adamw":
        raise ConfigurationError("training.optimizer.name must be 'adamw'")
    learning_rate = _optional_positive_float(
        optimizer_mapping, "learning_rate", 0.001, "training.optimizer"
    )
    weight_decay = _optional_nonnegative_float(
        optimizer_mapping, "weight_decay", 0.0, "training.optimizer"
    )
    scheduler_mapping = _optional_mapping(training_mapping, "scheduler")
    scheduler_name = _optional_text(
        scheduler_mapping, "name", "none", "training.scheduler"
    )
    if scheduler_name not in {"none", "step_lr"}:
        raise ConfigurationError(
            "training.scheduler.name must be 'none' or 'step_lr'"
        )
    step_size = _optional_positive_integer(
        scheduler_mapping, "step_size", 1, "training.scheduler"
    )
    scheduler_gamma = _optional_positive_float(
        scheduler_mapping, "gamma", 1.0, "training.scheduler"
    )

    return ProjectConfig(
        dataset=DatasetConfig(
            root=root,
            train_split=train_split,
            validation_split=validation_split,
            class_names=class_names,
            class_mode=class_mode,
            merged_class_name=merged_class_name,
            collision_policy=collision_policy,
        ),
        model=ModelConfig(
            input_size=input_size,
            output_stride=output_stride,
            backbone=backbone,
            width_multiplier=width_multiplier,
            head_channels=head_channels,
        ),
        loss=LossConfig(
            name=loss_name,
            gamma=loss_gamma,
            class_weights=class_weights,
        ),
        training=TrainingConfig(
            device=device,
            amp=amp,
            amp_initial_scale=amp_initial_scale,
            num_workers=num_workers,
            pin_memory=pin_memory,
            batch_size=batch_size,
            epochs=epochs,
            seed=seed,
            output_dir=output_dir,
            resume=resume,
            early_stopping_patience=early_stopping_patience,
            early_stopping_min_delta=early_stopping_min_delta,
            optimizer=OptimizerConfig(
                name=optimizer_name,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
            ),
            scheduler=SchedulerConfig(
                name=scheduler_name,
                step_size=step_size,
                gamma=scheduler_gamma,
            ),
        ),
        source_path=source_path,
    )


def _required_mapping(payload: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = payload.get(name)
    if value is None:
        raise ConfigurationError("{} is required".format(name))
    if not isinstance(value, Mapping):
        raise ConfigurationError("{} must be a mapping".format(name))
    return value


def _optional_mapping(payload: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    """Return an optional YAML mapping, or an empty mapping when absent."""

    value = payload.get(name, {})
    if not isinstance(value, Mapping):
        raise ConfigurationError("{} must be a mapping".format(name))
    return value


def _training_mapping(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """Read canonical ``training`` or legacy uppercase ``TRAIN`` YAML settings."""

    has_lowercase = "training" in payload
    has_uppercase = "TRAIN" in payload
    if has_lowercase and has_uppercase:
        raise ConfigurationError("training and TRAIN must not both be provided")
    if has_lowercase:
        return _optional_mapping(payload, "training")
    if not has_uppercase:
        return {}

    uppercase_mapping = _optional_mapping(payload, "TRAIN")
    normalized: dict[str, Any] = {}
    for key, value in uppercase_mapping.items():
        if not isinstance(key, str):
            raise ConfigurationError("TRAIN keys must be strings")
        normalized_key = key.lower()
        if normalized_key in normalized:
            raise ConfigurationError(
                "TRAIN must not contain duplicate keys differing only by case"
            )
        normalized[normalized_key] = value
    return normalized


def _required_path(
    payload: Mapping[str, Any], name: str, section: str
) -> Path:
    value = payload.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError("{}.{} must be a non-empty path string".format(section, name))
    return _expand_environment_path(value, section, name)


def _optional_path(
    payload: Mapping[str, Any], name: str, default: str, section: str
) -> Path:
    """Return a non-empty YAML path without resolving it against a machine path."""

    value = payload.get(name, default)
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError("{}.{} must be a non-empty path string".format(section, name))
    return _expand_environment_path(value, section, name)


def _optional_nullable_path(
    payload: Mapping[str, Any], name: str, section: str
) -> Optional[Path]:
    """Return a nullable YAML checkpoint path used for training resume."""

    value = payload.get(name)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(
            "{}.{} must be null or a non-empty path string".format(section, name)
        )
    return _expand_environment_path(value, section, name)


def _expand_environment_path(value: str, section: str, name: str) -> Path:
    """Expand Windows/POSIX environment variables and reject unresolved placeholders."""

    expanded = os.path.expandvars(value)
    if _UNRESOLVED_ENVIRONMENT_VARIABLE.search(expanded):
        raise ConfigurationError(
            "{}.{} references an undefined environment variable: {}".format(
                section, name, value
            )
        )
    return Path(expanded)


def _optional_text(
    payload: Mapping[str, Any], name: str, default: str, section: str
) -> str:
    value = payload.get(name, default)
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError("{}.{} must be a non-empty string".format(section, name))
    return value


def _optional_positive_float(
    payload: Mapping[str, Any], name: str, default: float, section: str
) -> float:
    """Return a positive YAML number while excluding booleans."""

    value = payload.get(name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ConfigurationError("{}.{} must be a positive number".format(section, name))
    return float(value)


def _optional_nonnegative_float(
    payload: Mapping[str, Any], name: str, default: float, section: str
) -> float:
    """Return a finite non-negative YAML number while excluding booleans."""

    value = payload.get(name, default)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(value)
        or value < 0
    ):
        raise ConfigurationError(
            "{}.{} must be a finite non-negative number".format(section, name)
        )
    return float(value)


def _optional_positive_integer(
    payload: Mapping[str, Any], name: str, default: int, section: str
) -> int:
    """Return a positive YAML integer while excluding booleans."""

    value = payload.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigurationError("{}.{} must be a positive integer".format(section, name))
    return value


def _optional_nonnegative_integer(
    payload: Mapping[str, Any], name: str, default: int, section: str
) -> int:
    """Return a non-negative YAML integer while excluding booleans."""

    value = payload.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ConfigurationError(
            "{}.{} must be a non-negative integer".format(section, name)
        )
    return value


def _optional_boolean(
    payload: Mapping[str, Any], name: str, default: bool, section: str
) -> bool:
    """Return a YAML boolean without accepting ambiguous truthy values."""

    value = payload.get(name, default)
    if not isinstance(value, bool):
        raise ConfigurationError("{}.{} must be a boolean".format(section, name))
    return value


def _optional_class_weights(
    payload: Mapping[str, Any], expected_count: int
) -> Tuple[float, ...]:
    """Validate background-plus-foreground class weights from the loss YAML block."""

    raw_weights = payload.get("class_weights", [1.0] * expected_count)
    if not isinstance(raw_weights, Sequence) or isinstance(raw_weights, (str, bytes)):
        raise ConfigurationError("loss.class_weights must be a list of positive numbers")
    if len(raw_weights) != expected_count:
        raise ConfigurationError(
            "loss.class_weights must contain {} values".format(expected_count)
        )
    weights: list[float] = []
    for index, value in enumerate(raw_weights):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(value)
            or value <= 0
        ):
            raise ConfigurationError(
                "loss.class_weights[{}] must be a finite positive number".format(index)
            )
        weights.append(float(value))
    return tuple(weights)


def _required_class_names(payload: Mapping[str, Any]) -> Tuple[str, ...]:
    value = payload.get("classes")
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ConfigurationError("dataset.classes must be a list of class names")
    if not value:
        raise ConfigurationError("dataset.classes must contain at least one class name")

    class_names = tuple(value)
    if any(not isinstance(name, str) or not name.strip() for name in class_names):
        raise ConfigurationError("dataset.classes must contain non-empty strings")
    if len(set(class_names)) != len(class_names):
        raise ConfigurationError("dataset.classes must not contain duplicates")
    return class_names


def _required_positive_integer(
    payload: Mapping[str, Any], name: str, section: str
) -> int:
    value = payload.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigurationError("{}.{} must be a positive integer".format(section, name))
    return value
