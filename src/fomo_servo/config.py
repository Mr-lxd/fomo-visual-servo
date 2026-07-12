"""YAML configuration loading and validation for the project skeleton."""

from __future__ import annotations

import os
import re
import warnings
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
    checkpoint_criterion: str = "grid_f1"
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
    class_weights: Optional[Tuple[float, ...]] = None
    class_weight_mode: str = "manual"
    background_weight: float = 1.0
    foreground_base_weight: float = 25.0
    class_balance: str = "sqrt_inverse_frequency"
    min_foreground_weight: float = 12.5
    max_foreground_weight: float = 75.0


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
class PostprocessConfig:
    """YAML-controlled logits postprocessing and target-selection settings."""

    inference_threshold: Optional[float] = None
    """Default probability threshold used by image/video inference."""

    # Retained only so callers constructing the old dataclass directly keep
    # working.  YAML loading warns when this legacy field is used.
    confidence_threshold: Optional[float] = None
    class_thresholds: Optional[Tuple[float, ...]] = None
    component_mode: str = "connected_components"
    confidence_mode: str = "max"
    selection_strategy: str = "highest_confidence"
    max_match_distance_pixels: float = 32.0
    max_lost_frames: int = 5
    allowed_class_ids: Optional[Tuple[int, ...]] = None

    def __post_init__(self) -> None:
        """Resolve the legacy constructor keyword without conflating its purpose."""

        if (
            self.inference_threshold is not None
            and self.confidence_threshold is not None
            and self.inference_threshold != self.confidence_threshold
        ):
            raise ValueError(
                "inference_threshold and legacy confidence_threshold cannot both be set"
            )
        if self.inference_threshold is None:
            object.__setattr__(
                self,
                "inference_threshold",
                0.05 if self.confidence_threshold is None else self.confidence_threshold,
            )


@dataclass(frozen=True)
class EvaluationConfig:
    """YAML-controlled centroid matching and validation threshold sweep settings."""

    matching_mode: str = "centroid_in_bbox"
    max_distance_pixels: float = 32.0
    checkpoint_threshold: float = 0.5
    threshold_sweep_enabled: bool = True
    threshold_sweep_minimum: float = 0.05
    threshold_sweep_maximum: float = 0.95
    threshold_sweep_step: float = 0.05
    threshold_sweep: Tuple[float, ...] = tuple(
        round(0.05 + index * 0.05, 10) for index in range(19)
    )


@dataclass(frozen=True)
class ExperimentConfig:
    """Optional append-only experiment recording settings."""

    name: Optional[str] = None
    summary_csv: Path = Path("outputs/experiments/experiments_summary.csv")


@dataclass(frozen=True)
class ProjectConfig:
    """Validated project configuration with derived FOMO output dimensions."""

    dataset: DatasetConfig
    model: ModelConfig
    loss: LossConfig
    training: TrainingConfig
    postprocess: PostprocessConfig
    evaluation: EvaluationConfig
    source_path: Path
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)

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
    (
        class_weight_mode,
        class_weights,
        background_weight,
        foreground_base_weight,
        class_balance,
        min_foreground_weight,
        max_foreground_weight,
    ) = _parse_loss_class_weight_settings(
        loss_mapping,
        expected_count=1 + len(class_names),
    )

    postprocess_mapping = _optional_mapping(payload, "postprocess")
    has_inference_threshold = "inference_threshold" in postprocess_mapping
    has_legacy_threshold = "confidence_threshold" in postprocess_mapping
    if has_inference_threshold and has_legacy_threshold:
        raise ConfigurationError(
            "postprocess.inference_threshold and deprecated "
            "postprocess.confidence_threshold must not both be provided"
        )
    legacy_confidence_threshold: Optional[float] = None
    if has_legacy_threshold:
        warnings.warn(
            "postprocess.confidence_threshold is deprecated; use "
            "postprocess.inference_threshold. It controls inference only; "
            "evaluation.checkpoint_threshold is independent.",
            DeprecationWarning,
            stacklevel=2,
        )
        legacy_confidence_threshold = _optional_probability(
            postprocess_mapping, "confidence_threshold", 0.05, "postprocess"
        )
    inference_threshold = _optional_probability(
        postprocess_mapping,
        "inference_threshold",
        0.05 if legacy_confidence_threshold is None else legacy_confidence_threshold,
        "postprocess",
    )
    class_thresholds = _optional_class_thresholds(
        postprocess_mapping, "class_thresholds", class_names, "postprocess"
    )
    component_mode = _optional_text(
        postprocess_mapping, "component_mode", "connected_components", "postprocess"
    )
    if component_mode not in {"connected_components", "local_peaks"}:
        raise ConfigurationError(
            "postprocess.component_mode must be 'connected_components' or 'local_peaks'"
        )
    confidence_mode = _optional_text(
        postprocess_mapping, "confidence_mode", "max", "postprocess"
    )
    if confidence_mode not in {"max", "mean"}:
        raise ConfigurationError("postprocess.confidence_mode must be 'max' or 'mean'")
    selection_strategy = _optional_text(
        postprocess_mapping, "selection_strategy", "highest_confidence", "postprocess"
    )
    if selection_strategy not in {
        "highest_confidence",
        "largest_component",
        "nearest_previous",
    }:
        raise ConfigurationError(
            "postprocess.selection_strategy is not a supported target strategy"
        )
    max_match_distance_pixels = _optional_nonnegative_float(
        postprocess_mapping, "max_match_distance_pixels", 32.0, "postprocess"
    )
    max_lost_frames = _optional_nonnegative_integer(
        postprocess_mapping, "max_lost_frames", 5, "postprocess"
    )
    allowed_class_ids = _optional_class_ids(
        postprocess_mapping, "allowed_class_ids", len(class_names), "postprocess"
    )

    evaluation_mapping = _optional_mapping(payload, "evaluation")
    matching_mode = _optional_text(
        evaluation_mapping, "matching_mode", "centroid_in_bbox", "evaluation"
    )
    if matching_mode not in {"centroid_in_bbox", "max_distance_pixels"}:
        raise ConfigurationError(
            "evaluation.matching_mode must be 'centroid_in_bbox' or 'max_distance_pixels'"
        )
    max_distance_pixels = _optional_nonnegative_float(
        evaluation_mapping, "max_distance_pixels", 32.0, "evaluation"
    )
    checkpoint_threshold = _optional_probability(
        evaluation_mapping, "checkpoint_threshold", 0.5, "evaluation"
    )
    threshold_sweep_value = evaluation_mapping.get("threshold_sweep")
    if isinstance(threshold_sweep_value, Mapping):
        threshold_sweep_enabled = _optional_boolean(
            threshold_sweep_value, "enabled", True, "evaluation.threshold_sweep"
        )
        threshold_sweep_minimum = _optional_probability(
            threshold_sweep_value, "minimum", 0.05, "evaluation.threshold_sweep"
        )
        threshold_sweep_maximum = _optional_probability(
            threshold_sweep_value, "maximum", 0.95, "evaluation.threshold_sweep"
        )
        threshold_sweep_step = _optional_positive_float(
            threshold_sweep_value, "step", 0.05, "evaluation.threshold_sweep"
        )
        if threshold_sweep_minimum > threshold_sweep_maximum:
            raise ConfigurationError(
                "evaluation.threshold_sweep.minimum must not exceed maximum"
            )
        threshold_sweep = _build_threshold_sweep(
            threshold_sweep_minimum,
            threshold_sweep_maximum,
            threshold_sweep_step,
            checkpoint_threshold,
            enabled=threshold_sweep_enabled,
        )
    else:
        threshold_sweep = _optional_probability_sequence(
            evaluation_mapping,
            "threshold_sweep",
            tuple(round(0.05 + index * 0.05, 10) for index in range(19)),
            "evaluation",
        )
        threshold_sweep_enabled = True
        threshold_sweep_minimum = threshold_sweep[0]
        threshold_sweep_maximum = threshold_sweep[-1]
        threshold_sweep_step = (
            threshold_sweep[1] - threshold_sweep[0]
            if len(threshold_sweep) > 1
            else 0.05
        )

    experiment_mapping = _optional_mapping(payload, "experiment")
    experiment_name = _optional_nullable_text(
        experiment_mapping, "name", None, "experiment"
    )
    experiment_summary_csv = _optional_path(
        experiment_mapping,
        "summary_csv",
        "outputs/experiments/experiments_summary.csv",
        "experiment",
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
    checkpoint_criterion = _optional_text(
        training_mapping, "checkpoint_criterion", "grid_f1", "training"
    )
    if checkpoint_criterion not in {"grid_f1", "centroid_f1"}:
        raise ConfigurationError(
            "training.checkpoint_criterion must be 'grid_f1' or 'centroid_f1'"
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
            class_weight_mode=class_weight_mode,
            background_weight=background_weight,
            foreground_base_weight=foreground_base_weight,
            class_balance=class_balance,
            min_foreground_weight=min_foreground_weight,
            max_foreground_weight=max_foreground_weight,
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
            checkpoint_criterion=checkpoint_criterion,
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
        postprocess=PostprocessConfig(
            inference_threshold=inference_threshold,
            confidence_threshold=legacy_confidence_threshold,
            class_thresholds=class_thresholds,
            component_mode=component_mode,
            confidence_mode=confidence_mode,
            selection_strategy=selection_strategy,
            max_match_distance_pixels=max_match_distance_pixels,
            max_lost_frames=max_lost_frames,
            allowed_class_ids=allowed_class_ids,
        ),
        evaluation=EvaluationConfig(
            matching_mode=matching_mode,
            max_distance_pixels=max_distance_pixels,
            checkpoint_threshold=checkpoint_threshold,
            threshold_sweep_enabled=threshold_sweep_enabled,
            threshold_sweep_minimum=threshold_sweep_minimum,
            threshold_sweep_maximum=threshold_sweep_maximum,
            threshold_sweep_step=threshold_sweep_step,
            threshold_sweep=threshold_sweep,
        ),
        source_path=source_path,
        experiment=ExperimentConfig(
            name=experiment_name,
            summary_csv=experiment_summary_csv,
        ),
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


def _optional_nullable_text(
    payload: Mapping[str, Any],
    name: str,
    default: Optional[str],
    section: str,
) -> Optional[str]:
    """Return an optional non-empty text value or ``None``."""

    value = payload.get(name, default)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(
            "{}.{} must be null or a non-empty string".format(section, name)
        )
    return value


def _optional_probability(
    payload: Mapping[str, Any], name: str, default: float, section: str
) -> float:
    value = payload.get(name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or not 0.0 <= value <= 1.0:
        raise ConfigurationError("{}.{} must be a finite probability in [0,1]".format(section, name))
    return float(value)


def _optional_probability_sequence(
    payload: Mapping[str, Any],
    name: str,
    default: Sequence[float],
    section: str,
) -> Tuple[float, ...]:
    value = payload.get(name, default)
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise ConfigurationError("{}.{} must be a non-empty probability list".format(section, name))
    return tuple(
        _optional_probability({"value": item}, "value", 0.0, section)
        for item in value
    )


def _build_threshold_sweep(
    minimum: float,
    maximum: float,
    step: float,
    checkpoint_threshold: float,
    *,
    enabled: bool,
) -> Tuple[float, ...]:
    """Build deterministic inclusive sweep values from YAML range settings."""

    if not enabled:
        # A disabled sweep still leaves a valid final report at the locked
        # checkpoint threshold, but it cannot select a different threshold.
        return (float(checkpoint_threshold),)
    values = []
    current = float(minimum)
    tolerance = max(1e-10, abs(step) * 1e-8)
    while current <= maximum + tolerance:
        values.append(round(min(current, maximum), 10))
        current += step
    if not values:
        values.append(float(minimum))
    if values[-1] < maximum - tolerance:
        values.append(float(maximum))
    return tuple(values)


def _optional_class_thresholds(
    payload: Mapping[str, Any],
    name: str,
    class_names: Sequence[str],
    section: str,
) -> Optional[Tuple[float, ...]]:
    value = payload.get(name)
    if value is None:
        return None
    if isinstance(value, Mapping):
        thresholds = [None] * len(class_names)
        for raw_key, raw_value in value.items():
            if isinstance(raw_key, bool) or not isinstance(raw_key, (int, str)):
                raise ConfigurationError("{}.{} keys must be class IDs or names".format(section, name))
            if isinstance(raw_key, int):
                class_id = raw_key
            else:
                if raw_key not in class_names:
                    raise ConfigurationError("unknown class name '{}' in {}.{}".format(raw_key, section, name))
                class_id = class_names.index(raw_key)
            if not 0 <= class_id < len(class_names):
                raise ConfigurationError("{}.{} class ID is outside dataset classes".format(section, name))
            thresholds[class_id] = _optional_probability({"value": raw_value}, "value", 0.0, section)
        if any(item is None for item in thresholds):
            raise ConfigurationError("{}.{} must specify every dataset class".format(section, name))
        return tuple(float(item) for item in thresholds)
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != len(class_names):
        raise ConfigurationError("{}.{} must contain one value per dataset class".format(section, name))
    return tuple(
        _optional_probability({"value": item}, "value", 0.0, section)
        for item in value
    )


def _optional_class_ids(
    payload: Mapping[str, Any], name: str, class_count: int, section: str
) -> Optional[Tuple[int, ...]]:
    value = payload.get(name)
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ConfigurationError("{}.{} must be a list of class IDs".format(section, name))
    output = []
    for index in value:
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < class_count:
            raise ConfigurationError("{}.{} contains an invalid class ID".format(section, name))
        output.append(index)
    return tuple(output)


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


def _parse_loss_class_weight_settings(
    payload: Mapping[str, Any], expected_count: int
) -> tuple[str, Optional[Tuple[float, ...]], float, float, str, float, float]:
    """Parse legacy manual lists or explicit ``manual``/``auto`` loss weighting.

    The legacy form ``class_weights: [background, class_0, ...]`` remains manual.
    The explicit form is a mapping with ``mode``.  In manual mode it must include
    ``values``; automatic mode receives its balancing parameters in the same
    mapping and derives foreground weights from the training heatmaps.
    """

    raw_weights = payload.get("class_weights", [1.0] * expected_count)
    defaults = (1.0, 25.0, "sqrt_inverse_frequency", 12.5, 75.0)
    if not isinstance(raw_weights, Mapping):
        return (
            "manual",
            _validate_class_weight_sequence(raw_weights, expected_count),
            *defaults,
        )

    mode = _optional_text(raw_weights, "mode", "manual", "loss.class_weights")
    if mode == "manual":
        if "values" not in raw_weights:
            raise ConfigurationError(
                "loss.class_weights.values is required when loss.class_weights.mode is 'manual'"
            )
        return (
            "manual",
            _validate_class_weight_sequence(raw_weights["values"], expected_count),
            *defaults,
        )
    if mode != "auto":
        raise ConfigurationError(
            "loss.class_weights.mode must be 'manual' or 'auto'"
        )

    background_weight = _optional_positive_float(
        raw_weights, "background_weight", 1.0, "loss.class_weights"
    )
    foreground_base_weight = _optional_positive_float(
        raw_weights, "foreground_base_weight", 25.0, "loss.class_weights"
    )
    class_balance = _optional_text(
        raw_weights, "class_balance", "sqrt_inverse_frequency", "loss.class_weights"
    )
    if class_balance != "sqrt_inverse_frequency":
        raise ConfigurationError(
            "loss.class_weights.class_balance must be 'sqrt_inverse_frequency'"
        )
    min_foreground_weight = _optional_positive_float(
        raw_weights, "min_foreground_weight", 12.5, "loss.class_weights"
    )
    max_foreground_weight = _optional_positive_float(
        raw_weights, "max_foreground_weight", 75.0, "loss.class_weights"
    )
    if min_foreground_weight > max_foreground_weight:
        raise ConfigurationError(
            "loss.class_weights.min_foreground_weight must not exceed max_foreground_weight"
        )
    return (
        "auto",
        None,
        background_weight,
        foreground_base_weight,
        class_balance,
        min_foreground_weight,
        max_foreground_weight,
    )


def _validate_class_weight_sequence(
    raw_weights: Any, expected_count: int
) -> Tuple[float, ...]:
    """Validate a background-plus-foreground manual weight sequence."""

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
