"""Public runtime helpers used by future FOMO training loops."""

from .runtime import (
    TrainingRuntime,
    TrainingRuntimeError,
    autocast_context,
    create_training_runtime,
    move_training_batch,
    prepare_model,
)
from fomo_servo.datasets import FOMOBatch, collate_fomo_samples
from .engine import (
    EpochMetrics,
    TrainingError,
    TrainingSummary,
    ensure_finite_gradients,
    run_training,
    set_random_seed,
)
from .class_weights import (
    AutoClassWeightSettings,
    ClassTrainingStatistics,
    ClassWeightError,
    ResolvedClassWeights,
    collect_training_heatmap_statistics,
    resolve_auto_class_weights,
    resolve_training_class_weights,
)

__all__ = [
    "TrainingRuntime",
    "TrainingRuntimeError",
    "TrainingError",
    "AutoClassWeightSettings",
    "ClassTrainingStatistics",
    "ClassWeightError",
    "ResolvedClassWeights",
    "TrainingSummary",
    "EpochMetrics",
    "FOMOBatch",
    "autocast_context",
    "create_training_runtime",
    "collate_fomo_samples",
    "ensure_finite_gradients",
    "move_training_batch",
    "prepare_model",
    "run_training",
    "collect_training_heatmap_statistics",
    "resolve_auto_class_weights",
    "resolve_training_class_weights",
    "set_random_seed",
]
