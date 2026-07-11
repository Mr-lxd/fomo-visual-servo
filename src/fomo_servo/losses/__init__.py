"""Public FOMO logits-to-class-index classification losses."""

from .classification import (
    FOMOClassificationLoss,
    LossConfigurationError,
    build_classification_loss,
)

__all__ = [
    "FOMOClassificationLoss",
    "LossConfigurationError",
    "build_classification_loss",
]
