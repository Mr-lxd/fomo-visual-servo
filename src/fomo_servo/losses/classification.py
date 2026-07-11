"""Weighted cross entropy and focal losses for FOMO class-index heatmaps."""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from fomo_servo.config import LossConfig


class LossConfigurationError(ValueError):
    """Raised when loss settings or logits/target contracts are incompatible."""


class FOMOClassificationLoss(nn.Module):
    """Compute weighted CE or focal CE from logits [B,C,G,G] and targets [B,G,G]."""

    def __init__(self, config: LossConfig) -> None:
        super().__init__()
        if not isinstance(config, LossConfig):
            raise LossConfigurationError("config must be a LossConfig instance")
        if config.name not in {"weighted_cross_entropy", "focal_cross_entropy"}:
            raise LossConfigurationError("unsupported FOMO classification loss")
        if config.gamma < 0:
            raise LossConfigurationError("gamma must be non-negative")
        self.name = config.name
        self.gamma = config.gamma
        self.register_buffer(
            "class_weights", torch.tensor(config.class_weights, dtype=torch.float32)
        )

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Return scalar loss for logits [B,C,G,G] and class-index targets [B,G,G]."""

        if not isinstance(logits, Tensor) or logits.ndim != 4:
            raise LossConfigurationError("logits must have shape [B,C,G,G]")
        if not isinstance(targets, Tensor) or targets.ndim != 3:
            raise LossConfigurationError("targets must have shape [B,G,G]")
        if logits.shape[0] != targets.shape[0] or logits.shape[-2:] != targets.shape[-2:]:
            raise LossConfigurationError("logits and targets batch/grid shapes must match")
        if targets.dtype != torch.int64:
            raise LossConfigurationError("targets must have dtype torch.int64")
        if logits.shape[1] != self.class_weights.numel():
            raise LossConfigurationError(
                "class_weights length must equal logits channel count (background + classes)"
            )
        weights = self.class_weights.to(device=logits.device, dtype=logits.dtype)
        if self.name == "weighted_cross_entropy":
            return functional.cross_entropy(logits, targets, weight=weights)

        log_probabilities = functional.log_softmax(logits, dim=1)
        target_log_probabilities = log_probabilities.gather(
            1, targets.unsqueeze(1)
        ).squeeze(1)
        target_probabilities = target_log_probabilities.exp()
        target_weights = weights[targets]
        focal_terms = (
            -(1.0 - target_probabilities).pow(self.gamma)
            * target_log_probabilities
            * target_weights
        )
        return focal_terms.sum() / target_weights.sum()


def build_classification_loss(config: LossConfig) -> FOMOClassificationLoss:
    """Build a YAML-configured FOMO loss for arbitrary single or multi-class outputs."""

    return FOMOClassificationLoss(config)
