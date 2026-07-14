"""Weighted FOMO classification losses for class-index heatmaps."""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from fomo_servo.config import LossConfig


class LossConfigurationError(ValueError):
    """Raised when loss settings or logits/target contracts are incompatible."""


class FOMOClassificationLoss(nn.Module):
    """Compute a configured loss from logits ``[B,C,G,G]`` and targets ``[B,G,G]``.

    ``weighted_cross_entropy`` and ``focal_cross_entropy`` are the historical
    per-class-weighted softmax losses.  ``weighted_softmax_ce`` applies one
    explicit weight to each target cell: background uses ``background_weight``
    and every foreground class uses ``object_weight``.  It deliberately ignores
    ``class_weights``.  ``ei_weighted_xent_legacy`` follows TensorFlow's
    ``weighted_cross_entropy_with_logits`` semantics on one-hot maps: each
    foreground positive channel receives ``object_weight`` while negative
    channel terms retain weight one.
    """

    def __init__(self, config: LossConfig) -> None:
        super().__init__()
        if not isinstance(config, LossConfig):
            raise LossConfigurationError("config must be a LossConfig instance")
        supported = {
            "weighted_cross_entropy",
            "focal_cross_entropy",
            "weighted_softmax_ce",
            "ei_weighted_xent_legacy",
        }
        if config.name not in supported:
            raise LossConfigurationError("unsupported FOMO classification loss")
        if config.gamma < 0:
            raise LossConfigurationError("gamma must be non-negative")
        self.object_weight_mode = config.name in {
            "weighted_softmax_ce",
            "ei_weighted_xent_legacy",
        }
        if self.object_weight_mode and config.gamma != 0.0:
            raise LossConfigurationError(
                "object-weight loss does not support focal gamma"
            )
        if not self.object_weight_mode and config.class_weights is None:
            raise LossConfigurationError(
                "automatic class weights must be resolved from the training dataset before loss construction"
            )
        self.name = config.name
        self.gamma = config.gamma
        self.background_weight = float(config.background_weight)
        self.object_weight = float(config.object_weight)
        if self.object_weight_mode and (
            not torch.isfinite(torch.tensor(self.background_weight))
            or not torch.isfinite(torch.tensor(self.object_weight))
            or self.background_weight <= 0.0
            or self.object_weight <= 0.0
        ):
            raise LossConfigurationError(
                "background_weight and object_weight must be finite positive numbers"
            )
        self.register_buffer(
            "class_weights",
            torch.tensor(
                config.class_weights if config.class_weights is not None else (),
                dtype=torch.float32,
            ),
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
        if not self.object_weight_mode and logits.shape[1] != self.class_weights.numel():
            raise LossConfigurationError(
                "class_weights length must equal logits channel count (background + classes)"
            )
        if targets.numel() and (
            int(targets.min().item()) < 0 or int(targets.max().item()) >= logits.shape[1]
        ):
            raise LossConfigurationError(
                "targets contain a class index outside the logits channel range"
            )

        if self.name == "weighted_softmax_ce":
            target_weights = self.target_cell_weights(targets, dtype=logits.dtype)
            cell_losses = functional.cross_entropy(logits, targets, reduction="none")
            return (cell_losses * target_weights).sum() / target_weights.sum()

        if self.name == "ei_weighted_xent_legacy":
            labels = functional.one_hot(targets, num_classes=logits.shape[1])
            labels = labels.permute(0, 3, 1, 2).to(dtype=logits.dtype)
            positive_weights = torch.full(
                (logits.shape[1],),
                self.object_weight,
                device=logits.device,
                dtype=logits.dtype,
            )
            positive_weights[0] = self.background_weight
            return functional.binary_cross_entropy_with_logits(
                logits,
                labels,
                pos_weight=positive_weights.view(1, -1, 1, 1),
                reduction="mean",
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

    def target_cell_weights(
        self,
        targets: Tensor,
        *,
        dtype: torch.dtype = torch.float32,
    ) -> Tensor:
        """Return explicit background/foreground weights with shape ``[B,G,G]``.

        This helper is meaningful for object-weight modes only and keeps the
        target-cell weighting observable for exact unit tests and audit logs.
        ``targets`` is ``int64 [B,G,G]``; the returned tensor is floating point
        on the same device.
        """

        if not self.object_weight_mode or self.name != "weighted_softmax_ce":
            raise LossConfigurationError(
                "target_cell_weights is available only for weighted_softmax_ce"
            )
        if not isinstance(targets, Tensor) or targets.ndim != 3 or targets.dtype != torch.int64:
            raise LossConfigurationError("targets must have dtype torch.int64 and shape [B,G,G]")
        return torch.where(
            targets == 0,
            torch.as_tensor(
                self.background_weight, device=targets.device, dtype=dtype
            ),
            torch.as_tensor(self.object_weight, device=targets.device, dtype=dtype),
        )


def build_classification_loss(config: LossConfig) -> FOMOClassificationLoss:
    """Build a YAML-configured FOMO loss for arbitrary single or multi-class outputs."""

    return FOMOClassificationLoss(config)
