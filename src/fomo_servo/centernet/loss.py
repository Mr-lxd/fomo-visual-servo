"""CenterNet penalty-reduced focal loss plus masked L1 offset and size losses."""

from __future__ import annotations

from typing import Mapping

import torch
from torch import Tensor


def centernet_loss(
    raw: Tensor,
    targets: Mapping[str, Tensor],
    *,
    alpha: float = 2.0,
    beta: float = 4.0,
    offset_weight: float = 1.0,
    size_weight: float = 0.1,
) -> tuple[Tensor, dict[str, float]]:
    """Return ``(total, parts)`` for raw logits ``[B,C+4,G,G]`` and dense targets.

    Heat is normalised by the number of positive cells; cells with
    ``heat_weight == 0`` (ignore regions) contribute nothing. Offset and size L1
    are normalised by the number of supervised centre cells.
    """

    raw = raw.float()
    c = targets["heat"].shape[1]
    gt = targets["heat"].float()
    weight = targets["heat_weight"].float().unsqueeze(1)
    pred = torch.sigmoid(raw[:, :c]).clamp(1e-4, 1.0 - 1e-4)

    positive = gt.eq(1.0).float() * weight
    negative = gt.lt(1.0).float() * weight
    positive_loss = torch.log(pred) * (1.0 - pred).pow(alpha) * positive
    negative_loss = torch.log(1.0 - pred) * pred.pow(alpha) * (1.0 - gt).pow(beta) * negative
    num_positive = positive.sum().clamp(min=1.0)
    heat_loss = -(positive_loss.sum() + negative_loss.sum()) / num_positive

    reg_mask = targets["reg_mask"].float().unsqueeze(1)
    offset_pred = torch.sigmoid(raw[:, c : c + 2])
    offset_loss = ((offset_pred - targets["offset"].float()).abs() * reg_mask).sum() / (
        reg_mask.sum() * 2.0
    ).clamp(min=1.0)

    size_mask = targets["size_mask"].float().unsqueeze(1)
    size_loss = ((raw[:, c + 2 : c + 4] - targets["size"].float()).abs() * size_mask).sum() / (
        size_mask.sum() * 2.0
    ).clamp(min=1.0)

    total = heat_loss + offset_weight * offset_loss + size_weight * size_loss
    return total, {
        "heat": float(heat_loss.detach()),
        "offset": float(offset_loss.detach()),
        "size": float(size_loss.detach()),
    }
