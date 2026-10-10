"""Experimental bbox-fill targets and legacy EI BCE for the Task 13 study."""

from __future__ import annotations

from math import ceil, floor
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as functional

from .targets import Box


def build_bbox_targets(
    boxes: Sequence[Box], *, grid_size: int, stride: int,
    central_half: bool, object_weight: float,
) -> dict[str, np.ndarray]:
    """Return int64 target and float32 positive_weight/loss_mask, all ``[G,G]``.

    Input boxes use letterbox pixel ``xyxy`` coordinates. Clip to the input
    extent, then rasterise half-open boxes with floor(min/stride) and
    ceil(max/stride). For ``central_half``, shrink around the centre before
    clipping; ignore regions always use the whole box. Foreground channels are
    class_id+1. Overlaps use keep-first in input order, with each object's
    positive weight normalised over its *owned* cells after overlap resolution.
    A fully overlapped object owns no cells. Positive objects override ignore.
    """

    target = np.zeros((grid_size, grid_size), dtype=np.int64)
    owner = np.full((grid_size, grid_size), -1, dtype=np.int64)
    ignore = np.zeros((grid_size, grid_size), dtype=bool)
    extent = grid_size * stride
    for index, box in enumerate(boxes):
        x0, y0, x1, y1 = box.x_min, box.y_min, box.x_max, box.y_max
        if central_half and box.visibility != "ignore":
            dx, dy = (x1 - x0) / 4, (y1 - y0) / 4
            x0, y0, x1, y1 = x0 + dx, y0 + dy, x1 - dx, y1 - dy
        x0, y0 = max(0.0, x0), max(0.0, y0)
        x1, y1 = min(extent, x1), min(extent, y1)
        if x1 <= x0 or y1 <= y0:
            continue
        xs = slice(floor(x0 / stride), ceil(x1 / stride))
        ys = slice(floor(y0 / stride), ceil(y1 / stride))
        if box.visibility == "ignore":
            ignore[ys, xs] = True
        else:
            free = owner[ys, xs] == -1
            owner[ys, xs][free] = index
            target[ys, xs][free] = box.class_id + 1
    positive_weight = np.ones((grid_size, grid_size), dtype=np.float32)
    for index in np.unique(owner[owner >= 0]):
        owned = owner == index
        positive_weight[owned] = object_weight / np.count_nonzero(owned)
    loss_mask = (~ignore | (owner >= 0)).astype(np.float32)
    return {"target": target, "positive_weight": positive_weight, "loss_mask": loss_mask}


def bbox_classification_loss(
    logits: torch.Tensor, targets: torch.Tensor,
    positive_weight: torch.Tensor, loss_mask: torch.Tensor,
) -> torch.Tensor:
    """Scalar legacy EI BCE for logits ``[B,C,G,G]`` and maps ``[B,G,G]``.

    Targets are int64; weights and masks are floating point. Only positive
    channel terms receive positive_weight; all negative terms retain weight 1.
    Background positives have weight 1. Ignored cells remove every channel term
    and their contribution to the mean denominator. An all-ignore batch has
    differentiable zero loss. The float32 BCE runs safely inside AMP autocast.
    """

    labels = functional.one_hot(targets, num_classes=logits.shape[1]).permute(0, 3, 1, 2).float()
    terms = functional.binary_cross_entropy_with_logits(
        logits.float(), labels, pos_weight=positive_weight[:, None].float(), reduction="none",
    )
    mask = loss_mask[:, None].float()
    return (terms * mask).sum() / (mask.sum() * logits.shape[1]).clamp_min(1.0)
