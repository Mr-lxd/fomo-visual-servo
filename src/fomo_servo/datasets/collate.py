"""PyTorch DataLoader collation for YOLOv5 FOMO samples."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np
import torch
from torch import Tensor

from .yolo import FOMOSample


@dataclass(frozen=True)
class FOMOBatch:
    """A FOMO tensor batch before device transfer.

    ``images`` is RGB float32 ``[B,3,S,S]`` in letterbox coordinates.
    ``targets`` is class-index int64 ``[B,S/8,S/8]`` with channel zero encoded as
    background. Both tensors begin on CPU and are moved by ``move_training_batch``.
    """

    images: Tensor
    targets: Tensor
    transforms: Tuple[object, ...]
    original_boxes: Tuple[Tuple[object, ...], ...]


def collate_fomo_samples(samples: Sequence[FOMOSample]) -> FOMOBatch:
    """Stack FOMO samples into CPU training tensors with explicit dtypes and shapes."""

    if not samples:
        raise ValueError("cannot collate an empty FOMO sample sequence")
    if any(not isinstance(sample, FOMOSample) for sample in samples):
        raise TypeError("samples must contain only FOMOSample instances")
    try:
        images = np.stack([sample.image for sample in samples], axis=0)
        targets = np.stack([sample.heatmap.class_index for sample in samples], axis=0)
    except ValueError as error:
        raise ValueError(
            "FOMO samples must have identical image and heatmap shapes for batching"
        ) from error
    if images.dtype != np.float32:
        raise TypeError("FOMO sample images must have dtype float32")
    if targets.dtype != np.int64:
        raise TypeError("FOMO sample targets must have dtype int64")
    return FOMOBatch(
        images=torch.from_numpy(np.ascontiguousarray(images)),
        targets=torch.from_numpy(np.ascontiguousarray(targets)),
        transforms=tuple(sample.transform for sample in samples),
        original_boxes=tuple(sample.original_boxes for sample in samples),
    )
