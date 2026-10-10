"""Peak decoding for CenterNet-lite outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
from torch.nn import functional as F

from fomo_servo.geometry.letterbox import LetterboxTransform


@dataclass(frozen=True)
class CenterDetection:
    """One decoded object; ``x/y/w/h`` are original-image pixels."""

    class_id: int
    score: float
    x: float
    y: float
    w: float
    h: float


def decode_centernet(
    output: np.ndarray,
    transform: LetterboxTransform,
    *,
    stride: int,
    threshold: float,
    num_classes: Optional[int] = None,
) -> Tuple[CenterDetection, ...]:
    """Decode ``[C+4,G,G]`` activated output (heat/offset sigmoid, raw log size).

    A peak is a 3x3 local maximum of one class channel with score >= ``threshold``.
    """

    c = (output.shape[0] - 4) if num_classes is None else num_classes
    heat = torch.from_numpy(np.ascontiguousarray(output[:c], dtype=np.float32))[None]
    local_max = F.max_pool2d(heat, kernel_size=3, stride=1, padding=1)
    peaks = ((heat == local_max) & (heat >= threshold))[0].numpy()
    detections = []
    for class_id, gy, gx in np.argwhere(peaks):
        score = float(output[class_id, gy, gx])
        dx, dy = float(output[c, gy, gx]), float(output[c + 1, gy, gx])
        w_cells = float(np.exp(output[c + 2, gy, gx]))
        h_cells = float(np.exp(output[c + 3, gy, gx]))
        x, y = transform.inverse_point((gx + dx) * stride, (gy + dy) * stride)
        detections.append(
            CenterDetection(
                class_id=int(class_id),
                score=score,
                x=min(max(x, 0.0), float(transform.original_width)),
                y=min(max(y, 0.0), float(transform.original_height)),
                w=w_cells * stride / transform.scale,
                h=h_cells * stride / transform.scale,
            )
        )
    detections.sort(key=lambda d: -d.score)
    return tuple(detections)
