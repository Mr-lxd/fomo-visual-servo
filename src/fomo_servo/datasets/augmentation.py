"""Train-only augmentation interface with an explicitly disabled no-op path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np

from fomo_servo.config import AugmentationConfig


class AugmentationNotImplementedError(NotImplementedError):
    """Raised when a future augmentation algorithm is enabled before implementation."""


@dataclass(frozen=True)
class AugmentationResult:
    """Image/bbox result before letterbox.

    ``image`` remains RGB ``uint8 [H,W,3]`` and ``boxes`` remain original-image
    pixel-coordinate bbox objects. ``applied`` is always false in this phase.
    """

    image: np.ndarray
    boxes: tuple[Any, ...]
    applied: bool


class AugmentationPipeline:
    """Apply future train-only augmentations before letterbox and heatmap creation."""

    def __init__(self, config: AugmentationConfig, *, is_train: bool) -> None:
        if not isinstance(config, AugmentationConfig):
            raise TypeError("config must be an AugmentationConfig instance")
        if not isinstance(is_train, bool):
            raise TypeError("is_train must be a boolean")
        self.config = config
        self.is_train = is_train

    @classmethod
    def disabled(cls) -> "AugmentationPipeline":
        """Return a non-training no-op pipeline for validation/test callers."""

        return cls(AugmentationConfig.disabled(), is_train=False)

    def apply(
        self,
        image: np.ndarray,
        boxes: Sequence[Any],
        rng: Optional[Any] = None,
    ) -> AugmentationResult:
        """Apply configured operations to RGB image/bboxes before letterbox.

        Args:
            image: Original RGB uint8 image ``[H,W,3]``.
            boxes: Original-image pixel-coordinate bbox objects.
            rng: Seeded NumPy generator/module reserved for future stochastic
                operations. DataLoader workers provide a worker-seeded NumPy module.

        Returns:
            ``AugmentationResult`` with the same image and boxes when disabled.

        Raises:
            AugmentationNotImplementedError: If train augmentation is enabled in this
                phase, because no actual color/geometric/degradation algorithm exists.
        """

        if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError("augmentation image must have shape [H,W,3]")
        if image.dtype != np.uint8:
            raise ValueError("augmentation image must have dtype uint8")
        if not isinstance(boxes, Sequence):
            raise TypeError("augmentation boxes must be a sequence")
        if not self.is_train or not self.config.enabled:
            return AugmentationResult(image=image, boxes=tuple(boxes), applied=False)
        raise AugmentationNotImplementedError(
            "augmentation algorithms are not implemented in the locked framework phase"
        )
