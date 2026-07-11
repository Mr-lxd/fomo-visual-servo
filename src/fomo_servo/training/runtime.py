"""Device, AMP, and DataLoader runtime helpers for fixed-shape FOMO training."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, ContextManager, Optional, Tuple

import torch
from torch import Tensor, nn

from fomo_servo.config import TrainingConfig
from fomo_servo.runtime import DeviceRequest, resolve_device


class TrainingRuntimeError(ValueError):
    """Raised when a model, batch, or training configuration is invalid."""


@dataclass(frozen=True)
class TrainingRuntime:
    """Effective CPU/CUDA settings shared by a train DataLoader and train step.

    Attributes:
        device: Device used for ``model.to(device)`` and all train-step tensors.
        amp_enabled: Whether CUDA float16 autocast is active for model forward/loss.
        num_workers: DataLoader process count; zero keeps loading in the main process.
        pin_memory: Effective DataLoader pin-memory value, enabled only for CUDA.
        diagnostics: Explicit notices when requested CUDA-only optimizations are disabled.
    """

    device: torch.device
    amp_enabled: bool
    num_workers: int
    pin_memory: bool
    diagnostics: Tuple[str, ...]

    @property
    def data_loader_kwargs(self) -> dict[str, int | bool]:
        """Return keyword arguments for ``DataLoader(..., **kwargs)``."""

        return {"num_workers": self.num_workers, "pin_memory": self.pin_memory}


def create_training_runtime(
    config: TrainingConfig, device_override: Optional[DeviceRequest] = None
) -> TrainingRuntime:
    """Resolve YAML/CLI settings into device, AMP, and DataLoader execution policy.

    Args:
        config: Validated YAML training configuration.
        device_override: Optional CLI request. When present it overrides ``config.device``.

    Returns:
        Immutable effective runtime settings. ``auto`` resolves through
        :func:`fomo_servo.runtime.resolve_device` and prefers CUDA when available.

    Raises:
        TrainingRuntimeError: If ``config`` is not a ``TrainingConfig`` instance.
        RuntimeDeviceError: If an explicit device request is invalid or unavailable.
    """

    if not isinstance(config, TrainingConfig):
        raise TrainingRuntimeError("config must be a TrainingConfig instance")

    requested_device = config.device if device_override is None else device_override
    device = resolve_device(requested_device)
    cuda_enabled = device.type == "cuda"
    amp_enabled = config.amp and cuda_enabled
    pin_memory = config.pin_memory and cuda_enabled
    diagnostics: list[str] = []
    if config.amp and not amp_enabled:
        diagnostics.append("AMP disabled because the effective device is CPU")
    if config.pin_memory and not pin_memory:
        diagnostics.append("pin_memory disabled because the effective device is CPU")

    return TrainingRuntime(
        device=device,
        amp_enabled=amp_enabled,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        diagnostics=tuple(diagnostics),
    )


def prepare_model(model: nn.Module, runtime: TrainingRuntime) -> nn.Module:
    """Move a FOMO model to ``runtime.device`` with ``model.to(device)``.

    Args:
        model: Any PyTorch module whose forward accepts FOMO images
            ``float32 [B,3,S,S]``.
        runtime: Effective training runtime settings.

    Returns:
        The same module after in-place device migration.
    """

    if not isinstance(model, nn.Module):
        raise TrainingRuntimeError("model must be a torch.nn.Module")
    _require_runtime(runtime)
    return model.to(runtime.device)


def move_training_batch(
    images: Tensor, targets: Tensor, runtime: TrainingRuntime
) -> tuple[Tensor, Tensor]:
    """Move FOMO tensors to a runtime device with non-blocking transfer requests.

    Args:
        images: RGB ``float32 [B,3,S,S]`` image tensor in letterbox coordinates.
        targets: Class-index ``int64 [B,S/8,S/8]`` heatmap tensor.
        runtime: Effective training runtime settings.

    Returns:
        ``(images, targets)`` on ``runtime.device``. ``non_blocking=True`` is supplied
        for both transfers; CUDA loaders use ``runtime.pin_memory`` to make it effective.
    """

    _require_runtime(runtime)
    if not isinstance(images, Tensor):
        raise TypeError("images must be a torch.Tensor")
    if not isinstance(targets, Tensor):
        raise TypeError("targets must be a torch.Tensor")
    return (
        images.to(runtime.device, non_blocking=True),
        targets.to(runtime.device, non_blocking=True),
    )


def autocast_context(runtime: TrainingRuntime) -> ContextManager[Any]:
    """Return CUDA float16 autocast when enabled, otherwise a no-op context manager."""

    _require_runtime(runtime)
    if runtime.amp_enabled:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _require_runtime(runtime: TrainingRuntime) -> None:
    """Validate a public runtime object before it controls model or tensor placement."""

    if not isinstance(runtime, TrainingRuntime):
        raise TrainingRuntimeError("runtime must be a TrainingRuntime instance")
