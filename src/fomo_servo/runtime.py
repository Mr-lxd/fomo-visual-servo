"""Runtime device selection shared by future train, evaluation, and export entry points."""

from __future__ import annotations

from typing import Union

import torch


DeviceRequest = Union[str, torch.device]


class RuntimeDeviceError(ValueError):
    """Raised when a requested CPU/CUDA execution device cannot be honored."""


def resolve_device(requested: DeviceRequest = "auto") -> torch.device:
    """Resolve the configured device without moving tensors or models.

    Args:
        requested: ``"auto"`` prefers CUDA when available and otherwise returns CPU.
            Explicit ``"cpu"`` remains CPU. Explicit CUDA requests must be available.

    Returns:
        A ``torch.device`` for callers to pass to ``model.to(device)`` and
        ``input.to(device)``.

    Raises:
        RuntimeDeviceError: If the request is invalid, unsupported, or an explicit CUDA
            device is unavailable.
    """

    if isinstance(requested, str) and requested.strip().lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        device = torch.device(requested)
    except (RuntimeError, TypeError) as error:
        raise RuntimeDeviceError(
            f"Unsupported device request {requested!r}; use 'auto', 'cpu', or 'cuda'"
        ) from error

    if device.type not in {"cpu", "cuda"}:
        raise RuntimeDeviceError(
            f"Unsupported device {device!s}; use 'auto', 'cpu', or 'cuda'"
        )
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeDeviceError(
                "CUDA was requested, but torch.cuda.is_available() is false"
            )
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeDeviceError(
                f"CUDA device index {device.index} is unavailable; "
                f"found {torch.cuda.device_count()} CUDA device(s)"
            )
    return device
