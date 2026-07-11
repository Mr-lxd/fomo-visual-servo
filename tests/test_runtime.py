"""Tests for runtime device policy kept outside model forward code."""

from __future__ import annotations

import importlib
from typing import Any, Callable

import pytest
import torch


def _runtime_api() -> tuple[type[Exception] | None, Callable[[Any], torch.device] | None]:
    """Return optional public APIs so the missing implementation fails clearly."""

    try:
        module = importlib.import_module("fomo_servo.runtime")
    except ModuleNotFoundError:
        return None, None
    error_class = getattr(module, "RuntimeDeviceError", None)
    resolver = getattr(module, "resolve_device", None)
    return error_class, resolver


def test_resolve_device_auto_prefers_cuda_and_falls_back_to_cpu() -> None:
    """`auto` follows the training policy without a model-internal device choice."""

    _, resolver = _runtime_api()
    assert callable(resolver), "fomo_servo.runtime.resolve_device must be available"

    device = resolver("auto")

    expected_type = "cuda" if torch.cuda.is_available() else "cpu"
    assert device.type == expected_type


def test_resolve_device_honors_explicit_cpu_request() -> None:
    """An explicit CPU configuration must remain CPU even when CUDA is present."""

    _, resolver = _runtime_api()
    assert callable(resolver), "fomo_servo.runtime.resolve_device must be available"

    assert resolver(torch.device("cpu")) == torch.device("cpu")


def test_resolve_device_rejects_invalid_or_unavailable_cuda_requests() -> None:
    """Invalid explicit requests must raise instead of silently changing device."""

    error_class, resolver = _runtime_api()
    assert isinstance(error_class, type), "RuntimeDeviceError must be available"
    assert callable(resolver), "fomo_servo.runtime.resolve_device must be available"

    with pytest.raises(error_class, match="Unsupported device"):
        resolver("accelerator")

    if torch.cuda.is_available():
        assert resolver("cuda").type == "cuda"
    else:
        with pytest.raises(error_class, match="CUDA was requested"):
            resolver("cuda")
