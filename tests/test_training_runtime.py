"""Tests for device-aware training runtime helpers without a training loop."""

from __future__ import annotations

import importlib
from typing import Any, Callable

import pytest
import torch

from fomo_servo.models import FOMONet


def _training_config_type() -> type[Any] | None:
    """Return TrainingConfig only when all requested YAML fields exist."""

    module = importlib.import_module("fomo_servo.config")
    config_type = getattr(module, "TrainingConfig", None)
    fields = getattr(config_type, "__dataclass_fields__", {})
    expected_fields = {"device", "amp", "num_workers", "pin_memory"}
    if not expected_fields.issubset(fields):
        return None
    return config_type


def _training_api() -> tuple[
    Callable[..., Any] | None,
    Callable[[torch.nn.Module, Any], torch.nn.Module] | None,
    Callable[[torch.Tensor, torch.Tensor, Any], tuple[torch.Tensor, torch.Tensor]] | None,
    Callable[[Any], Any] | None,
]:
    """Return optional public APIs so missing behavior fails through assertions."""

    try:
        module = importlib.import_module("fomo_servo.training")
    except ModuleNotFoundError:
        return None, None, None, None
    return (
        getattr(module, "create_training_runtime", None),
        getattr(module, "prepare_model", None),
        getattr(module, "move_training_batch", None),
        getattr(module, "autocast_context", None),
    )


def _make_training_config(**overrides: Any) -> Any:
    """Create a config through the public dataclass with explicit defaults."""

    config_type = _training_config_type()
    assert config_type is not None, (
        "TrainingConfig must expose device, amp, num_workers, and pin_memory"
    )
    values = {
        "device": "cpu",
        "amp": True,
        "num_workers": 4,
        "pin_memory": True,
    }
    values.update(overrides)
    return config_type(**values)


def test_cpu_runtime_moves_model_and_fomo_batch_without_cuda() -> None:
    """CPU runtime must apply the documented model/images/targets device movement."""

    factory, prepare_model, move_batch, autocast_context = _training_api()
    assert callable(factory), "fomo_servo.training.create_training_runtime must exist"
    assert callable(prepare_model), "fomo_servo.training.prepare_model must exist"
    assert callable(move_batch), "fomo_servo.training.move_training_batch must exist"
    assert callable(autocast_context), "fomo_servo.training.autocast_context must exist"

    runtime = factory(_make_training_config(device="cpu"))
    model = prepare_model(FOMONet(num_classes=1, input_size=96), runtime)
    source_images = torch.randn(2, 3, 96, 96, dtype=torch.float32)
    source_targets = torch.zeros(2, 12, 12, dtype=torch.int64)
    images, targets = move_batch(source_images, source_targets, runtime)

    with autocast_context(runtime):
        logits = model(images)

    assert runtime.device == torch.device("cpu")
    assert runtime.amp_enabled is False
    assert runtime.data_loader_kwargs == {"num_workers": 4, "pin_memory": False}
    assert any("AMP disabled" in message for message in runtime.diagnostics)
    assert any("pin_memory disabled" in message for message in runtime.diagnostics)
    assert images.device == targets.device == torch.device("cpu")
    assert logits.shape == (2, 2, 12, 12)


def test_device_override_has_priority_over_yaml_device() -> None:
    """The command-line request must override YAML without mutating the config."""

    factory, _, _, _ = _training_api()
    assert callable(factory), "fomo_servo.training.create_training_runtime must exist"

    runtime = factory(_make_training_config(device="auto"), device_override="cpu")

    assert runtime.device == torch.device("cpu")


CUDA_AVAILABLE = torch.cuda.is_available()


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is not available in this environment")
def test_cuda_runtime_enables_amp_and_moves_batch_to_cuda() -> None:
    """CUDA runtime must use AMP and non-blocking model/batch-compatible placement."""

    factory, prepare_model, move_batch, autocast_context = _training_api()
    assert callable(factory), "fomo_servo.training.create_training_runtime must exist"
    assert callable(prepare_model), "fomo_servo.training.prepare_model must exist"
    assert callable(move_batch), "fomo_servo.training.move_training_batch must exist"
    assert callable(autocast_context), "fomo_servo.training.autocast_context must exist"

    runtime = factory(_make_training_config(device="cuda"))
    model = prepare_model(FOMONet(num_classes=1, input_size=96), runtime).eval()
    images, targets = move_batch(
        torch.randn(1, 3, 96, 96, dtype=torch.float32),
        torch.zeros(1, 12, 12, dtype=torch.int64),
        runtime,
    )

    with torch.no_grad(), autocast_context(runtime):
        logits = model(images)

    assert runtime.device.type == "cuda"
    assert runtime.amp_enabled is True
    assert runtime.data_loader_kwargs == {"num_workers": 4, "pin_memory": True}
    assert images.device.type == targets.device.type == logits.device.type == "cuda"


def test_training_runtime_rejects_non_tensor_batch_values() -> None:
    """A training batch must fail visibly rather than silently skip device movement."""

    factory, _, move_batch, _ = _training_api()
    assert callable(factory), "fomo_servo.training.create_training_runtime must exist"
    assert callable(move_batch), "fomo_servo.training.move_training_batch must exist"

    runtime = factory(_make_training_config(device="cpu"))

    with pytest.raises(TypeError, match="images must be a torch.Tensor"):
        move_batch("not-a-tensor", torch.zeros(1, 12, 12, dtype=torch.int64), runtime)
