"""Tests for the fixed-size stride-8 MobileNetV2-lite FOMO model."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest
import torch

from fomo_servo.config import load_config


def _model_api() -> tuple[Callable[..., Any] | None, Callable[[Any], int] | None]:
    """Return optional public APIs so missing code produces assertion failures."""

    module = importlib.import_module("fomo_servo.models")
    model_class = getattr(module, "FOMONet", None)
    counter = getattr(module, "count_trainable_parameters", None)
    return model_class, counter


def _make_model(
    *, input_size: int = 160, num_classes: int = 1
) -> torch.nn.Module:
    """Construct the model through its intended public API."""

    model_class, _ = _model_api()
    assert callable(model_class), "fomo_servo.models.FOMONet must be available"
    return model_class(
        num_classes=num_classes,
        input_size=input_size,
        width_multiplier=0.35,
        head_channels=32,
    )


@pytest.mark.parametrize(
    ("input_size", "num_classes"),
    [
        (96, 1),
        (192, 1),
        (224, 7),
    ],
)
def test_fomo_model_returns_raw_stride_eight_logits(
    input_size: int, num_classes: int
) -> None:
    """RGB float32 [B,3,S,S] must map to float32 [B,1+N,S/8,S/8]."""

    model = _make_model(input_size=input_size, num_classes=num_classes).eval()
    images = torch.randn(2, 3, input_size, input_size, dtype=torch.float32)

    with torch.no_grad():
        logits = model(images)

    assert logits.shape == (2, num_classes + 1, input_size // 8, input_size // 8)
    assert logits.dtype == torch.float32
    assert logits.device == images.device
    assert not torch.allclose(logits.softmax(dim=1), logits)


def test_fomo_model_rejects_non_square_or_wrong_sized_images() -> None:
    """The model must report invalid fixed-input geometry before convolutions run."""

    model = _make_model(input_size=96)

    with pytest.raises(ValueError, match=r"square 96x96"):
        model(torch.randn(1, 3, 96, 192))
    with pytest.raises(ValueError, match=r"square 96x96"):
        model(torch.randn(1, 3, 192, 192))


def test_fomo_model_cpu_backward_produces_parameter_gradients() -> None:
    """CPU forward/backward must work without CUDA-specific operators."""

    model = _make_model(input_size=96, num_classes=1).train().to("cpu")
    images = torch.randn(1, 3, 96, 96, device="cpu", requires_grad=True)

    loss = model(images).square().mean()
    loss.backward()

    assert images.grad is not None
    assert any(parameter.grad is not None for parameter in model.parameters())
    assert all(
        parameter.device.type == "cpu" for parameter in model.parameters()
    )


def test_parameter_count_is_trainable_and_lightweight() -> None:
    """Parameter accounting must be explicit and suitable for the Pi-targeted baseline."""

    model = _make_model(input_size=160, num_classes=1)
    _, counter = _model_api()
    assert callable(counter), (
        "fomo_servo.models.count_trainable_parameters must be available"
    )

    parameter_count = counter(model)

    assert parameter_count == sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    assert 0 < parameter_count < 1_000_000


def test_build_fomo_model_uses_yaml_class_and_model_fields(tmp_path: Path) -> None:
    """The public factory must map YAML classes and model fields to the logits contract."""

    config_path = tmp_path / "multiclass.yaml"
    config_path.write_text(
        """
dataset:
  root: data/aquarium_creature
  classes: [fish, crab, shrimp, ray, eel, turtle, creature]
model:
  backbone: mobilenet_v2_lite
  width_multiplier: 0.35
  head_channels: 32
  input_size: 96
  output_stride: 8
training:
  device: auto
""".lstrip(),
        encoding="utf-8",
    )
    module = importlib.import_module("fomo_servo.models")
    builder = getattr(module, "build_fomo_model", None)
    assert callable(builder), "fomo_servo.models.build_fomo_model must be available"

    model = builder(load_config(config_path)).eval()
    with torch.no_grad():
        logits = model(torch.randn(1, 3, 96, 96))

    assert logits.shape == (1, 8, 12, 12)


CUDA_AVAILABLE = torch.cuda.is_available()


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is not available in this environment")
def test_fomo_model_cuda_forward_backward_and_shape_match_cpu() -> None:
    """CUDA tensors and CPU tensors must have the same output shape and valid gradients."""

    torch.manual_seed(7)
    cpu_model = _make_model(input_size=96, num_classes=7).eval().to("cpu")
    cuda_model = _make_model(input_size=96, num_classes=7).eval().to("cuda")
    cuda_model.load_state_dict(cpu_model.state_dict())
    cpu_images = torch.randn(1, 3, 96, 96, dtype=torch.float32)

    with torch.no_grad():
        cpu_logits = cpu_model(cpu_images)
        cuda_logits = cuda_model(cpu_images.to("cuda")).cpu()

    assert cuda_logits.shape == cpu_logits.shape
    torch.testing.assert_close(cuda_logits, cpu_logits, rtol=1e-4, atol=5e-5)

    train_model = _make_model(input_size=96, num_classes=1).train().to("cuda")
    cuda_images = torch.randn(1, 3, 96, 96, device="cuda", requires_grad=True)
    loss = train_model(cuda_images).square().mean()
    loss.backward()

    assert cuda_images.grad is not None
    assert any(parameter.grad is not None for parameter in train_model.parameters())
    assert all(
        parameter.device.type == "cuda" for parameter in train_model.parameters()
    )


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is not available in this environment")
def test_fomo_model_cuda_mixed_precision_forward_smoke() -> None:
    """CUDA autocast must run a forward pass with output [B,1+N,G,G]."""

    model = _make_model(input_size=96, num_classes=1).eval().to("cuda")
    images = torch.randn(1, 3, 96, 96, device="cuda", dtype=torch.float32)

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        logits = model(images)

    assert logits.shape == (1, 2, 12, 12)
    assert logits.device.type == "cuda"


def _export_fixed_onnx(tmp_path: Path) -> tuple[Path, torch.nn.Module, torch.Tensor]:
    """Export fixed float32 [1,3,160,160] input with raw float32 logits."""

    pytest.importorskip("onnx")
    torch.manual_seed(11)
    model = _make_model(input_size=160, num_classes=1).eval().to("cpu")
    images = torch.randn(1, 3, 160, 160, dtype=torch.float32)
    output_path = tmp_path / "fomo_160.onnx"

    torch.onnx.export(
        model,
        images,
        output_path,
        input_names=["images"],
        output_names=["logits"],
        opset_version=17,
        dynamic_axes=None,
    )
    return output_path, model, images


def test_fomo_model_fixed_input_onnx_export_smoke(tmp_path: Path) -> None:
    """The fixed model must export a valid ONNX graph without custom operators."""

    onnx = pytest.importorskip("onnx")
    output_path, _, _ = _export_fixed_onnx(tmp_path)

    exported_model = onnx.load(str(output_path))
    onnx.checker.check_model(exported_model)

    assert output_path.is_file()
    assert all(node.domain in ("", "ai.onnx") for node in exported_model.graph.node)


def test_onnxruntime_cpu_matches_pytorch_logits(tmp_path: Path) -> None:
    """ONNX Runtime CPU logits must numerically match PyTorch eval logits."""

    onnxruntime = pytest.importorskip("onnxruntime")
    output_path, model, images = _export_fixed_onnx(tmp_path)
    session = onnxruntime.InferenceSession(
        str(output_path), providers=["CPUExecutionProvider"]
    )

    with torch.no_grad():
        pytorch_logits = model(images).cpu().numpy()
    onnx_logits = session.run(
        ["logits"], {"images": images.cpu().numpy()}
    )[0]

    assert onnx_logits.shape == (1, 2, 20, 20)
    np.testing.assert_allclose(onnx_logits, pytorch_logits, rtol=1e-4, atol=1e-5)
