"""Contracts for the standard MobileNetV2 block-6 expansion FOMO topology."""

from __future__ import annotations

from pathlib import Path
from dataclasses import asdict
import hashlib

import numpy as np
import pytest
import torch
from torch import nn

from fomo_servo.config import ConfigurationError, load_config
from fomo_servo.models import (
    BLOCK_6_EXPAND_RELU,
    STANDARD_MOBILENET_V2_BLOCK_SPECS,
    FOMONet,
    MobileNetV2FOMOBackbone,
    MobileNetV2FOMONet,
    ModelConfigurationError,
    build_fomo_model,
    count_trainable_parameters,
    describe_model,
)


def _write_config(
    path: Path,
    *,
    backbone: str = "mobilenet_v2_fomo",
    cut_point: str | None = "block_6_expand_relu",
    pretrained: bool = False,
) -> Path:
    model_extra = ""
    if cut_point is not None:
        model_extra += f"  cut_point: {cut_point}\n"
    model_extra += f"  pretrained: {str(pretrained).lower()}\n"
    path.write_text(
        f"""
dataset:
  root: data
  classes: [a, b, c, d, e, f, g]
  class_mode: preserve
model:
  backbone: {backbone}
  width_multiplier: 0.35
  head_channels: 32
  input_size: 192
  output_stride: 8
{model_extra}training:
  output_dir: outputs/test
""",
        encoding="utf-8",
    )
    return path


def test_standard_block_mapping_and_cut_point_identity() -> None:
    assert BLOCK_6_EXPAND_RELU == "block_6_expand_relu"
    assert [spec.block_index for spec in STANDARD_MOBILENET_V2_BLOCK_SPECS] == list(
        range(7)
    )
    assert [spec.expansion for spec in STANDARD_MOBILENET_V2_BLOCK_SPECS] == [
        1,
        6,
        6,
        6,
        6,
        6,
        6,
    ]
    assert [spec.base_output_channels for spec in STANDARD_MOBILENET_V2_BLOCK_SPECS] == [
        16,
        24,
        24,
        32,
        32,
        32,
        64,
    ]
    assert [spec.stride for spec in STANDARD_MOBILENET_V2_BLOCK_SPECS] == [
        1,
        2,
        1,
        2,
        1,
        1,
        2,
    ]


def test_cut_point_shape_and_depthwise_are_strictly_excluded() -> None:
    backbone = MobileNetV2FOMOBackbone(width_multiplier=0.35).eval()
    called_convolutions: list[nn.Conv2d] = []
    handles = [
        module.register_forward_hook(
            lambda current, _inputs, _output: called_convolutions.append(current)
        )
        for module in backbone.modules()
        if isinstance(module, nn.Conv2d)
    ]
    try:
        with torch.inference_mode():
            features = backbone(torch.zeros(1, 3, 192, 192))
    finally:
        for handle in handles:
            handle.remove()

    assert features.shape == (1, 96, 24, 24)
    assert backbone.cut_point_input_channels == 16
    assert backbone.output_channels == 96
    assert backbone.output_stride == 8
    assert backbone.block_6_expansion[0] in called_convolutions
    assert tuple(type(module) for module in backbone.block_6_expansion) == (
        nn.Conv2d,
        nn.BatchNorm2d,
        nn.ReLU6,
    )
    expansion = backbone.block_6_expansion[0]
    assert isinstance(expansion, nn.Conv2d)
    assert expansion.kernel_size == (1, 1)
    assert expansion.stride == (1, 1)
    assert expansion.bias is None
    assert not any(
        isinstance(module, nn.Conv2d)
        and module.kernel_size == (3, 3)
        and module.stride == (2, 2)
        and module.groups == 96
        for module in backbone.modules()
    ), "block 6 depthwise convolution must not be instantiated"
    assert not any(
        convolution in called_convolutions
        and convolution.kernel_size == (3, 3)
        and convolution.stride == (2, 2)
        and convolution.groups == 96
        for convolution in backbone.modules()
        if isinstance(convolution, nn.Conv2d)
    ), "block 6 stride-2 depthwise convolution must not execute"


def test_new_model_exact_shape_head_and_parameter_counts() -> None:
    model = MobileNetV2FOMONet(
        num_classes=7,
        input_size=192,
        width_multiplier=0.35,
        head_channels=32,
        output_stride=8,
        cut_point=BLOCK_6_EXPAND_RELU,
        pretrained=False,
    ).eval()
    with torch.inference_mode():
        logits = model(torch.zeros(1, 3, 192, 192))

    assert logits.shape == (1, 8, 24, 24)
    assert isinstance(model.head[0], nn.Conv2d)
    assert (model.head[0].in_channels, model.head[0].out_channels) == (96, 32)
    assert isinstance(model.head[2], nn.Conv2d)
    assert (model.head[2].in_channels, model.head[2].out_channels) == (32, 8)
    assert count_trainable_parameters(model.backbone) == 15_840
    assert count_trainable_parameters(model.head) == 3_368
    assert count_trainable_parameters(model) == 19_208


def test_ei_backbone_preserves_keras_batchnorm_and_head_activation_contract() -> None:
    """The FOMO cut model must match EI's unfused training-time BN contract."""

    model = MobileNetV2FOMONet(
        num_classes=7,
        input_size=192,
        width_multiplier=0.35,
        head_channels=32,
        output_stride=8,
        cut_point=BLOCK_6_EXPAND_RELU,
        pretrained=False,
    )

    batch_norms = [
        module for module in model.backbone.modules() if isinstance(module, nn.BatchNorm2d)
    ]
    assert batch_norms
    assert all(module.eps == pytest.approx(1e-3) for module in batch_norms)
    assert all(module.momentum == pytest.approx(0.999) for module in batch_norms)
    assert isinstance(model.head[1], nn.ReLU)


def test_ei_stride_two_same_padding_is_right_bottom_asymmetric() -> None:
    """Even-sized EI SAME convolutions pad one pixel on the right and bottom."""

    model = MobileNetV2FOMONet(num_classes=7, input_size=192, pretrained=False)
    stride_two_wrappers = [
        module
        for module in model.backbone.modules()
        if getattr(module, "stride", None) == (2, 2)
        and hasattr(module, "explicit_padding")
    ]

    assert len(stride_two_wrappers) == 3
    assert all(module.explicit_padding == (0, 1, 0, 1) for module in stride_two_wrappers)
    assert all(module.conv.padding == (0, 0) for module in stride_two_wrappers)


def test_factory_dispatch_metadata_and_legacy_defaults(tmp_path: Path) -> None:
    new_config = load_config(_write_config(tmp_path / "new.yaml"))
    new_model = build_fomo_model(new_config)
    assert isinstance(new_model, MobileNetV2FOMONet)
    assert describe_model(new_config, new_model) == {
        "backbone_name": "mobilenet_v2_fomo",
        "width_multiplier": 0.35,
        "cut_point": "block_6_expand_relu",
        "cut_point_input_channels": 16,
        "cut_point_output_channels": 96,
        "output_stride": 8,
        "head_channels": 32,
        "pretrained": False,
        "initialization": "pytorch_module_defaults",
        "backbone_parameter_count": 15_840,
        "head_parameter_count": 3_368,
        "parameter_count": 19_208,
    }

    old_config = load_config(
        _write_config(
            tmp_path / "old.yaml",
            backbone="mobilenet_v2_lite",
            cut_point=None,
        )
    )
    old_model = build_fomo_model(old_config)
    assert isinstance(old_model, FOMONet)
    assert old_config.model.cut_point == "lite_stride8_output"


def test_legacy_state_dict_round_trip_is_unchanged(tmp_path: Path) -> None:
    config = load_config(
        _write_config(
            tmp_path / "legacy.yaml",
            backbone="mobilenet_v2_lite",
            cut_point=None,
        )
    )
    original = build_fomo_model(config)
    original_keys = tuple(original.state_dict())
    key_fingerprint = hashlib.sha256("\n".join(original_keys).encode("utf-8")).hexdigest()
    assert key_fingerprint == "5cec26d9746e879ad68365f091c96701cb4011c715a61c193f6e8c3cc7a6270b"
    checkpoint = tmp_path / "legacy.pt"
    torch.save({"model_state": original.state_dict()}, checkpoint)
    restored = build_fomo_model(config)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    restored.load_state_dict(payload["model_state"])
    assert tuple(restored.state_dict()) == original_keys
    for key, tensor in original.state_dict().items():
        assert torch.equal(tensor, restored.state_dict()[key])


def test_pretrained_request_fails_without_attempting_download(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        torch.hub,
        "load_state_dict_from_url",
        lambda *_args, **_kwargs: pytest.fail("network weight loading was attempted"),
    )
    MobileNetV2FOMONet(num_classes=1, pretrained=False)
    with pytest.raises(ModelConfigurationError, match="pretrained"):
        MobileNetV2FOMONet(num_classes=1, pretrained=True)


def test_new_model_cpu_backward_is_finite() -> None:
    model = MobileNetV2FOMONet(num_classes=7, input_size=192)
    logits = model(torch.rand(1, 3, 192, 192))
    loss = torch.nn.functional.cross_entropy(
        logits, torch.zeros(1, 24, 24, dtype=torch.long)
    )
    loss.backward()
    assert torch.isfinite(logits).all()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


@pytest.mark.parametrize(
    ("input_size", "num_classes"), ((96, 1), (160, 7), (192, 7), (224, 1))
)
def test_supported_input_and_class_shapes(input_size: int, num_classes: int) -> None:
    model = MobileNetV2FOMONet(
        num_classes=num_classes, input_size=input_size
    ).eval()
    with torch.inference_mode():
        logits = model(torch.zeros(1, 3, input_size, input_size))
    assert logits.shape == (
        1,
        num_classes + 1,
        input_size // 8,
        input_size // 8,
    )


def test_new_model_state_dict_round_trip(tmp_path: Path) -> None:
    torch.manual_seed(7)
    original = MobileNetV2FOMONet(num_classes=7, input_size=192).eval()
    inputs = torch.rand(1, 3, 192, 192)
    checkpoint = tmp_path / "new.pt"
    torch.save(original.state_dict(), checkpoint)
    restored = MobileNetV2FOMONet(num_classes=7, input_size=192).eval()
    restored.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=True)
    )
    with torch.inference_mode():
        assert torch.equal(original(inputs), restored(inputs))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cpu_cuda_eval_outputs_and_backward_are_compatible() -> None:
    torch.manual_seed(9)
    cpu_model = MobileNetV2FOMONet(num_classes=7, input_size=192).eval()
    cuda_model = MobileNetV2FOMONet(num_classes=7, input_size=192).cuda().eval()
    cuda_model.load_state_dict(cpu_model.state_dict())
    inputs = torch.rand(1, 3, 192, 192)
    with torch.inference_mode():
        cpu_logits = cpu_model(inputs)
        cuda_logits = cuda_model(inputs.cuda()).cpu()
    assert cpu_logits.shape == cuda_logits.shape
    # cuDNN permits TF32 convolution by default; this tolerance covers its
    # measured accumulation error while still detecting material divergence.
    torch.testing.assert_close(cuda_logits, cpu_logits, rtol=1e-3, atol=3e-5)

    cuda_model.train()
    training_logits = cuda_model(inputs.cuda())
    training_logits.square().mean().backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in cuda_model.parameters()
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_mixed_precision_forward_smoke() -> None:
    model = MobileNetV2FOMONet(num_classes=7, input_size=192).cuda().eval()
    inputs = torch.rand(1, 3, 192, 192, device="cuda")
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
        logits = model(inputs)
    assert logits.shape == (1, 8, 24, 24)
    assert torch.isfinite(logits).all()


def test_new_model_onnx_export_smoke(tmp_path: Path) -> None:
    onnx = pytest.importorskip("onnx")
    model = MobileNetV2FOMONet(num_classes=7, input_size=192).eval()
    onnx_path = tmp_path / "mobilenet_v2_fomo.onnx"
    torch.onnx.export(
        model,
        torch.zeros(1, 3, 192, 192),
        onnx_path,
        input_names=["images"],
        output_names=["logits"],
        opset_version=17,
        dynamic_axes=None,
    )
    graph = onnx.load(str(onnx_path))
    onnx.checker.check_model(graph)
    assert all(node.domain in ("", "ai.onnx") for node in graph.graph.node)


def test_new_model_onnx_runtime_cpu_matches_pytorch(tmp_path: Path) -> None:
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    torch.manual_seed(11)
    model = MobileNetV2FOMONet(num_classes=7, input_size=192).eval()
    inputs = torch.rand(1, 3, 192, 192)
    onnx_path = tmp_path / "mobilenet_v2_fomo_runtime.onnx"
    torch.onnx.export(
        model,
        inputs,
        onnx_path,
        input_names=["images"],
        output_names=["logits"],
        opset_version=17,
        dynamic_axes=None,
    )
    session = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    with torch.inference_mode():
        expected = model(inputs).numpy()
    actual = session.run(["logits"], {"images": inputs.numpy()})[0]
    assert actual.shape == (1, 8, 24, 24)
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize(
    ("backbone", "cut_point", "pretrained", "message"),
    (
        ("mobilenet_v2_lite", "block_6_expand_relu", False, "cut_point"),
        ("mobilenet_v2_lite", None, True, "pretrained"),
        ("mobilenet_v2_fomo", "block_6_expand_relu", True, "pretrained"),
    ),
)
def test_config_rejects_contradictory_model_identity(
    tmp_path: Path,
    backbone: str,
    cut_point: str | None,
    pretrained: bool,
    message: str,
) -> None:
    with pytest.raises(ConfigurationError, match=message):
        load_config(
            _write_config(
                tmp_path / "invalid.yaml",
                backbone=backbone,
                cut_point=cut_point,
                pretrained=pretrained,
            )
        )


def test_formal_config_changes_only_model_identity_and_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path(__file__).resolve().parents[1]
    monkeypatch.setenv("FOMO_DATASET_ROOT", str(root / "tests" / "fixtures" / "yolo_micro"))
    baseline = load_config(
        root / "configs" / "experiments" / "aug03_underwater_conservative.yaml"
    )
    candidate = load_config(
        root
        / "configs"
        / "experiments"
        / "model01_mobilenet_v2_fomo_aug03.yaml"
    )

    baseline_values = asdict(baseline)
    candidate_values = asdict(candidate)
    for values in (baseline_values, candidate_values):
        values.pop("source_path")
        values.pop("model")
        values["experiment"].pop("name")
        values["training"].pop("output_dir")
    assert candidate_values == baseline_values

    assert candidate.model.input_size == baseline.model.input_size == 192
    assert candidate.model.output_stride == baseline.model.output_stride == 8
    assert candidate.model.width_multiplier == baseline.model.width_multiplier == 0.35
    assert candidate.model.head_channels == baseline.model.head_channels == 32
    assert candidate.model.backbone == "mobilenet_v2_fomo"
    assert candidate.model.cut_point == BLOCK_6_EXPAND_RELU
    assert candidate.model.pretrained is False
