"""Contracts for Stage E torchvision-backed stride-8 FOMO candidates."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch
from torch import nn
import torchvision
from torchvision.models import mobilenet_v3_small, squeezenet1_1

from fomo_servo.config import load_config
from fomo_servo.models import (
    MobileNetV3SmallFOMONet,
    ModelConfigurationError,
    SqueezeNet1_1FOMONet,
    build_fomo_model,
)


MOBILENET_V3_SMALL_URL = (
    "https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth"
)
SQUEEZENET1_1_URL = "https://download.pytorch.org/models/squeezenet1_1-b8a52dc0.pth"


def _sha256(path: Path) -> str:
    """Return the whole-file SHA-256 used by strict local pretrained loading."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_config(tmp_path: Path, *, backbone: str) -> Path:
    """Write a minimal seven-class fixed-192 random-init FOMO config."""

    path = tmp_path / (backbone + ".yaml")
    path.write_text(
        """
dataset:
  root: data/aquarium_pretrain
  classes: [fish, jellyfish, penguin, puffin, shark, starfish, stingray]
model:
  backbone: {backbone}
  input_size: 192
  output_stride: 8
  head_channels: 32
  pretrained: false
training:
  device: cpu
""".format(backbone=backbone).lstrip(),
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize(
    ("backbone", "cut_point"),
    [
        ("mobilenet_v3_small_fomo", "features.2"),
        ("squeezenet1_1_fomo", "features.6.fire4"),
    ],
)
def test_alternative_backbone_config_and_registry_return_stride_eight_logits(
    tmp_path: Path, backbone: str, cut_point: str
) -> None:
    """Each registered Stage E model maps RGB `[2,3,192,192]` to `[2,8,24,24]`."""

    config = load_config(_write_config(tmp_path, backbone=backbone))

    assert config.model.backbone == backbone
    assert config.model.cut_point == cut_point
    model = build_fomo_model(config).eval()

    with torch.no_grad():
        logits = model(torch.randn(2, 3, 192, 192, dtype=torch.float32))

    assert logits.shape == (2, 8, 24, 24)
    assert logits.dtype == torch.float32
    assert model.backbone.output_stride == 8


def test_unknown_backbone_has_actionable_factory_error(tmp_path: Path) -> None:
    """Unknown model identities fail explicitly instead of silently falling back."""

    config = load_config(_write_config(tmp_path, backbone="not_a_backbone"))

    with pytest.raises(ModelConfigurationError, match="model.backbone"):
        build_fomo_model(config)


def test_squeezenet_uses_explicit_input_padding_not_feature_interpolation(
    tmp_path: Path,
) -> None:
    """SqueezeNet 1.1 keeps its Fire4 stride-8 stage and pads only the input grid."""

    config = load_config(_write_config(tmp_path, backbone="squeezenet1_1_fomo"))
    model = build_fomo_model(config)

    assert model.backbone.input_padding == (0, 1, 0, 1)
    assert not any(isinstance(module, nn.Upsample) for module in model.modules())


def test_mobilenet_v3_pretrained_load_is_hash_checked_and_records_provenance(
    tmp_path: Path,
) -> None:
    """A full local torchvision state dict strictly initializes the selected feature prefix."""

    source = tmp_path / "mobilenet_v3_small.pth"
    state_dict = mobilenet_v3_small(weights=None).state_dict()
    torch.save(state_dict, source)

    model = MobileNetV3SmallFOMONet(
        num_classes=7,
        input_size=192,
        pretrained=True,
        pretrained_source=source,
        pretrained_sha256=_sha256(source),
        pretrained_torchvision_version=torchvision.__version__,
        pretrained_weights_enum="MobileNet_V3_Small_Weights.IMAGENET1K_V1",
        pretrained_url=MOBILENET_V3_SMALL_URL,
    )

    report = model.pretrained_load_report
    assert report.loaded_tensor_count == len(model.backbone.state_dict())
    assert report.missing_keys == ()
    assert report.unexpected_keys == ()
    assert report.source_sha256 == _sha256(source)
    assert model.initialization == "torchvision_imagenet_pretrained"
    assert torch.equal(
        model.backbone.state_dict()["features.0.0.weight"],
        state_dict["features.0.0.weight"],
    )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        ("hash", "SHA-256 mismatch"),
        ("missing", "missing required tensors"),
    ],
)
def test_mobilenet_v3_pretrained_rejects_unverifiable_local_weights(
    tmp_path: Path, mutate: str, message: str
) -> None:
    """A wrong digest or incomplete full torchvision checkpoint never partially loads."""

    source = tmp_path / "invalid_mobilenet_v3_small.pth"
    state_dict = mobilenet_v3_small(weights=None).state_dict()
    if mutate == "missing":
        state_dict.pop("classifier.0.weight")
    torch.save(state_dict, source)
    expected_sha256 = "0" * 64 if mutate == "hash" else _sha256(source)

    with pytest.raises(ModelConfigurationError, match=message):
        MobileNetV3SmallFOMONet(
            num_classes=7,
            input_size=192,
            pretrained=True,
            pretrained_source=source,
            pretrained_sha256=expected_sha256,
            pretrained_torchvision_version=torchvision.__version__,
            pretrained_weights_enum="MobileNet_V3_Small_Weights.IMAGENET1K_V1",
            pretrained_url=MOBILENET_V3_SMALL_URL,
        )


def test_squeezenet_pretrained_strictly_loads_the_fire_four_prefix(tmp_path: Path) -> None:
    """SqueezeNet1.1 validates the whole official state dict before selecting Fire4."""

    source = tmp_path / "squeezenet1_1.pth"
    state_dict = squeezenet1_1(weights=None).state_dict()
    torch.save(state_dict, source)

    model = SqueezeNet1_1FOMONet(
        num_classes=7,
        input_size=192,
        pretrained=True,
        pretrained_source=source,
        pretrained_sha256=_sha256(source),
        pretrained_torchvision_version=torchvision.__version__,
        pretrained_weights_enum="SqueezeNet1_1_Weights.IMAGENET1K_V1",
        pretrained_url=SQUEEZENET1_1_URL,
    )

    assert model.pretrained_load_report.loaded_tensor_count == len(
        model.backbone.state_dict()
    )
    assert torch.equal(
        model.backbone.state_dict()["features.0.weight"], state_dict["features.0.weight"]
    )


def test_alternative_pretrained_config_requires_torchvision_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stage E YAML persists local source, digest, exact package version, enum and URL."""

    source = tmp_path / "official.pth"
    source.write_bytes(b"fixture")
    monkeypatch.setenv("FOMO_MOBILENET_V3_SMALL_WEIGHTS", str(source))
    path = tmp_path / "pretrained.yaml"
    path.write_text(
        """
dataset:
  root: data/aquarium_pretrain
  classes: [fish, jellyfish, penguin, puffin, shark, starfish, stingray]
model:
  backbone: mobilenet_v3_small_fomo
  input_size: 192
  output_stride: 8
  pretrained: true
  pretrained_format: torchvision_state_dict
  pretrained_source: ${{FOMO_MOBILENET_V3_SMALL_WEIGHTS}}
  pretrained_sha256: {sha256}
  pretrained_torchvision_version: {version}
  pretrained_weights_enum: MobileNet_V3_Small_Weights.IMAGENET1K_V1
  pretrained_url: {url}
""".format(sha256=_sha256(source), version=torchvision.__version__, url=MOBILENET_V3_SMALL_URL).lstrip(),
        encoding="utf-8",
    )

    config = load_config(path)

    assert config.model.pretrained_torchvision_version == torchvision.__version__
    assert config.model.pretrained_format == "torchvision_state_dict"
    assert config.model.pretrained_weights_enum == "MobileNet_V3_Small_Weights.IMAGENET1K_V1"
    assert config.model.pretrained_url == MOBILENET_V3_SMALL_URL
