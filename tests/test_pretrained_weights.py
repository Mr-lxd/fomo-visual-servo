from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fomo_servo.config import ConfigurationError, load_config
from fomo_servo.models import MobileNetV2FOMONet, ModelConfigurationError, describe_model


h5py = None


def _write_dataset(file: object, path: str, shape: tuple[int, ...], value: float) -> None:
    group_path, dataset_name = path.rsplit("/", 1)
    group = file.require_group(group_path)
    group.create_dataset(dataset_name, data=np.full(shape, value, dtype=np.float32))


def _write_minimal_ei_backbone_h5(path: Path, *, omit: str | None = None) -> None:
    global h5py
    if h5py is None:
        h5py = pytest.importorskip("h5py")
    model = MobileNetV2FOMONet(num_classes=7, input_size=192)
    backbone = model.backbone
    entries: list[tuple[str, tuple[int, ...], float]] = []

    def add(path_name: str, shape: tuple[int, ...], value: float) -> None:
        entries.append((path_name, shape, value))

    add("Conv1/Conv1/kernel:0", (3, 3, 3, 16), 0.11)
    add_bn("bn_Conv1", 16, add, 0.21)
    for block_index in range(6):
        block = backbone.blocks_0_to_5[block_index].block
        if block_index == 0:
            depthwise = block[0].conv
            project = block[1]
            add(
                f"mobl{block_index}_conv_{block_index}_depthwise/mobl{block_index}_conv_{block_index}_depthwise/depthwise_kernel:0",
                (3, 3, depthwise.in_channels, 1),
                0.31,
            )
            add_bn(
                f"bn{block_index}_conv_{block_index}_bn_depthwise",
                depthwise.out_channels,
                add,
                0.41,
            )
            add(
                f"mobl{block_index}_conv_{block_index}_project/mobl{block_index}_conv_{block_index}_project/kernel:0",
                (1, 1, project.in_channels, project.out_channels),
                0.51,
            )
            add_bn(
                f"bn{block_index}_conv_{block_index}_bn_project",
                project.out_channels,
                add,
                0.61,
            )
            continue
        expand = block[0].conv
        depthwise = block[1].conv
        project = block[2]
        add(
            f"mobl{block_index}_conv_{block_index}_expand/mobl{block_index}_conv_{block_index}_expand/kernel:0",
            (1, 1, expand.in_channels, expand.out_channels),
            0.71,
        )
        add_bn(
            f"bn{block_index}_conv_{block_index}_bn_expand",
            expand.out_channels,
            add,
            0.81,
        )
        add(
            f"mobl{block_index}_conv_{block_index}_depthwise/mobl{block_index}_conv_{block_index}_depthwise/depthwise_kernel:0",
            (3, 3, depthwise.in_channels, 1),
            0.91,
        )
        add_bn(
            f"bn{block_index}_conv_{block_index}_bn_depthwise",
            depthwise.out_channels,
            add,
            1.01,
        )
        add(
            f"mobl{block_index}_conv_{block_index}_project/mobl{block_index}_conv_{block_index}_project/kernel:0",
            (1, 1, project.in_channels, project.out_channels),
            1.11,
        )
        add_bn(
            f"bn{block_index}_conv_{block_index}_bn_project",
            project.out_channels,
            add,
            1.21,
        )

    block_6 = backbone.block_6_expansion
    add(
        "mobl6_conv_6_expand/mobl6_conv_6_expand/kernel:0",
        (1, 1, block_6[0].in_channels, block_6[0].out_channels),
        1.31,
    )
    add_bn("bn6_conv_6_bn_expand", block_6[1].num_features, add, 1.41)

    with h5py.File(path, "w") as file:
        for source_path, shape, value in entries:
            if source_path != omit:
                _write_dataset(file, source_path, shape, value)


def add_bn(
    prefix: str,
    channels: int,
    add: object,
    value: float,
) -> None:
    for name, offset in (
        ("gamma:0", 0.0),
        ("beta:0", 0.1),
        ("moving_mean:0", 0.2),
        ("moving_variance:0", 0.3),
    ):
        add(f"{prefix}/{prefix}/{name}", (channels,), value + offset)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_ei_h5_loader_transposes_backbone_and_keeps_head_new(tmp_path: Path) -> None:
    source = tmp_path / "weights.h5"
    _write_minimal_ei_backbone_h5(source)
    torch.manual_seed(7)
    random_model = MobileNetV2FOMONet(num_classes=7, input_size=192)
    head_before = {name: value.detach().clone() for name, value in random_model.head.state_dict().items()}

    torch.manual_seed(7)
    loaded_model = MobileNetV2FOMONet(
        num_classes=7,
        input_size=192,
        pretrained=True,
        pretrained_source=source,
        pretrained_sha256=_sha256(source),
    )

    assert loaded_model.pretrained_load_report.loaded_tensor_count == 95
    assert loaded_model.pretrained_load_report.missing_keys == ()
    assert loaded_model.pretrained_load_report.unexpected_keys == ()
    assert torch.allclose(loaded_model.backbone.stem[0].weight, torch.full_like(loaded_model.backbone.stem[0].weight, 0.11))
    assert torch.allclose(loaded_model.backbone.stem[1].weight, torch.full_like(loaded_model.backbone.stem[1].weight, 0.21))
    assert torch.allclose(loaded_model.backbone.stem[1].bias, torch.full_like(loaded_model.backbone.stem[1].bias, 0.31))
    assert torch.allclose(loaded_model.backbone.stem[1].running_mean, torch.full_like(loaded_model.backbone.stem[1].running_mean, 0.41))
    assert torch.allclose(loaded_model.backbone.stem[1].running_var, torch.full_like(loaded_model.backbone.stem[1].running_var, 0.51))
    assert loaded_model.initialization == "ei_keras_mobilenet_v2_035_96"
    for name, value in loaded_model.head.state_dict().items():
        assert torch.equal(value, random_model.head.state_dict()[name])

    metadata = describe_model(
        SimpleNamespace(
            model=SimpleNamespace(
                backbone="mobilenet_v2_fomo",
                width_multiplier=0.35,
                cut_point="block_6_expand_relu",
                output_stride=8,
                head_channels=32,
                pretrained=True,
            )
        ),
        loaded_model,
    )
    report = metadata["pretrained_load_report"]
    assert report["sha256"] == _sha256(source)
    assert report["loaded_tensor_count"] == 95
    assert report["skipped_tensor_count"] == 0
    assert report["missing_keys"] == []
    assert report["unexpected_keys"] == []
    assert "head" in report["initialization_policy"]


def test_ei_h5_loader_rejects_hash_mismatch_before_loading(tmp_path: Path) -> None:
    source = tmp_path / "weights.h5"
    _write_minimal_ei_backbone_h5(source)

    with pytest.raises(ModelConfigurationError, match="SHA-256 mismatch"):
        MobileNetV2FOMONet(
            num_classes=7,
            pretrained=True,
            pretrained_source=source,
            pretrained_sha256="0" * 64,
        )


def test_ei_h5_loader_rejects_missing_required_source_tensor(tmp_path: Path) -> None:
    source = tmp_path / "weights_missing.h5"
    missing = "mobl6_conv_6_expand/mobl6_conv_6_expand/kernel:0"
    _write_minimal_ei_backbone_h5(source, omit=missing)

    with pytest.raises(ModelConfigurationError, match="missing required tensors"):
        MobileNetV2FOMONet(
            num_classes=7,
            pretrained=True,
            pretrained_source=source,
            pretrained_sha256=_sha256(source),
        )


def test_pretrained_config_requires_source_and_sha256(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = tmp_path / "pretrained.yaml"
    config_path.write_text(
        """
dataset:
  root: data
  classes: [fish]
model:
  backbone: mobilenet_v2_fomo
  input_size: 192
  output_stride: 8
  pretrained: true
  pretrained_source: ${FOMO_PRETRAINED_WEIGHTS}
  pretrained_sha256: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
""".lstrip(),
        encoding="utf-8",
    )
    with pytest.raises(ConfigurationError, match="FOMO_PRETRAINED_WEIGHTS"):
        load_config(config_path)
    monkeypatch.setenv("FOMO_PRETRAINED_WEIGHTS", str(tmp_path / "weights.h5"))
    config = load_config(config_path)
    assert config.model.pretrained is True
    assert config.model.pretrained_source == tmp_path / "weights.h5"
    assert config.model.pretrained_sha256 == "a" * 64
