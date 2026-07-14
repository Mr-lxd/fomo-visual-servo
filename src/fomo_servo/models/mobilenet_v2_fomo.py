"""Standard MobileNetV2 blocks 0-5 plus the block-6 expansion FOMO cut."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import torch
from torch import Tensor, nn

from .fomo import (
    OUTPUT_STRIDE,
    ModelConfigurationError,
    _make_divisible,
    _require_positive_float,
    _require_positive_integer,
)


BLOCK_6_EXPAND_RELU: Final[str] = "block_6_expand_relu"


@dataclass(frozen=True)
class MobileNetV2BlockSpec:
    """One standard MobileNetV2 block before width scaling."""

    block_index: int
    expansion: int
    base_output_channels: int
    stride: int


STANDARD_MOBILENET_V2_BLOCK_SPECS: Final[tuple[MobileNetV2BlockSpec, ...]] = (
    MobileNetV2BlockSpec(0, 1, 16, 1),
    MobileNetV2BlockSpec(1, 6, 24, 2),
    MobileNetV2BlockSpec(2, 6, 24, 1),
    MobileNetV2BlockSpec(3, 6, 32, 2),
    MobileNetV2BlockSpec(4, 6, 32, 1),
    MobileNetV2BlockSpec(5, 6, 32, 1),
    MobileNetV2BlockSpec(6, 6, 64, 2),
)


class _ConvBNReLU6(nn.Sequential):
    """ONNX-safe Conv2d-BatchNorm2d-ReLU6 for `[B,C,H,W]` tensors."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        stride: int,
        groups: int = 1,
    ) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=(kernel_size - 1) // 2,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.999),
            nn.ReLU6(inplace=False),
        )


class _StandardInvertedResidual(nn.Module):
    """Complete standard MobileNetV2 block mapping `[B,C,H,W]` features."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        expansion: int,
        stride: int,
    ) -> None:
        super().__init__()
        hidden_channels = in_channels * expansion
        layers: list[nn.Module] = []
        if expansion != 1:
            layers.append(
                _ConvBNReLU6(
                    in_channels,
                    hidden_channels,
                    kernel_size=1,
                    stride=1,
                )
            )
        layers.extend(
            (
                _ConvBNReLU6(
                    hidden_channels,
                    hidden_channels,
                    kernel_size=3,
                    stride=stride,
                    groups=hidden_channels,
                ),
                nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.999),
            )
        )
        self.block = nn.Sequential(*layers)
        self.use_residual = stride == 1 and in_channels == out_channels

    def forward(self, features: Tensor) -> Tensor:
        """Return transformed features on the input tensor's device and dtype."""

        output = self.block(features)
        return features + output if self.use_residual else output


class MobileNetV2FOMOBackbone(nn.Module):
    """Map RGB `[B,3,S,S]` to block-6 expansion `[B,96,S/8,S/8]` at alpha 0.35."""

    def __init__(self, width_multiplier: float = 0.35) -> None:
        super().__init__()
        self.width_multiplier = _require_positive_float(
            "width_multiplier", width_multiplier
        )
        stem_channels = _make_divisible(32 * self.width_multiplier)
        self.stem = _ConvBNReLU6(
            3, stem_channels, kernel_size=3, stride=2
        )

        in_channels = stem_channels
        blocks: list[nn.Module] = []
        for spec in STANDARD_MOBILENET_V2_BLOCK_SPECS[:6]:
            out_channels = _make_divisible(
                spec.base_output_channels * self.width_multiplier
            )
            blocks.append(
                _StandardInvertedResidual(
                    in_channels,
                    out_channels,
                    expansion=spec.expansion,
                    stride=spec.stride,
                )
            )
            in_channels = out_channels
        self.blocks_0_to_5 = nn.Sequential(*blocks)

        block_6 = STANDARD_MOBILENET_V2_BLOCK_SPECS[6]
        self.cut_point_input_channels = in_channels
        self.output_channels = in_channels * block_6.expansion
        self.output_stride = OUTPUT_STRIDE
        self.cut_point = BLOCK_6_EXPAND_RELU
        self.block_6_expansion = nn.Sequential(
            nn.Conv2d(
                self.cut_point_input_channels,
                self.output_channels,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
            nn.BatchNorm2d(self.output_channels, eps=1e-3, momentum=0.999),
            nn.ReLU6(inplace=False),
        )

    def forward(self, images: Tensor) -> Tensor:
        """Return cut-point features `[B,F,S/8,S/8]`; block-6 depthwise is absent."""

        return self.block_6_expansion(self.blocks_0_to_5(self.stem(images)))


class MobileNetV2FOMONet(nn.Module):
    """Fixed-size FOMO model returning raw logits `[B,1+N,S/8,S/8]`."""

    initialization: Final[str] = "pytorch_module_defaults"

    def __init__(
        self,
        *,
        num_classes: int,
        input_size: int = 192,
        width_multiplier: float = 0.35,
        head_channels: int = 32,
        output_stride: int = OUTPUT_STRIDE,
        cut_point: str = BLOCK_6_EXPAND_RELU,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        self.num_classes = _require_positive_integer("num_classes", num_classes)
        self.input_size = _require_positive_integer("input_size", input_size)
        self.width_multiplier = _require_positive_float(
            "width_multiplier", width_multiplier
        )
        self.head_channels = _require_positive_integer("head_channels", head_channels)
        if output_stride != OUTPUT_STRIDE:
            raise ModelConfigurationError(
                f"output_stride must be {OUTPUT_STRIDE} for MobileNetV2FOMONet"
            )
        if self.input_size % OUTPUT_STRIDE != 0:
            raise ModelConfigurationError(
                f"input_size must be divisible by output stride {OUTPUT_STRIDE}"
            )
        if cut_point != BLOCK_6_EXPAND_RELU:
            raise ModelConfigurationError(
                f"cut_point must be '{BLOCK_6_EXPAND_RELU}'"
            )
        if not isinstance(pretrained, bool):
            raise ModelConfigurationError("pretrained must be a boolean")
        if pretrained:
            raise ModelConfigurationError(
                "pretrained=true is unsupported: no pretrained source or download path is configured"
            )
        self.output_stride = OUTPUT_STRIDE
        self.cut_point = cut_point
        self.pretrained = pretrained
        self.backbone = MobileNetV2FOMOBackbone(self.width_multiplier)
        self.head = nn.Sequential(
            nn.Conv2d(self.backbone.output_channels, self.head_channels, kernel_size=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(self.head_channels, self.num_classes + 1, kernel_size=1),
        )

    def forward(self, images: Tensor) -> Tensor:
        """Return raw logits for float32 RGB images `[B,3,S,S]`."""

        self._validate_images(images)
        logits = self.head(self.backbone(images))
        expected_size = self.input_size // self.output_stride
        if logits.shape[-2:] != (expected_size, expected_size):
            raise RuntimeError(
                "MobileNetV2FOMONet violated its stride-8 contract: expected "
                f"{expected_size}x{expected_size}, got {tuple(logits.shape[-2:])}"
            )
        return logits

    def _validate_images(self, images: Tensor) -> None:
        if not isinstance(images, Tensor):
            raise ValueError("images must be a torch.Tensor")
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images must have shape [B,3,S,S] with three RGB channels")
        if images.shape[-2:] != (self.input_size, self.input_size):
            raise ValueError(
                f"images must have square {self.input_size}x{self.input_size} geometry"
            )
        if images.dtype != torch.float32:
            raise ValueError("images must have dtype torch.float32")
