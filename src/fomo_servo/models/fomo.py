"""MobileNetV2-lite FOMO model with a fixed square stride-8 logits contract."""

from __future__ import annotations

from typing import Final

import torch
from torch import Tensor, nn

from fomo_servo.config import ProjectConfig


OUTPUT_STRIDE: Final[int] = 8


class ModelConfigurationError(ValueError):
    """Raised when FOMO model construction or input geometry is invalid."""


def _require_positive_integer(name: str, value: int) -> int:
    """Validate an integer model hyperparameter and return it unchanged."""

    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ModelConfigurationError(f"{name} must be a positive integer")
    return value


def _require_positive_float(name: str, value: float) -> float:
    """Validate a positive scalar model hyperparameter and return it as float."""

    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ModelConfigurationError(f"{name} must be a positive number")
    return float(value)


def _make_divisible(value: float, divisor: int = 8) -> int:
    """Round a positive channel count to an ONNX-friendly channel multiple."""

    rounded = max(divisor, int(value + divisor / 2) // divisor * divisor)
    if rounded < 0.9 * value:
        rounded += divisor
    return rounded


class ConvBNReLU6(nn.Sequential):
    """Conv2d → BatchNorm2d → ReLU6 for float tensors [B,C,H,W]."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        stride: int,
        groups: int = 1,
    ) -> None:
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=False),
        )


class InvertedResidual(nn.Module):
    """MobileNetV2 residual block preserving [B,C,H,W] tensor device and dtype."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int,
        expansion: int,
    ) -> None:
        super().__init__()
        if stride not in (1, 2):
            raise ModelConfigurationError("inverted residual stride must be 1 or 2")
        _require_positive_integer("expansion", expansion)

        hidden_channels = in_channels * expansion
        layers: list[nn.Module] = []
        if expansion != 1:
            layers.append(
                ConvBNReLU6(
                    in_channels,
                    hidden_channels,
                    kernel_size=1,
                    stride=1,
                )
            )
        layers.extend(
            [
                ConvBNReLU6(
                    hidden_channels,
                    hidden_channels,
                    kernel_size=3,
                    stride=stride,
                    groups=hidden_channels,
                ),
                nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            ]
        )
        self.block = nn.Sequential(*layers)
        self.use_residual = stride == 1 and in_channels == out_channels

    def forward(self, features: Tensor) -> Tensor:
        """Transform float [B,C,H,W] features into same-device output features."""

        output = self.block(features)
        return features + output if self.use_residual else output


class MobileNetV2LiteBackbone(nn.Module):
    """MobileNetV2-style encoder mapping [B,3,S,S] to [B,F,S/8,S/8]."""

    def __init__(self, width_multiplier: float) -> None:
        super().__init__()
        width_multiplier = _require_positive_float(
            "width_multiplier", width_multiplier
        )
        stem_channels = _make_divisible(32 * width_multiplier)
        layers: list[nn.Module] = [
            ConvBNReLU6(3, stem_channels, kernel_size=3, stride=2)
        ]
        in_channels = stem_channels
        # (expansion, base_channels, repeats, first_stride). Total stride: 2 * 2 * 2.
        for expansion, base_channels, repeats, first_stride in (
            (1, 16, 1, 1),
            (6, 24, 2, 2),
            (6, 32, 3, 2),
            (6, 64, 2, 1),
        ):
            out_channels = _make_divisible(base_channels * width_multiplier)
            for repeat_index in range(repeats):
                layers.append(
                    InvertedResidual(
                        in_channels,
                        out_channels,
                        stride=first_stride if repeat_index == 0 else 1,
                        expansion=expansion,
                    )
                )
                in_channels = out_channels

        self.features = nn.Sequential(*layers)
        self.output_channels = in_channels

    def forward(self, images: Tensor) -> Tensor:
        """Return float [B,F,S/8,S/8] features on ``images.device``."""

        return self.features(images)


class FOMONet(nn.Module):
    """Fixed-square FOMO classifier returning raw logits [B,1+N,S/8,S/8].

    Args:
        num_classes: Foreground class count ``N``; output channel zero is background.
        input_size: Fixed square RGB side length ``S``. It must be divisible by 8.
        width_multiplier: YAML-configurable MobileNetV2-lite channel multiplier.
        head_channels: YAML-configurable intermediate 1×1 FOMO-head width.

    Inputs:
        ``images`` is float32 RGB ``[B,3,S,S]`` in letterbox input coordinates.

    Returns:
        Raw float32 logits ``[B,1+N,S/8,S/8]``. No device selection or softmax occurs.
    """

    def __init__(
        self,
        *,
        num_classes: int,
        input_size: int = 160,
        width_multiplier: float = 0.35,
        head_channels: int = 32,
    ) -> None:
        super().__init__()
        self.num_classes = _require_positive_integer("num_classes", num_classes)
        self.input_size = _require_positive_integer("input_size", input_size)
        if self.input_size % OUTPUT_STRIDE != 0:
            raise ModelConfigurationError(
                f"input_size must be divisible by output stride {OUTPUT_STRIDE}"
            )
        self.width_multiplier = _require_positive_float(
            "width_multiplier", width_multiplier
        )
        self.head_channels = _require_positive_integer("head_channels", head_channels)
        self.output_stride = OUTPUT_STRIDE

        self.backbone = MobileNetV2LiteBackbone(self.width_multiplier)
        self.head = nn.Sequential(
            nn.Conv2d(self.backbone.output_channels, self.head_channels, kernel_size=1),
            nn.ReLU6(inplace=False),
            nn.Conv2d(self.head_channels, self.num_classes + 1, kernel_size=1),
        )

    def forward(self, images: Tensor) -> Tensor:
        """Return raw logits for fixed float32 RGB images [B,3,S,S]."""

        self._validate_images(images)
        logits = self.head(self.backbone(images))
        expected_grid_size = self.input_size // self.output_stride
        if logits.shape[-2:] != (expected_grid_size, expected_grid_size):
            raise RuntimeError(
                "FOMONet backbone violated the stride-8 output contract: expected "
                f"{expected_grid_size}x{expected_grid_size}, got "
                f"{logits.shape[-2]}x{logits.shape[-1]}"
            )
        return logits

    def _validate_images(self, images: Tensor) -> None:
        """Validate float32 fixed-square input without allocating a device-specific tensor."""

        if not isinstance(images, Tensor):
            raise ValueError("images must be a torch.Tensor")
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(
                "images must have shape [B,3,S,S] with three RGB channels"
            )
        if images.shape[-2:] != (self.input_size, self.input_size):
            raise ValueError(
                f"images must have square {self.input_size}x{self.input_size} geometry"
            )
        if images.dtype != torch.float32:
            raise ValueError("images must have dtype torch.float32")


def build_fomo_model(config: ProjectConfig) -> FOMONet:
    """Build a YAML-configured FOMO model with logits [B,1+N,S/8,S/8]."""

    if config.model.backbone != "mobilenet_v2_lite":
        raise ModelConfigurationError(
            "model.backbone must be 'mobilenet_v2_lite' for the first FOMO version"
        )
    if config.model.output_stride != OUTPUT_STRIDE:
        raise ModelConfigurationError(
            f"model.output_stride must be {OUTPUT_STRIDE} for FOMONet"
        )
    return FOMONet(
        num_classes=len(config.dataset.class_names),
        input_size=config.model.input_size,
        width_multiplier=config.model.width_multiplier,
        head_channels=config.model.head_channels,
    )


def count_trainable_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters in a model without allocating tensors."""

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
