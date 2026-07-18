"""Torchvision MobileNetV3-Small and SqueezeNet1.1 FOMO logits models."""

from __future__ import annotations

from pathlib import Path
from typing import Final, Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.models import mobilenet_v3_small, squeezenet1_1

from .fomo import OUTPUT_STRIDE, ModelConfigurationError, _require_positive_integer


MOBILENET_V3_SMALL_CUT_POINT: Final[str] = "features.2"
SQUEEZENET1_1_CUT_POINT: Final[str] = "features.6.fire4"


class TorchvisionFeatureBackbone(nn.Module):
    """Map float RGB `[B,3,S,S]` to a fixed stride-8 feature map `[B,F,S/8,S/8]`."""

    def __init__(
        self,
        features: nn.Sequential,
        *,
        output_channels: int,
        cut_point: str,
        input_padding: tuple[int, int, int, int] = (0, 0, 0, 0),
    ) -> None:
        super().__init__()
        self.features = features
        self.output_channels = _require_positive_integer(
            "output_channels", output_channels
        )
        self.output_stride = OUTPUT_STRIDE
        self.cut_point = cut_point
        self.input_padding = input_padding

    def forward(self, images: Tensor) -> Tensor:
        """Return same-dtype stride-8 features; SqueezeNet pads only right/bottom input."""

        if any(self.input_padding):
            images = F.pad(images, self.input_padding)
        return self.features(images)


class _AlternativeFOMONet(nn.Module):
    """Common raw-logit FOMO wrapper for a torchvision stride-8 feature encoder."""

    initialization: Final[str] = "pytorch_module_defaults"

    def __init__(
        self,
        *,
        backbone: TorchvisionFeatureBackbone,
        num_classes: int,
        input_size: int,
        head_channels: int,
        output_stride: int,
        cut_point: str,
        pretrained: bool,
        backbone_name: str,
        pretrained_source: Optional[Path] = None,
        pretrained_sha256: Optional[str] = None,
        pretrained_torchvision_version: Optional[str] = None,
        pretrained_weights_enum: Optional[str] = None,
        pretrained_url: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.num_classes = _require_positive_integer("num_classes", num_classes)
        self.input_size = _require_positive_integer("input_size", input_size)
        self.head_channels = _require_positive_integer("head_channels", head_channels)
        if output_stride != OUTPUT_STRIDE:
            raise ModelConfigurationError(
                f"output_stride must be {OUTPUT_STRIDE} for alternative FOMO models"
            )
        if self.input_size % OUTPUT_STRIDE != 0:
            raise ModelConfigurationError(
                f"input_size must be divisible by output stride {OUTPUT_STRIDE}"
            )
        if cut_point != backbone.cut_point:
            raise ModelConfigurationError(
                f"cut_point must be '{backbone.cut_point}' for this backbone"
            )
        self.output_stride = OUTPUT_STRIDE
        self.cut_point = cut_point
        self.pretrained = pretrained
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Conv2d(self.backbone.output_channels, self.head_channels, kernel_size=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(self.head_channels, self.num_classes + 1, kernel_size=1),
        )
        self.pretrained_load_report: Optional[object] = None
        self.initialization = "pytorch_module_defaults"
        if pretrained:
            if None in (
                pretrained_source,
                pretrained_sha256,
                pretrained_torchvision_version,
                pretrained_weights_enum,
                pretrained_url,
            ):
                raise ModelConfigurationError(
                    "pretrained=true requires source, SHA-256, torchvision version, "
                    "weights enum, and URL"
                )
            from .torchvision_pretrained import (
                TorchvisionPretrainedWeightsError,
                load_torchvision_backbone_weights,
            )

            try:
                self.pretrained_load_report = load_torchvision_backbone_weights(
                    self.backbone,
                    backbone_name=backbone_name,
                    source=pretrained_source,
                    expected_sha256=pretrained_sha256,
                    expected_torchvision_version=pretrained_torchvision_version,
                    expected_weights_enum=pretrained_weights_enum,
                    expected_url=pretrained_url,
                )
            except TorchvisionPretrainedWeightsError as error:
                raise ModelConfigurationError(str(error)) from error
            self.initialization = "torchvision_imagenet_pretrained"

    def forward(self, images: Tensor) -> Tensor:
        """Return raw float logits `[B,1+N,S/8,S/8]` for RGB float32 `[B,3,S,S]`."""

        self._validate_images(images)
        logits = self.head(self.backbone(images))
        expected_shape = (self.input_size // self.output_stride,) * 2
        if tuple(logits.shape[-2:]) != expected_shape:
            raise RuntimeError(
                "alternative FOMO backbone violated the stride-8 contract: expected "
                f"{expected_shape}, got {tuple(logits.shape[-2:])}"
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


class MobileNetV3SmallFOMONet(_AlternativeFOMONet):
    """MobileNetV3-Small through torchvision `features.2`, then a FOMO logits head."""

    def __init__(
        self,
        *,
        num_classes: int,
        input_size: int = 192,
        head_channels: int = 32,
        output_stride: int = OUTPUT_STRIDE,
        cut_point: str = MOBILENET_V3_SMALL_CUT_POINT,
        pretrained: bool = False,
        pretrained_source: Optional[Path] = None,
        pretrained_sha256: Optional[str] = None,
        pretrained_torchvision_version: Optional[str] = None,
        pretrained_weights_enum: Optional[str] = None,
        pretrained_url: Optional[str] = None,
    ) -> None:
        source = mobilenet_v3_small(weights=None)
        super().__init__(
            backbone=TorchvisionFeatureBackbone(
                source.features[:3],
                output_channels=24,
                cut_point=MOBILENET_V3_SMALL_CUT_POINT,
            ),
            num_classes=num_classes,
            input_size=input_size,
            head_channels=head_channels,
            output_stride=output_stride,
            cut_point=cut_point,
            pretrained=pretrained,
            pretrained_source=pretrained_source,
            pretrained_sha256=pretrained_sha256,
            pretrained_torchvision_version=pretrained_torchvision_version,
            pretrained_weights_enum=pretrained_weights_enum,
            pretrained_url=pretrained_url,
            backbone_name="mobilenet_v3_small_fomo",
        )


class SqueezeNet1_1FOMONet(_AlternativeFOMONet):
    """SqueezeNet 1.1 through Fire4 with explicit right/bottom input padding."""

    def __init__(
        self,
        *,
        num_classes: int,
        input_size: int = 192,
        head_channels: int = 32,
        output_stride: int = OUTPUT_STRIDE,
        cut_point: str = SQUEEZENET1_1_CUT_POINT,
        pretrained: bool = False,
        pretrained_source: Optional[Path] = None,
        pretrained_sha256: Optional[str] = None,
        pretrained_torchvision_version: Optional[str] = None,
        pretrained_weights_enum: Optional[str] = None,
        pretrained_url: Optional[str] = None,
    ) -> None:
        source = squeezenet1_1(weights=None)
        super().__init__(
            backbone=TorchvisionFeatureBackbone(
                source.features[:7],
                output_channels=256,
                cut_point=SQUEEZENET1_1_CUT_POINT,
                input_padding=(0, 1, 0, 1),
            ),
            num_classes=num_classes,
            input_size=input_size,
            head_channels=head_channels,
            output_stride=output_stride,
            cut_point=cut_point,
            pretrained=pretrained,
            pretrained_source=pretrained_source,
            pretrained_sha256=pretrained_sha256,
            pretrained_torchvision_version=pretrained_torchvision_version,
            pretrained_weights_enum=pretrained_weights_enum,
            pretrained_url=pretrained_url,
            backbone_name="squeezenet1_1_fomo",
        )


__all__ = [
    "MOBILENET_V3_SMALL_CUT_POINT",
    "SQUEEZENET1_1_CUT_POINT",
    "MobileNetV3SmallFOMONet",
    "SqueezeNet1_1FOMONet",
    "TorchvisionFeatureBackbone",
]
