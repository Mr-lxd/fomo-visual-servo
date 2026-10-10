"""CenterNet-lite network: FOMO backbone, shared 1x1 head and three 1x1 branches."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn

from fomo_servo.models.fomo import OUTPUT_STRIDE
from fomo_servo.models.mobilenet_v2_fomo import MobileNetV2FOMOBackbone, MobileNetV2FOMONet


class CenterNetLiteNet(nn.Module):
    """Map RGB ``[B,3,S,S]`` to ``[B,C+4,S/8,S/8]``.

    Channels: ``C`` sigmoid centre heat-maps, 2 sigmoid sub-cell offsets
    (dx, dy in 0..1), 2 raw log sizes (log w, log h in grid cells).
    """

    def __init__(
        self,
        *,
        num_classes: int = 7,
        input_size: int = 192,
        width_multiplier: float = 0.35,
        head_channels: int = 32,
        heat_bias: float = -2.19,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        self.output_stride = OUTPUT_STRIDE
        self.backbone = MobileNetV2FOMOBackbone(width_multiplier)
        self.shared = nn.Sequential(
            nn.Conv2d(self.backbone.output_channels, head_channels, kernel_size=1),
            nn.ReLU(inplace=False),
        )
        self.heat_head = nn.Conv2d(head_channels, num_classes, kernel_size=1)
        self.offset_head = nn.Conv2d(head_channels, 2, kernel_size=1)
        self.size_head = nn.Conv2d(head_channels, 2, kernel_size=1)
        nn.init.constant_(self.heat_head.bias, heat_bias)

    def new_head_parameters(self) -> list[nn.Parameter]:
        """Parameters of the three branches that are not initialised from FOMO weights."""

        return [
            *self.heat_head.parameters(),
            *self.offset_head.parameters(),
            *self.size_head.parameters(),
        ]

    def forward_raw(self, images: Tensor) -> Tensor:
        """Return pre-activation ``[B,C+4,G,G]``."""

        features = self.shared(self.backbone(images))
        return torch.cat(
            [self.heat_head(features), self.offset_head(features), self.size_head(features)],
            dim=1,
        )

    def forward(self, images: Tensor) -> Tensor:
        raw = self.forward_raw(images)
        c = self.num_classes
        return torch.cat([torch.sigmoid(raw[:, : c + 2]), raw[:, c + 2 :]], dim=1)


def load_fomo_initialisation(model: CenterNetLiteNet, checkpoint: Path, sha256: str) -> dict:
    """Init backbone and shared 1x1 conv from a verified FOMO epoch snapshot."""

    from fomo_servo.training.engine import initialize_model_weights

    fomo = MobileNetV2FOMONet(
        num_classes=model.num_classes,
        input_size=model.input_size,
        width_multiplier=model.backbone.width_multiplier,
        head_channels=model.shared[0].out_channels,
    )
    report = initialize_model_weights(fomo, checkpoint, sha256)
    model.backbone.load_state_dict(fomo.backbone.state_dict())
    model.shared[0].load_state_dict(fomo.head[0].state_dict())
    return report
