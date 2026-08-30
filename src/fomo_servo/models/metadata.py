"""Stable model identity metadata for checkpoints and experiment artifacts."""

from __future__ import annotations

from typing import Any

from torch import nn

from fomo_servo.config import ProjectConfig

from .fomo import count_trainable_parameters


def describe_model(config: ProjectConfig, model: nn.Module) -> dict[str, Any]:
    """Describe topology and trainable parameters without changing model state."""

    backbone = getattr(model, "backbone", None)
    head = getattr(model, "head", None)
    if not isinstance(backbone, nn.Module) or not isinstance(head, nn.Module):
        raise ValueError("model must expose nn.Module backbone and head attributes")
    cut_input = getattr(backbone, "cut_point_input_channels", None)
    result: dict[str, Any] = {
        "backbone_name": config.model.backbone,
        "width_multiplier": config.model.width_multiplier,
        "cut_point": config.model.cut_point,
        "cut_point_output_channels": int(getattr(backbone, "output_channels")),
        "output_stride": config.model.output_stride,
        "head_channels": config.model.head_channels,
        "pretrained": config.model.pretrained,
        "initialization": getattr(model, "initialization", "pytorch_module_defaults"),
        "backbone_parameter_count": count_trainable_parameters(backbone),
        "head_parameter_count": count_trainable_parameters(head),
        "parameter_count": count_trainable_parameters(model),
    }
    if cut_input is not None:
        result["cut_point_input_channels"] = int(cut_input)
    load_report = getattr(model, "pretrained_load_report", None)
    if load_report is not None:
        result["pretrained_load_report"] = load_report.as_dict()
    return result
