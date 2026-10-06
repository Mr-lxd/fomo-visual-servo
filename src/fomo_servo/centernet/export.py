"""Fixed-shape ONNX export for CenterNet-lite."""

from __future__ import annotations

from pathlib import Path

import torch

from .model import CenterNetLiteNet


def export_centernet_onnx(model: CenterNetLiteNet, path: Path | str, *, opset: int = 17) -> Path:
    """Export ``[1,3,S,S] -> [1,C+4,S/8,S/8]`` (activations included)."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    model = model.eval().cpu()
    dummy = torch.zeros(1, 3, model.input_size, model.input_size)
    torch.onnx.export(
        model,
        dummy,
        str(destination),
        input_names=["images"],
        output_names=["centernet_lite"],
        opset_version=opset,
        dynamo=False,
    )
    return destination
