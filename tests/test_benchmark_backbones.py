"""Behavioral contract for the Stage E fixed-shape CPU benchmark utility."""

from __future__ import annotations

from pathlib import Path

import torch

from scripts.benchmark_backbones import benchmark_model


class _TinyStrideEightModel(torch.nn.Module):
    """Small logits model used only to exercise the benchmark artifact schema."""

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Map float32 `[B,3,192,192]` inputs to logits `[B,8,24,24]`."""

        return images[:, :1, ::8, ::8].repeat(1, 8, 1, 1)


def test_benchmark_model_writes_static_onnx_and_cpu_latency_metrics(tmp_path: Path) -> None:
    """The benchmark records parameter, ONNX and CPU timing facts for one fixed model."""

    report = benchmark_model(
        name="tiny",
        model=_TinyStrideEightModel().eval(),
        input_size=192,
        warmup_iterations=1,
        measured_iterations=2,
        onnx_path=tmp_path / "tiny.onnx",
    )

    assert report["name"] == "tiny"
    assert report["input_shape"] == [1, 3, 192, 192]
    assert report["output_shape"] == [1, 8, 24, 24]
    assert report["parameter_count"] == 0
    assert report["onnx_size_bytes"] > 0
    assert report["pytorch_cpu_latency_ms_median"] >= 0.0
    assert report["onnxruntime_cpu_latency_ms_median"] >= 0.0
    assert (tmp_path / "tiny.onnx").is_file()
