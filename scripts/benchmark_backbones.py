"""Export and benchmark fixed-shape FOMO backbones on CPU without dataset access."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from fomo_servo.config import ConfigurationError, load_config
from fomo_servo.models import ModelConfigurationError, build_fomo_model, describe_model


def _median_latency_ms(callable_: Any, *, warmup_iterations: int, measured_iterations: int) -> float:
    """Return median CPU latency in milliseconds after deterministic warm-up calls."""

    if warmup_iterations < 0 or measured_iterations <= 0:
        raise ValueError("warmup_iterations must be non-negative and measured_iterations positive")
    for _ in range(warmup_iterations):
        callable_()
    samples = []
    for _ in range(measured_iterations):
        started = time.perf_counter()
        callable_()
        samples.append((time.perf_counter() - started) * 1_000.0)
    return float(statistics.median(samples))


def benchmark_model(
    *,
    name: str,
    model: nn.Module,
    input_size: int,
    warmup_iterations: int,
    measured_iterations: int,
    onnx_path: Path,
    threads: int = 1,
) -> dict[str, object]:
    """Export one fixed CPU FOMO model and return its latency/parity metrics.

    Inputs are deterministic float32 RGB `[1,3,S,S]` tensors; models must return
    float logits `[1,C,S/8,S/8]`. This function neither loads a dataset nor touches
    any train/validation/test split.
    """

    if not name:
        raise ValueError("name must be non-empty")
    if input_size <= 0 or input_size % 8 != 0:
        raise ValueError("input_size must be a positive multiple of 8")
    if threads <= 0:
        raise ValueError("threads must be positive")
    try:
        import onnx
        import onnxruntime as ort
    except ImportError as error:
        raise RuntimeError(
            "benchmark_backbones requires the 'export' and 'deployment' dependencies"
        ) from error

    torch.set_num_threads(threads)
    model = model.to(device="cpu", dtype=torch.float32).eval()
    generator = torch.Generator(device="cpu").manual_seed(42)
    images = torch.randn((1, 3, input_size, input_size), generator=generator)
    with torch.inference_mode():
        pytorch_logits = model(images)
    expected_shape = (1, int(pytorch_logits.shape[1]), input_size // 8, input_size // 8)
    if tuple(pytorch_logits.shape) != expected_shape:
        raise RuntimeError(
            "model violates fixed stride-8 logits contract: expected {}, got {}".format(
                expected_shape, tuple(pytorch_logits.shape)
            )
        )

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        images,
        onnx_path,
        input_names=["images"],
        output_names=["logits"],
        opset_version=17,
        dynamic_axes=None,
        do_constant_folding=True,
    )
    onnx.checker.check_model(str(onnx_path))
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = threads
    session_options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(onnx_path), session_options, providers=["CPUExecutionProvider"]
    )
    ort_inputs = {"images": images.numpy()}
    ort_logits = session.run(["logits"], ort_inputs)[0]
    pytorch_array = pytorch_logits.detach().cpu().numpy()
    difference = np.abs(pytorch_array - ort_logits)
    if not np.allclose(pytorch_array, ort_logits, rtol=1e-4, atol=1e-5):
        raise RuntimeError(
            "ONNX Runtime logits differ from PyTorch beyond rtol=1e-4, atol=1e-5; "
            "max_abs_error={:.8g}".format(float(difference.max()))
        )

    with torch.inference_mode():
        pytorch_latency = _median_latency_ms(
            lambda: model(images),
            warmup_iterations=warmup_iterations,
            measured_iterations=measured_iterations,
        )
    ort_latency = _median_latency_ms(
        lambda: session.run(["logits"], ort_inputs),
        warmup_iterations=warmup_iterations,
        measured_iterations=measured_iterations,
    )
    return {
        "name": name,
        "input_shape": list(images.shape),
        "output_shape": list(pytorch_logits.shape),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "onnx_path": str(onnx_path),
        "onnx_size_bytes": onnx_path.stat().st_size,
        "onnx_opset": 17,
        "cpu_threads": threads,
        "warmup_iterations": warmup_iterations,
        "measured_iterations": measured_iterations,
        "pytorch_cpu_latency_ms_median": pytorch_latency,
        "onnxruntime_cpu_latency_ms_median": ort_latency,
        "onnx_max_absolute_error": float(difference.max()),
        "onnx_mean_absolute_error": float(difference.mean()),
        "onnx_rtol": 1e-4,
        "onnx_atol": 1e-5,
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for comparison runs across one or more YAML configs."""

    parser = argparse.ArgumentParser(
        description="Benchmark fixed-shape FOMO models on PyTorch CPU and ONNX Runtime CPU."
    )
    parser.add_argument("--config", type=Path, action="append", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--onnx-dir", type=Path, required=True)
    parser.add_argument("--warmup-iterations", type=int, default=20)
    parser.add_argument("--measured-iterations", type=int, default=100)
    parser.add_argument("--threads", type=int, default=1)
    return parser


def main(arguments: Optional[Sequence[str]] = None) -> int:
    """Run benchmark for YAML-configured models and write a JSON comparison report."""

    args = build_parser().parse_args(arguments)
    try:
        reports: list[dict[str, object]] = []
        for config_path in args.config:
            config = load_config(config_path)
            model = build_fomo_model(config)
            name = config.experiment.name or config_path.stem
            report = benchmark_model(
                name=name,
                model=model,
                input_size=config.model.input_size,
                warmup_iterations=args.warmup_iterations,
                measured_iterations=args.measured_iterations,
                onnx_path=args.onnx_dir / (config_path.stem + ".onnx"),
                threads=args.threads,
            )
            report["config"] = str(config_path)
            report["model"] = describe_model(config, model)
            reports.append(report)
        payload = {
            "protocol": {
                "device": "cpu",
                "batch_size": 1,
                "static_input": True,
                "dataset_access": "none",
            },
            "models": reports,
        }
    except (ConfigurationError, ModelConfigurationError, OSError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    args.output_json.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
