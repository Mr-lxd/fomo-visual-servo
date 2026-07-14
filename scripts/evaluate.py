"""Evaluate a checkpoint on the configured validation split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

import torch

from fomo_servo.config import ConfigurationError, load_config
from fomo_servo.evaluation import evaluate_validation_dataset
from fomo_servo.inference import InferenceError, load_inference_model
from fomo_servo.models import ModelConfigurationError
from fomo_servo.runtime import RuntimeDeviceError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate FOMO grid and centroid metrics.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser


def main(arguments: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(arguments)
    try:
        config = load_config(args.config)
        request = config.training.device if args.device is None else args.device
        model, device = load_inference_model(config, args.checkpoint, request)
        report = evaluate_validation_dataset(config, model, device)
        class_weight_metadata = _read_checkpoint_class_weight_metadata(args.checkpoint)
    except (ConfigurationError, ModelConfigurationError, RuntimeDeviceError, InferenceError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    payload = report.as_dict(**class_weight_metadata)
    payload["checkpoint"] = str(args.checkpoint)
    payload["device"] = str(device)
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n", encoding="utf-8")
    return 0


def _read_checkpoint_class_weight_metadata(checkpoint_path: Path) -> dict[str, object]:
    """Read optional resolved training weights without changing checkpoint loading."""

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, ValueError) as error:
        raise InferenceError(
            "unable to read checkpoint metadata '{}': {}".format(checkpoint_path, error)
        ) from error
    if not isinstance(checkpoint, dict):
        return {}
    weights = checkpoint.get("class_weights")
    statistics = checkpoint.get("class_statistics")
    if weights is not None and (
        not isinstance(weights, list)
        or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in weights)
    ):
        raise InferenceError("checkpoint class_weights must be a numeric list")
    if statistics is not None and (
        not isinstance(statistics, list)
        or any(not isinstance(item, dict) for item in statistics)
    ):
        raise InferenceError("checkpoint class_statistics must be a list of mappings")
    mode = checkpoint.get("class_weight_mode")
    if mode is not None and mode not in {"manual", "auto", "disabled"}:
        raise InferenceError(
            "checkpoint class_weight_mode must be 'manual', 'auto', or 'disabled'"
        )
    return {
        "class_weights": weights,
        "class_weight_mode": mode,
        "class_statistics": statistics,
    }


if __name__ == "__main__":
    raise SystemExit(main())
