"""YAML-driven FOMO training and validation entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from fomo_servo.config import ConfigurationError, load_config
from fomo_servo.models import ModelConfigurationError, build_fomo_model
from fomo_servo.runtime import RuntimeDeviceError
from fomo_servo.training import TrainingError, run_training


def build_parser() -> argparse.ArgumentParser:
    """Build the ``train.py`` CLI parser for config loading and device override."""

    parser = argparse.ArgumentParser(
        description="Train and validate a YAML-configured FOMO model."
    )
    parser.add_argument("--config", type=Path, required=True, help="Path to project YAML")
    parser.add_argument(
        "--device",
        default=None,
        help="Override YAML device: auto, cpu, cuda, or cuda:N",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Override YAML training.resume checkpoint path",
    )
    return parser


def main(arguments: Optional[Sequence[str]] = None) -> int:
    """Train/validate from YAML and report F1-selected checkpoint paths on completion."""

    parsed_arguments = build_parser().parse_args(arguments)
    try:
        config = load_config(parsed_arguments.config)
        summary = run_training(
            config,
            device_override=parsed_arguments.device,
            resume_override=parsed_arguments.resume,
        )
    except (
        ConfigurationError,
        ModelConfigurationError,
        RuntimeDeviceError,
        TrainingError,
    ) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print("Training complete")
    print(f"Config: {parsed_arguments.config}")
    print(f"Device: {summary.device}")
    print("AMP: {}".format("enabled" if summary.amp_enabled else "disabled"))
    print(f"Start epoch: {summary.start_epoch}")
    print(f"Completed epoch: {summary.completed_epochs}")
    print("Best validation F1: {:.6f}".format(summary.best_val_f1))
    print("Early stopped: {}".format(summary.stopped_early))
    print(f"Output directory: {summary.output_dir}")
    print(f"Last checkpoint: {summary.output_dir / 'last.pt'}")
    print(f"Best checkpoint: {summary.output_dir / 'best_val_f1.pt'}")
    print(f"History: {summary.output_dir / 'history.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
