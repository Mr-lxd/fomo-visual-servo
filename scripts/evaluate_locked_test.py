"""Run one locked Stage B test evaluation with threshold sweep disabled."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

from fomo_servo.config import ConfigurationError, load_config
from fomo_servo.evaluation import StageBProtocolError, run_locked_test


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate exactly one checkpoint with the validation-locked threshold."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--threshold-sweep",
        action="store_true",
        help="Rejected deliberately; final test never sweeps thresholds.",
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        help="Rejected deliberately; the manifest is the only checkpoint source.",
    )
    return parser


def main(arguments: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(arguments)
    try:
        config = load_config(args.config)
        payload = run_locked_test(
            config,
            args.manifest,
            args.device,
            args.output_dir,
            threshold_sweep_requested=args.threshold_sweep,
            checkpoint_overrides=tuple(args.checkpoint),
        )
        print(json.dumps(payload, ensure_ascii=False))
    except (ConfigurationError, StageBProtocolError, RuntimeError, OSError, ValueError) as error:
        print("Error: {}".format(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
