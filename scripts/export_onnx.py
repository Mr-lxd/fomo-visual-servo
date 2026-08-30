"""Export a validated formal FOMO epoch snapshot to fixed-shape ONNX."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

from fomo_servo.deployment.onnx_export import (
    OnnxExportError,
    export_checkpoint_to_onnx,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the explicit config/checkpoint/output command-line interface."""

    parser = argparse.ArgumentParser(
        description=(
            "Export a formal weights-only FOMO checkpoint and verify PyTorch/ONNX "
            "Runtime raw-logits parity on a deterministic fixed input."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser


def main(arguments: Optional[Sequence[str]] = None) -> int:
    """Run formal export and return zero only after checker and parity pass."""

    args = build_parser().parse_args(arguments)
    try:
        report = export_checkpoint_to_onnx(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            onnx_path=args.output,
            report_path=args.report,
        )
    except OnnxExportError as error:
        print("Error: {}".format(error), file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
