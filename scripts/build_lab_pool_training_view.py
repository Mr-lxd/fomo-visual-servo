"""Build the deterministic D2 train-only view for lab-pool adaptation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from fomo_servo.datasets.lab_pool_view import (
    LabPoolConversionError,
    build_lab_pool_training_view,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--clamp-epsilon", type=float, default=1e-6)
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(arguments)
    try:
        manifest = build_lab_pool_training_view(
            args.source, args.destination, clamp_epsilon=args.clamp_epsilon
        )
    except LabPoolConversionError as error:
        print("Error: {}".format(error), file=sys.stderr)
        return 1
    print(json.dumps(manifest["counts"], sort_keys=True))
    print("Manifest: {}".format(args.destination / "conversion_manifest.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
