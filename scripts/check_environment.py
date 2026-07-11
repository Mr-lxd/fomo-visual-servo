"""Report whether the current machine satisfies the project skeleton requirements."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from fomo_servo.config import ConfigurationError, load_config


SUPPORTED_PYTHON_VERSIONS = {(3, 10), (3, 11)}


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/aquarium_creature_192.yaml",
        help="Path to a project YAML configuration, relative to the repository root.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Print environment diagnostics and return 0 only for a supported setup."""

    args = _parse_args(argv)
    errors = []
    python_version = sys.version_info[:2]

    if python_version in SUPPORTED_PYTHON_VERSIONS:
        print("Python: {}.{} (supported)".format(*python_version))
    else:
        print("Python: {}.{} (unsupported)".format(*python_version))
        errors.append("Python 3.10 or 3.11 is required")

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = REPOSITORY_ROOT / config_path
    try:
        config = load_config(config_path)
    except ConfigurationError as error:
        print("Config: invalid ({})".format(error))
        errors.append("Project YAML configuration is invalid")
    else:
        print(
            "Config: {} (input={} stride={} channels={})".format(
                config_path.name,
                config.model.input_size,
                config.model.output_stride,
                config.output_channels,
            )
        )

    try:
        import torch
    except ModuleNotFoundError:
        print("PyTorch: not installed")
        errors.append("PyTorch is required before model, training, or export work")
    else:
        print(
            "PyTorch: {} (cuda_available={})".format(
                torch.__version__, torch.cuda.is_available()
            )
        )

    if errors:
        for error in errors:
            print("ERROR: {}".format(error), file=sys.stderr)
        print("Status: failed")
        return 1

    print("Status: ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
