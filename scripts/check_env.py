"""Inspect the active training, deployment, or combined project environment."""

from __future__ import annotations

import argparse
import sys
from typing import Dict, Optional, Sequence, Tuple


SUPPORTED_PYTHON_VERSION = (3, 10)
PROFILE_REQUIREMENTS = {
    "training": ("torch", "opencv"),
    "deployment": ("opencv", "onnxruntime"),
    "all": ("torch", "opencv", "onnxruntime"),
}


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=tuple(PROFILE_REQUIREMENTS),
        default="all",
        help="Dependency set to require from the active environment.",
    )
    return parser.parse_args(argv)


def _probe_torch() -> Tuple[bool, str, str, str]:
    try:
        import torch
    except Exception as error:  # Import failures must be reported, not hidden.
        reason = "{}: {}".format(type(error).__name__, error)
        return False, "not available ({})".format(reason), "unavailable", "unavailable"

    cuda_available = torch.cuda.is_available()
    cuda_status = "available" if cuda_available else "unavailable"
    device = "cuda" if cuda_available else "cpu"
    return True, str(torch.__version__), cuda_status, device


def _probe_opencv() -> Tuple[bool, str]:
    try:
        import cv2
    except Exception as error:  # Import failures must be reported, not hidden.
        return False, "not available ({}: {})".format(type(error).__name__, error)
    return True, str(cv2.__version__)


def _probe_onnxruntime() -> Tuple[bool, str]:
    try:
        import onnxruntime
    except Exception as error:  # Import failures must be reported, not hidden.
        return False, "not available ({}: {})".format(type(error).__name__, error)
    return True, str(onnxruntime.__version__)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Report active-environment capabilities and return profile-specific status."""

    args = _parse_args(argv)
    requirements = PROFILE_REQUIREMENTS[args.profile]
    checks: Dict[str, bool] = {}
    errors = []

    python_version = sys.version_info[:2]
    python_supported = python_version == SUPPORTED_PYTHON_VERSION
    checks["python"] = python_supported
    python_status = "supported" if python_supported else "unsupported; requires 3.10"
    print("Python: {}.{} ({})".format(*python_version, python_status))
    if not python_supported:
        errors.append("Python 3.10 is required")

    torch_available, torch_version, cuda_status, device = _probe_torch()
    checks["torch"] = torch_available
    print("PyTorch: {}".format(torch_version))
    print("CUDA: {}".format(cuda_status))
    print("Device: {}".format(device))

    opencv_available, opencv_version = _probe_opencv()
    checks["opencv"] = opencv_available
    print("OpenCV: {}".format(opencv_version))

    onnxruntime_available, onnxruntime_version = _probe_onnxruntime()
    checks["onnxruntime"] = onnxruntime_available
    print("ONNX Runtime: {}".format(onnxruntime_version))

    for requirement in requirements:
        if not checks[requirement]:
            errors.append("{} is required for the {} profile".format(requirement, args.profile))

    print("Profile: {}".format(args.profile))
    if errors:
        for error in errors:
            print("ERROR: {}".format(error), file=sys.stderr)
        print("Status: failed")
        return 1

    print("Status: ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
