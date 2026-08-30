from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest


PACKAGE_MODULES = (
    "fomo_servo",
    "fomo_servo.config",
    "fomo_servo.datasets",
    "fomo_servo.datasets.augmentation",
    "fomo_servo.models",
    "fomo_servo.training",
    "fomo_servo.losses",
    "fomo_servo.metrics",
    "fomo_servo.postprocess",
    "fomo_servo.inference",
    "fomo_servo.export",
)


@pytest.mark.parametrize("module_name", PACKAGE_MODULES)
def test_skeleton_package_modules_import(module_name: str) -> None:
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError:
        module = None

    assert module is not None, f"{module_name} must be importable"


def test_deployment_runtime_package_import_does_not_require_torch() -> None:
    root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(root / "src")

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.modules['torch'] = None; import fomo_servo.deployment",
        ],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_onnx_inference_modules_import_without_torch() -> None:
    root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(root / "src")

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; sys.modules['torch'] = None; "
                "import fomo_servo.inference; "
                "import fomo_servo.inference.ort_predictor"
            ),
        ],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_onnx_cli_modules_import_without_torch() -> None:
    root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(root), str(root / "src"))
    )

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; sys.modules['torch'] = None; "
                "import scripts.predict_image; import scripts.predict_video; "
                "import scripts.select_smoke_test_assets"
            ),
        ],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_ort_numpy_postprocess_modules_never_probe_torch_on_import() -> None:
    """Regression: the ORT-only path must not probe torch at module import.

    Unlike the ``sys.modules['torch'] = None`` poison used above, a meta-path
    blocker fails the very attempt to import torch, so even a probe that is
    caught and degraded at import time cannot hide. The recorded probe list
    also catches attempts swallowed by a broad except clause.
    """

    root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(root / "src")
    script = "\n".join(
        (
            "import sys",
            "probed = []",
            "class _TorchProbeBlocker:",
            "    def find_spec(self, fullname, path=None, target=None):",
            "        if fullname == 'torch' or fullname.startswith('torch.'):",
            "            probed.append(fullname)",
            "            raise ImportError(",
            "                'torch import attempted during module import: ' + fullname)",
            "        return None",
            "sys.meta_path.insert(0, _TorchProbeBlocker())",
            "import fomo_servo.postprocess",
            "import fomo_servo.inference",
            "import fomo_servo.inference.preprocessing",
            "import fomo_servo.inference.ort_predictor",
            "assert 'torch' not in sys.modules, 'torch ended up in sys.modules'",
            "assert not probed, 'torch import probed at module import: ' + repr(probed)",
            "print('ok')",
        )
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
