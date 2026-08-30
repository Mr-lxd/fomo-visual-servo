from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
FUTURE_COMMANDS = (
    "evaluate.py",
    "predict_image.py",
    "predict_video.py",
    "export_onnx.py",
    "benchmark.py",
)

IMPLEMENTED_COMMANDS = {
    "evaluate.py",
    "predict_image.py",
    "predict_video.py",
    "export_onnx.py",
}


@pytest.mark.parametrize("script_name", FUTURE_COMMANDS)
def test_future_commands_fail_explicitly_until_implemented(script_name: str) -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / script_name)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    if script_name in IMPLEMENTED_COMMANDS:
        assert "usage:" in result.stderr.lower()
    else:
        assert f"{Path(script_name).stem} is not implemented" in result.stderr


def test_environment_check_reports_current_environment() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS_DIR / "check_environment.py"),
            "--config",
            "configs/aquarium_creature_192.yaml",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    supported_python = sys.version_info[:2] in {(3, 10), (3, 11)}
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError:
        torch_available = False
    else:
        torch_available = True

    expected_exit_code = 0 if supported_python and torch_available else 1
    assert result.returncode == expected_exit_code
    assert "Python:" in result.stdout
    assert "PyTorch:" in result.stdout
    assert "Config:" in result.stdout


PROFILE_DEPENDENCIES = {
    "training": ("torch", "cv2"),
    "deployment": ("cv2", "onnxruntime"),
    "all": ("torch", "cv2", "onnxruntime"),
}


@pytest.mark.parametrize("profile", tuple(PROFILE_DEPENDENCIES))
def test_check_env_reports_every_required_capability(profile: str) -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / "check_env.py"), "--profile", profile],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    supported_python = sys.version_info[:2] == (3, 10)
    dependencies_available = all(
        importlib.util.find_spec(name) is not None
        for name in PROFILE_DEPENDENCIES[profile]
    )
    expected_exit_code = 0 if supported_python and dependencies_available else 1

    assert result.returncode == expected_exit_code
    for label in (
        "Python:",
        "PyTorch:",
        "CUDA:",
        "OpenCV:",
        "ONNX Runtime:",
        "Device:",
    ):
        assert label in result.stdout
