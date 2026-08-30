"""Bundle launcher (run.py) tests: root resolution, dispatch, and usage errors."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

STUB_PREDICT_VIDEO = (
    "def main(arguments):\n"
    "    print('stub predict_video', list(arguments))\n"
    "    return 0\n"
)


def _write_mini_bundle(tmp_path: Path) -> Path:
    """Fabricate a minimal src-layout bundle with a stub predict_video entry."""

    root = tmp_path / "bundle"
    (root / "src" / "scripts").mkdir(parents=True)
    shutil.copyfile(REPO_ROOT / "run.py", root / "run.py")
    (root / "src" / "scripts" / "__init__.py").write_text("", encoding="utf-8")
    (root / "src" / "scripts" / "predict_video.py").write_text(
        STUB_PREDICT_VIDEO, encoding="utf-8"
    )
    return root


def _run_launcher(
    bundle_root: Path, arguments: list[str], cwd: Path
) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    return subprocess.run(
        [sys.executable, str(bundle_root / "run.py"), *arguments],
        cwd=cwd,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def test_launcher_resolves_bundle_root_from_own_location(tmp_path: Path) -> None:
    root = _write_mini_bundle(tmp_path)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()

    result = _run_launcher(
        root,
        ["predict_video", "--source", "/dev/video0", "--max-frames", "5"],
        cwd=elsewhere,
    )

    assert result.returncode == 0, result.stderr
    assert "stub predict_video" in result.stdout
    assert "--source" in result.stdout
    assert "/dev/video0" in result.stdout
    assert "--max-frames" in result.stdout


def test_launcher_rejects_unknown_entry(tmp_path: Path) -> None:
    root = _write_mini_bundle(tmp_path)

    result = _run_launcher(root, ["no_such_entry"], cwd=tmp_path)

    assert result.returncode == 2
    assert "unknown entry" in result.stderr
    assert "predict_video" in result.stderr
    assert "predict_image" in result.stderr
    assert "stub predict_video" not in result.stdout


def test_launcher_prints_usage_without_arguments(tmp_path: Path) -> None:
    root = _write_mini_bundle(tmp_path)

    bare = _run_launcher(root, [], cwd=tmp_path)
    help_result = _run_launcher(root, ["--help"], cwd=tmp_path)

    assert bare.returncode == 2
    assert "predict_video" in bare.stdout
    assert help_result.returncode == 0
    assert "predict_video" in help_result.stdout


def test_launcher_requires_src_layout_next_to_launcher(tmp_path: Path) -> None:
    broken_root = tmp_path / "broken"
    broken_root.mkdir()
    shutil.copyfile(REPO_ROOT / "run.py", broken_root / "run.py")

    result = _run_launcher(broken_root, ["predict_video"], cwd=tmp_path)

    assert result.returncode == 2
    assert "src" in result.stderr


def test_repo_launcher_dispatches_real_cli_without_pythonpath() -> None:
    """From the repo root the launcher must reach the real predict_video parser."""

    result = _run_launcher(REPO_ROOT, ["predict_video", "--help"], cwd=REPO_ROOT)

    assert result.returncode == 0, result.stderr
    assert "--source" in result.stdout
    assert "--max-frames" in result.stdout
    assert "--duration-seconds" in result.stdout
    assert "--display" in result.stdout


def test_pi_runtime_requirement_profiles_keep_gui_optional() -> None:
    headless = (
        REPO_ROOT / "requirements-pi4-headless.txt"
    ).read_text(encoding="utf-8").splitlines()
    preview = (
        REPO_ROOT / "requirements-pi4-preview.txt"
    ).read_text(encoding="utf-8").splitlines()

    assert headless == [
        "numpy==2.5.2",
        "onnxruntime==1.29.0",
        "opencv-python-headless==5.0.0.93",
    ]
    assert preview == [
        "numpy==2.5.2",
        "onnxruntime==1.29.0",
        "opencv-python==5.0.0.93",
    ]
    joined = "\n".join(headless + preview).lower()
    for forbidden in ("torch", "torchvision", "cuda", "training", "models"):
        assert forbidden not in joined
