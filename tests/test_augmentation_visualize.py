"""CLI smoke test for the disabled augmentation visualization interface."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]


def test_visualize_augmentations_writes_fixture_panel(tmp_path: Path) -> None:
    """The future visualization entry point must consume no-op dataset outputs."""

    output_path = tmp_path / "augmentation.jpg"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/visualize_augmentations.py",
            "--dataset-root",
            "tests/fixtures/yolo_micro",
            "--split",
            "train",
            "--index",
            "0",
            "--input-size",
            "96",
            "--output",
            str(output_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    panel = cv2.imread(str(output_path), cv2.IMREAD_COLOR)
    assert panel is not None
    assert panel.shape == (192, 192, 3)
