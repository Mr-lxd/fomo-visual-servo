from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = ROOT / "tests" / "fixtures" / "yolo_micro"


def test_visualizer_writes_four_panel_image(tmp_path: Path) -> None:
    output_path = tmp_path / "fomo_panels.jpg"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "visualize_yolo_heatmap.py"),
            "--dataset-root",
            str(FIXTURE_ROOT),
            "--split",
            "train",
            "--index",
            "3",
            "--input-size",
            "192",
            "--stride",
            "8",
            "--class-mode",
            "preserve",
            "--output",
            str(output_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    panel_image = cv2.imread(str(output_path), cv2.IMREAD_COLOR)
    assert panel_image is not None
    assert panel_image.shape == (384, 384, 3)
