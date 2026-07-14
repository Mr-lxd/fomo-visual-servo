"""CLI contract tests for Stage C validation snapshot scans."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _script_module():
    path = Path(__file__).parents[1] / "scripts" / "evaluate_stage_c_snapshots.py"
    spec = importlib.util.spec_from_file_location("evaluate_stage_c_snapshots", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_stage_c_cli_requires_config_snapshot_and_output_paths() -> None:
    module = _script_module()
    arguments = module.build_parser().parse_args(
        [
            "--config",
            "stage_c.yaml",
            "--snapshot-dir",
            "snapshots",
            "--output-dir",
            "validation",
        ]
    )

    assert arguments.device == "cpu"
    assert arguments.snapshot_dir == Path("snapshots")
