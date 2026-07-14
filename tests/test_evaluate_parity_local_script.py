"""CLI contract tests for the fixed local-checkpoint parity evaluator."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _script_module():
    path = Path(__file__).parents[1] / "scripts" / "evaluate_parity_local.py"
    spec = importlib.util.spec_from_file_location("evaluate_parity_local", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_requires_explicit_cleaning_manifest_and_fixed_threshold() -> None:
    module = _script_module()
    arguments = module.build_parser().parse_args(
        [
            "--config",
            "config.yaml",
            "--dataset-root",
            "dataset",
            "--checkpoint",
            "epoch58.pt",
            "--expected-checkpoint-sha256",
            "a" * 64,
            "--cleaning-manifest",
            "parity-clean-v1.json",
            "--threshold",
            "0.5",
            "--output-dir",
            "result",
        ]
    )

    assert arguments.threshold == 0.5
    assert arguments.device == "cpu"


def test_output_directory_is_refused_when_it_already_exists(tmp_path: Path) -> None:
    module = _script_module()
    destination = tmp_path / "existing"
    destination.mkdir()

    with pytest.raises(module.LocalParityError, match="refusing to overwrite"):
        module.prepare_output_dir(destination)
