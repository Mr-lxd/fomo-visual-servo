"""CLI contract tests for fixed-threshold Edge Impulse TFLite parity."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import zipfile

import pytest


def _script_module():
    path = Path(__file__).parents[1] / "scripts" / "evaluate_edge_impulse_tflite.py"
    spec = importlib.util.spec_from_file_location("evaluate_edge_impulse_tflite", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_cli_fixes_threshold_to_ei_parity_protocol() -> None:
    module = _script_module()
    args = module.build_parser().parse_args(
        [
            "--model",
            "edge-impulse.zip",
            "--config",
            "config.yaml",
            "--dataset-root",
            "dataset",
            "--cleaning-manifest",
            "parity-clean-v1.json",
            "--output-dir",
            "result",
            "--raw-output-cache",
            "prior-result",
        ]
    )

    assert args.expected_image_count == 63
    assert args.raw_output_cache == Path("prior-result")
    assert not hasattr(args, "threshold")


def test_zip_with_multiple_candidates_requires_explicit_entry(tmp_path: Path) -> None:
    module = _script_module()
    archive = tmp_path / "models.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("a.tflite", b"a")
        handle.writestr("b.tflite", b"b")

    with pytest.raises(module.EdgeImpulseTFLiteError, match="multiple .tflite"):
        module.resolve_tflite_model(archive, model_entry=None)
