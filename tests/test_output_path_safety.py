"""Output artifact paths must never alias protected inputs or each other."""

import os
from pathlib import Path

import pytest


def test_rejects_output_that_aliases_protected_input(tmp_path: Path) -> None:
    from fomo_servo.inference.path_safety import OutputPathError, validate_output_paths

    model = tmp_path / "formal.onnx"
    model.write_bytes(b"model")

    with pytest.raises(OutputPathError, match="output_model.*model"):
        validate_output_paths(
            protected_inputs={"model": model},
            outputs={"output_model": model},
        )


def test_rejects_outputs_that_alias_each_other(tmp_path: Path) -> None:
    from fomo_servo.inference.path_safety import OutputPathError, validate_output_paths

    shared = tmp_path / "result.bin"

    with pytest.raises(OutputPathError, match="output_json.*output_image"):
        validate_output_paths(
            protected_inputs={},
            outputs={"output_image": shared, "output_json": shared},
        )


@pytest.mark.skipif(os.name != "nt", reason="Win32 path normalization applies on Windows")
@pytest.mark.parametrize("unsafe_suffix", (".", " "))
def test_rejects_nonexistent_output_with_win32_trailing_alias(
    tmp_path: Path, unsafe_suffix: str
) -> None:
    from fomo_servo.inference.path_safety import OutputPathError, validate_output_paths

    ordinary = tmp_path / "result.json"
    unsafe_alias = tmp_path / ("result.json" + unsafe_suffix)

    with pytest.raises(OutputPathError, match="unsafe Win32 output path component"):
        validate_output_paths(
            protected_inputs={},
            outputs={"ordinary": ordinary, "unsafe_alias": unsafe_alias},
        )


@pytest.mark.skipif(os.name != "nt", reason="Win32 path normalization applies on Windows")
@pytest.mark.parametrize("unsafe_parent", ("parent.", "parent "))
def test_rejects_win32_trailing_alias_in_parent_output_component(
    tmp_path: Path, unsafe_parent: str
) -> None:
    from fomo_servo.inference.path_safety import OutputPathError, validate_output_paths

    with pytest.raises(OutputPathError, match="unsafe Win32 output path component"):
        validate_output_paths(
            protected_inputs={},
            outputs={"result": tmp_path / unsafe_parent / "result.json"},
        )


@pytest.mark.skipif(os.name != "nt", reason="Win32 path normalization applies on Windows")
def test_win32_output_safety_allows_parent_navigation_component(
    tmp_path: Path,
) -> None:
    from fomo_servo.inference.path_safety import validate_output_paths

    validate_output_paths(
        protected_inputs={},
        outputs={"result": tmp_path / "child" / ".." / "result.json"},
    )


@pytest.mark.skipif(os.name != "nt", reason="Win32 path normalization applies on Windows")
def test_exporter_and_image_cli_share_win32_output_path_rejection(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from fomo_servo.deployment.onnx_export import (
        OnnxExportError,
        export_checkpoint_to_onnx,
    )
    from scripts.predict_image import main as predict_image_main

    unsafe_parent = tmp_path / "unsafe."
    shared_message = "unsafe Win32 output path component"

    with pytest.raises(OnnxExportError, match=shared_message):
        export_checkpoint_to_onnx(
            config_path=tmp_path / "unused-export.yaml",
            checkpoint_path=tmp_path / "unused-checkpoint.pt",
            onnx_path=unsafe_parent / "formal.onnx",
            report_path=tmp_path / "formal.onnx.json",
        )

    exit_code = predict_image_main(
        [
            "--config",
            str(tmp_path / "unused-project.yaml"),
            "--checkpoint",
            str(tmp_path / "unused-checkpoint.pt"),
            "--image",
            str(tmp_path / "unused-image.png"),
            "--output-image",
            str(unsafe_parent / "prediction.png"),
            "--output-json",
            str(tmp_path / "prediction.json"),
        ]
    )

    assert exit_code == 1
    assert shared_message in capsys.readouterr().err
