"""Formal checkpoint-to-ONNX export contract and parity tests."""

from __future__ import annotations

import hashlib
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from fomo_servo.models import MobileNetV2FOMONet


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    ("relative_path", "expected_sha256"),
    (
        (
            "configs/experiments/stage_d2_fomo_ei_w100_pretrained.yaml",
            "f6e35683a6df8f98537c4ca870487858a9e089f4d76d965176d51f08fe88fb2e",
        ),
        (
            "configs/export/d2_seed42_epoch40_onnx.yaml",
            "a05219c21ddaa8e24ee262ffc7d9d30d104268dc1bd3d8cff970a41cc81becf7",
        ),
    ),
)
def test_formal_provenance_yaml_bytes_match_locked_sha256(
    relative_path: str, expected_sha256: str
) -> None:
    payload = (ROOT / relative_path).read_bytes()

    assert b"\r\n" not in payload
    assert hashlib.sha256(payload).hexdigest() == expected_sha256


def _checkpoint_metadata() -> dict[str, object]:
    return {
        "backbone_name": "mobilenet_v2_fomo",
        "width_multiplier": 0.35,
        "cut_point": "block_6_expand_relu",
        "cut_point_input_channels": 16,
        "cut_point_output_channels": 96,
        "output_stride": 8,
        "head_channels": 32,
        "pretrained": True,
        "initialization": "ei_keras_mobilenet_v2_035_96",
        "backbone_parameter_count": 15_840,
        "head_parameter_count": 3_368,
        "parameter_count": 19_208,
        "pretrained_load_report": {
            "sha256": "a" * 64,
            "loaded_tensor_count": 95,
            "missing_keys": [],
            "unexpected_keys": [],
        },
    }


def _write_checkpoint(path: Path, *, epoch: int = 40) -> Path:
    torch.manual_seed(7)
    model = MobileNetV2FOMONet(num_classes=7, input_size=192, pretrained=False)
    payload = {
        "checkpoint_kind": "epoch_snapshot",
        "weights_only": True,
        "resumable": False,
        "format": "weights_only",
        "model_state": model.state_dict(),
        "epoch": epoch,
        "model_metadata": _checkpoint_metadata(),
        "parameter_count": 19_208,
        "config_fingerprint": "b" * 64,
        "dataset_content_hash": "c" * 64,
        "git_commit_sha": "d" * 40,
        "seed": 42,
        "augmentation_preset": "underwater_conservative",
        "checkpoint_threshold": 0.5,
        "loss": {"type": "ei_weighted_xent_legacy"},
    }
    torch.save(payload, path)
    return path


def _write_export_config(path: Path, checkpoint: Path, **overrides: object) -> Path:
    source_config = path.parent / "source_experiment.yaml"
    source_config.write_text(
        yaml.safe_dump(
            {
                "dataset": {
                    "classes": [
                        "fish",
                        "jellyfish",
                        "penguin",
                        "puffin",
                        "shark",
                        "starfish",
                        "stingray",
                    ]
                },
                "model": {
                    "backbone": "mobilenet_v2_fomo",
                    "width_multiplier": 0.35,
                    "head_channels": 32,
                    "input_size": 192,
                    "output_stride": 8,
                    "pretrained": True,
                    "pretrained_sha256": "a" * 64,
                },
                "training": {"seed": 42},
                "export": {"onnx_opset": 17, "input_size": 192},
                "postprocess": {
                    "inference_threshold": 0.5,
                    "class_thresholds": None,
                    "component_mode": "connected_components",
                    "confidence_mode": "max",
                    "selection_strategy": "highest_confidence",
                    "max_match_distance_pixels": 32.0,
                    "max_lost_frames": 5,
                    "allowed_class_ids": None,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    payload: dict[str, object] = {
        "artifact": {
            "name": "d2_mobilenet_v2_fomo_seed42_epoch40",
            "source_experiment_config": source_config.name,
            "source_experiment_config_sha256": _sha256(source_config),
            "validation_threshold": 0.40,
        },
        "checkpoint": {
            "sha256": _sha256(checkpoint),
            "epoch": 40,
            "seed": 42,
            "parameter_count": 19_208,
            "config_fingerprint": "b" * 64,
        },
        "model": {
            "backbone": "mobilenet_v2_fomo",
            "width_multiplier": 0.35,
            "cut_point": "block_6_expand_relu",
            "head_channels": 32,
            "output_stride": 8,
            "classes": [
                "fish",
                "jellyfish",
                "penguin",
                "puffin",
                "shark",
                "starfish",
                "stingray",
            ],
            "pretrained": True,
            "initialization": "ei_keras_mobilenet_v2_035_96",
            "pretrained_sha256": "a" * 64,
        },
        "input": {
            "name": "images",
            "shape": [1, 3, 192, 192],
            "dtype": "float32",
            "color_order": "RGB",
            "value_range": [0.0, 1.0],
        },
        "output": {
            "name": "logits",
            "shape": [1, 8, 24, 24],
            "dtype": "float32",
            "semantic": "raw_logits",
        },
        "onnx": {"opset": 17},
        "parity": {"seed": 42, "rtol": 0.0001, "atol": 0.00001},
        "postprocess": {
            "confidence_threshold": 0.4,
            "class_thresholds": None,
            "component_mode": "connected_components",
            "confidence_mode": "max",
            "selection_strategy": "highest_confidence",
            "max_match_distance_pixels": 32.0,
            "max_lost_frames": 5,
            "allowed_class_ids": None,
        },
    }
    for dotted_key, value in overrides.items():
        section, key = dotted_key.split("__", maxsplit=1)
        section_payload = payload[section]
        assert isinstance(section_payload, dict)
        section_payload[key] = value
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_load_formal_checkpoint_validates_metadata_and_state_dict(tmp_path: Path) -> None:
    from fomo_servo.deployment.onnx_export import (
        load_checkpoint_model,
        load_export_contract,
    )

    checkpoint = _write_checkpoint(tmp_path / "epoch_040_weights.pt")
    config = _write_export_config(tmp_path / "export.yaml", checkpoint)

    contract = load_export_contract(config)
    model, provenance = load_checkpoint_model(contract, checkpoint)

    assert model.training is False
    assert provenance["epoch"] == 40
    assert provenance["seed"] == 42
    assert provenance["parameter_count"] == 19_208
    with torch.inference_mode():
        assert model(torch.zeros(1, 3, 192, 192)).shape == (1, 8, 24, 24)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("wrong_epoch", "epoch"),
        ("wrong_metadata", "model metadata"),
        ("wrong_state", "state dict"),
    ),
)
def test_formal_checkpoint_rejects_incompatible_payload(
    tmp_path: Path, mutation: str, message: str
) -> None:
    from fomo_servo.deployment.onnx_export import (
        OnnxExportError,
        load_checkpoint_model,
        load_export_contract,
    )

    checkpoint = _write_checkpoint(tmp_path / "epoch_040_weights.pt")
    config = _write_export_config(tmp_path / "export.yaml", checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if mutation == "wrong_epoch":
        payload["epoch"] = 39
    elif mutation == "wrong_metadata":
        payload["model_metadata"]["backbone_name"] = "wrong_backbone"
    else:
        payload["model_state"]["head.2.weight"] = torch.zeros(7, 32, 1, 1)
    torch.save(payload, checkpoint)
    _write_export_config(config, checkpoint)

    with pytest.raises(OnnxExportError, match=message):
        load_checkpoint_model(load_export_contract(config), checkpoint)


def test_export_config_rejects_shape_contract_mismatch(tmp_path: Path) -> None:
    from fomo_servo.deployment.onnx_export import OnnxExportError, load_export_contract

    checkpoint = _write_checkpoint(tmp_path / "epoch_040_weights.pt")
    config = _write_export_config(
        tmp_path / "invalid.yaml", checkpoint, output__shape=[1, 8, 23, 24]
    )

    with pytest.raises(OnnxExportError, match="output.shape"):
        load_export_contract(config)


def test_export_config_rejects_class_order_drift_from_source_experiment(
    tmp_path: Path,
) -> None:
    from fomo_servo.deployment.onnx_export import OnnxExportError, load_export_contract

    checkpoint = _write_checkpoint(tmp_path / "epoch_040_weights.pt")
    config = _write_export_config(
        tmp_path / "invalid.yaml",
        checkpoint,
        model__classes=[
            "jellyfish",
            "fish",
            "penguin",
            "puffin",
            "shark",
            "starfish",
            "stingray",
        ],
    )

    with pytest.raises(OnnxExportError, match="class mapping"):
        load_export_contract(config)


@pytest.mark.parametrize(
    "collision",
    (
        "output_checkpoint",
        "normalized_output_checkpoint",
        "hardlink_output_checkpoint",
        "report_config",
        "report_output",
    ),
)
def test_export_rejects_path_aliases_before_modifying_sources(
    tmp_path: Path, collision: str
) -> None:
    from fomo_servo.deployment.onnx_export import OnnxExportError, export_checkpoint_to_onnx

    checkpoint = _write_checkpoint(tmp_path / "epoch_040_weights.pt")
    config = _write_export_config(tmp_path / "export.yaml", checkpoint)
    output = tmp_path / "formal.onnx"
    report = tmp_path / "formal.onnx.json"
    if collision == "output_checkpoint":
        output = checkpoint
    elif collision == "normalized_output_checkpoint":
        output = tmp_path / "unused" / ".." / checkpoint.name
    elif collision == "hardlink_output_checkpoint":
        output = tmp_path / "checkpoint-hardlink.pt"
        os.link(checkpoint, output)
    elif collision == "report_config":
        report = config
    else:
        report = output
    checkpoint_before = checkpoint.read_bytes()
    config_before = config.read_bytes()

    with pytest.raises(OnnxExportError, match="distinct paths"):
        export_checkpoint_to_onnx(
            config_path=config,
            checkpoint_path=checkpoint,
            onnx_path=output,
            report_path=report,
        )

    assert checkpoint.read_bytes() == checkpoint_before
    assert config.read_bytes() == config_before


def test_export_runs_checker_and_pytorch_ort_logits_parity(tmp_path: Path) -> None:
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    from fomo_servo.deployment.onnx_export import export_checkpoint_to_onnx

    checkpoint = _write_checkpoint(tmp_path / "epoch_040_weights.pt")
    config = _write_export_config(tmp_path / "export.yaml", checkpoint)
    onnx_path = tmp_path / "formal.onnx"
    report_path = tmp_path / "formal.onnx.json"

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        report = export_checkpoint_to_onnx(
            config_path=config,
            checkpoint_path=checkpoint,
            onnx_path=onnx_path,
            report_path=report_path,
        )

    graph = onnx.load(str(onnx_path))
    onnx.checker.check_model(graph)
    assert graph.opset_import[0].version == 17
    assert report["input_shape"] == [1, 3, 192, 192]
    assert report["output_shape"] == [1, 8, 24, 24]
    assert report["output_stride"] == 8
    assert report["postprocess"] == {
        "confidence_threshold": 0.4,
        "class_thresholds": None,
        "component_mode": "connected_components",
        "confidence_mode": "max",
        "selection_strategy": "highest_confidence",
        "max_match_distance_pixels": 32.0,
        "max_lost_frames": 5,
        "allowed_class_ids": None,
    }
    assert report["checkpoint_sha256"] == _sha256(checkpoint)
    assert report["onnx_sha256"] == _sha256(onnx_path)
    assert report["onnx_size_bytes"] == onnx_path.stat().st_size
    assert report["onnx_file"] == onnx_path.name
    assert report["checkpoint_file"] == checkpoint.name
    assert report["epoch"] == 40
    assert report["seed"] == 42
    assert report["onnx_checker"] == "passed"
    assert report["parity"]["passed"] is True
    assert caught_warnings == []
    assert json.loads(report_path.read_text(encoding="utf-8")) == report

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    assert session.get_inputs()[0].shape == [1, 3, 192, 192]
    assert session.get_inputs()[0].type == "tensor(float)"
    assert session.get_outputs()[0].shape == [1, 8, 24, 24]
    assert session.get_outputs()[0].type == "tensor(float)"


@pytest.mark.parametrize("failed_publication", ("onnx", "report"))
def test_publish_failure_restores_existing_artifact_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failed_publication: str,
) -> None:
    from fomo_servo.deployment import onnx_export

    destination = tmp_path / "formal.onnx"
    report_destination = tmp_path / "formal.onnx.json"
    staged_onnx = tmp_path / "staged.onnx"
    staged_report = tmp_path / "staged.onnx.json"
    destination.write_bytes(b"old-onnx")
    report_destination.write_bytes(b"old-report")
    staged_onnx.write_bytes(b"new-onnx")
    staged_report.write_bytes(b"new-report")
    real_replace = os.replace
    failed = False

    def fail_selected_publication(source: object, target: object) -> None:
        nonlocal failed
        source_path = Path(source)
        target_path = Path(target)
        selected_source = staged_onnx if failed_publication == "onnx" else staged_report
        selected_target = destination if failed_publication == "onnx" else report_destination
        if not failed and source_path == selected_source and target_path == selected_target:
            failed = True
            raise OSError("simulated {} publication failure".format(failed_publication))
        real_replace(source, target)

    monkeypatch.setattr(onnx_export.os, "replace", fail_selected_publication)

    with pytest.raises(
        onnx_export.OnnxExportError,
        match="unable to publish ONNX artifact pair",
    ):
        onnx_export._publish_staged_artifact_pair(
            staged_onnx=staged_onnx,
            staged_report=staged_report,
            onnx_path=destination,
            report_path=report_destination,
        )

    assert destination.read_bytes() == b"old-onnx"
    assert report_destination.read_bytes() == b"old-report"


@pytest.mark.parametrize("failed_publication", ("onnx", "report"))
def test_publish_failure_without_existing_pair_leaves_no_formal_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failed_publication: str,
) -> None:
    from fomo_servo.deployment import onnx_export

    destination = tmp_path / "formal.onnx"
    report_destination = tmp_path / "formal.onnx.json"
    staged_onnx = tmp_path / "staged.onnx"
    staged_report = tmp_path / "staged.onnx.json"
    staged_onnx.write_bytes(b"new-onnx")
    staged_report.write_bytes(b"new-report")
    real_replace = os.replace
    failed = False

    def fail_selected_publication(source: object, target: object) -> None:
        nonlocal failed
        source_path = Path(source)
        target_path = Path(target)
        selected_source = staged_onnx if failed_publication == "onnx" else staged_report
        selected_target = destination if failed_publication == "onnx" else report_destination
        if not failed and source_path == selected_source and target_path == selected_target:
            failed = True
            raise OSError("simulated {} publication failure".format(failed_publication))
        real_replace(source, target)

    monkeypatch.setattr(onnx_export.os, "replace", fail_selected_publication)

    with pytest.raises(
        onnx_export.OnnxExportError,
        match="unable to publish ONNX artifact pair",
    ):
        onnx_export._publish_staged_artifact_pair(
            staged_onnx=staged_onnx,
            staged_report=staged_report,
            onnx_path=destination,
            report_path=report_destination,
        )

    assert not destination.exists()
    assert not report_destination.exists()


def test_export_cli_reports_diagnostic_checkpoint_hash_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from scripts.export_onnx import main

    checkpoint = _write_checkpoint(tmp_path / "epoch_040_weights.pt")
    config = _write_export_config(
        tmp_path / "export.yaml", checkpoint, checkpoint__sha256="0" * 64
    )

    exit_code = main(
        [
            "--config",
            str(config),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(tmp_path / "formal.onnx"),
            "--report",
            str(tmp_path / "formal.onnx.json"),
        ]
    )

    assert exit_code == 1
    assert "checkpoint SHA-256 mismatch" in capsys.readouterr().err
    assert not (tmp_path / "formal.onnx").exists()


def test_export_wraps_checker_failure_with_stage_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from fomo_servo.deployment.onnx_export import OnnxExportError, export_checkpoint_to_onnx

    class _CheckerFailure(Exception):
        pass

    checkpoint = _write_checkpoint(tmp_path / "epoch_040_weights.pt")
    config = _write_export_config(tmp_path / "export.yaml", checkpoint)
    monkeypatch.setattr(
        onnx.checker,
        "check_model",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(_CheckerFailure("bad graph")),
    )

    with pytest.raises(OnnxExportError, match="ONNX checker failed: bad graph"):
        export_checkpoint_to_onnx(
            config_path=config,
            checkpoint_path=checkpoint,
            onnx_path=tmp_path / "formal.onnx",
            report_path=tmp_path / "formal.onnx.json",
        )


def test_export_config_wraps_invalid_numeric_value(tmp_path: Path) -> None:
    from fomo_servo.deployment.onnx_export import OnnxExportError, load_export_contract

    checkpoint = _write_checkpoint(tmp_path / "epoch_040_weights.pt")
    config = _write_export_config(
        tmp_path / "invalid.yaml", checkpoint, input__value_range=["bad", 1.0]
    )

    with pytest.raises(OnnxExportError, match="input.value_range"):
        load_export_contract(config)


@pytest.mark.parametrize("failure_stage", ("session", "inference"))
def test_export_wraps_onnxruntime_failures_with_stage_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_stage: str
) -> None:
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    from fomo_servo.deployment.onnx_export import OnnxExportError, export_checkpoint_to_onnx

    real_session = ort.InferenceSession

    class _OrtFailure(Exception):
        pass

    if failure_stage == "session":
        monkeypatch.setattr(
            ort,
            "InferenceSession",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(_OrtFailure("bad session")),
        )
        message = "ONNX Runtime session creation failed: bad session"
    else:
        class _InferenceFailureSession:
            def __init__(self, *args: object, **kwargs: object) -> None:
                self._session = real_session(*args, **kwargs)

            def get_inputs(self) -> object:
                return self._session.get_inputs()

            def get_outputs(self) -> object:
                return self._session.get_outputs()

            def run(self, *_args: object, **_kwargs: object) -> object:
                raise _OrtFailure("bad inference")

        monkeypatch.setattr(ort, "InferenceSession", _InferenceFailureSession)
        message = "ONNX Runtime inference failed: bad inference"

    checkpoint = _write_checkpoint(tmp_path / "epoch_040_weights.pt")
    config = _write_export_config(tmp_path / "export.yaml", checkpoint)

    with pytest.raises(OnnxExportError, match=message):
        export_checkpoint_to_onnx(
            config_path=config,
            checkpoint_path=checkpoint,
            onnx_path=tmp_path / "formal.onnx",
            report_path=tmp_path / "formal.onnx.json",
        )
