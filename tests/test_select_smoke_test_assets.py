"""Deterministic deployment smoke-test asset selection script tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from torch import nn

from scripts.select_smoke_test_assets import main


class _BrightCellModel(nn.Module):
    """Map ``[1,3,24,24]`` RGB to raw logits ``[1,2,3,3]`` with stride 8.

    The foreground logit is ``2v - 1`` at each stride-8 sample of the red
    channel, so bright cells (v=1) produce probability ``sigmoid(2) > 0.4``
    detections and dark cells (v=0) stay below the 0.4 threshold. The 3x3
    grid allows two disconnected bright cells to yield two detections.
    """

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        foreground = images[:, :1, ::8, ::8] * 2.0 - 1.0
        return torch.cat((-foreground, foreground), dim=1)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_model_and_report(tmp_path: Path) -> tuple[Path, Path]:
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    model = _BrightCellModel().eval()
    onnx_path = tmp_path / "tiny.onnx"
    torch.onnx.export(
        model,
        torch.zeros(1, 3, 24, 24, dtype=torch.float32),
        onnx_path,
        input_names=["images"],
        output_names=["logits"],
        opset_version=17,
        dynamic_axes=None,
    )
    report = {
        "artifact_name": "tiny_fixture",
        "source_experiment_config": "experiment.yaml",
        "source_experiment_config_sha256": "1" * 64,
        "export_config_file": "export.yaml",
        "export_config_sha256": "2" * 64,
        "checkpoint_file": "epoch_040_weights.pt",
        "checkpoint_sha256": "3" * 64,
        "epoch": 40,
        "seed": 42,
        "parameter_count": 1,
        "config_fingerprint": "4" * 64,
        "validation_threshold": 0.4,
        "validation_threshold_usage": "provenance_only_raw_logits_export",
        "onnx_file": onnx_path.name,
        "onnx_sha256": _sha256(onnx_path),
        "onnx_size_bytes": onnx_path.stat().st_size,
        "onnx_opset": 17,
        "onnx_checker": "passed",
        "input_name": "images",
        "input_shape": [1, 3, 24, 24],
        "input_dtype": "float32",
        "input_color_order": "RGB",
        "input_value_range": [0.0, 1.0],
        "output_name": "logits",
        "output_shape": [1, 2, 3, 3],
        "output_dtype": "float32",
        "output_semantic": "raw_logits",
        "output_stride": 8,
        "class_names": ["creature"],
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
        "pytorch_version": "fixture",
        "onnx_version": "fixture",
        "onnxruntime_version": "fixture",
        "exported_at_utc": "2026-08-29T00:00:00+00:00",
        "parity": {
            "passed": True,
            "input_seed": 42,
            "rtol": 1e-4,
            "atol": 1e-5,
            "max_absolute_error": 0.0,
            "mean_absolute_error": 0.0,
        },
    }
    report_path = tmp_path / "tiny.onnx.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    return onnx_path, report_path


def _write_png(path: Path, bright_cells: tuple[tuple[int, int], ...]) -> None:
    """Write a 24x24 RGB PNG with the given stride-8 grid cells at value 255."""

    image = np.zeros((24, 24, 3), dtype=np.uint8)
    for row, col in bright_cells:
        image[row * 8 : (row + 1) * 8, col * 8 : (col + 1) * 8] = 255
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), image)


def _selection_arguments(
    dataset_root: Path,
    onnx_path: Path,
    report_path: Path,
    manifest_path: Path,
    *,
    media_dir: Path | None = None,
    max_positive: int = 2,
    max_negative: int = 2,
) -> list[str]:
    arguments = [
        "--dataset-root",
        str(dataset_root),
        "--split",
        "train",
        "--onnx",
        str(onnx_path),
        "--onnx-report",
        str(report_path),
        "--output-manifest",
        str(manifest_path),
        "--max-positive",
        str(max_positive),
        "--max-negative",
        str(max_negative),
    ]
    if media_dir is not None:
        arguments += ["--media-output-dir", str(media_dir)]
    return arguments


def test_selection_is_deterministic_and_records_contract(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    _write_png(
        dataset_root / "train" / "images" / "a_bright.png", ((0, 0), (0, 2))
    )
    _write_png(dataset_root / "train" / "images" / "b_dark.png", ())
    _write_png(dataset_root / "train" / "images" / "c_dark.png", ())
    _write_png(dataset_root / "train" / "images" / "d_bright.png", ((1, 1),))
    _write_png(dataset_root / "train" / "images" / "e_bright.png", ((0, 0),))
    _write_png(
        dataset_root / "valid" / "images" / "z_bright.png", ((0, 0), (0, 2))
    )
    onnx_path, report_path = _write_model_and_report(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    media_dir = tmp_path / "media"

    exit_code = main(
        _selection_arguments(
            dataset_root,
            onnx_path,
            report_path,
            manifest_path,
            media_dir=media_dir,
        )
    )

    assert exit_code == 0
    manifest_text = manifest_path.read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)
    assert str(tmp_path) not in manifest_text, "manifest must not embed machine paths"

    contract_section = manifest["model_contract"]
    assert contract_section["onnx_file"] == onnx_path.name
    assert contract_section["onnx_sha256"] == _sha256(onnx_path)
    assert contract_section["onnx_size_bytes"] == onnx_path.stat().st_size
    assert contract_section["onnx_report_sha256"] == _sha256(report_path)
    assert contract_section["confidence_threshold"] == 0.4
    assert contract_section["class_names"] == ["creature"]

    selection = manifest["selection"]
    assert selection["dataset_split"] == "train"
    assert selection["train_image_count"] == 5
    assert selection["scanned_image_count"] == 4

    cases = manifest["cases"]
    assert [case["image_path"] for case in cases] == [
        "train/images/a_bright.png",
        "train/images/b_dark.png",
        "train/images/c_dark.png",
        "train/images/d_bright.png",
    ]
    by_path = {case["image_path"]: case for case in cases}
    assert by_path["train/images/a_bright.png"]["role"] == "positive"
    assert by_path["train/images/a_bright.png"]["detection_count"] == 2
    assert by_path["train/images/d_bright.png"]["role"] == "positive"
    assert by_path["train/images/d_bright.png"]["detection_count"] == 1
    assert by_path["train/images/b_dark.png"]["role"] == "negative_control"
    assert by_path["train/images/b_dark.png"]["detection_count"] == 0
    assert by_path["train/images/b_dark.png"]["detections"] == []
    assert by_path["train/images/c_dark.png"]["role"] == "negative_control"
    for case in cases:
        assert case["split"] == "train"
        assert case["image_sha256"] == _sha256(dataset_root / case["image_path"])
        assert case["image_width"] == 24
        assert case["image_height"] == 24
    two_detections = by_path["train/images/a_bright.png"]["detections"]
    assert {
        (round(item["original_x"]), round(item["original_y"]))
        for item in two_detections
    } == {(4, 4), (20, 4)}
    assert all(
        item["class_id"] == 0
        and item["class_name"] == "creature"
        and item["confidence"] > 0.4
        for item in two_detections
    )

    copied_names = sorted(path.name for path in media_dir.iterdir())
    assert copied_names == [
        "a_bright.png",
        "b_dark.png",
        "c_dark.png",
        "d_bright.png",
    ]
    for name in copied_names:
        assert _sha256(media_dir / name) == _sha256(
            dataset_root / "train" / "images" / name
        )


def test_scan_stops_after_quotas_are_filled(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    _write_png(dataset_root / "train" / "images" / "a_bright.png", ((0, 0),))
    _write_png(dataset_root / "train" / "images" / "b_dark.png", ())
    _write_png(dataset_root / "train" / "images" / "c_bright.png", ((0, 0),))
    _write_png(dataset_root / "train" / "images" / "d_dark.png", ())
    onnx_path, report_path = _write_model_and_report(tmp_path)
    manifest_path = tmp_path / "manifest.json"

    exit_code = main(
        _selection_arguments(
            dataset_root,
            onnx_path,
            report_path,
            manifest_path,
            max_positive=1,
            max_negative=1,
        )
    )

    assert exit_code == 0
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["selection"]["scanned_image_count"] == 2
    assert [case["image_path"] for case in manifest["cases"]] == [
        "train/images/a_bright.png",
        "train/images/b_dark.png",
    ]


def test_no_positive_detection_reports_fact_and_writes_no_manifest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset_root = tmp_path / "dataset"
    _write_png(dataset_root / "train" / "images" / "only_dark.png", ())
    onnx_path, report_path = _write_model_and_report(tmp_path)
    manifest_path = tmp_path / "manifest.json"

    exit_code = main(
        _selection_arguments(dataset_root, onnx_path, report_path, manifest_path)
    )

    assert exit_code == 1
    assert not manifest_path.exists()
    captured = capsys.readouterr()
    assert "non-empty detection" in captured.err
    assert "0.4" in captured.err


def test_other_split_images_are_never_selected(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    _write_png(dataset_root / "train" / "images" / "only_dark.png", ())
    _write_png(dataset_root / "valid" / "images" / "bright.png", ((0, 0), (1, 1)))
    onnx_path, report_path = _write_model_and_report(tmp_path)
    manifest_path = tmp_path / "manifest.json"

    exit_code = main(
        _selection_arguments(dataset_root, onnx_path, report_path, manifest_path)
    )

    assert exit_code == 1
    assert not manifest_path.exists()


def test_missing_split_images_directory_fails_clearly(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    onnx_path, report_path = _write_model_and_report(tmp_path)
    manifest_path = tmp_path / "manifest.json"

    exit_code = main(
        _selection_arguments(dataset_root, onnx_path, report_path, manifest_path)
    )

    assert exit_code == 1
    assert not manifest_path.exists()
    captured = capsys.readouterr()
    assert "train" in captured.err
