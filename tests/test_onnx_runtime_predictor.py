"""ONNX Runtime image predictor and shared pipeline parity tests."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
import cv2


class _TinyImageLogitModel(nn.Module):
    """Map `[1,3,16,16]` RGB input to raw logits `[1,2,2,2]`."""

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        foreground = images[:, :1, ::8, ::8]
        return torch.cat((-foreground, foreground), dim=1)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_model_and_report(tmp_path: Path) -> tuple[Path, Path, nn.Module]:
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    model = _TinyImageLogitModel().eval()
    onnx_path = tmp_path / "tiny.onnx"
    torch.onnx.export(
        model,
        torch.zeros(1, 3, 16, 16, dtype=torch.float32),
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
        "input_shape": [1, 3, 16, 16],
        "input_dtype": "float32",
        "input_color_order": "RGB",
        "input_value_range": [0.0, 1.0],
        "output_name": "logits",
        "output_shape": [1, 2, 2, 2],
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
        "exported_at_utc": "2026-08-28T00:00:00+00:00",
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
    return onnx_path, report_path, model


def _run_cli_without_torch(module_name: str, arguments: list[str]) -> subprocess.CompletedProcess[str]:
    root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join((str(root), str(root / "src")))
    code = (
        "import json,sys; sys.modules['torch']=None; "
        "from {} import main; raise SystemExit(main(json.loads(sys.argv[1])))".format(
            module_name
        )
    )
    return subprocess.run(
        [sys.executable, "-c", code, json.dumps(arguments)],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def test_shared_preprocessing_is_rgb_letterbox_nchw_float32() -> None:
    from fomo_servo.geometry import letterbox_rgb
    from fomo_servo.inference.preprocessing import preprocess_rgb_image

    image = np.zeros((8, 16, 3), dtype=np.uint8)
    image[:, :, 0] = 255

    prepared = preprocess_rgb_image(image, input_size=16)
    legacy_letterbox, legacy_transform = letterbox_rgb(image, 16)
    legacy_tensor = (
        torch.from_numpy(legacy_letterbox.transpose(2, 0, 1).copy())
        .float()
        .div(255.0)
        .unsqueeze(0)
        .numpy()
    )

    assert prepared.original_image is image
    assert prepared.letterbox_image.shape == (16, 16, 3)
    assert prepared.input_tensor.shape == (1, 3, 16, 16)
    assert prepared.input_tensor.dtype == np.float32
    assert prepared.input_tensor.flags.c_contiguous
    assert prepared.transform == legacy_transform
    np.testing.assert_array_equal(prepared.letterbox_image, legacy_letterbox)
    np.testing.assert_array_equal(prepared.input_tensor, legacy_tensor)
    np.testing.assert_array_equal(
        prepared.input_tensor[0, :, 4:12, :],
        prepared.letterbox_image[4:12].transpose(2, 0, 1).astype(np.float32) / 255.0,
    )
    assert prepared.input_tensor.min() >= 0.0
    assert prepared.input_tensor.max() <= 1.0


def test_onnx_runtime_and_pytorch_match_full_image_pipeline(tmp_path: Path) -> None:
    from fomo_servo.inference.ort_predictor import OnnxRuntimePredictor
    from fomo_servo.inference.predictor import predict_rgb_image
    from fomo_servo.inference.preprocessing import preprocess_rgb_image

    onnx_path, report_path, model = _write_model_and_report(tmp_path)
    predictor = OnnxRuntimePredictor.from_files(onnx_path, report_path)
    image = np.zeros((8, 16, 3), dtype=np.uint8)
    image[:, :, 1] = 64

    prepared = preprocess_rgb_image(image, input_size=16)
    with torch.inference_mode():
        pytorch_logits = model(torch.from_numpy(prepared.input_tensor)).numpy()
    config = SimpleNamespace(
        model=SimpleNamespace(input_size=16, output_stride=8),
        dataset=SimpleNamespace(class_names=("creature",)),
        postprocess=SimpleNamespace(
            inference_threshold=0.4,
            class_thresholds=None,
            component_mode="connected_components",
            confidence_mode="max",
        ),
    )
    pytorch_prediction = predict_rgb_image(
        model,
        image,
        config=config,
        device=torch.device("cpu"),
        confidence_threshold=0.4,
    )
    ort_logits = predictor.predict_logits(prepared.input_tensor)
    ort_prediction = predictor.predict_rgb_image(image)

    np.testing.assert_allclose(ort_logits, pytorch_logits, rtol=1e-6, atol=1e-7)
    np.testing.assert_array_equal(
        ort_prediction.letterbox_image, pytorch_prediction.letterbox_image
    )
    assert ort_prediction.transform == pytorch_prediction.transform
    assert len(ort_prediction.detections) == len(pytorch_prediction.detections)
    for ort_detection, pytorch_detection in zip(
        ort_prediction.detections, pytorch_prediction.detections
    ):
        assert ort_detection.class_id == pytorch_detection.class_id
        assert ort_detection.class_name == pytorch_detection.class_name
        for field in (
            "confidence",
            "mean_confidence",
            "heatmap_x",
            "heatmap_y",
            "input_x",
            "input_y",
            "original_x",
            "original_y",
        ):
            assert getattr(ort_detection, field) == pytest.approx(
                getattr(pytorch_detection, field), abs=1e-6
            )


def test_pipeline_parity_report_covers_preprocess_logits_and_detections(
    tmp_path: Path,
) -> None:
    from fomo_servo.inference.ort_predictor import OnnxRuntimePredictor
    from fomo_servo.inference.parity import compare_rgb_image_pipeline

    onnx_path, report_path, model = _write_model_and_report(tmp_path)
    predictor = OnnxRuntimePredictor.from_files(onnx_path, report_path)
    original_session = predictor.session

    class _CountingSession:
        def __init__(self) -> None:
            self.run_count = 0

        def get_inputs(self):
            return original_session.get_inputs()

        def get_outputs(self):
            return original_session.get_outputs()

        def get_providers(self):
            return original_session.get_providers()

        def run(self, *args, **kwargs):
            self.run_count += 1
            return original_session.run(*args, **kwargs)

    counting_session = _CountingSession()
    predictor = OnnxRuntimePredictor(counting_session, predictor.contract)
    image = np.zeros((8, 16, 3), dtype=np.uint8)
    image[:, :, 2] = 96

    report = compare_rgb_image_pipeline(
        model,
        predictor,
        image,
        logits_rtol=1e-6,
        logits_atol=1e-7,
        detection_atol=1e-6,
    )

    assert report["passed"] is True
    assert report["preprocessing"]["passed"] is True
    assert report["preprocessing"]["input_shape"] == [1, 3, 16, 16]
    assert report["logits"]["passed"] is True
    assert report["logits"]["pytorch_shape"] == [1, 2, 2, 2]
    assert report["logits"]["onnxruntime_shape"] == [1, 2, 2, 2]
    assert report["detections"]["passed"] is True
    assert report["detections"]["pytorch_count"] == report["detections"][
        "onnxruntime_count"
    ]
    assert counting_session.run_count == 1


def test_onnx_runtime_predictor_rejects_model_sha_mismatch(tmp_path: Path) -> None:
    from fomo_servo.inference.ort_predictor import OnnxRuntimePredictor, OrtPredictorError

    onnx_path, report_path, _model = _write_model_and_report(tmp_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["onnx_sha256"] = "0" * 64
    report_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(OrtPredictorError, match="SHA-256 mismatch"):
        OnnxRuntimePredictor.from_files(onnx_path, report_path)


def test_onnx_runtime_predictor_rejects_input_shape(tmp_path: Path) -> None:
    from fomo_servo.inference.ort_predictor import OnnxRuntimePredictor, OrtPredictorError

    onnx_path, report_path, _model = _write_model_and_report(tmp_path)
    predictor = OnnxRuntimePredictor.from_files(onnx_path, report_path)

    with pytest.raises(OrtPredictorError, match="input shape"):
        predictor.predict_logits(np.zeros((1, 3, 8, 8), dtype=np.float32))


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (lambda report: report.update({"onnx_opset": 16}), "onnx_opset must be 17"),
        (
            lambda report: report["postprocess"].update(
                {"component_mode": "local_peaks"}
            ),
            "postprocess.component_mode",
        ),
        (
            lambda report: report["postprocess"].update(
                {"class_thresholds": "invalid"}
            ),
            "postprocess.class_thresholds",
        ),
    ),
)
def test_onnx_runtime_predictor_rejects_invalid_sidecar_contract(
    tmp_path: Path, mutate: object, message: str
) -> None:
    from fomo_servo.inference.ort_predictor import (
        OnnxRuntimePredictor,
        OrtPredictorError,
    )

    onnx_path, report_path, _model = _write_model_and_report(tmp_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    mutate(report)
    report_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(OrtPredictorError, match=message):
        OnnxRuntimePredictor.from_files(onnx_path, report_path)


def test_onnx_runtime_predictor_requires_checkpoint_provenance(
    tmp_path: Path,
) -> None:
    from fomo_servo.inference.ort_predictor import (
        OnnxRuntimePredictor,
        OrtPredictorError,
    )

    onnx_path, report_path, _model = _write_model_and_report(tmp_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    del report["checkpoint_sha256"]
    report_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(OrtPredictorError, match="checkpoint_sha256"):
        OnnxRuntimePredictor.from_files(onnx_path, report_path)


def test_parity_rejects_postprocess_sidecar_drift(tmp_path: Path) -> None:
    from scripts.verify_onnx_pipeline_parity import _validate_contract_pair
    from fomo_servo.inference.ort_predictor import OnnxRuntimePredictor
    from fomo_servo.inference.parity import PipelineParityError

    onnx_path, report_path, _model = _write_model_and_report(tmp_path)
    original = OnnxRuntimePredictor.from_files(onnx_path, report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["postprocess"]["class_thresholds"] = [1.0]
    report_path.write_text(json.dumps(report), encoding="utf-8")
    drifted = OnnxRuntimePredictor.from_files(onnx_path, report_path)

    with pytest.raises(PipelineParityError, match="contract mismatch"):
        _validate_contract_pair(original.contract, drifted)


def test_sidecar_normalizes_json_numeric_class_threshold_keys(tmp_path: Path) -> None:
    from fomo_servo.inference.ort_predictor import load_ort_model_contract

    _onnx_path, report_path, _model = _write_model_and_report(tmp_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["postprocess"]["class_thresholds"] = {"0": 0.6}
    report_path.write_text(json.dumps(report), encoding="utf-8")

    contract = load_ort_model_contract(report_path)

    assert contract.class_thresholds == {0: 0.6}


def test_contract_pair_normalizes_sequence_class_thresholds(tmp_path: Path) -> None:
    from dataclasses import replace

    from scripts.verify_onnx_pipeline_parity import _validate_contract_pair
    from fomo_servo.inference.ort_predictor import OnnxRuntimePredictor

    onnx_path, report_path, _model = _write_model_and_report(tmp_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["postprocess"]["class_thresholds"] = [0.4]
    report_path.write_text(json.dumps(report), encoding="utf-8")
    predictor = OnnxRuntimePredictor.from_files(onnx_path, report_path)
    export_contract = replace(predictor.contract, class_thresholds=[0.4])

    _validate_contract_pair(export_contract, predictor)


def test_parity_video_rejects_early_decode_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import scripts.verify_onnx_pipeline_parity as parity_script

    class _TruncatedCapture:
        def __init__(self) -> None:
            self.read_count = 0

        def isOpened(self) -> bool:
            return True

        def get(self, property_id: int) -> float:
            if property_id == cv2.CAP_PROP_FRAME_COUNT:
                return 2.0
            return 0.0

        def read(self):
            self.read_count += 1
            if self.read_count == 1:
                return True, np.zeros((8, 16, 3), dtype=np.uint8)
            return False, None

        def release(self) -> None:
            pass

    monkeypatch.setattr(parity_script.cv2, "VideoCapture", lambda _path: _TruncatedCapture())
    monkeypatch.setattr(
        parity_script,
        "compare_rgb_image_pipeline",
        lambda *args, **kwargs: {"passed": True},
    )

    with pytest.raises(parity_script.PipelineParityError, match="expected 2"):
        parity_script._compare_video(
            object(),
            object(),  # type: ignore[arg-type]
            tmp_path / "truncated.avi",
            max_frames=2,
            logits_rtol=1e-4,
            logits_atol=1e-5,
            pytorch_contract=object(),
        )


def test_predict_image_cli_runs_with_onnx_and_sidecar_only(tmp_path: Path) -> None:
    from scripts.predict_image import main

    onnx_path, report_path, _model = _write_model_and_report(tmp_path)
    image_path = tmp_path / "input.png"
    output_image = tmp_path / "prediction.png"
    output_json = tmp_path / "prediction.json"
    image_bgr = np.zeros((8, 16, 3), dtype=np.uint8)
    image_bgr[:, :, 1] = 64
    assert cv2.imwrite(str(image_path), image_bgr)

    exit_code = main(
        [
            "--onnx",
            str(onnx_path),
            "--onnx-report",
            str(report_path),
            "--image",
            str(image_path),
            "--output-image",
            str(output_image),
            "--output-json",
            str(output_json),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["runtime"] == "onnxruntime"
    assert payload["model_sha256"] == _sha256(onnx_path)
    assert payload["image_width"] == 16
    assert payload["image_height"] == 8
    assert output_image.is_file()


def test_predict_image_cli_refuses_to_overwrite_onnx(tmp_path: Path) -> None:
    from scripts.predict_image import main

    onnx_path, report_path, _model = _write_model_and_report(tmp_path)
    image_path = tmp_path / "input.png"
    assert cv2.imwrite(str(image_path), np.zeros((8, 16, 3), dtype=np.uint8))
    original_sha256 = _sha256(onnx_path)

    exit_code = main(
        [
            "--onnx", str(onnx_path),
            "--onnx-report", str(report_path),
            "--image", str(image_path),
            "--output-image", str(onnx_path),
            "--output-json", str(tmp_path / "prediction.json"),
        ]
    )

    assert exit_code == 1
    assert _sha256(onnx_path) == original_sha256


def test_predict_video_cli_runs_with_onnx_and_sidecar_only(tmp_path: Path) -> None:
    from scripts.predict_video import main

    onnx_path, report_path, _model = _write_model_and_report(tmp_path)
    input_video = tmp_path / "input.avi"
    writer = cv2.VideoWriter(
        str(input_video), cv2.VideoWriter_fourcc(*"MJPG"), 5.0, (16, 8)
    )
    assert writer.isOpened()
    try:
        for index in range(4):
            frame = np.zeros((8, 16, 3), dtype=np.uint8)
            frame[:, :, 1] = 32 + index
            writer.write(frame)
    finally:
        writer.release()

    output_dir = tmp_path / "nested" / "video-output"
    output_video = output_dir / "prediction.mp4"
    output_csv = output_dir / "prediction.csv"
    output_jsonl = output_dir / "prediction.jsonl"
    exit_code = main(
        [
            "--onnx",
            str(onnx_path),
            "--onnx-report",
            str(report_path),
            "--source",
            str(input_video),
            "--output-video",
            str(output_video),
            "--output-csv",
            str(output_csv),
            "--output-jsonl",
            str(output_jsonl),
        ]
    )

    assert exit_code == 0
    assert output_video.is_file() and output_video.stat().st_size > 0
    csv_lines = output_csv.read_text(encoding="utf-8").splitlines()
    jsonl_lines = output_jsonl.read_text(encoding="utf-8").splitlines()
    assert len(csv_lines) >= 2
    assert len(jsonl_lines) == len(csv_lines) - 1
    assert all(json.loads(line)["runtime"] == "onnxruntime" for line in jsonl_lines)


def test_predict_video_cli_rejects_zero_frame_source(tmp_path: Path) -> None:
    from scripts.predict_video import main

    onnx_path, report_path, _model = _write_model_and_report(tmp_path)
    input_video = tmp_path / "empty.avi"
    writer = cv2.VideoWriter(
        str(input_video), cv2.VideoWriter_fourcc(*"MJPG"), 5.0, (16, 8)
    )
    assert writer.isOpened()
    writer.release()

    exit_code = main(
        [
            "--onnx", str(onnx_path),
            "--onnx-report", str(report_path),
            "--source", str(input_video),
            "--output-video", str(tmp_path / "output.mp4"),
            "--output-csv", str(tmp_path / "output.csv"),
            "--output-jsonl", str(tmp_path / "output.jsonl"),
        ]
    )

    assert exit_code == 1


def test_onnx_image_and_video_clis_execute_without_torch(tmp_path: Path) -> None:
    onnx_path, report_path, _model = _write_model_and_report(tmp_path)
    image_path = tmp_path / "input.png"
    assert cv2.imwrite(str(image_path), np.zeros((8, 16, 3), dtype=np.uint8))
    image_result = _run_cli_without_torch(
        "scripts.predict_image",
        [
            "--onnx", str(onnx_path),
            "--onnx-report", str(report_path),
            "--image", str(image_path),
            "--output-image", str(tmp_path / "image-output.png"),
            "--output-json", str(tmp_path / "image-output.json"),
        ],
    )
    assert image_result.returncode == 0, image_result.stderr

    input_video = tmp_path / "input.avi"
    writer = cv2.VideoWriter(
        str(input_video), cv2.VideoWriter_fourcc(*"MJPG"), 5.0, (16, 8)
    )
    assert writer.isOpened()
    writer.write(np.zeros((8, 16, 3), dtype=np.uint8))
    writer.release()
    video_result = _run_cli_without_torch(
        "scripts.predict_video",
        [
            "--onnx", str(onnx_path),
            "--onnx-report", str(report_path),
            "--source", str(input_video),
            "--output-video", str(tmp_path / "video-output.mp4"),
            "--output-csv", str(tmp_path / "video-output.csv"),
            "--output-jsonl", str(tmp_path / "video-output.jsonl"),
        ],
    )
    assert video_result.returncode == 0, video_result.stderr
