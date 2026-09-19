from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from fomo_servo.vision.service import VisionServiceConfig


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "vision_live.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("vision_live_script", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_slice1_service_defaults_match_frozen_hardware_target() -> None:
    config = VisionServiceConfig()

    assert config.source == "/dev/video0"
    assert config.width == 640
    assert config.height == 480
    assert config.fps == 25.0
    assert config.fourcc == "YUYV"
    assert config.bind_host == "0.0.0.0"
    assert config.port == 47010
    assert config.jpeg_quality == 80
    assert config.send_buffer_bytes == 64 * 1024
    assert config.write_timeout == 0.5
    assert config.control_port == 47011
    assert config.capture_output_root == Path("datasets_raw/robobeetle")
    assert config.capture_queue_bytes == 64 * 1024 * 1024
    assert config.capture_min_free_bytes == 512 * 1024 * 1024


def test_cli_defaults_map_to_service_config_with_separate_capture_control() -> None:
    module = _load_script()
    parser = module.build_parser()
    args = parser.parse_args([])
    config = module.service_config_from_args(args)

    assert config == VisionServiceConfig()
    destinations = {action.dest for action in parser._actions}
    assert "control_port" in destinations
    assert "capture_output_root" in destinations
    assert "capture_queue_mib" in destinations
    assert "capture_min_free_mib" in destinations
    assert "device" not in destinations
    assert "authority" not in destinations
    assert "rbrp" not in destinations
    assert "stm32" not in destinations


def test_cli_forwards_both_inference_artifacts_exactly() -> None:
    module = _load_script()
    onnx_path = Path("artifacts/model.onnx")
    report_path = Path("artifacts/report.json")

    args = module.build_parser().parse_args(
        [
            "--inference-onnx",
            str(onnx_path),
            "--inference-report",
            str(report_path),
        ]
    )

    config = module.service_config_from_args(args)

    assert config.inference_onnx == onnx_path
    assert config.inference_report == report_path


@pytest.mark.parametrize(
    "argv",
    [
        ["--inference-onnx", "artifacts/model.onnx"],
        ["--inference-report", "artifacts/report.json"],
    ],
)
def test_main_rejects_incomplete_inference_pair_before_service_startup(
    argv: list[str], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _load_script()
    constructed: list[object] = []

    def unexpected_service(config: object) -> object:
        constructed.append(config)
        raise AssertionError("VisionService construction was reached")

    monkeypatch.setattr(module, "VisionService", unexpected_service)

    with pytest.raises(SystemExit) as error:
        module.main(argv)

    assert error.value.code == 2
    assert constructed == []
    assert "inference_onnx and inference_report" in capsys.readouterr().err


@pytest.mark.parametrize(
    "unsupported_option", ["--confidence-threshold", "--strategy", "--tracking"]
)
def test_parser_rejects_unsupported_inference_options(unsupported_option: str) -> None:
    module = _load_script()

    with pytest.raises(SystemExit) as error:
        module.build_parser().parse_args([unsupported_option])

    assert error.value.code == 2
