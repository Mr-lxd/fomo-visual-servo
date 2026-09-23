from __future__ import annotations

import importlib.util
import logging
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import fomo_servo.vision.service as service_module
from fomo_servo.vision.detection_streaming import DETECTION_STREAM_VERSION
from fomo_servo.vision.service import VisionService, VisionServiceConfig


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "vision_live.py"


def test_process_memory_status_reports_rss_and_total_memory_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        service_module.Path,
        "read_text",
        lambda _path, encoding: "100 7 3 2 1 0 0",
    )
    monkeypatch.setattr(
        os,
        "sysconf",
        lambda name: {"SC_PAGE_SIZE": 4096, "SC_PHYS_PAGES": 1000}[name],
        raising=False,
    )

    assert service_module.process_memory_status() == (7 * 4096, 1000 * 4096)


@pytest.mark.parametrize("statm", [OSError("statm unavailable"), "100 malformed"])
def test_process_memory_status_reports_none_for_unavailable_or_malformed_rss(
    monkeypatch: pytest.MonkeyPatch,
    statm: OSError | str,
) -> None:
    def read_statm(_path, encoding):
        if isinstance(statm, OSError):
            raise statm
        return statm

    monkeypatch.setattr(service_module.Path, "read_text", read_statm)
    monkeypatch.setattr(
        os,
        "sysconf",
        lambda name: {"SC_PAGE_SIZE": 4096, "SC_PHYS_PAGES": 1000}[name],
        raising=False,
    )

    assert service_module.process_memory_status() == (None, 1000 * 4096)


def test_process_memory_status_reports_none_when_total_memory_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        service_module.Path,
        "read_text",
        lambda _path, encoding: "100 7 3 2 1 0 0",
    )

    def sysconf(name: str) -> int:
        if name == "SC_PHYS_PAGES":
            raise OSError("physical page count unavailable")
        return 4096

    monkeypatch.setattr(os, "sysconf", sysconf, raising=False)

    assert service_module.process_memory_status() == (7 * 4096, None)


def test_service_inference_status_includes_memory_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        service_module,
        "process_memory_status",
        lambda: (327786496, 4089446400),
        raising=False,
    )
    service = VisionService(VisionServiceConfig())
    try:
        status = service._inference_status()
    finally:
        service.shutdown()

    assert status["vision_process_rss_bytes"] == 327786496
    assert status["system_total_memory_bytes"] == 4089446400


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
    assert config.detection_port == 47012
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
    assert "detection_port" in destinations
    assert "capture_output_root" in destinations
    assert "capture_queue_mib" in destinations
    assert "capture_min_free_mib" in destinations
    assert "device" not in destinations
    assert "authority" not in destinations
    assert "rbrp" not in destinations
    assert "stm32" not in destinations


def test_cli_forwards_detection_port_exactly() -> None:
    module = _load_script()
    args = module.build_parser().parse_args(["--detection-port", "48012"])

    config = module.service_config_from_args(args)

    assert config.detection_port == 48012


def test_service_status_advertises_actual_bound_detection_port() -> None:
    service = VisionService(VisionServiceConfig(detection_port=0))
    service.detection_server.start()
    try:
        status = service._inference_status()
        assert status["configured"] is False
        assert status["detection_stream_supported"] is True
        assert status["detection_stream_version"] == DETECTION_STREAM_VERSION
        assert isinstance(status["detection_stream_port"], int)
        assert status["detection_stream_port"] > 0
        assert status["detection_stream_port"] == service.detection_server.bound_port

        service.detection_server.last_error = RuntimeError(
            "metadata listener failed"
        )
        failed_status = service._inference_status()
        assert failed_status["detection_stream_supported"] is False
        assert failed_status["detection_stream_version"] == DETECTION_STREAM_VERSION
    finally:
        service.detection_server.stop()


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


def test_detection_runtime_failure_warns_without_killing_video_control(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    module = _load_script()
    instances: list[object] = []

    class _Finished:
        def __init__(self) -> None:
            self.calls = 0

        def wait(self, timeout: float) -> bool:
            assert timeout == 0.5
            self.calls += 1
            return self.calls >= 2

    class _Flag:
        def is_set(self) -> bool:
            return False

    class _FakeService:
        def __init__(self, _config) -> None:
            instances.append(self)
            self.shutdown_called = False
            self.camera_owner = SimpleNamespace(
                facts=SimpleNamespace(
                    observed_width=640,
                    observed_height=480,
                    observed_fps=25.0,
                    observed_fourcc="YUYV",
                ),
                finished=_Finished(),
                error=None,
                frames_captured=1,
                measured_capture_fps=25.0,
            )
            self.stream_server = SimpleNamespace(
                bound_port=47010,
                last_error=None,
                last_metrics=None,
                client_connected=_Flag(),
            )
            self.control_server = SimpleNamespace(
                bound_port=47011,
                last_error=None,
            )
            self.detection_server = SimpleNamespace(
                bound_port=47012,
                last_error=RuntimeError("metadata transport failed"),
                last_metrics=None,
                client_connected=_Flag(),
            )
            self.capture_manager = SimpleNamespace(
                status=lambda: {
                    "state": "idle",
                    "recorded_frames": 0,
                    "snapshot_count": 0,
                    "queue_bytes": 0,
                }
            )
            self.hub = SimpleNamespace(snapshot=lambda: None)

        def start_live(self) -> None:
            return None

        def shutdown(self) -> None:
            self.shutdown_called = True

    monkeypatch.setattr(module, "VisionService", _FakeService)

    with caplog.at_level(logging.WARNING):
        assert module.main([]) == 0

    assert len(instances) == 1
    assert instances[0].shutdown_called is True
    assert "video/control remain live" in caplog.text
    assert "metadata transport failed" in caplog.text
