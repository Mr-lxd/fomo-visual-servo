from __future__ import annotations

import pytest

from fomo_servo.vision.mode import VisionMode, VisionModeManager


class _Hub:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def clear(self) -> None:
        self.events.append("hub.clear")


class _Camera:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def start(self):
        self.events.append("camera.start")
        return self

    def stop(self) -> None:
        self.events.append("camera.stop")


class _Server:
    def __init__(self, events: list[str], *, fail_start: bool = False) -> None:
        self.events = events
        self.fail_start = fail_start

    def start(self):
        self.events.append("server.start")
        if self.fail_start:
            raise RuntimeError("bind failed")
        return self

    def stop(self) -> None:
        self.events.append("server.stop")


def test_live_off_transition_closes_stream_before_camera_generation_ends() -> None:
    events: list[str] = []
    manager = VisionModeManager(
        _Hub(events),  # type: ignore[arg-type]
        _Camera(events),  # type: ignore[arg-type]
        _Server(events),  # type: ignore[arg-type]
    )

    manager.set_mode(VisionMode.LIVE)
    assert manager.mode is VisionMode.LIVE
    manager.set_mode(VisionMode.OFF)

    assert manager.mode is VisionMode.OFF
    assert events == [
        "camera.start",
        "server.start",
        "server.stop",
        "camera.stop",
        "hub.clear",
    ]


def test_repeated_same_mode_is_idempotent() -> None:
    events: list[str] = []
    manager = VisionModeManager(
        _Hub(events),  # type: ignore[arg-type]
        _Camera(events),  # type: ignore[arg-type]
        _Server(events),  # type: ignore[arg-type]
    )

    manager.set_mode(VisionMode.OFF)
    manager.set_mode(VisionMode.LIVE)
    manager.set_mode(VisionMode.LIVE)

    assert events == ["camera.start", "server.start"]


def test_server_start_failure_releases_camera_and_stays_off() -> None:
    events: list[str] = []
    manager = VisionModeManager(
        _Hub(events),  # type: ignore[arg-type]
        _Camera(events),  # type: ignore[arg-type]
        _Server(events, fail_start=True),  # type: ignore[arg-type]
    )

    with pytest.raises(RuntimeError, match="bind failed"):
        manager.set_mode(VisionMode.LIVE)

    assert manager.mode is VisionMode.OFF
    assert events == [
        "camera.start",
        "server.start",
        "camera.stop",
        "hub.clear",
    ]


class _FailingStopServer(_Server):
    def stop(self) -> None:
        self.events.append("server.stop")
        raise RuntimeError("server stop failed")


def test_off_still_stops_camera_when_server_stop_fails() -> None:
    events: list[str] = []
    manager = VisionModeManager(
        _Hub(events),  # type: ignore[arg-type]
        _Camera(events),  # type: ignore[arg-type]
        _FailingStopServer(events),  # type: ignore[arg-type]
    )
    manager.set_mode(VisionMode.LIVE)

    with pytest.raises(RuntimeError, match="server stop failed"):
        manager.set_mode(VisionMode.OFF)

    assert manager.mode is VisionMode.OFF
    assert events == [
        "camera.start",
        "server.start",
        "server.stop",
        "camera.stop",
        "hub.clear",
    ]


class _FailingStopCamera(_Camera):
    def stop(self) -> None:
        self.events.append("camera.stop")
        raise RuntimeError("camera stop failed")


def test_start_failure_preserves_server_error_and_still_clears_generation() -> None:
    events: list[str] = []
    manager = VisionModeManager(
        _Hub(events),  # type: ignore[arg-type]
        _FailingStopCamera(events),  # type: ignore[arg-type]
        _Server(events, fail_start=True),  # type: ignore[arg-type]
    )

    with pytest.raises(RuntimeError, match="bind failed") as captured:
        manager.set_mode(VisionMode.LIVE)

    assert captured.value.__cause__ is not None
    assert "camera stop failed" in str(captured.value.__cause__)
    assert manager.mode is VisionMode.OFF
    assert events == [
        "camera.start",
        "server.start",
        "camera.stop",
        "hub.clear",
    ]
