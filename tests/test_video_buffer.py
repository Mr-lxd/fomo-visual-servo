from __future__ import annotations

from fomo_servo.inference import FramePacket, LatestFrameBuffer


def test_latest_frame_buffer_drops_stale_packet() -> None:
    buffer = LatestFrameBuffer()
    buffer.put_latest(FramePacket(1, 1.0, "old"))
    buffer.put_latest(FramePacket(2, 2.0, "new"))

    packet = buffer.get(timeout=0.1)

    assert packet is not None
    assert packet.frame_index == 2
    assert packet.frame == "new"
