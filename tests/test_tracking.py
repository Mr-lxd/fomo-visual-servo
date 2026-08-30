from __future__ import annotations

from fomo_servo.postprocess import Detection, TargetTracker


def _detection(x: float) -> Detection:
    return Detection(
        class_id=0,
        class_name="creature",
        confidence=0.9,
        mean_confidence=0.8,
        component_area_cells=1,
        heatmap_x=x / 8.0,
        heatmap_y=1.0,
        input_x=x,
        input_y=8.0,
        original_x=x,
        original_y=8.0,
    )


def test_tracker_detected_lost_reacquired_state_machine() -> None:
    tracker = TargetTracker(
        strategy="nearest_previous", max_match_distance=20.0, max_lost_frames=2
    )

    assert tracker.update((_detection(10.0),)).status == "detected"
    assert tracker.update((_detection(12.0),)).status == "detected"
    lost = tracker.update(())
    assert lost.status == "lost"
    assert lost.lost_frames == 1
    assert tracker.update((_detection(14.0),)).status == "reacquired"


def test_tracker_returns_idle_after_max_lost_frames() -> None:
    tracker = TargetTracker(strategy="highest_confidence", max_lost_frames=1)
    tracker.update((_detection(10.0),))
    tracker.update(())
    assert tracker.update(()).status == "idle"
