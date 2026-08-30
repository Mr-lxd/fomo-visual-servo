from __future__ import annotations

import pytest

from fomo_servo.metrics import SequenceStatistics, normalize_centroid
from fomo_servo.postprocess import Detection


def _detection(x: float, y: float = 20.0) -> Detection:
    return Detection(
        class_id=0,
        class_name="creature",
        confidence=0.9,
        mean_confidence=0.9,
        component_area_cells=1,
        heatmap_x=x / 8.0,
        heatmap_y=y / 8.0,
        input_x=x,
        input_y=y,
        original_x=x,
        original_y=y,
    )


def test_normalized_coordinates_have_expected_signs() -> None:
    assert normalize_centroid(0.0, 0.0, 100, 100) == (-1.0, -1.0)
    assert normalize_centroid(50.0, 50.0, 100, 100) == pytest.approx((0.0, 0.0))
    assert normalize_centroid(99.0, 99.0, 100, 100) == pytest.approx((0.98, 0.98))


def test_sequence_statistics_report_jitter_availability_and_reacquisition() -> None:
    statistics = SequenceStatistics()
    statistics.update("detected", _detection(10.0), 100, 100)
    statistics.update("detected", _detection(13.0), 100, 100)
    statistics.update("lost", None, 100, 100)
    statistics.update("reacquired", _detection(15.0), 100, 100)

    result = statistics.summary()
    assert result.processed_frame_count == 4
    assert result.detection_availability == pytest.approx(3.0 / 4.0)
    assert result.target_loss_rate == pytest.approx(1.0 / 4.0)
    assert result.reacquisition_count == 1
    assert result.jitter_mean_pixels > 0.0
