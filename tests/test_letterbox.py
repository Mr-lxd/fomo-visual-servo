from __future__ import annotations

import importlib

import pytest


@pytest.fixture
def letterbox_api():
    try:
        module = importlib.import_module("fomo_servo.geometry.letterbox")
    except ModuleNotFoundError:
        module = None
    assert module is not None, "fomo_servo.geometry.letterbox must be importable"
    return module


@pytest.mark.parametrize(
    ("width", "height", "point"),
    [
        (160, 80, (143.25, 17.5)),
        (80, 160, (12.75, 140.5)),
        (100, 100, (50.0, 50.0)),
    ],
    ids=("landscape", "portrait", "square"),
)
def test_letterbox_point_round_trip_stays_within_one_pixel(
    letterbox_api, width: int, height: int, point: tuple[float, float]
) -> None:
    transform_type = getattr(letterbox_api, "LetterboxTransform", None)
    assert transform_type is not None, "LetterboxTransform must be available"

    transform = transform_type.from_image_size(width, height, input_size=192)
    letterbox_point = transform.forward_point(*point)
    restored_point = transform.inverse_point(*letterbox_point)

    assert restored_point[0] == pytest.approx(point[0], abs=1.0)
    assert restored_point[1] == pytest.approx(point[1], abs=1.0)
    assert transform.resized_width <= 192
    assert transform.resized_height <= 192
    assert transform.resized_width == 192 or transform.resized_height == 192


@pytest.mark.parametrize(
    ("width", "height", "point"),
    [
        (160, 80, (143.25, 17.5)),
        (80, 160, (12.75, 140.5)),
        (100, 100, (50.0, 50.0)),
    ],
    ids=("landscape", "portrait", "square"),
)
def test_continuous_heatmap_coordinate_inverse_stays_within_one_pixel(
    letterbox_api, width: int, height: int, point: tuple[float, float]
) -> None:
    transform_type = getattr(letterbox_api, "LetterboxTransform", None)
    assert transform_type is not None, "LetterboxTransform must be available"

    transform = transform_type.from_image_size(width, height, input_size=192)
    letterbox_point = transform.forward_point(*point)
    heatmap_point = transform.letterbox_to_heatmap(*letterbox_point, stride=8)
    restored_point = transform.heatmap_to_original(*heatmap_point, stride=8)

    assert restored_point[0] == pytest.approx(point[0], abs=1.0)
    assert restored_point[1] == pytest.approx(point[1], abs=1.0)
