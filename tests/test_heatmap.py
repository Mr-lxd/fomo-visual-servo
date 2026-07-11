from __future__ import annotations

import importlib

import numpy as np
import pytest


@pytest.fixture
def heatmap_api():
    try:
        module = importlib.import_module("fomo_servo.datasets.heatmap")
    except ModuleNotFoundError:
        module = None
    assert module is not None, "fomo_servo.datasets.heatmap must be importable"
    return module


def test_heatmap_has_background_plus_foreground_channels(heatmap_api) -> None:
    generate = getattr(heatmap_api, "generate_fomo_heatmap", None)
    assert callable(generate), "generate_fomo_heatmap must be available"

    target = generate(
        centroids=((16.0, 16.0, 0), (176.0, 176.0, 1)),
        input_size=192,
        stride=8,
        num_foreground_classes=2,
    )

    assert target.class_index.shape == (24, 24)
    assert target.class_index.dtype == np.int64
    assert target.one_hot.shape == (3, 24, 24)
    assert target.one_hot.dtype == np.uint8
    assert target.class_index[2, 2] == 1
    assert target.class_index[22, 22] == 2
    assert np.all(target.one_hot.sum(axis=0) == 1)


def test_empty_heatmap_is_entirely_background(heatmap_api) -> None:
    generate = getattr(heatmap_api, "generate_fomo_heatmap", None)
    assert callable(generate), "generate_fomo_heatmap must be available"

    target = generate(
        centroids=(), input_size=192, stride=8, num_foreground_classes=1
    )

    assert np.all(target.class_index == 0)
    assert np.all(target.one_hot[0] == 1)
    assert np.all(target.one_hot[1] == 0)


def test_conflicting_classes_in_one_cell_raise_clear_error(heatmap_api) -> None:
    generate = getattr(heatmap_api, "generate_fomo_heatmap", None)
    collision_error = getattr(heatmap_api, "HeatmapCollisionError", None)
    assert callable(generate), "generate_fomo_heatmap must be available"
    assert isinstance(collision_error, type), "HeatmapCollisionError must be available"

    with pytest.raises(collision_error, match="different foreground classes"):
        generate(
            centroids=((10.0, 10.0, 0), (11.0, 11.0, 1)),
            input_size=192,
            stride=8,
            num_foreground_classes=2,
        )


def test_conflicting_classes_can_keep_first_cell_label_explicitly(heatmap_api) -> None:
    """Configured keep-first policy preserves a deterministic class and counts loss."""

    generate = getattr(heatmap_api, "generate_fomo_heatmap", None)
    assert callable(generate), "generate_fomo_heatmap must be available"

    target = generate(
        centroids=((10.0, 10.0, 0), (11.0, 11.0, 1)),
        input_size=192,
        stride=8,
        num_foreground_classes=2,
        collision_policy="keep_first",
    )

    assert target.class_index[1, 1] == 1
    assert target.different_class_collision_count == 1
