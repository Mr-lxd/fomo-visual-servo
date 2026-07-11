from __future__ import annotations

import importlib
import shutil
from pathlib import Path

import numpy as np
import pytest


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "yolo_micro"


@pytest.fixture
def yolo_api():
    try:
        module = importlib.import_module("fomo_servo.datasets.yolo")
    except ModuleNotFoundError:
        module = None
    assert module is not None, "fomo_servo.datasets.yolo must be importable"
    return module


def _sample_by_stem(dataset, stem: str):
    for index, path in enumerate(dataset.image_paths):
        if path.stem == stem:
            return dataset[index]
    raise AssertionError("Fixture image not found: {}".format(stem))


def test_single_class_merge_maps_all_foreground_to_creature(yolo_api) -> None:
    dataset_type = getattr(yolo_api, "YOLOv5FOMODataset", None)
    assert dataset_type is not None, "YOLOv5FOMODataset must be available"

    dataset = dataset_type(
        FIXTURE_ROOT,
        split="train",
        input_size=192,
        stride=8,
        class_mode="merge_single",
        merged_class_name="creature",
    )
    sample = _sample_by_stem(dataset, "multi")

    assert dataset.class_names == ("creature",)
    assert sample.image.shape == (3, 192, 192)
    assert sample.image.dtype == np.float32
    assert sample.heatmap.class_index.shape == (24, 24)
    assert sample.heatmap.one_hot.shape == (2, 24, 24)
    assert set(np.unique(sample.heatmap.class_index)) <= {0, 1}


def test_preserved_multiclass_mode_keeps_foreground_channels(yolo_api) -> None:
    dataset_type = getattr(yolo_api, "YOLOv5FOMODataset", None)
    assert dataset_type is not None, "YOLOv5FOMODataset must be available"

    dataset = dataset_type(
        FIXTURE_ROOT,
        split="train",
        input_size=192,
        stride=8,
        class_mode="preserve",
    )
    sample = _sample_by_stem(dataset, "multi")
    empty_sample = _sample_by_stem(dataset, "empty")
    edge_sample = _sample_by_stem(dataset, "edge")

    assert dataset.class_names == ("fish", "crab")
    assert sample.heatmap.one_hot.shape == (3, 24, 24)
    assert {1, 2}.issubset(set(np.unique(sample.heatmap.class_index)))
    assert np.all(empty_sample.heatmap.class_index == 0)
    assert edge_sample.heatmap.class_index[0, 23] == 2


@pytest.mark.parametrize("stem", ("landscape", "portrait", "square"))
def test_bbox_centroids_round_trip_to_original_within_one_pixel(yolo_api, stem: str) -> None:
    dataset_type = getattr(yolo_api, "YOLOv5FOMODataset", None)
    assert dataset_type is not None, "YOLOv5FOMODataset must be available"

    dataset = dataset_type(
        FIXTURE_ROOT,
        split="train",
        input_size=192,
        stride=8,
        class_mode="preserve",
    )
    sample = _sample_by_stem(dataset, stem)

    for box in sample.original_boxes:
        original_centroid = box.center
        letterbox_centroid = sample.transform.forward_point(*original_centroid)
        restored_centroid = sample.transform.inverse_point(*letterbox_centroid)
        assert restored_centroid[0] == pytest.approx(original_centroid[0], abs=1.0)
        assert restored_centroid[1] == pytest.approx(original_centroid[1], abs=1.0)


def test_invalid_yolo_label_reports_line_and_reason(yolo_api) -> None:
    parser = getattr(yolo_api, "parse_yolo_label_file", None)
    label_error = getattr(yolo_api, "YoloLabelError", None)
    assert callable(parser), "parse_yolo_label_file must be available"
    assert isinstance(label_error, type), "YoloLabelError must be available"

    with pytest.raises(label_error, match="line 1.*x_center"):
        parser(FIXTURE_ROOT / "invalid_labels.txt", num_source_classes=2)


def test_roboflow_layout_resolves_valid_directory_for_val_split(
    yolo_api, tmp_path: Path
) -> None:
    """Roboflow ``valid/images`` must serve the project logical ``val`` split."""

    dataset_type = getattr(yolo_api, "YOLOv5FOMODataset", None)
    assert dataset_type is not None, "YOLOv5FOMODataset must be available"
    root = tmp_path / "roboflow_export"
    for split, source_split, stem in (
        ("train", "train", "landscape"),
        ("valid", "val", "val_square"),
    ):
        image_directory = root / split / "images"
        label_directory = root / split / "labels"
        image_directory.mkdir(parents=True)
        label_directory.mkdir(parents=True)
        shutil.copy2(
            FIXTURE_ROOT / "images" / source_split / "{}.jpg".format(stem),
            image_directory / "{}.jpg".format(stem),
        )
        shutil.copy2(
            FIXTURE_ROOT / "labels" / source_split / "{}.txt".format(stem),
            label_directory / "{}.txt".format(stem),
        )
    (root / "data.yaml").write_text(
        """
train: ../train/images
val: ../valid/images
names: [fish, crab]
""".lstrip(),
        encoding="utf-8",
    )

    train_dataset = dataset_type(
        root, split="train", input_size=96, stride=8, class_mode="preserve"
    )
    validation_dataset = dataset_type(
        root, split="val", input_size=96, stride=8, class_mode="preserve"
    )

    assert train_dataset.images_dir == root / "train" / "images"
    assert train_dataset.labels_dir == root / "train" / "labels"
    assert validation_dataset.images_dir == root / "valid" / "images"
    assert validation_dataset.labels_dir == root / "valid" / "labels"
    assert validation_dataset[0].heatmap.class_index.shape == (12, 12)
