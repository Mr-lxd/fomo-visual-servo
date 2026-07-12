"""Visualize the disabled augmentation framework and its downstream outputs.

This command intentionally does not implement any augmentation algorithm.  It
loads one sample through the same train-only no-op pipeline used by the
dataset, then renders original boxes, augmentation output, letterbox boxes,
and the stride heatmap with decoded grid centroids.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from fomo_servo.config import AugmentationConfig  # noqa: E402
from fomo_servo.datasets import (  # noqa: E402
    AbsoluteBox,
    YOLOv5FOMODataset,
    FOMOSample,
    decode_class_index_heatmap,
)
from fomo_servo.geometry.letterbox import letterbox_rgb  # noqa: E402


COLORS = ((0, 255, 0), (0, 128, 255), (255, 0, 255), (255, 255, 0))


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--input-size", type=int, default=192)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument(
        "--class-mode", choices=("merge_single", "preserve"), default="preserve"
    )
    parser.add_argument("--merged-class-name", default="creature")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args(argv)


def _draw_boxes(
    image_rgb: np.ndarray, boxes: Sequence[AbsoluteBox], class_names: Sequence[str]
) -> np.ndarray:
    """Draw boxes on RGB ``uint8 [S,S,3]`` and return a BGR panel."""

    panel = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    for box in boxes:
        color = COLORS[box.foreground_class_id % len(COLORS)]
        cv2.rectangle(
            panel,
            (round(box.x_min), round(box.y_min)),
            (round(box.x_max), round(box.y_max)),
            color,
            2,
        )
        cv2.putText(
            panel,
            class_names[box.foreground_class_id],
            (round(box.x_min), max(14, round(box.y_min) - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    return panel


def _panel_title(panel: np.ndarray, title: str) -> np.ndarray:
    """Add a small title to a BGR ``uint8 [S,S,3]`` panel."""

    cv2.rectangle(panel, (0, 0), (panel.shape[1], 24), (32, 32, 32), -1)
    cv2.putText(
        panel,
        title,
        (6, 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return panel


def _display_image_and_boxes(
    image: np.ndarray,
    boxes: Sequence[AbsoluteBox],
    class_names: Sequence[str],
    input_size: int,
) -> np.ndarray:
    """Letterbox a pre-letterbox image for display and transform its boxes."""

    display_image, display_transform = letterbox_rgb(image, input_size)
    display_boxes = []
    for box in boxes:
        coordinates = display_transform.forward_box(
            box.x_min, box.y_min, box.x_max, box.y_max
        )
        display_boxes.append(AbsoluteBox(box.foreground_class_id, *coordinates))
    return _draw_boxes(display_image, display_boxes, class_names)


def _heatmap_panel(sample: FOMOSample, input_size: int, stride: int) -> np.ndarray:
    """Render ``sample.heatmap.class_index [G,G]`` as BGR ``uint8 [S,S,3]``."""

    class_index = sample.heatmap.class_index
    grid_height, grid_width = class_index.shape
    grid = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
    for value in np.unique(class_index):
        if value > 0:
            grid[class_index == value] = COLORS[(int(value) - 1) % len(COLORS)]
    panel = cv2.resize(grid, (input_size, input_size), interpolation=cv2.INTER_NEAREST)
    for decoded in decode_class_index_heatmap(class_index):
        center = (round((decoded.grid_x + 0.5) * stride), round((decoded.grid_y + 0.5) * stride))
        cv2.drawMarker(
            panel,
            center,
            (255, 255, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=10,
            thickness=2,
        )
    return panel


def build_visualization(
    sample: FOMOSample,
    class_names: Sequence[str],
    input_size: int,
    stride: int,
) -> np.ndarray:
    """Create BGR ``uint8 [2*S,2*S,3]`` framework diagnostic output."""

    panels = (
        _panel_title(
            _display_image_and_boxes(
                sample.original_image,
                sample.original_boxes,
                class_names,
                input_size,
            ),
            "original bbox",
        ),
        _panel_title(
            _display_image_and_boxes(
                sample.augmented_image,
                sample.augmented_boxes,
                class_names,
                input_size,
            ),
            "augmentation output (disabled)",
        ),
        _panel_title(
            _draw_boxes(sample.letterbox_image, sample.letterbox_boxes, class_names),
            "letterbox bbox",
        ),
        _panel_title(_heatmap_panel(sample, input_size, stride), "heatmap + decoded centroid"),
    )
    return cv2.vconcat((cv2.hconcat(panels[:2]), cv2.hconcat(panels[2:])))


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Render one sample without model inference or training."""

    args = _parse_args(argv)
    dataset = YOLOv5FOMODataset(
        args.dataset_root,
        split=args.split,
        input_size=args.input_size,
        stride=args.stride,
        class_mode=args.class_mode,
        merged_class_name=args.merged_class_name,
        augmentation=AugmentationConfig.disabled(),
        train_split="train",
    )
    sample = dataset[args.index]
    visualization = build_visualization(
        sample, dataset.class_names, args.input_size, args.stride
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), visualization):
        raise RuntimeError("unable to write visualization: {}".format(args.output))
    print("Wrote {}".format(args.output))
    if args.show:
        cv2.imshow("FOMO augmentation framework", visualization)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
