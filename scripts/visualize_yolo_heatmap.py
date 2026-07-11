"""Visualize YOLO boxes, letterbox geometry, FOMO heatmaps, and decoded centroids."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from fomo_servo.datasets.heatmap import decode_class_index_heatmap
from fomo_servo.datasets.yolo import AbsoluteBox, YOLOv5FOMODataset
from fomo_servo.geometry.letterbox import letterbox_rgb


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


def _color_for_foreground(foreground_class_id: int) -> Tuple[int, int, int]:
    return COLORS[foreground_class_id % len(COLORS)]


def _draw_boxes(
    image_rgb: np.ndarray, boxes: Sequence[AbsoluteBox], class_names: Sequence[str]
) -> np.ndarray:
    panel = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    for box in boxes:
        color = _color_for_foreground(box.foreground_class_id)
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


def _add_title(panel: np.ndarray, title: str) -> np.ndarray:
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 24), (32, 32, 32), thickness=-1)
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


def _original_display_panel(
    image_rgb: np.ndarray,
    boxes: Sequence[AbsoluteBox],
    class_names: Sequence[str],
    input_size: int,
) -> Tuple[np.ndarray, object]:
    display_image, display_transform = letterbox_rgb(image_rgb, input_size)
    display_boxes = []
    for box in boxes:
        x_min, y_min, x_max, y_max = display_transform.forward_box(
            box.x_min, box.y_min, box.x_max, box.y_max
        )
        display_boxes.append(
            AbsoluteBox(box.foreground_class_id, x_min, y_min, x_max, y_max)
        )
    return _draw_boxes(display_image, display_boxes, class_names), display_transform


def _heatmap_panel(class_index: np.ndarray, input_size: int) -> np.ndarray:
    grid_height, grid_width = class_index.shape
    color_grid = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
    for class_index_value in np.unique(class_index):
        if class_index_value == 0:
            continue
        color_grid[class_index == class_index_value] = _color_for_foreground(
            int(class_index_value) - 1
        )
    return cv2.resize(color_grid, (input_size, input_size), interpolation=cv2.INTER_NEAREST)


def build_visualization(sample, class_names: Sequence[str], input_size: int, stride: int) -> np.ndarray:
    """Create a BGR ``uint8 [2*S,2*S,3]`` four-panel diagnostic image."""

    original_panel, display_transform = _original_display_panel(
        sample.original_image, sample.original_boxes, class_names, input_size
    )
    letterbox_panel = _draw_boxes(
        sample.letterbox_image, sample.letterbox_boxes, class_names
    )
    heatmap_panel = _heatmap_panel(sample.heatmap.class_index, input_size)
    decoded_panel, _ = _original_display_panel(
        sample.original_image, (), class_names, input_size
    )
    for decoded in decode_class_index_heatmap(sample.heatmap.class_index):
        original_x, original_y = sample.transform.grid_cell_center_to_original(
            decoded.grid_x, decoded.grid_y, stride
        )
        panel_x, panel_y = display_transform.forward_point(original_x, original_y)
        color = _color_for_foreground(decoded.class_index - 1)
        cv2.drawMarker(
            decoded_panel,
            (round(panel_x), round(panel_y)),
            color,
            markerType=cv2.MARKER_CROSS,
            markerSize=12,
            thickness=2,
        )
        cv2.putText(
            decoded_panel,
            class_names[decoded.class_index - 1],
            (round(panel_x) + 4, max(14, round(panel_y) - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    panels = (
        _add_title(original_panel, "original bbox"),
        _add_title(letterbox_panel, "letterbox bbox"),
        _add_title(heatmap_panel, "stride heatmap"),
        _add_title(decoded_panel, "decoded centroid"),
    )
    return cv2.vconcat((cv2.hconcat(panels[:2]), cv2.hconcat(panels[2:])))


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Render one dataset sample without invoking a model or training loop."""

    args = _parse_args(argv)
    dataset = YOLOv5FOMODataset(
        args.dataset_root,
        split=args.split,
        input_size=args.input_size,
        stride=args.stride,
        class_mode=args.class_mode,
        merged_class_name=args.merged_class_name,
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
        cv2.imshow("YOLOv5 FOMO heatmap", visualization)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
