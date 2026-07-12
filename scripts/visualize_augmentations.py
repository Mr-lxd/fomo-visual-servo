"""Visualize train-only color jitter and write a deterministic audit contact sheet."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Optional, Sequence

import cv2
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from fomo_servo.config import AugmentationConfig, ProjectConfig, load_config  # noqa: E402
from fomo_servo.datasets import (  # noqa: E402
    AbsoluteBox,
    ColorJitterFactors,
    FOMOSample,
    YOLOv5FOMODataset,
    apply_color_jitter,
    decode_class_index_heatmap,
)
from fomo_servo.geometry.letterbox import LetterboxTransform, letterbox_rgb  # noqa: E402


COLORS = ((0, 255, 0), (0, 128, 255), (255, 0, 255), (255, 255, 0))


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--num-images", type=int, default=16)
    parser.add_argument("--input-size", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument(
        "--class-mode", choices=("merge_single", "preserve"), default=None
    )
    parser.add_argument("--merged-class-name", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
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
    """Add a title to a BGR ``uint8 [S,S,3]`` panel."""

    cv2.rectangle(panel, (0, 0), (panel.shape[1], 24), (32, 32, 32), -1)
    cv2.putText(
        panel,
        title[:48],
        (6, 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
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
    display_boxes = _transform_boxes(boxes, display_transform)
    return _draw_boxes(display_image, display_boxes, class_names)


def _transform_boxes(
    boxes: Sequence[AbsoluteBox], transform: LetterboxTransform
) -> tuple[AbsoluteBox, ...]:
    """Map pixel-space boxes through one letterbox transform."""

    transformed = []
    for box in boxes:
        coordinates = transform.forward_box(
            box.x_min, box.y_min, box.x_max, box.y_max
        )
        transformed.append(AbsoluteBox(box.foreground_class_id, *coordinates))
    return tuple(transformed)


def _draw_decoded_centroids(
    panel: np.ndarray, sample: FOMOSample, stride: int
) -> np.ndarray:
    """Draw decoded grid-cell centers on a BGR ``uint8 [S,S,3]`` panel."""

    for decoded in decode_class_index_heatmap(sample.heatmap.class_index):
        center = (
            round((decoded.grid_x + 0.5) * stride),
            round((decoded.grid_y + 0.5) * stride),
        )
        color = COLORS[(decoded.class_index - 1) % len(COLORS)]
        cv2.drawMarker(
            panel,
            center,
            color,
            markerType=cv2.MARKER_CROSS,
            markerSize=10,
            thickness=2,
        )
    return panel


def _heatmap_panel(sample: FOMOSample, input_size: int, stride: int) -> np.ndarray:
    """Render class-index heatmap ``[G,G]`` as BGR ``uint8 [S,S,3]``."""

    class_index = sample.heatmap.class_index
    grid_height, grid_width = class_index.shape
    grid = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
    for value in np.unique(class_index):
        if value > 0:
            grid[class_index == value] = COLORS[(int(value) - 1) % len(COLORS)]
    panel = cv2.resize(grid, (input_size, input_size), interpolation=cv2.INTER_NEAREST)
    return _draw_decoded_centroids(panel, sample, stride)


def build_visualization(
    sample: FOMOSample,
    class_names: Sequence[str],
    input_size: int,
    stride: int,
) -> np.ndarray:
    """Create legacy BGR ``uint8 [2*S,2*S,3]`` diagnostic output."""

    panels = (
        _panel_title(
            _display_image_and_boxes(
                sample.original_image, sample.original_boxes, class_names, input_size
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
            "color jitter output",
        ),
        _panel_title(
            _draw_boxes(sample.letterbox_image, sample.letterbox_boxes, class_names),
            "letterbox bbox",
        ),
        _panel_title(
            _heatmap_panel(sample, input_size, stride),
            "heatmap + decoded centroid",
        ),
    )
    return cv2.vconcat((cv2.hconcat(panels[:2]), cv2.hconcat(panels[2:])))


def _render_variant_panel(
    image: np.ndarray,
    boxes: Sequence[AbsoluteBox],
    sample: FOMOSample,
    class_names: Sequence[str],
    input_size: int,
    stride: int,
    title: str,
    draw_center_line: bool = False,
) -> np.ndarray:
    """Render one RGB variant with its unchanged bbox and centroid heatmap."""

    letterboxed, transform = letterbox_rgb(image, input_size)
    panel = _draw_boxes(
        letterboxed, _transform_boxes(boxes, transform), class_names
    )
    panel = _draw_decoded_centroids(panel, sample, stride)
    if draw_center_line:
        center_x = input_size // 2
        cv2.line(panel, (center_x, 0), (center_x, input_size - 1), (255, 255, 255), 1)
    return _panel_title(panel, title)


def _factor_text(factors: ColorJitterFactors) -> str:
    return "b{:.2f} c{:.2f} s{:.2f} h{:+.3f}".format(
        factors.brightness_factor,
        factors.contrast_factor,
        factors.saturation_factor,
        factors.hue_shift,
    )


def _metadata_record(
    dataset: YOLOv5FOMODataset,
    index: int,
    seed: int,
    case: str,
    sample: FOMOSample,
    *,
    sampled: bool = True,
) -> dict[str, Any]:
    """Convert one sample's color metadata to a JSON-safe record."""

    metadata = sample.augmentation_metadata
    return {
        "relative_image_path": dataset.image_paths[index]
        .relative_to(dataset.root)
        .as_posix(),
        "seed": int(seed),
        "case": case,
        "sampled": sampled,
        "applied": bool(metadata.applied),
        "brightness_factor": float(metadata.brightness_factor),
        "contrast_factor": float(metadata.contrast_factor),
        "saturation_factor": float(metadata.saturation_factor),
        "hue_shift": float(metadata.hue_shift),
    }


def _boundary_factors(config: Any) -> tuple[ColorJitterFactors, ...]:
    """Return minimum, neutral, maximum factors from YAML parameter magnitudes."""

    color = config.augmentation.color_jitter
    return (
        ColorJitterFactors(
            1.0 - color.brightness,
            1.0 - color.contrast,
            1.0 - color.saturation,
            -color.hue,
        ),
        ColorJitterFactors.neutral(),
        ColorJitterFactors(
            1.0 + color.brightness,
            1.0 + color.contrast,
            1.0 + color.saturation,
            color.hue,
        ),
    )


def _metadata_from_factors(factors: ColorJitterFactors) -> Any:
    from fomo_servo.datasets.augmentation import ColorJitterMetadata

    return ColorJitterMetadata(
        applied=factors != ColorJitterFactors.neutral(),
        brightness_factor=factors.brightness_factor,
        contrast_factor=factors.contrast_factor,
        saturation_factor=factors.saturation_factor,
        hue_shift=factors.hue_shift,
    )


def _serialize_boxes(boxes: Sequence[AbsoluteBox]) -> list[dict[str, float | int]]:
    """Serialize continuous xyxy boxes without absolute image paths."""

    return [
        {
            "class_id": int(box.foreground_class_id),
            "x_min": float(box.x_min),
            "y_min": float(box.y_min),
            "x_max": float(box.x_max),
            "y_max": float(box.y_max),
        }
        for box in boxes
    ]


def _serialize_centroids(
    boxes: Sequence[AbsoluteBox],
) -> list[dict[str, float | int]]:
    """Serialize bbox centers in original-image continuous pixel coordinates."""

    return [
        {
            "class_id": int(box.foreground_class_id),
            "x": float(box.center[0]),
            "y": float(box.center[1]),
        }
        for box in boxes
    ]


def _horizontal_flip_record(
    dataset: YOLOv5FOMODataset,
    index: int,
    seed: int,
    case: str,
    sample: FOMOSample,
) -> dict[str, Any]:
    """Build the aug02 JSON record with original and transformed geometry."""

    metadata = sample.augmentation_metadata
    return {
        "relative_image_path": dataset.image_paths[index]
        .relative_to(dataset.root)
        .as_posix(),
        "seed": int(seed),
        "case": case,
        "horizontal_flip_applied": bool(metadata.horizontal_flip_applied),
        "original_boxes": _serialize_boxes(sample.original_boxes),
        "flipped_boxes": _serialize_boxes(sample.augmented_boxes),
        "original_centroids": _serialize_centroids(sample.original_boxes),
        "flipped_centroids": _serialize_centroids(sample.augmented_boxes),
        "color_jitter_factors": {
            "brightness_factor": float(metadata.brightness_factor),
            "contrast_factor": float(metadata.contrast_factor),
            "saturation_factor": float(metadata.saturation_factor),
            "hue_shift": float(metadata.hue_shift),
        },
    }


def _dataset_with_augmentation(
    dataset: YOLOv5FOMODataset, augmentation: AugmentationConfig
) -> YOLOv5FOMODataset:
    """Clone dataset geometry settings with one visualization-only augmentation config."""

    return YOLOv5FOMODataset(
        dataset.root,
        split=dataset.split,
        input_size=dataset.input_size,
        stride=dataset.stride,
        class_mode=dataset.class_mode,
        merged_class_name=dataset.class_names[0],
        collision_policy=dataset.collision_policy,
        augmentation=augmentation,
        train_split=dataset.train_split,
        augmentation_seed=dataset.augmentation_seed,
    )


def render_horizontal_flip_contact_sheet(
    dataset: YOLOv5FOMODataset,
    config: ProjectConfig,
    output_dir: Path,
    *,
    num_images: int,
    seed: int,
) -> tuple[Path, Path]:
    """Render aug02 geometry panels with neutral forced-flip controls."""

    if num_images <= 0:
        raise ValueError("num_images must be positive")
    if len(dataset) < num_images:
        raise ValueError(
            "requested {} images but split contains only {}".format(
                num_images, len(dataset)
            )
        )

    neutral_color = replace(
        config.augmentation.color_jitter,
        enabled=False,
        probability=0.0,
        brightness=0.0,
        contrast=0.0,
        saturation=0.0,
        hue=0.0,
    )
    forced_flip_config = replace(
        config.augmentation,
        color_jitter=neutral_color,
        horizontal_flip=replace(
            config.augmentation.horizontal_flip,
            enabled=True,
            probability=1.0,
        ),
    )
    raw_dataset = _dataset_with_augmentation(
        dataset, AugmentationConfig.disabled()
    )
    forced_dataset = _dataset_with_augmentation(dataset, forced_flip_config)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    records = []
    for index in range(num_images):
        image_seed = seed + index * 10
        raw = raw_dataset.get_sample(index, np.random.default_rng(image_seed))
        forced = forced_dataset.get_sample(index, np.random.default_rng(image_seed))
        variants = [
            dataset.get_sample(index, np.random.default_rng(image_seed + variant))
            for variant in range(4)
        ]
        panels = [
            _render_variant_panel(
                raw.original_image,
                raw.original_boxes,
                raw,
                dataset.class_names,
                dataset.input_size,
                dataset.stride,
                "original",
                draw_center_line=True,
            ),
            _render_variant_panel(
                forced.augmented_image,
                forced.augmented_boxes,
                forced,
                dataset.class_names,
                dataset.input_size,
                dataset.stride,
                "forced flip neutral",
                draw_center_line=True,
            ),
        ]
        records.append(_horizontal_flip_record(dataset, index, image_seed, "original", raw))
        records.append(
            _horizontal_flip_record(dataset, index, image_seed, "forced_flip", forced)
        )
        for variant, sample in enumerate(variants):
            metadata = sample.augmentation_metadata
            factors = ColorJitterFactors(
                metadata.brightness_factor,
                metadata.contrast_factor,
                metadata.saturation_factor,
                metadata.hue_shift,
            )
            case = "typical" if variant == 0 else "random{}".format(variant)
            panels.append(
                _render_variant_panel(
                    sample.augmented_image,
                    sample.augmented_boxes,
                    sample,
                    dataset.class_names,
                    dataset.input_size,
                    dataset.stride,
                    "{} {}".format(case, _factor_text(factors)),
                    draw_center_line=True,
                )
            )
            records.append(
                _horizontal_flip_record(dataset, index, image_seed + variant, case, sample)
            )
        rows.append(np.hstack(panels))

    contact_sheet = np.vstack(rows)
    contact_path = output_dir / "horizontal_flip_contact_sheet.jpg"
    json_path = output_dir / "horizontal_flip_samples.json"
    if not cv2.imwrite(str(contact_path), contact_sheet):
        raise RuntimeError("unable to write contact sheet: {}".format(contact_path))
    json_path.write_text(
        json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return contact_path, json_path


def render_contact_sheet(
    dataset: YOLOv5FOMODataset,
    config: ProjectConfig,
    output_dir: Path,
    *,
    num_images: int,
    seed: int,
) -> tuple[Path, Path]:
    """Render 16-by-default samples and return contact-sheet/JSON paths."""

    if num_images <= 0:
        raise ValueError("num_images must be positive")
    if len(dataset) < num_images:
        raise ValueError(
            "requested {} images but split contains only {}".format(
                num_images, len(dataset)
            )
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    records = []
    boundary_names = ("minimum", "neutral", "maximum")
    for index in range(num_images):
        image_seed = seed + index * 10
        variant_samples = [
            dataset.get_sample(index, np.random.default_rng(image_seed + variant))
            for variant in range(4)
        ]
        reference = variant_samples[0]
        panels = [
            _render_variant_panel(
                reference.original_image,
                reference.original_boxes,
                reference,
                dataset.class_names,
                dataset.input_size,
                dataset.stride,
                "original",
            )
        ]
        records.append(
            {
                "relative_image_path": dataset.image_paths[index]
                .relative_to(dataset.root)
                .as_posix(),
                "seed": int(image_seed),
                "case": "original",
                "sampled": False,
                "applied": False,
                "brightness_factor": 1.0,
                "contrast_factor": 1.0,
                "saturation_factor": 1.0,
                "hue_shift": 0.0,
            }
        )
        for variant, sample in enumerate(variant_samples):
            metadata = sample.augmentation_metadata
            factors = ColorJitterFactors(
                metadata.brightness_factor,
                metadata.contrast_factor,
                metadata.saturation_factor,
                metadata.hue_shift,
            )
            panels.append(
                _render_variant_panel(
                    sample.augmented_image,
                    sample.augmented_boxes,
                    sample,
                    dataset.class_names,
                    dataset.input_size,
                    dataset.stride,
                    "{} {}".format(
                        "typical" if variant == 0 else "random{}".format(variant),
                        _factor_text(factors),
                    ),
                )
            )
            records.append(
                _metadata_record(
                    dataset,
                    index,
                    image_seed + variant,
                    "typical" if variant == 0 else "random{}".format(variant),
                    sample,
                )
            )

        for name, factors in zip(boundary_names, _boundary_factors(config)):
            boundary_image = apply_color_jitter(reference.original_image, factors)
            boundary_sample = reference
            metadata = _metadata_from_factors(factors)
            panels.append(
                _render_variant_panel(
                    boundary_image,
                    reference.original_boxes,
                    boundary_sample,
                    dataset.class_names,
                    dataset.input_size,
                    dataset.stride,
                    "{} {}".format(name, _factor_text(factors)),
                )
            )
            records.append(
                {
                    "relative_image_path": dataset.image_paths[index]
                    .relative_to(dataset.root)
                    .as_posix(),
                    "seed": int(image_seed),
                    "case": name,
                    "sampled": False,
                    "applied": bool(metadata.applied),
                    "brightness_factor": float(metadata.brightness_factor),
                    "contrast_factor": float(metadata.contrast_factor),
                    "saturation_factor": float(metadata.saturation_factor),
                    "hue_shift": float(metadata.hue_shift),
                }
            )
        rows.append(np.hstack(panels))

    contact_sheet = np.vstack(rows)
    contact_path = output_dir / "color_jitter_contact_sheet.jpg"
    json_path = output_dir / "color_jitter_samples.json"
    if not cv2.imwrite(str(contact_path), contact_sheet):
        raise RuntimeError("unable to write contact sheet: {}".format(contact_path))
    json_path.write_text(
        json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return contact_path, json_path


def _load_dataset_and_context(
    args: argparse.Namespace,
) -> tuple[YOLOv5FOMODataset, Optional[ProjectConfig], int]:
    """Load config-driven or legacy CLI dataset settings without hard-coded paths."""

    if args.config is not None:
        config = load_config(args.config)
        root = args.dataset_root or config.dataset.root
        input_size = args.input_size or config.model.input_size
        stride = args.stride or config.model.output_stride
        class_mode = args.class_mode or config.dataset.class_mode
        merged_name = args.merged_class_name or config.dataset.merged_class_name
        seed = config.training.seed if args.seed is None else args.seed
        dataset = YOLOv5FOMODataset(
            root,
            split=args.split,
            input_size=input_size,
            stride=stride,
            class_mode=class_mode,
            merged_class_name=merged_name,
            collision_policy=config.dataset.collision_policy,
            augmentation=config.augmentation,
            train_split=config.dataset.train_split,
            augmentation_seed=seed,
        )
        return dataset, config, seed

    if args.dataset_root is None:
        raise ValueError("--dataset-root is required when --config is omitted")
    input_size = args.input_size or 192
    stride = args.stride or 8
    class_mode = args.class_mode or "preserve"
    dataset = YOLOv5FOMODataset(
        args.dataset_root,
        split=args.split,
        input_size=input_size,
        stride=stride,
        class_mode=class_mode,
        merged_class_name=args.merged_class_name or "creature",
        augmentation=AugmentationConfig.disabled(),
        train_split="train",
        augmentation_seed=args.seed or 0,
    )
    return dataset, None, args.seed or 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Render one legacy panel or the config-driven aug01 contact sheet."""

    args = _parse_args(argv)
    dataset, config, seed = _load_dataset_and_context(args)
    if args.output is not None:
        sample = dataset.get_sample(args.index, np.random.default_rng(seed))
        visualization = build_visualization(
            sample, dataset.class_names, dataset.input_size, dataset.stride
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(args.output), visualization):
            raise RuntimeError("unable to write visualization: {}".format(args.output))
        print("Wrote {}".format(args.output))
        if args.show:
            cv2.imshow("FOMO color jitter", visualization)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return 0

    if config is None:
        raise ValueError("--config is required for contact-sheet mode")
    output_dir = args.output_dir or (config.training.output_dir / "visualization")
    if config.augmentation.horizontal_flip.enabled:
        contact_path, json_path = render_horizontal_flip_contact_sheet(
            dataset,
            config,
            output_dir,
            num_images=args.num_images,
            seed=seed,
        )
    else:
        contact_path, json_path = render_contact_sheet(
            dataset,
            config,
            output_dir,
            num_images=args.num_images,
            seed=seed,
        )
    print("Wrote {}".format(contact_path))
    print("Wrote {}".format(json_path))
    if args.show:
        contact_sheet = cv2.imread(str(contact_path), cv2.IMREAD_COLOR)
        if contact_sheet is None:
            raise RuntimeError("unable to read contact sheet: {}".format(contact_path))
        cv2.imshow("FOMO color jitter contact sheet", contact_sheet)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
