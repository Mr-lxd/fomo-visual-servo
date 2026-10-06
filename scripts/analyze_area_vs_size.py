"""Offline check: does component_area_cells track the apparent size of the target?

Runs the deployed ONNX pipeline (shared letterbox + postprocess via
``OnnxRuntimePredictor.predict_rgb_image``) over a YOLO-labelled image set,
matches detections to ground truth with the repository's centroid evaluator,
and relates ``component_area_cells`` to ``gt_size_px = sqrt(w*h)``.

Read-only on the dataset and the weights. Writes only to ``--output-dir``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from fomo_servo.datasets.yolo import AbsoluteBox, parse_yolo_label_file
from fomo_servo.inference import OnnxRuntimePredictor, read_rgb_image
from fomo_servo.metrics.centroid import CentroidEvaluator, ground_truths_from_boxes

THRESHOLDS = (0.30, 0.40, 0.50)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start
        while end + 1 < len(values) and sorted_values[end + 1] == sorted_values[start]:
            end += 1
        ranks[order[start : end + 1]] = (start + end) / 2.0 + 1.0
        start = end + 1
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    if len(x) < 3:
        return None
    rx, ry = _rank(np.asarray(x, float)), _rank(np.asarray(y, float))
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def size_bins(sizes: np.ndarray, count: int = 5) -> list[tuple[float, float]]:
    edges = np.unique(np.quantile(sizes, np.linspace(0.0, 1.0, count + 1)))
    return [(float(a), float(b)) for a, b in zip(edges[:-1], edges[1:])]


def bin_stats(sizes: np.ndarray, areas: np.ndarray) -> list[dict]:
    if len(sizes) == 0:
        return []
    bins = size_bins(sizes)
    rows = []
    for index, (low, high) in enumerate(bins):
        last = index == len(bins) - 1
        mask = (sizes >= low) & ((sizes <= high) if last else (sizes < high))
        if not mask.any():
            continue
        a = areas[mask]
        rows.append(
            {
                "size_low_px": low,
                "size_high_px": high,
                "n": int(mask.sum()),
                "area_median": float(np.median(a)),
                "area_q1": float(np.percentile(a, 25)),
                "area_q3": float(np.percentile(a, 75)),
            }
        )
    return rows


def is_monotonic(rows: list[dict]) -> bool:
    medians = [row["area_median"] for row in rows]
    return all(b > a for a, b in zip(medians, medians[1:]))


def collect(predictor, images: list[Path], labels_dir: Path, class_names, threshold: float):
    evaluator = CentroidEvaluator(class_names)
    pairs, unmatched_gt, false_positives = [], 0, 0
    for image_path in images:
        image = read_rgb_image(image_path)
        height, width = image.shape[:2]
        detections = predictor.predict_rgb_image(image, confidence_threshold=threshold).detections
        boxes = [
            AbsoluteBox(
                b.source_class_id,
                (b.x_center - b.width / 2) * width,
                (b.y_center - b.height / 2) * height,
                (b.x_center + b.width / 2) * width,
                (b.y_center + b.height / 2) * height,
            )
            for b in parse_yolo_label_file(
                labels_dir / (image_path.stem + ".txt"), len(class_names)
            )
        ]
        gts = ground_truths_from_boxes(boxes, class_names)
        matches, unmatched_predictions, unmatched_gts = evaluator._match(detections, gts)
        unmatched_gt += len(unmatched_gts)
        false_positives += len(unmatched_predictions)
        for p, g, _ in matches:
            gt, det = gts[g], detections[p]
            w, h = gt.x_max - gt.x_min, gt.y_max - gt.y_min
            pairs.append(
                {
                    "image": image_path.name,
                    "class_name": gt.class_name,
                    "gt_w_px": w,
                    "gt_h_px": h,
                    "gt_size_px": float(np.sqrt(w * h)),
                    "component_area_cells": det.component_area_cells,
                    "confidence": det.confidence,
                    "mean_confidence": det.mean_confidence,
                }
            )
    n_gt = len(pairs) + unmatched_gt
    return pairs, {
        "matched": len(pairs),
        "unmatched_gt": unmatched_gt,
        "false_positives": false_positives,
        "match_rate": (len(pairs) / n_gt) if n_gt else None,
    }


def summarize(pairs: list[dict]) -> dict:
    def block(rows):
        sizes = np.array([r["gt_size_px"] for r in rows], float)
        areas = np.array([r["component_area_cells"] for r in rows], float)
        bins = bin_stats(sizes, areas)
        return {
            "n": len(rows),
            "spearman_rho": spearman(sizes, areas),
            "bins": bins,
            "bin_medians_strictly_increasing": is_monotonic(bins) if bins else None,
        }

    result = {"overall": block(pairs), "per_class": {}}
    for name in sorted({r["class_name"] for r in pairs}):
        result["per_class"][name] = block([r for r in pairs if r["class_name"] == name])
    return result


def write_scatter(pairs: list[dict], summary: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    for name in sorted({r["class_name"] for r in pairs}):
        rows = [r for r in pairs if r["class_name"] == name]
        ax.scatter(
            [r["gt_size_px"] for r in rows],
            [r["component_area_cells"] for r in rows],
            s=14,
            alpha=0.6,
            label=f"{name} (n={len(rows)})",
        )
    bins = summary["overall"]["bins"]
    centers = [(b["size_low_px"] + b["size_high_px"]) / 2 for b in bins]
    ax.plot(centers, [b["area_median"] for b in bins], "k-o", label="bin median")
    ax.fill_between(
        centers,
        [b["area_q1"] for b in bins],
        [b["area_q3"] for b in bins],
        color="k",
        alpha=0.15,
        label="bin IQR",
    )
    rho = summary["overall"]["spearman_rho"]
    ax.set_xlabel("gt_size_px = sqrt(w*h) in the original frame")
    ax.set_ylabel("component_area_cells")
    ax.set_title(
        "Area vs size (training images) rho={}".format("n/a" if rho is None else f"{rho:.2f}")
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True, help="ONNX sidecar JSON")
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--unlabeled-images", type=Path, help="optional: count detections and area only")
    args = parser.parse_args(argv)

    predictor = OnnxRuntimePredictor.from_files(args.onnx, args.report)
    class_names = predictor.contract.class_names
    images = sorted(p for p in args.images.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "onnx_sha256": predictor.contract.onnx_sha256,
        "images": len(images),
        "thresholds": {},
    }
    for threshold in THRESHOLDS:
        pairs, counts = collect(predictor, images, args.labels, class_names, threshold)
        entry = {**counts, **summarize(pairs)}
        summary["thresholds"][f"{threshold:.2f}"] = entry
        if threshold == 0.40:
            with (args.output_dir / "pairs.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(pairs[0].keys()) if pairs else ["image"])
                writer.writeheader()
                writer.writerows(pairs)
            write_scatter(pairs, entry, args.output_dir / "scatter_area_vs_size.png")

    if args.unlabeled_images is not None:
        frames = sorted(
            p for p in args.unlabeled_images.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES
        )
        counts, areas = [], []
        for frame in frames:
            detections = predictor.predict_rgb_image(read_rgb_image(frame), confidence_threshold=0.40).detections
            counts.append(len(detections))
            areas.extend(d.component_area_cells for d in detections)
        summary["unlabeled_frames"] = {
            "path": str(args.unlabeled_images),
            "frames": len(frames),
            "frames_with_detection": int(sum(c > 0 for c in counts)),
            "detections": len(areas),
            "area_median": float(np.median(areas)) if areas else None,
            "area_q1_q3": [float(np.percentile(areas, 25)), float(np.percentile(areas, 75))] if areas else None,
            "area_min_max": [int(min(areas)), int(max(areas))] if areas else None,
        }

    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
