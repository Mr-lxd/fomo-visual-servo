"""Task 14 round 2: does a keypoint run reproduce task 13 B-box50 detection exactly?

Compares, per fold, the training detection-loss history and every cached held-out
logits array of a keypoint work root (control or KP-detach) against the task 13
round 2 B-box50 work root, and optionally re-decodes both at one threshold to
show the detection metrics are identical. Read-only; writes one JSON summary.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from run_centernet_cv import _load_cfg  # noqa: E402
from evaluate_bbox_study import decode, predictions, probabilities  # noqa: E402
from fomo_servo.centernet.evaluation import evaluate_threshold  # noqa: E402
from fomo_servo.centernet.keypoints import load_keypoint_samples  # noqa: E402
from fomo_servo.geometry.letterbox import LetterboxTransform  # noqa: E402

import cv2  # noqa: E402


def image_size(path: Path) -> tuple[int, int]:
    annotation = path.with_suffix(".json")
    if annotation.exists():
        payload = json.loads(annotation.read_text(encoding="utf-8"))
        return int(payload["imageWidth"]), int(payload["imageHeight"])
    height, width = cv2.imread(str(path)).shape[:2]
    return width, height


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "dataset-root", "variant-work", "baseline-work", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--variant-family", default="B-box50-kp")
    parser.add_argument("--baseline-family", default="B-box50")
    parser.add_argument("--threshold", type=float, default=0.96)
    parser.add_argument("--metrics-epoch", type=int, default=250)
    args = parser.parse_args()

    cfg = _load_cfg(args.config)
    issues = []
    samples = load_keypoint_samples(args.dataset_root, issues=issues)
    by_name = {s.image_path.name: s for s in samples}

    folds, identical_history, identical_logits, all_bitwise = [], True, True, True
    metrics = {}
    for fold in cfg["data"]["held_out_sessions"]:
        variant = args.variant_work / "{}__{}".format(fold, args.variant_family)
        baseline = args.baseline_work / "{}__{}".format(fold, args.baseline_family)
        if not (variant / "meta.json").exists() or not (baseline / "meta.json").exists():
            print("skip (missing meta)", fold, flush=True)
            continue
        vmeta = json.loads((variant / "meta.json").read_text(encoding="utf-8"))
        bmeta = json.loads((baseline / "meta.json").read_text(encoding="utf-8"))
        names_equal = vmeta["test_images_list"] == bmeta["test_images_list"]
        if not names_equal:
            identical_history = identical_logits = all_bitwise = False
        shared = min(len(vmeta["history"]), len(bmeta["history"]))
        variant_loss = [float(vmeta["history"][i].get("detection_loss", vmeta["history"][i].get("loss")))
                        for i in range(shared)]
        baseline_loss = [float(bmeta["history"][i].get("detection_loss", bmeta["history"][i].get("loss")))
                         for i in range(shared)]
        loss_diffs = [abs(a - b) for a, b in zip(variant_loss, baseline_loss)]
        exact_loss = sum(a == b for a, b in zip(variant_loss, baseline_loss))
        caches = []
        for path in sorted(variant.glob("outputs_e*.npz")):
            epoch = int(path.stem.split("_e")[1])
            other = baseline / path.name
            if not other.exists():
                continue
            with np.load(path) as z:
                ours = z["outputs"]
            theirs = np.load(other)["outputs"]
            same_shape = ours.shape == theirs.shape
            equal = bool(same_shape and np.array_equal(ours, theirs))
            difference = float(np.max(np.abs(ours - theirs))) if same_shape else float("nan")
            caches.append({"epoch": epoch, "shape": list(ours.shape), "dtype": str(ours.dtype),
                           "bitwise_equal": equal, "max_abs_difference": difference,
                           "sha256_variant": hashlib.sha256(path.read_bytes()).hexdigest(),
                           "sha256_baseline": hashlib.sha256(other.read_bytes()).hexdigest()})
            identical_logits &= equal
            all_bitwise &= equal
        if not caches:
            identical_logits = False
        folds.append({"fold": fold, "test_images_list_equal": names_equal,
                      "history_epochs_compared": shared, "history_epochs_exact": exact_loss,
                      "detection_loss_max_abs_difference": max(loss_diffs) if loss_diffs else None,
                      "caches": caches})

        # Optional end-to-end check: re-decode both caches at one threshold.
        target = variant / "outputs_e{}.npz".format(args.metrics_epoch)
        other_target = baseline / "outputs_e{}.npz".format(args.metrics_epoch)
        if target.exists() and other_target.exists():
            pooled = [by_name[n] for n in vmeta["test_images_list"]]
            transforms = [LetterboxTransform.from_image_size(*image_size(s.image_path), cfg["model"]["input_size"])
                          for s in pooled]
            gt = [s.boxes for s in pooled]
            row = {}
            for label, path in (("variant", target), ("baseline", other_target)):
                with np.load(path) as z:
                    data = probabilities(z["outputs"])
                result = evaluate_threshold(
                    predictions(decode(data, transforms, cfg, args.threshold)), gt,
                    class_agnostic=True, adjacent_distance=cfg["data"]["distance_threshold_px"])
                row[label] = {k: result[k] for k in ("tp", "fp", "fn", "precision", "recall", "f1")}
            row["identical_metrics"] = row["variant"] == row["baseline"]
            all_bitwise &= row["identical_metrics"]
            metrics[fold] = row

    summary = {"variant_work": str(args.variant_work.resolve()),
               "baseline_work": str(args.baseline_work.resolve()),
               "config": str(args.config.resolve()),
               "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
               "folds": folds, "detection_metrics_at_threshold": metrics,
               "threshold": args.threshold, "metrics_epoch": args.metrics_epoch,
               "all_history_epochs_exact": all(f["history_epochs_exact"] == f["history_epochs_compared"] for f in folds),
               "all_held_out_logits_bitwise_equal": bool(identical_logits) and bool(folds),
               "all_detection_evidence_identical": bool(all_bitwise) and bool(folds),
               "annotation_issues": issues}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("all_history_epochs_exact", "all_held_out_logits_bitwise_equal",
                                              "all_detection_evidence_identical")}, indent=2))
    for fold in folds:
        print(fold["fold"], "loss_max_abs_diff", fold["detection_loss_max_abs_difference"],
              "exact_epochs", "{}/{}".format(fold["history_epochs_exact"], fold["history_epochs_compared"]),
              "caches", [(c["epoch"], c["bitwise_equal"], c["max_abs_difference"]) for c in fold["caches"]])


if __name__ == "__main__":
    main()
