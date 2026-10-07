"""Task 13 offline bbox-fill CV evaluation; train-only annotation access."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_centernet_cv import _load_cfg, _sweep  # noqa: E402
from run_epoch_study import EPOCHS, select  # noqa: E402
from fomo_servo.centernet.annotations import class_names, load_pool_samples  # noqa: E402
from fomo_servo.centernet.evaluation import Pred, adjacent_pairs, evaluate_threshold, match_image, spearman  # noqa: E402
from fomo_servo.datasets.lab_pool_view import CLASS_MAPPING  # noqa: E402
from fomo_servo.geometry.letterbox import LetterboxTransform  # noqa: E402
from fomo_servo.postprocess import postprocess_numpy_probabilities  # noqa: E402
from fomo_servo.postprocess.connected_components import find_connected_components  # noqa: E402

FAMILIES = ("B", "B-box", "B-box50")
SOURCE_CLASSES = ("jellyfish", "fish", "tuna")
SIZE_GROUPS = ("<80", "80-<150", "150-<250", "250-<350", ">=350")


def source_metadata(sample, input_size: int) -> tuple[list[str], LetterboxTransform]:
    """Recover source labels and geometry from train JSON or unlabelled images."""
    annotation = sample.image_path.with_suffix(".json")
    if not annotation.is_file():
        import cv2

        image = cv2.imread(str(sample.image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError("unable to read background image: {}".format(sample.image_path))
        height, width = image.shape[:2]
        return [], LetterboxTransform.from_image_size(width, height, input_size)
    payload = json.loads(annotation.read_text(encoding="utf-8"))
    labels = [s["label"] for s in payload.get("shapes", []) if CLASS_MAPPING[s["label"]] is not None]
    if len(labels) != len(sample.boxes):
        raise ValueError("source-label alignment mismatch: {}".format(sample.image_path.name))
    transform = LetterboxTransform.from_image_size(int(payload["imageWidth"]), int(payload["imageHeight"]), input_size)
    return labels, transform


def probabilities(logits: np.ndarray) -> np.ndarray:
    """Float32 ``[N,C,G,G]`` softmax, identical to shared NumPy logits entry point.

    The shared API does not expose its softmax separately. Compute its exact
    formula once, then use shared probability decoding and shared components.
    """
    values = logits.astype(np.float32, copy=False)
    shifted = values - values.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def decode(probs, transforms, cfg, threshold):
    """Shared 8-neighbour decode; return detections retaining component area."""
    return postprocess_numpy_probabilities(
        probs, class_names=class_names(), stride=cfg["model"]["output_stride"],
        transforms=transforms, confidence_threshold=threshold,
    )


def predictions(detections):
    """Convert shared detections to original-pixel evaluation predictions."""
    return [[Pred(d.class_id, d.confidence, d.original_x, d.original_y) for d in ds] for ds in detections]


def gt_size(box):
    """GT geometric mean side length in original-image pixels."""
    return float(np.sqrt((box.x_max - box.x_min) * (box.y_max - box.y_min)))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def merge_stats(probs, samples, transforms, threshold: float, stride: int, distance: float) -> dict:
    """Merge iff same-class GT centres occupy one threshold component.

    Ignore GT is excluded. Map centres to half-open grid cells by floor after
    letterbox transform; off-grid centres cannot merge. Numerator is same-class
    pairs only; report both all-adjacent and same-model-class denominators.
    This diagnoses component bridging, independent of centroid matching.
    """
    total = same_class = merged = 0
    for probability, sample, transform in zip(probs, samples, transforms):
        scored = [g for g in sample.boxes if g.visibility != "ignore"]
        component_ids = {}
        for cid in {g.class_id for g in scored}:
            ids = np.full(probability.shape[-2:], -1, dtype=np.int64)
            for ci, component in enumerate(find_connected_components(probability[cid + 1] >= threshold, connectivity=8)):
                for x, y in component.cells:
                    ids[y, x] = ci
            component_ids[cid] = ids

        def component_of(g):
            x, y = transform.forward_point((g.x_min + g.x_max) / 2, (g.y_min + g.y_max) / 2)
            ix, iy = int(np.floor(x / stride)), int(np.floor(y / stride))
            ids = component_ids[g.class_id]
            return int(ids[iy, ix]) if 0 <= iy < ids.shape[0] and 0 <= ix < ids.shape[1] else -1

        for i, j in adjacent_pairs(sample.boxes, distance):
            total += 1
            if scored[i].class_id == scored[j].class_id:
                same_class += 1
                ci, cj = component_of(scored[i]), component_of(scored[j])
                merged += ci >= 0 and ci == cj
    return {"merged_pairs": int(merged), "all_adjacent_pairs": total, "same_model_class_adjacent_pairs": same_class,
            "merged_fraction_all_adjacent": merged / total if total else float("nan"),
            "merged_fraction_same_model_class": merged / same_class if same_class else float("nan")}


def detailed(detections, samples, sources, boundaries, family, epoch, threshold):
    """Return agnostic recall groups and matched area/size records; ignore excluded."""
    groups = {(scope, label): [0, 0] for scope, labels in (
        ("source_class", SOURCE_CLASSES), ("size", SIZE_GROUPS),
        ("source_class_size", tuple(c + "/" + s for c in SOURCE_CLASSES for s in SIZE_GROUPS)),
    ) for label in labels}
    rows = []
    for ds, sample, labels in zip(detections, samples, sources):
        scored = [(g, label) for g, label in zip(sample.boxes, labels) if g.visibility != "ignore"]
        ps = [Pred(d.class_id, d.confidence, d.original_x, d.original_y) for d in ds]
        match = match_image(ps, sample.boxes, class_agnostic=True)
        hits = {gi for _, gi, _ in match.pairs}
        for gi, (g, label) in enumerate(scored):
            size_group = SIZE_GROUPS[int(np.searchsorted(boundaries, gt_size(g), side="right"))]
            for key in (("source_class", label), ("size", size_group), ("source_class_size", label + "/" + size_group)):
                groups[key][1] += 1
                groups[key][0] += gi in hits
        for pi, gi, distance in match.pairs:
            g, label = scored[gi]
            rows.append({"family": family, "epoch": epoch, "threshold": threshold,
                         "image": sample.image_path.name, "source_class": label, "visibility": g.visibility,
                         "gt_size_sqrt_area_px": gt_size(g), "component_area_cells": ds[pi].component_area_cells,
                         "gt_class_id": g.class_id, "pred_class_id": ds[pi].class_id, "center_error_px": distance})
    recall = [{"family": family, "epoch": epoch, "threshold": threshold, "scope": scope, "group": label,
               "hit": hit, "total": total, "recall": hit / total if total else float("nan")}
              for (scope, label), (hit, total) in groups.items()]
    correlations = []
    for visibility in ("all_nonignore", "full_only"):
        for label in ("all",) + SOURCE_CLASSES:
            selected_rows = [r for r in rows if (visibility != "full_only" or r["visibility"] == "full")
                             and (label == "all" or r["source_class"] == label)]
            correlations.append({"family": family, "epoch": epoch, "threshold": threshold,
                                 "visibility": visibility, "source_class": label, "n": len(selected_rows),
                                 "spearman_rho": spearman([r["component_area_cells"] for r in selected_rows],
                                                          [r["gt_size_sqrt_area_px"] for r in selected_rows])})
    return recall, correlations, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "dataset-root", "baseline-work", "work", "results"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    cfg = _load_cfg(args.config)
    samples = load_pool_samples(args.dataset_root, use_visibility=True)
    by_name = {s.image_path.name: s for s in samples}
    metadata = {s.image_path.name: source_metadata(s, cfg["model"]["input_size"]) for s in samples}
    boundaries = np.asarray((80.0, 150.0, 250.0, 350.0))  # Task10 original-pixel bins, lower inclusive
    thresholds = sorted(set(_sweep(cfg) + [0.6]))
    folds = cfg["data"]["held_out_sessions"]
    distance = cfg["data"]["distance_threshold_px"]
    data = {}
    pooled_samples, pooled_sources, pooled_transforms = None, None, None
    sweep_rows = []
    tables = {}
    for family in FAMILIES:
        family_samples, family_sources, family_transforms = [], [], []
        outputs = {epoch: [] for epoch in EPOCHS}
        for fold in folds:
            directory = (args.baseline_work if family == "B" else args.work) / (fold + "__" + family)
            meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
            names = meta["test_images_list"]
            heldout = [by_name[name] for name in names]
            if any(s.session != fold for s in heldout):
                raise ValueError("held-out image/session mismatch in {}".format(directory))
            family_samples.extend(heldout)
            family_sources.extend(metadata[name][0] for name in names)
            family_transforms.extend(metadata[name][1] for name in names)
            for epoch in EPOCHS:
                with np.load(directory / "outputs_e{}.npz".format(epoch)) as archive:
                    raw = archive["outputs"]
                if len(raw) != len(names):
                    raise ValueError("output/image count mismatch in {} e{}".format(directory, epoch))
                outputs[epoch].append(raw)
        if pooled_samples is None:
            pooled_samples, pooled_sources, pooled_transforms = family_samples, family_sources, family_transforms
        elif [s.image_path.name for s in family_samples] != [s.image_path.name for s in pooled_samples]:
            raise ValueError("family held-out image order differs")
        data[family] = {epoch: probabilities(np.concatenate(outputs[epoch])) for epoch in EPOCHS}
        tables[family] = {}
        for epoch in EPOCHS:
            for threshold in thresholds:
                dets = decode(data[family][epoch], pooled_transforms, cfg, threshold)
                ps = predictions(dets)
                gs = [s.boxes for s in pooled_samples]
                metrics = evaluate_threshold(ps, gs, class_agnostic=True, adjacent_distance=distance)
                aware = evaluate_threshold(ps, gs, class_agnostic=False, adjacent_distance=distance)
                row = {"family": family, "epoch": epoch, "threshold": threshold,
                       **{key: metrics[key] for key in ("tp", "fp", "fn", "precision", "recall", "f1")},
                       **{"class_aware_" + key: aware[key] for key in ("precision", "recall", "f1")}}
                sweep_rows.append(row)
                tables[family][(epoch, threshold)] = row
        print("threshold scan complete:", family, flush=True)
    selected = {f: select({k: v["f1"] for k, v in tables[f].items()}) for f in FAMILIES}
    recall_rows, correlation_rows, matched_rows, merge_rows = [], [], [], []
    for family in FAMILIES:
        for epoch in EPOCHS:
            threshold = selected[family][1]
            points = [threshold] + ([0.6] if family == "B" and epoch == 150 and threshold != 0.6 else [])
            for threshold in points:
                probs = data[family][epoch]
                dets = decode(probs, pooled_transforms, cfg, threshold)
                recall, corr, matched = detailed(dets, pooled_samples, pooled_sources, boundaries, family, epoch, threshold)
                recall_rows.extend(recall); correlation_rows.extend(corr); matched_rows.extend(matched)
                merge_rows.append({"family": family, "epoch": epoch, "threshold": threshold,
                                   **merge_stats(probs, pooled_samples, pooled_transforms, threshold,
                                                 cfg["model"]["output_stride"], distance)})
    out = args.results
    out.mkdir(parents=True, exist_ok=True)
    for filename, rows in (("threshold_scan.csv", sweep_rows), ("recall_groups.csv", recall_rows),
                           ("area_size_correlation.csv", correlation_rows), ("matched_area_size.csv", matched_rows),
                           ("adjacent_merge.csv", merge_rows)):
        write_csv(out / filename, rows)
    baseline = tables["B"][(150, 0.6)]
    stability = {}
    for family in FAMILIES:
        stability[family] = {}
        for visibility in ("all_nonignore", "full_only"):
            values = [r["spearman_rho"] for r in correlation_rows if r["family"] == family
                      and r["threshold"] == selected[family][1] and r["source_class"] == "all"
                      and r["visibility"] == visibility]
            finite = np.asarray([v for v in values if np.isfinite(v)])
            stability[family][visibility] = {
                "defined_epochs": len(finite), "total_epochs": len(EPOCHS),
                "min_rho": float(finite.min()) if len(finite) else float("nan"),
                "max_rho": float(finite.max()) if len(finite) else float("nan"),
                "std_rho": float(finite.std()) if len(finite) else float("nan"),
                "all_six_epochs_rho_ge_0.6": bool(len(finite) == len(EPOCHS) and np.all(finite >= 0.6)),
            }
    summary = {
        "selection_rule": "Task08: max pooled agnostic F1; within <0.01 of max -> fewer epochs then threshold nearest0.4",
        "selection_optimism": "same four CV folds select and report; no independent labelled test set",
        "matching": "shared centroid-in-box greedy nearest, class-agnostic; ignored GT excluded",
        "size_metric": "component_area_cells vs GT sqrt(area) original pixels; matched-only correlation",
        "size_group_boundaries_px": boundaries.tolist(), "size_group_rule": "Task10 fixed80/150/250/350 pixels, lower inclusive",
        "merge_definition": "same model-class GT centres in same shared8-neighbour threshold component; all adjacent and same-class denominators",
        "adjacent_distance_px": distance, "baseline_B150_at_0.60": baseline,
        "selected": {f: {**tables[f][k], "f1_difference_vs_B150_at_0.60": tables[f][k]["f1"] - baseline["f1"]} for f, k in selected.items()},
        "rho_stability_at_family_selected_threshold": stability,
        "recall_groups": recall_rows,
        "selected_threshold_correlations": correlation_rows, "selected_threshold_merges": merge_rows,
    }
    def finite_json(value):
        if isinstance(value, dict):
            return {k: finite_json(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [finite_json(v) for v in value]
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value
    (out / "summary.json").write_text(json.dumps(finite_json(summary), indent=2, allow_nan=False), encoding="utf-8")
    make_plots(out, selected, tables, correlation_rows, matched_rows, boundaries)
    print(json.dumps(summary["selected"], indent=2), flush=True)


def make_plots(out, selected, tables, correlations, matches, boundaries):
    """Export epoch curves and matched-only GT-bin median component areas."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for family in FAMILIES:
        threshold = selected[family][1]
        axes[0].plot(EPOCHS, [tables[family][(e, threshold)]["f1"] for e in EPOCHS], marker="o", label=family + " selected thr")
        axes[0].plot(EPOCHS, [max(v["f1"] for (ep, _), v in tables[family].items() if ep == e) for e in EPOCHS], ls="--", alpha=0.5)
        for vis, style in (("all_nonignore", "-"), ("full_only", "--")):
            axes[1].plot(EPOCHS, [next(r["spearman_rho"] for r in correlations if r["family"] == family and r["epoch"] == e
                                      and r["threshold"] == threshold and r["source_class"] == "all" and r["visibility"] == vis) for e in EPOCHS],
                         marker="o", ls=style, label=family + " " + vis)
    axes[0].axhline(tables["B"][(150, 0.6)]["f1"], color="black", ls=":", label="B150@0.60")
    axes[0].set_ylabel("Pooled agnostic F1"); axes[1].set_ylabel("Spearman rho (matched targets)")
    axes[1].axhline(0.6, color="black", ls=":")
    for ax in axes:
        ax.set_xlabel("Epoch"); ax.grid(alpha=0.25); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / "f1_rho_vs_epochs.png", dpi=160); plt.close(fig)
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharex=True)
    bins_rows = []
    for vi, visibility in enumerate(("all_nonignore", "full_only")):
        for ci, label in enumerate(("all",) + SOURCE_CLASSES):
            ax = axes[vi, ci]
            for family in FAMILIES:
                epoch, threshold = (150, 0.6) if family == "B" else selected[family]
                rows = [r for r in matches if r["family"] == family and r["epoch"] == epoch and r["threshold"] == threshold
                        and (label == "all" or r["source_class"] == label) and (visibility != "full_only" or r["visibility"] == "full")]
                xs, ys = [], []
                for bin_index in range(len(SIZE_GROUPS)):
                    bucket = [r for r in rows if int(np.searchsorted(boundaries, r["gt_size_sqrt_area_px"], side="right")) == bin_index]
                    x = float(np.median([r["gt_size_sqrt_area_px"] for r in bucket])) if bucket else float("nan")
                    y = float(np.median([r["component_area_cells"] for r in bucket])) if bucket else float("nan")
                    xs.append(x); ys.append(y)
                    bins_rows.append({"family": family, "epoch": epoch, "threshold": threshold, "visibility": visibility,
                                      "source_class": label, "size_group": SIZE_GROUPS[bin_index], "n": len(bucket),
                                      "median_gt_size_px": x, "median_component_area_cells": y})
                ax.plot(xs, ys, marker="o", label=family)
            ax.set_title(label + "/" + visibility, fontsize=9); ax.grid(alpha=0.25); ax.legend(fontsize=7)
            if vi == 1:
                ax.set_xlabel("Median GT sqrt(area), px")
            if ci == 0:
                ax.set_ylabel("Median component area, cells")
    fig.tight_layout(); fig.savefig(out / "area_size_binned_medians.png", dpi=160); plt.close(fig)
    write_csv(out / "area_size_binned_medians.csv", bins_rows)


if __name__ == "__main__":
    main()
