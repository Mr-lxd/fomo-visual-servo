"""Task13 round2: evaluate cached train-only CV logits, never train or overwrite round1.

``cached`` scans the original six snapshots. ``final`` additionally reads ONLY
B-box50 e250/e300 from --extended-work; old snapshots always retain their original
cache provenance. Outputs are exclusive results/round2/{cached,final} directories.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from evaluate_bbox_study import (
    FAMILIES, SIZE_GROUPS, SOURCE_CLASSES, class_names, decode, detailed, gt_size, merge_stats,
    predictions, probabilities, source_metadata, write_csv,
)
from run_centernet_cv import _load_cfg
from run_epoch_study import EPOCHS, select
from fomo_servo.centernet.annotations import IMAGE_SUFFIXES, load_pool_samples, session_of
from fomo_servo.centernet.evaluation import evaluate_threshold

EXTRA_EPOCHS = (250, 300)
LATE_EPOCHS = (100, 150, 200, 250, 300)
THRESHOLDS = tuple(round(i / 100, 3) for i in range(5, 96, 5)) + tuple(
    round(i / 1000, 3) for i in range(960, 996, 5)
)
BOUNDARIES = np.asarray((80.0, 150.0, 250.0, 350.0))


def digest(path: Path) -> str:
    """Return SHA256 of an input file without changing it."""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def finite_json(value):
    """Convert undefined scalar metrics to JSON null, recursively."""
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def gt_ranges(samples, sources):
    """Original-pixel sqrt(area) min/median/max over GT, including missed targets."""
    rows = []
    for visibility in ("all_nonignore", "full_only"):
        for source in ("all",) + SOURCE_CLASSES:
            values = [gt_size(box) for sample, labels in zip(samples, sources)
                      for box, label in zip(sample.boxes, labels)
                      if box.visibility != "ignore"
                      and (visibility != "full_only" or box.visibility == "full")
                      and (source == "all" or label == source)]
            rows.append({"visibility": visibility, "source_class": source, "n": len(values),
                         "min_gt_size_px": min(values) if values else float("nan"),
                         "median_gt_size_px": float(np.median(values)) if values else float("nan"),
                         "max_gt_size_px": max(values) if values else float("nan")})
    return rows


def stability(correlations, scope, family, threshold):
    """Fixed-threshold matched-only rho; missing late snapshots cannot pass."""
    values = {r["epoch"]: r["spearman_rho"] for r in correlations
              if r["selection_scope"] == scope and r["family"] == family
              and r["threshold"] == threshold and r["source_class"] == "all"
              and r["visibility"] == "all_nonignore"}
    available_late = [e for e in LATE_EPOCHS if e in values]
    complete = all(e in values for e in LATE_EPOCHS)
    passes = lambda epochs: bool(all(e in values and np.isfinite(values[e]) and values[e] >= 0.6
                                    for e in epochs))
    return {"fixed_threshold": threshold, "rho_by_epoch": values,
            "reference_old_six_all_ge_0.6": passes(EPOCHS),
            "available_late_epochs": available_late,
            "available_late_all_ge_0.6": passes(available_late),
            "required_late_epochs": list(LATE_EPOCHS),
            "missing_late_epochs": [e for e in LATE_EPOCHS if e not in values],
            "complete_late_standard_confirmed": complete,
            "late_all_five_ge_0.6": passes(LATE_EPOCHS) if complete else None,
            "status": ("pass" if passes(LATE_EPOCHS) else "fail") if complete else
                      "unconfirmed: not extended to 250/300"}


def make_plots(out, selections, tables, correlations, matches):
    """Save fixed-threshold rho/F1 curves and source/size bin medians (cells vs px)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bin_rows = []
    for scope, chosen in selections.items():
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for family, (epoch, threshold) in chosen.items():
            epochs = sorted({e for e, _ in tables[family]})
            axes[0].plot(epochs, [tables[family][(e, threshold)]["f1"] for e in epochs],
                         marker="o", label="{} @{:.3f}".format(family, threshold))
            for visibility, style in (("all_nonignore", "-"), ("full_only", "--")):
                rows = [r for r in correlations if r["selection_scope"] == scope
                        and r["family"] == family and r["source_class"] == "all"
                        and r["visibility"] == visibility and r["threshold"] == threshold]
                axes[1].plot([r["epoch"] for r in rows], [r["spearman_rho"] for r in rows],
                             marker="o", ls=style, label=family + " " + visibility)
        axes[0].axhline(tables["B"][(150, 0.6)]["f1"], color="black", ls=":", label="B150@0.60")
        axes[1].axhline(0.6, color="black", ls=":")
        axes[0].set_ylabel("Pooled agnostic F1")
        axes[1].set_ylabel("Spearman rho (matched GT)")
        for ax in axes:
            ax.set_xlabel("Epoch"); ax.grid(alpha=0.25); ax.legend(fontsize=7)
        fig.suptitle(scope + ": threshold fixed across epochs")
        fig.tight_layout(); fig.savefig(out / (scope + "_f1_rho_vs_epochs.png"), dpi=160); plt.close(fig)

        fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharex=True)
        for vi, visibility in enumerate(("all_nonignore", "full_only")):
            for ci, source in enumerate(("all",) + SOURCE_CLASSES):
                ax = axes[vi, ci]
                # Show all selected models AND the locked B150 comparison point.
                points = [(f, e, t, f) for f, (e, t) in chosen.items()]
                points.append(("B", 150, 0.6, "B150@0.60"))
                for family, epoch, threshold, point in points:
                    rows = [r for r in matches if r["selection_scope"] == scope
                            and r["family"] == family and r["epoch"] == epoch and r["threshold"] == threshold
                            and (source == "all" or r["source_class"] == source)
                            and (visibility != "full_only" or r["visibility"] == "full")]
                    xs, ys = [], []
                    for bi, group in enumerate(SIZE_GROUPS):
                        bucket = [r for r in rows if int(np.searchsorted(BOUNDARIES, r["gt_size_sqrt_area_px"], side="right")) == bi]
                        x = float(np.median([r["gt_size_sqrt_area_px"] for r in bucket])) if bucket else float("nan")
                        y = float(np.median([r["component_area_cells"] for r in bucket])) if bucket else float("nan")
                        xs.append(x); ys.append(y)
                        bin_rows.append({"selection_scope": scope, "point": point, "family": family,
                                         "epoch": epoch, "threshold": threshold, "visibility": visibility,
                                         "source_class": source, "size_group": group, "n": len(bucket),
                                         "median_gt_size_px": x, "median_component_area_cells": y})
                    ax.plot(xs, ys, marker="o", label=point)
                ax.set_title(source + "/" + visibility, fontsize=9); ax.grid(alpha=0.25); ax.legend(fontsize=7)
                if vi == 1:
                    ax.set_xlabel("Median GT sqrt(area), px")
                if ci == 0:
                    ax.set_ylabel("Median component area, cells")
        fig.suptitle(scope + ": matched-only bin medians")
        fig.tight_layout(); fig.savefig(out / (scope + "_area_size_binned_medians.png"), dpi=160); plt.close(fig)
    write_csv(out / "area_size_binned_medians.csv", bin_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("cached", "final"), required=True)
    for name in ("config", "dataset-root", "baseline-work", "work", "results"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--extended-work", type=Path, help="final only: B-box50 fold caches containing e250/e300 and meta.json")
    args = parser.parse_args()
    if args.stage == "final" and args.extended_work is None:
        parser.error("--extended-work is required for --stage final")
    out = args.results / "round2" / args.stage
    if out.exists():
        raise FileExistsError("refusing to overwrite existing round2 stage: {}".format(out))
    cfg = _load_cfg(args.config)
    folds = cfg["data"]["held_out_sessions"]
    allowed = {"pool-20260831-{:03d}".format(i) for i in range(1, 5)}
    if len(folds) != 4 or set(folds) != allowed:
        raise ValueError("round2 requires exactly sessions 001-004")
    # Check names before shared loader can open annotations; never read 005/test.
    train_dir = args.dataset_root / "images" / "train"
    for path in train_dir.iterdir():
        if path.suffix.lower() in IMAGE_SUFFIXES and session_of(path) not in allowed:
            raise ValueError("non-approved session in images/train: {}".format(path.name))
    samples = load_pool_samples(args.dataset_root, use_visibility=True)
    by_name = {s.image_path.name: s for s in samples}
    metadata = {name: source_metadata(sample, cfg["model"]["input_size"]) for name, sample in by_name.items()}
    manifest = []
    for sample in samples:
        annotation = sample.image_path.with_suffix(".json")
        manifest.append({"kind": "dataset", "image": str(sample.image_path.resolve()),
                         "image_sha256": digest(sample.image_path), "session": sample.session,
                         "annotation": str(annotation.resolve()) if annotation.exists() else None,
                         "annotation_sha256": digest(annotation) if annotation.exists() else None})
    data, tables = {}, {}
    pooled_samples, pooled_sources, pooled_transforms, pooled_names = [], [], [], []
    for family in FAMILIES:
        epochs = EPOCHS + (EXTRA_EPOCHS if args.stage == "final" and family == "B-box50" else ())
        outputs = {e: [] for e in epochs}
        family_names = []
        for fold in folds:
            directory = (args.baseline_work if family == "B" else args.work) / (fold + "__" + family)
            meta_path = directory / "meta.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("family") != family or meta.get("held_out") != fold:
                raise ValueError("cache family/fold metadata mismatch: {}".format(directory))
            names = meta["test_images_list"]
            expected = {s.image_path.name for s in samples if s.session == fold}
            if len(names) != len(expected) or set(names) != expected:
                raise ValueError("cache held-out image list must cover exactly fold {}: {}".format(fold, directory))
            family_names.extend(names)
            for epoch in epochs:
                source = directory
                source_meta = meta_path
                if epoch in EXTRA_EPOCHS:
                    source = args.extended_work / (fold + "__B-box50")
                    source_meta = source / "meta.json"
                    extended = json.loads(source_meta.read_text(encoding="utf-8"))
                    if extended.get("family") != family or extended.get("held_out") != fold:
                        raise ValueError("extended cache family/fold metadata mismatch: {}".format(source))
                    if extended["test_images_list"] != names:
                        raise ValueError("extended/original cache held-out order differs: {}".format(source))
                path = source / "outputs_e{}.npz".format(epoch)
                with np.load(path, allow_pickle=False) as archive:
                    raw = archive["outputs"]
                grid = cfg["model"]["input_size"] // cfg["model"]["output_stride"]
                expected_shape = (len(names), 1 + len(class_names()), grid, grid)
                if raw.shape != expected_shape or not np.issubdtype(raw.dtype, np.floating) or not np.isfinite(raw).all():
                    raise ValueError("invalid finite floating logits [N,C,G,G]: {} {}".format(path, raw.shape))
                outputs[epoch].append(raw)
                manifest.append({"kind": "cache", "family": family, "fold": fold, "epoch": epoch,
                                 "path": str(path.resolve()), "sha256": digest(path), "shape": list(raw.shape),
                                 "dtype": str(raw.dtype), "meta_path": str(source_meta.resolve()),
                                 "meta_sha256": digest(source_meta), "meta": extended if epoch in EXTRA_EPOCHS else meta})
        if not pooled_names:
            pooled_names = family_names
            pooled_samples = [by_name[name] for name in pooled_names]
            pooled_sources = [metadata[name][0] for name in pooled_names]
            pooled_transforms = [metadata[name][1] for name in pooled_names]
        elif family_names != pooled_names:
            raise ValueError("original family held-out image order differs")
        data[family] = {e: probabilities(np.concatenate(parts)) for e, parts in outputs.items()}

    sweep_rows, source_scan = [], []
    distance = cfg["data"]["distance_threshold_px"]
    ground_truth = [s.boxes for s in pooled_samples]
    for family in FAMILIES:
        tables[family] = {}
        for epoch, probs in data[family].items():
            for threshold in THRESHOLDS:
                dets = decode(probs, pooled_transforms, cfg, threshold)
                ps = predictions(dets)
                metrics = evaluate_threshold(ps, ground_truth, class_agnostic=True, adjacent_distance=distance)
                aware = evaluate_threshold(ps, ground_truth, class_agnostic=False, adjacent_distance=distance)
                row = {"family": family, "epoch": epoch, "threshold": threshold,
                       **{key: metrics[key] for key in ("tp", "fp", "fn", "precision", "recall", "f1")},
                       **{"class_aware_" + key: aware[key] for key in ("precision", "recall", "f1")}}
                sweep_rows.append(row); tables[family][(epoch, threshold)] = row
                recall, _, _ = detailed(dets, pooled_samples, pooled_sources, BOUNDARIES, family, epoch, threshold)
                source_scan.extend(recall)
            print("scan complete: {} e{}".format(family, epoch), flush=True)
    old_selected = {f: select({k: v["f1"] for k, v in tables[f].items() if k[0] in EPOCHS}) for f in FAMILIES}
    selections = {"old_six": old_selected}
    if args.stage == "final":
        selections["final_eight"] = {**old_selected, "B-box50": select({k: v["f1"] for k, v in tables["B-box50"].items()})}
    recall_rows, correlations, matches, merges = [], [], [], []
    for scope, selected in selections.items():
        for family in FAMILIES:
            for epoch, probs in data[family].items():
                points = [selected[family][1]]
                if family == "B" and epoch == 150 and 0.6 not in points:
                    points.append(0.6)
                for threshold in points:
                    dets = decode(probs, pooled_transforms, cfg, threshold)
                    recall, corr, matched = detailed(dets, pooled_samples, pooled_sources, BOUNDARIES, family, epoch, threshold)
                    for destination, rows in ((recall_rows, recall), (correlations, corr), (matches, matched)):
                        destination.extend({"selection_scope": scope, **r} for r in rows)
                    merges.append({"selection_scope": scope, "family": family, "epoch": epoch, "threshold": threshold,
                                   **merge_stats(probs, pooled_samples, pooled_transforms, threshold,
                                                 cfg["model"]["output_stride"], distance)})
    baseline = tables["B"][(150, 0.6)]
    stability_rows = {scope: {f: stability(correlations, scope, f, selected[f][1]) for f in FAMILIES}
                      for scope, selected in selections.items()}
    selected_rows = {scope: {f: {**tables[f][k], "f1_difference_vs_B150_at_0.60": tables[f][k]["f1"] - baseline["f1"],
                                "f1_not_below_B150_minus_0.03": tables[f][k]["f1"] >= baseline["f1"] - 0.03,
                                "joint_criteria_met": (tables[f][k]["f1"] >= baseline["f1"] - 0.03
                                                       and stability_rows[scope][f]["late_all_five_ge_0.6"])
                                if stability_rows[scope][f]["complete_late_standard_confirmed"] else None}
                             for f, k in selected.items()} for scope, selected in selections.items()}
    ranges = gt_ranges(pooled_samples, pooled_sources)
    summary = {"stage": args.stage, "thresholds": THRESHOLDS, "epochs_by_family": {f: list(data[f]) for f in FAMILIES},
               "selection_rule": "Task08 shared select: max pooled agnostic F1; within <0.01 -> fewer epochs, threshold nearest 0.4",
               "selection_scope_note": "old_six selects 20/40/60/100/150/200; final_eight adds 250/300 only for B-box50; B/B-box remain six-snapshot selections",
               "selection_optimism": "same four train-only CV folds used for selection and report; no independent labelled test",
               "size_metric": "matched component_area_cells vs original-pixel GT sqrt(area); ignore excluded",
               "size_group_boundaries_px": BOUNDARIES.tolist(), "baseline_B150_at_0.60": baseline,
               "selected": selected_rows, "rho_stability": stability_rows, "gt_size_ranges": ranges,
               "selected_threshold_correlations": correlations, "selected_threshold_merges": merges,
               "recall_groups": recall_rows,
               "config_path": str(args.config.resolve()), "config_sha256": digest(args.config),
               "dataset_root": str(args.dataset_root.resolve()), "input_manifest": manifest}
    out.mkdir(parents=True, exist_ok=False)
    for name, rows in (("threshold_scan", sweep_rows), ("source_size_threshold_scan", source_scan),
                       ("recall_groups", recall_rows), ("area_size_correlation", correlations),
                       ("matched_area_size", matches), ("adjacent_merge", merges), ("gt_size_ranges", ranges)):
        write_csv(out / (name + ".csv"), rows)
    (out / "summary.json").write_text(json.dumps(finite_json(summary), indent=2, allow_nan=False), encoding="utf-8")
    (out / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    make_plots(out, selections, tables, correlations, matches)
    print(json.dumps(finite_json(selected_rows), indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
