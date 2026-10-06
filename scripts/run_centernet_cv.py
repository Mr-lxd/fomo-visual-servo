"""Leave-one-session-out CV for B20 / B60 / C60 (task 07). Subcommands: train, evaluate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fomo_servo.centernet.annotations import (  # noqa: E402
    load_pool_samples, session_of, sessions, split_fold, class_names,
)
from fomo_servo.centernet.decode import decode_centernet  # noqa: E402
from fomo_servo.centernet.evaluation import (  # noqa: E402
    Pred, evaluate_threshold, match_image, spearman,
)
from fomo_servo.postprocess import postprocess_numpy_logits  # noqa: E402


def _load_cfg(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def cmd_train(args: argparse.Namespace) -> None:
    import torch
    from fomo_servo.centernet.training import predict_raw, train_model

    cfg = _load_cfg(args.config)
    samples = load_pool_samples(args.dataset_root, use_visibility=args.use_visibility)
    assert cfg["data"]["frozen_test_session"] not in sessions(samples)
    work = args.work
    work.mkdir(parents=True, exist_ok=True)
    names = args.runs or list(cfg["training"]["runs"])
    for held_out in cfg["data"]["held_out_sessions"]:
        train_samples, test_samples = split_fold(samples, held_out)
        for run in names:
            spec = cfg["training"]["runs"][run]
            target = work / "{}__{}".format(held_out, run)
            if (target / "outputs.npz").exists():
                print("skip existing", target.name)
                continue
            print("[{}] {}: train {} imgs, test {} imgs".format(held_out, run, len(train_samples), len(test_samples)))
            started = time.time()
            model, history, report = train_model(spec["kind"], spec["epochs"], train_samples, cfg, args.init_weights)
            outputs, _ = predict_raw(model, spec["kind"], test_samples, cfg["model"]["input_size"], cfg["training"]["device"])
            target.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(target / "outputs.npz", outputs=outputs)
            torch.save(model.state_dict(), target / "weights.pt")
            meta = {
                "run": run, "kind": spec["kind"], "epochs": spec["epochs"], "held_out": held_out,
                "train_images": len(train_samples), "test_images": len(test_samples),
                "train_targets": sum(len(s.boxes) for s in train_samples),
                "test_targets": sum(len(s.boxes) for s in test_samples),
                "init": report, "history": history, "seconds": time.time() - started,
                "test_images_list": [s.image_path.name for s in test_samples],
                "weights_sha256": _sha256(target / "weights.pt"),
            }
            (target / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
            print("  done in {:.0f}s, amp skipped steps {}".format(meta["seconds"], report["amp_skipped_steps"]))


def _sweep(cfg: dict) -> list[float]:
    s = cfg["evaluation"]["sweep"]
    return [round(float(v), 4) for v in np.arange(s["start"], s["stop"] + 1e-9, s["step"])]


def _predictions(kind: str, outputs: np.ndarray, transforms, cfg: dict, thresholds: list[float]):
    """Return ``{threshold: [preds per image]}``."""

    stride = cfg["model"]["output_stride"]
    names = class_names()
    result = {}
    if kind == "centernet":
        low = min(thresholds)
        decoded = [decode_centernet(o, t, stride=stride, threshold=low) for o, t in zip(outputs, transforms)]
        for thr in thresholds:
            result[thr] = [
                [Pred(d.class_id, d.score, d.x, d.y, d.w, d.h) for d in img if d.score >= thr - 1e-9]
                for img in decoded
            ]
    else:
        for thr in thresholds:
            per_image = []
            for o, t in zip(outputs, transforms):
                dets = postprocess_numpy_logits(
                    o[None], class_names=names, stride=stride, transforms=(t,), confidence_threshold=thr,
                )[0]
                per_image.append([Pred(d.class_id, d.confidence, d.original_x, d.original_y) for d in dets])
            result[thr] = per_image
    return result


def cmd_evaluate(args: argparse.Namespace) -> None:
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from fomo_servo.geometry.letterbox import LetterboxTransform

    cfg = _load_cfg(args.config)
    ev = cfg["evaluation"]
    samples = load_pool_samples(args.dataset_root, use_visibility=args.use_visibility)
    by_name = {s.image_path.name: s for s in samples}
    thresholds = _sweep(cfg)
    reported = [round(t, 4) for t in ev["thresholds_report"]]
    adj = cfg["data"]["distance_threshold_px"]
    runs = list(cfg["training"]["runs"])
    folds = cfg["data"]["held_out_sessions"]
    out = args.results
    out.mkdir(parents=True, exist_ok=True)

    # per_run[run][fold] -> (preds {thr: [...]}, gts, names, meta)
    per_run: dict = {r: {} for r in runs}
    for fold in folds:
        for run in runs:
            d = args.work / "{}__{}".format(fold, run)
            meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
            outputs = np.load(d / "outputs.npz")["outputs"]
            test = [by_name[n] for n in meta["test_images_list"]]
            transforms = []
            for s in test:
                h, w = cv2.imread(str(s.image_path)).shape[:2]
                transforms.append(LetterboxTransform.from_image_size(w, h, cfg["model"]["input_size"]))
            preds = _predictions(meta["kind"], outputs, transforms, cfg, thresholds)
            per_run[run][fold] = {"preds": preds, "gts": [list(s.boxes) for s in test], "samples": test, "meta": meta}

    def pooled(run: str, thr: float, agnostic: bool, subset=None) -> dict:
        ps, gs = [], []
        for fold in (subset or folds):
            e = per_run[run][fold]
            ps += e["preds"][thr]
            gs += e["gts"]
        return evaluate_threshold(ps, gs, class_agnostic=agnostic, adjacent_distance=adj)

    # ---- tables ----
    summary: dict = {"folds": {}, "overall": {}}
    for fold in folds:
        m = per_run[runs[0]][fold]["meta"]
        summary["folds"][fold] = {"train_images": m["train_images"], "test_images": m["test_images"],
                                  "train_targets": m["train_targets"], "test_targets": m["test_targets"]}
    for run in runs:
        summary["overall"][run] = {}
        for thr in thresholds:
            for agn in (True, False):
                key = "{}@{:.2f}".format("agnostic" if agn else "class", thr)
                summary["overall"][run][key] = pooled(run, thr, agn)
    summary["by_fold"] = {
        fold: {
            run: {
                "{}@{:.2f}".format("agnostic" if agn else "class", thr):
                    pooled(run, thr, agn, [fold])
                for thr in reported for agn in (True, False)
            } for run in runs
        } for fold in folds
    }

    # size usability (C runs only; full GT only; agnostic matching at the headline threshold)
    headline = round(ev["headline_threshold"], 4)
    size_stats = {}
    pairs_rows = []
    for run in runs:
        if cfg["training"]["runs"][run]["kind"] != "centernet":
            continue
        pred_sizes, gt_sizes = [], []
        for fold in folds:
            e = per_run[run][fold]
            for sample, preds, gts in zip(e["samples"], e["preds"][headline], e["gts"]):
                scored = [g for g in gts if g.visibility != "ignore"]
                m = match_image(preds, gts, class_agnostic=True)
                for pi, gi, dist in m.pairs:
                    g, p = scored[gi], preds[pi]
                    gw, gh = g.x_max - g.x_min, g.y_max - g.y_min
                    pairs_rows.append({
                        "run": run, "fold": fold, "image": sample.image_path.name,
                        "gt_class": class_names()[g.class_id], "pred_class": class_names()[p.class_id],
                        "visibility": g.visibility, "score": round(p.score, 4), "center_error_px": round(dist, 2),
                        "gt_w": round(gw, 1), "gt_h": round(gh, 1), "pred_w": round(p.w, 1), "pred_h": round(p.h, 1),
                    })
                    if g.visibility == "full":
                        pred_sizes.append((p.w * p.h) ** 0.5)
                        gt_sizes.append((gw * gh) ** 0.5)
        pred_sizes, gt_sizes = np.array(pred_sizes), np.array(gt_sizes)
        size_stats[run] = {
            "n": int(len(gt_sizes)),
            "spearman_rho": spearman(pred_sizes, gt_sizes),
            "relative_error_median": float(np.median(np.abs(pred_sizes - gt_sizes) / gt_sizes)) if len(gt_sizes) else float("nan"),
        }
    summary["size_usability"] = size_stats
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")
    if pairs_rows:
        with (out / "pairs.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(pairs_rows[0]))
            w.writeheader()
            w.writerows(pairs_rows)

    # ---- markdown tables ----
    lines = ["# Summary tables (generated)\n"]
    lines.append("## Folds\n\n| held-out | train imgs | test imgs | train targets | test targets |\n|---|---|---|---|---|")
    for fold, v in summary["folds"].items():
        lines.append("| {} | {train_images} | {test_images} | {train_targets} | {test_targets} |".format(fold, **v))

    def fmt(r: dict) -> str:
        return "{:.3f} / {:.3f} / {:.3f}".format(r["precision"], r["recall"], r["f1"])

    for thr in reported:
        lines.append("\n## Pooled over 4 folds, threshold {:.2f}\n".format(thr))
        lines.append("| model | agnostic P/R/F1 | class-aware P/R/F1 | center err median (px) | adjacent separation | TP/FP/FN (agnostic) |\n|---|---|---|---|---|---|")
        for run in runs:
            a = summary["overall"][run]["agnostic@{:.2f}".format(thr)]
            c = summary["overall"][run]["class@{:.2f}".format(thr)]
            lines.append("| {} | {} | {} | {:.1f} | {}/{} = {:.3f} | {}/{}/{} |".format(
                run, fmt(a), fmt(c), a["center_error_median_px"], a["adjacent_both_detected"], a["adjacent_pairs"],
                a["adjacent_separation_rate"], a["tp"], a["fp"], a["fn"]))
    lines.append("\n## Per fold, threshold {:.2f}, agnostic P/R/F1 (TP/FP/FN)\n".format(headline))
    lines.append("| held-out | " + " | ".join(runs) + " |\n|---|" + "---|" * len(runs))
    for fold in folds:
        cells = []
        for run in runs:
            r = summary["by_fold"][fold][run]["agnostic@{:.2f}".format(headline)]
            cells.append("{} ({}/{}/{})".format(fmt(r), r["tp"], r["fp"], r["fn"]))
        lines.append("| {} | ".format(fold) + " | ".join(cells) + " |")
    lines.append("\n## C60 size usability (full targets, agnostic matches at {:.2f})\n".format(headline))
    for run, st in size_stats.items():
        lines.append("- {}: n={}, Spearman rho={:.3f}, median relative error={:.3f}".format(run, st["n"], st["spearman_rho"], st["relative_error_median"]))
    vis_lines = []
    for run in runs:
        v = summary["overall"][run]["agnostic@{:.2f}".format(headline)]["recall_by_visibility"]
        vis_lines.append("- {}: ".format(run) + ", ".join("{} {}/{}".format(k, d["hit"], d["total"]) for k, d in sorted(v.items())))
    lines.append("\n## Recall by visibility (agnostic, {:.2f})\n".format(headline))
    lines += vis_lines
    (out / "tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---- PR-like curves ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, agn in zip(axes, (True, False)):
        for run in runs:
            pts = [summary["overall"][run]["{}@{:.2f}".format("agnostic" if agn else "class", t)] for t in thresholds]
            ax.plot([p["recall"] for p in pts], [p["precision"] for p in pts], marker="o", ms=3, label=run)
        ax.set_xlabel("recall"); ax.set_ylabel("precision"); ax.grid(alpha=0.3)
        ax.set_title("pooled held-out, {}".format("class-agnostic" if agn else "class-aware"))
        ax.legend()
    fig.tight_layout(); fig.savefig(out / "pr_curves.png", dpi=130); plt.close(fig)

    # ---- overlays: 6 frames per model chosen from GT only (most targets, adjacent first) ----
    all_items = []
    for fold in folds:
        e = per_run[runs[0]][fold]
        for i, (s, gts) in enumerate(zip(e["samples"], e["gts"])):
            all_items.append((len(adjacent_pairs_safe(gts, adj)), len(gts), s.image_path.name, fold, i))
    all_items.sort(key=lambda t: (-t[0], -t[1], t[2]))
    chosen = []
    per_fold_count: dict = {}
    for item in all_items:  # spread over folds, at most 2 per held-out session
        if per_fold_count.get(item[3], 0) < 2:
            chosen.append(item)
            per_fold_count[item[3]] = per_fold_count.get(item[3], 0) + 1
        if len(chosen) == ev["overlay_samples_per_model"]:
            break
    ov_dir = out / "overlays"
    ov_dir.mkdir(exist_ok=True)
    for run in runs:
        tiles = []
        for _, _, name, fold, i in chosen:
            e = per_run[run][fold]
            img = cv2.imread(str(e["samples"][i].image_path))
            for g in e["gts"][i]:
                color = (0, 200, 0) if g.visibility != "ignore" else (160, 160, 160)
                cv2.rectangle(img, (int(g.x_min), int(g.y_min)), (int(g.x_max), int(g.y_max)), color, 2)
            for p in e["preds"][headline][i]:
                cv2.drawMarker(img, (int(p.x), int(p.y)), (0, 0, 255), cv2.MARKER_CROSS, 18, 2)
                if p.w:
                    cv2.rectangle(img, (int(p.x - p.w / 2), int(p.y - p.h / 2)), (int(p.x + p.w / 2), int(p.y + p.h / 2)), (0, 0, 255), 1)
            cv2.putText(img, "{} {}".format(run, name[-18:]), (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            tiles.append(img)
        rows = [np.hstack(tiles[k:k + 3]) for k in range(0, len(tiles), 3)]
        cv2.imwrite(str(ov_dir / "{}_6samples.jpg".format(run)), np.vstack(rows))
    print("wrote", out)


def adjacent_pairs_safe(gts, adj):
    from fomo_servo.centernet.evaluation import adjacent_pairs
    return adjacent_pairs(gts, adj)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "evaluate"):
        p = sub.add_parser(name)
        p.add_argument("--config", type=Path, default=ROOT / "configs/experiments/centernet_lite_cv.yaml")
        p.add_argument("--dataset-root", type=Path, required=True)
        p.add_argument("--work", type=Path, required=True, help="directory for per-fold weights and raw outputs")
        p.add_argument("--use-visibility", action="store_true", help="read visibility attributes (round 2)")
    sub.choices["train"].add_argument("--init-weights", type=Path, required=True)
    sub.choices["train"].add_argument("--runs", nargs="*")
    sub.choices["evaluate"].add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    cmd_train(args) if args.command == "train" else cmd_evaluate(args)


if __name__ == "__main__":
    main()
