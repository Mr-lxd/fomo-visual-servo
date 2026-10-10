"""Task 10 C: read Task 08 held-out logits; report source/size recall, no training.

The reference checkout supplies its unchanged Task 08 decoder and matcher.
All probability maps are float32 [classes,24,24]; boxes/peaks use original pixels.
"""
import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.dont_write_bytecode = True


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference-root", type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--results", type=Path, required=True)
    a = p.parse_args()
    sys.path[:0] = [str(a.reference_root / "src"), str(a.reference_root / "scripts")]
    import cv2
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from run_centernet_cv import _load_cfg, _predictions
    from fomo_servo.centernet.annotations import load_pool_samples, read_visibility
    from fomo_servo.centernet.evaluation import match_image, evaluate_threshold
    from fomo_servo.geometry.letterbox import LetterboxTransform

    cfg_path = a.reference_root / "configs/experiments/centernet_lite_cv.yaml"
    cfg = _load_cfg(cfg_path)
    samples = {s.image_path.name: s for s in load_pool_samples(a.dataset_root, use_visibility=True)}
    bins = ("<80", "80-150", "150-250", "250-350", ">=350")
    def size_bin(size):
        return bins[int(size >= 80) + int(size >= 150) + int(size >= 250) + int(size >= 350)]

    counts = defaultdict(lambda: [0, 0])
    objects, reflections, provenance, pooled = [], [], [], {}
    for family, epoch, thresholds in (("B", 150, [0.60, 0.40]), ("C", 100, [0.35])):
        ps = {t: [] for t in thresholds}
        gs = []
        for fold in cfg["data"]["held_out_sessions"]:
            d = a.work / (fold + "__" + family)
            meta_path, cache_path = d / "meta.json", d / f"outputs_e{epoch}.npz"
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            test = [samples[n] for n in meta["test_images_list"]]
            assert all(s.session == fold and "005" not in s.session for s in test)
            transforms = []
            payloads = []
            for s in test:
                annotation = s.image_path.with_suffix(".json")
                payload = json.loads(annotation.read_text(encoding="utf-8")) if annotation.exists() else None
                h, w = cv2.imread(str(s.image_path)).shape[:2]
                transforms.append(LetterboxTransform.from_image_size(w, h, cfg["model"]["input_size"]))
                payloads.append(payload)
                provenance.append({"path": str(s.image_path), "sha256": sha(s.image_path)})
                if payload:
                    provenance.append({"path": str(annotation), "sha256": sha(annotation)})
            outputs = np.load(cache_path)["outputs"]
            assert len(outputs) == len(test)
            decoded = _predictions("fomo" if family == "B" else "centernet", outputs, transforms, cfg, thresholds)
            provenance.extend({"path": str(x), "sha256": sha(x)} for x in (meta_path, cache_path))
            for i, (sample, tr, payload) in enumerate(zip(test, transforms, payloads)):
                shapes = payload.get("shapes", []) if payload else []
                source_shapes = [s for s in shapes if s["label"] in ("fish", "tuna", "jellyfish")]
                assert len(source_shapes) == len(sample.boxes)
                scored = [(g, shape) for g, shape in zip(sample.boxes, source_shapes) if g.visibility != "ignore"]
                gs.append(list(sample.boxes))
                if family == "B":
                    z = outputs[i].astype(np.float32)
                    ex = np.exp(z - z.max(axis=0, keepdims=True))
                    prob = ex / ex.sum(axis=0, keepdims=True)
                    fg = prob[1:].max(axis=0)
                    yy, xx = np.indices(fg.shape)
                    ox = ((xx + .5) * cfg["model"]["output_stride"] - tr.pad_left) / tr.scale
                    oy = ((yy + .5) * cfg["model"]["output_stride"] - tr.pad_top) / tr.scale
                    def box_peak(box):
                        x0, y0, x1, y1 = box
                        mask = (ox >= x0) & (ox <= x1) & (oy >= y0) & (oy <= y1)
                        if not mask.any():
                            return {"max_foreground": None, "peak_x": None, "peak_y": None, "peak_class_id": None}
                        v = np.where(mask, fg, -1)
                        gy, gx = np.unravel_index(v.argmax(), v.shape)
                        return {"max_foreground": float(fg[gy, gx]), "peak_x": float(ox[gy, gx]),
                                "peak_y": float(oy[gy, gx]), "peak_class_id": int(prob[1:, gy, gx].argmax())}
                    for si, shape in enumerate(shapes):
                        if shape["label"] != "reflection tuna":
                            continue
                        pts = shape["points"]
                        box = (max(0, min(v[0] for v in pts)), max(0, min(v[1] for v in pts)),
                               min(tr.original_width, max(v[0] for v in pts)), min(tr.original_height, max(v[1] for v in pts)))
                        reflections.append({"fold": fold, "image": sample.image_path.name, "shape_index": si,
                                            "visibility": read_visibility(shape, sample.image_path.name), **box_peak(box)})
                for thr in thresholds:
                    model = f"{family}{epoch}@{thr:.2f}"
                    preds = decoded[thr][i]
                    ps[thr].append(preds)
                    matched = {gi for _, gi, _ in match_image(preds, sample.boxes, class_agnostic=True).pairs}
                    for gi, (g, shape) in enumerate(scored):
                        size = ((g.x_max-g.x_min)*(g.y_max-g.y_min)) ** .5
                        hit = int(gi in matched)
                        source = shape["label"]
                        group = "jellyfish" if source == "jellyfish" else "fish+tuna"
                        categories = list(dict.fromkeys(("all", group, source)))
                        if source == "tuna":
                            categories.append("tuna/" + g.visibility)
                        for category in categories:
                            for b in ("all", size_bin(size)):
                                c = counts[model, category, b]
                                c[0] += hit; c[1] += 1
                        row = {"model": model, "fold": fold, "image": sample.image_path.name,
                               "source_class": source, "visibility": g.visibility, "size_px": size,
                               "size_bin": size_bin(size), "hit": hit, "gt_index": gi,
                               "x0": g.x_min, "y0": g.y_min, "x1": g.x_max, "y1": g.y_max}
                        if family == "B":
                            row.update(box_peak((g.x_min, g.y_min, g.x_max, g.y_max)))
                        else:
                            row.update(dict.fromkeys(("max_foreground", "peak_x", "peak_y", "peak_class_id")))
                        objects.append(row)
            for thr in thresholds:
                pooled[f"{family}{epoch}@{thr:.2f}"] = evaluate_threshold(ps[thr], gs, class_agnostic=True, adjacent_distance=cfg["data"]["distance_threshold_px"])

    a.results.mkdir(parents=True, exist_ok=True)
    rows = [{"model": m, "group": c, "size_bin": b, "hit": counts[m,c,b][0], "total": counts[m,c,b][1],
             "recall": counts[m,c,b][0]/counts[m,c,b][1] if counts[m,c,b][1] else None}
            for m in pooled for c in ("all", "jellyfish", "fish+tuna", "fish", "tuna", "tuna/full", "tuna/truncated", "tuna/occluded")
            for b in ("all", *bins)]
    write_csv(a.results / "recall.csv", rows)
    write_csv(a.results / "objects.csv", objects)
    write_csv(a.results / "reflection_tuna.csv", reflections)
    missed = [r for r in objects if r["model"] == "B150@0.60" and r["source_class"] == "tuna" and not r["hit"]]
    write_csv(a.results / "missed_tuna.csv", missed)
    true = [r for r in objects if r["model"] == "B150@0.60" and r["source_class"] == "tuna"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist([r["max_foreground"] for r in missed if r["max_foreground"] is not None], bins=np.linspace(0,1,21))
    axes[0].set_title(f"B150 @0.60 missed tuna (n={len(missed)})")
    axes[1].hist([[r["max_foreground"] for r in true if r["max_foreground"] is not None],
                  [r["max_foreground"] for r in reflections if r["max_foreground"] is not None]],
                 bins=np.linspace(0,1,21), label=[f"true tuna (n={len(true)})", f"reflection tuna (n={len(reflections)})"])
    axes[1].legend(); axes[1].set_title("Held-out foreground maxima inside boxes")
    for ax in axes:
        ax.set_xlabel("max foreground probability"); ax.set_ylabel("boxes")
        ax.axvline(.4, color="orange", ls="--"); ax.axvline(.6, color="red", ls="--")
    fig.tight_layout(); fig.savefig(a.results / "tuna_scores.png", dpi=150)
    provenance.append({"path": str(cfg_path), "sha256": sha(cfg_path)})
    (a.results / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    summary = {"pooled": pooled, "missed_tuna_count": len(missed), "missed_score_bands": {
        label: sum(lo <= r["max_foreground"] < hi for r in missed if r["max_foreground"] is not None)
        for label, lo, hi in (("<0.2",0,.2),("0.2-0.3",.2,.3),("0.3-0.6",.3,.6),(">=0.6",.6,1.01))}}
    (a.results / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
