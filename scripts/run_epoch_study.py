"""Task 08 section 2: epoch study (B and C trained once to 200 epochs, snapshots evaluated).

The optimiser has a constant learning rate (no scheduler), so a snapshot at epoch N equals a
separate N-epoch run with the same seed. Subcommands: train, evaluate.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_centernet_cv import _load_cfg, _predictions  # noqa: E402

from fomo_servo.centernet.annotations import load_pool_samples, split_fold  # noqa: E402
from fomo_servo.centernet.evaluation import evaluate_threshold, match_image, spearman  # noqa: E402

FAMILIES = {"B": "fomo", "C": "centernet"}
EPOCHS = (20, 40, 60, 100, 150, 200)


def cmd_train(args: argparse.Namespace) -> None:
    from fomo_servo.centernet.training import predict_raw, train_model

    cfg = _load_cfg(args.config)
    samples = load_pool_samples(args.dataset_root, use_visibility=True)
    args.work.mkdir(parents=True, exist_ok=True)
    for held_out in cfg["data"]["held_out_sessions"]:
        train_samples, test_samples = split_fold(samples, held_out)
        for family, kind in FAMILIES.items():
            target = args.work / "{}__{}".format(held_out, family)
            if (target / "meta.json").exists():
                print("skip", target.name)
                continue
            target.mkdir(parents=True, exist_ok=True)
            started = time.time()

            def snapshot(epoch: int, model, target=target, test_samples=test_samples, kind=kind) -> None:
                outputs, _ = predict_raw(model, kind, test_samples, cfg["model"]["input_size"], cfg["training"]["device"])
                np.savez_compressed(target / "outputs_e{}.npz".format(epoch), outputs=outputs)

            print("[{}] {} -> 200 epochs".format(held_out, family), flush=True)
            _, history, report = train_model(
                kind, max(EPOCHS), train_samples, cfg, args.init_weights, log=lambda m: print(m, flush=True),
                snapshot_epochs=EPOCHS, on_snapshot=snapshot,
            )
            meta = {
                "family": family, "held_out": held_out, "epochs": list(EPOCHS), "init": report,
                "history": history, "seconds": time.time() - started,
                "test_images_list": [s.image_path.name for s in test_samples],
            }
            (target / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")


def select(table: dict[tuple[int, float], float]) -> tuple[int, float]:
    """Highest pooled agnostic F1; combinations within <0.01 of it -> fewer epochs, then threshold nearest 0.4."""

    best = max(table.values())
    candidates = [k for k, v in table.items() if best - v < 0.01]
    return min(candidates, key=lambda k: (k[0], abs(k[1] - 0.4), k[1]))


def cmd_evaluate(args: argparse.Namespace) -> None:
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from fomo_servo.geometry.letterbox import LetterboxTransform

    cfg = _load_cfg(args.config)
    s = cfg["evaluation"]["sweep"]
    thresholds = [round(float(v), 4) for v in np.arange(s["start"], s["stop"] + 1e-9, s["step"])]
    adj = cfg["data"]["distance_threshold_px"]
    folds = cfg["data"]["held_out_sessions"]
    samples = load_pool_samples(args.dataset_root, use_visibility=True)
    by_name = {x.image_path.name: x for x in samples}
    out = args.results
    out.mkdir(parents=True, exist_ok=True)

    data: dict = {}  # data[family][epoch][fold] -> (preds, gts, samples)
    for family, kind in FAMILIES.items():
        data[family] = {e: {} for e in EPOCHS}
        for fold in folds:
            d = args.work / "{}__{}".format(fold, family)
            meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
            test = [by_name[n] for n in meta["test_images_list"]]
            transforms = []
            for x in test:
                h, w = cv2.imread(str(x.image_path)).shape[:2]
                transforms.append(LetterboxTransform.from_image_size(w, h, cfg["model"]["input_size"]))
            for e in EPOCHS:
                outputs = np.load(d / "outputs_e{}.npz".format(e))["outputs"]
                data[family][e][fold] = (_predictions(kind, outputs, transforms, cfg, thresholds), [list(x.boxes) for x in test], test)

    def pooled(family: str, e: int, thr: float, agnostic: bool, subset=None) -> dict:
        ps, gs = [], []
        for fold in subset or folds:
            preds, gts, _ = data[family][e][fold]
            ps += preds[thr]
            gs += gts
        return evaluate_threshold(ps, gs, class_agnostic=agnostic, adjacent_distance=adj)

    def size_stats(family: str, e: int, thr: float) -> dict:
        ps, gs = [], []
        for fold in folds:
            preds, gts, _ = data[family][e][fold]
            for pl, gl in zip(preds[thr], gts):
                scored = [g for g in gl if g.visibility != "ignore"]
                for pi, gi, _ in match_image(pl, gl, class_agnostic=True).pairs:
                    g = scored[gi]
                    if g.visibility == "full":
                        ps.append((pl[pi].w * pl[pi].h) ** 0.5)
                        gs.append(((g.x_max - g.x_min) * (g.y_max - g.y_min)) ** 0.5)
        gs_a, ps_a = np.array(gs), np.array(ps)
        return {
            "n": len(gs),
            "spearman_rho": spearman(ps_a, gs_a),
            "relative_error_median": float(np.median(np.abs(ps_a - gs_a) / gs_a)) if len(gs) else float("nan"),
        }

    result: dict = {"selection_rule": "max pooled agnostic F1 over epoch x threshold; F1 within <0.01 of max -> fewer epochs, then threshold nearest 0.4"}
    grid: dict = {}
    for family in FAMILIES:
        grid[family] = {
            (e, t): pooled(family, e, t, True) for e in EPOCHS for t in thresholds
        }
    selected = {f: select({k: v["f1"] for k, v in grid[f].items()}) for f in FAMILIES}
    result["selected"] = {f: {"epochs": k[0], "threshold": k[1]} for f, k in selected.items()}

    lines = ["# Epoch study (generated)\n", "Learning rate constant (no scheduler): snapshots equal separate runs. 4 folds pooled, agnostic matching unless stated.\n"]
    result["by_epoch"] = {}
    for family in FAMILIES:
        e_sel, t_sel = selected[family]
        lines.append("\n## {} - selected: {} epochs, threshold {:.2f}\n".format(family, e_sel, t_sel))
        lines.append("| epochs | thr | agnostic P/R/F1 | class-aware P/R/F1 | centre err (px) | adjacent sep | per-fold agnostic F1 (001/002/003/004) |")
        lines.append("|---|---|---|---|---|---|---|")
        result["by_epoch"][family] = {}
        for e in EPOCHS:
            best_t = max((t for t in thresholds), key=lambda t: (grid[family][(e, t)]["f1"], -abs(t - 0.4)))
            rows = [("0.40", 0.4), ("best", best_t)] + ([("sel", t_sel)] if e == e_sel else [])
            for label, t in rows:
                a, c = grid[family][(e, t)], pooled(family, e, t, False)
                pf = [pooled(family, e, t, True, [f])["f1"] for f in folds]
                lines.append("| {} | {:.2f} ({}) | {:.3f}/{:.3f}/{:.3f} | {:.3f}/{:.3f}/{:.3f} | {:.1f} | {}/{} | {} |".format(
                    e, t, label, a["precision"], a["recall"], a["f1"], c["precision"], c["recall"], c["f1"],
                    a["center_error_median_px"], a["adjacent_both_detected"], a["adjacent_pairs"],
                    " / ".join("{:.3f}".format(v) for v in pf)))
                result["by_epoch"][family].setdefault(str(e), {})[label] = {"threshold": t, "agnostic": a, "class_aware": c, "per_fold_f1": pf}
        lines.append("\nRecall by visibility (agnostic):\n")
        for e in EPOCHS:
            for label, t in (("0.40", 0.4), ("selected thr", t_sel)):
                v = grid[family][(e, t)]["recall_by_visibility"]
                lines.append("- {} e{} @{} ({:.2f}): ".format(family, e, label, t) + ", ".join("{} {}/{}".format(k, d["hit"], d["total"]) for k, d in sorted(v.items())))
    c_rows = []
    lines.append("\n## C size regression vs epochs (full targets, agnostic matches)\n")
    lines.append("| epochs | thr 0.40: n / rho / rel.err | selected thr: n / rho / rel.err |\n|---|---|---|")
    for e in EPOCHS:
        a, b = size_stats("C", e, 0.4), size_stats("C", e, selected["C"][1])
        c_rows.append((e, a, b))
        lines.append("| {} | {} / {:.3f} / {:.3f} | {} / {:.3f} / {:.3f} |".format(e, a["n"], a["spearman_rho"], a["relative_error_median"], b["n"], b["spearman_rho"], b["relative_error_median"]))
    result["size_vs_epoch"] = {str(e): {"thr_0.4": a, "selected_thr": b} for e, a, b in c_rows}
    sel_b, sel_c = selected["B"], selected["C"]
    lines.append("\n## Selection\n\n- B: {} epochs @ {:.2f}\n- C: {} epochs @ {:.2f}\n".format(sel_b[0], sel_b[1], sel_c[0], sel_c[1]))
    lines.append("Note: the 4 CV folds were used for selection, so the selected scores are slightly optimistic; there is no independent labelled test set.")
    (out / "tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(result, indent=1, default=float), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for family in FAMILIES:
        axes[0].plot(EPOCHS, [grid[family][(e, 0.4)]["f1"] for e in EPOCHS], marker="o", label="{} @0.40".format(family))
        axes[0].plot(EPOCHS, [max(grid[family][(e, t)]["f1"] for t in thresholds) for e in EPOCHS], marker="s", ls="--", label="{} best thr per epoch".format(family))
    axes[0].set_xlabel("epochs"); axes[0].set_ylabel("pooled agnostic F1"); axes[0].grid(alpha=0.3); axes[0].legend()
    axes[1].plot(EPOCHS, [a["spearman_rho"] for _, a, _ in c_rows], marker="o", label="C size rho @0.40")
    axes[1].plot(EPOCHS, [b["spearman_rho"] for _, _, b in c_rows], marker="s", ls="--", label="C size rho @selected thr")
    axes[1].set_xlabel("epochs"); axes[1].set_ylabel("Spearman rho (full targets)"); axes[1].grid(alpha=0.3); axes[1].legend()
    fig.tight_layout(); fig.savefig(out / "f1_and_size_vs_epochs.png", dpi=130)
    print("selected", result["selected"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "evaluate"):
        p = sub.add_parser(name)
        p.add_argument("--config", type=Path, default=ROOT / "configs/experiments/centernet_lite_cv.yaml")
        p.add_argument("--dataset-root", type=Path, required=True)
        p.add_argument("--work", type=Path, required=True)
    sub.choices["train"].add_argument("--init-weights", type=Path, required=True)
    sub.choices["evaluate"].add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    cmd_train(args) if args.command == "train" else cmd_evaluate(args)


if __name__ == "__main__":
    main()
