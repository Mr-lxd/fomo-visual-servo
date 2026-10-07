"""Task 13 round 2: restart B-box50 for 300 epochs; retain only 250/300 logits."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from run_centernet_cv import _load_cfg  # noqa: E402
from fomo_servo.centernet.annotations import load_pool_samples, split_fold  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "dataset-root", "init-weights", "reference-work", "work"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    from fomo_servo.centernet.training import predict_raw, train_model

    cfg = _load_cfg(args.config)
    samples = load_pool_samples(args.dataset_root, use_visibility=True)
    for held_out in cfg["data"]["held_out_sessions"]:
        train_samples, heldout_samples = split_fold(samples, held_out)
        directory = args.work / (held_out + "__B-box50")
        if (directory / "meta.json").exists():
            print("skip", directory.name, flush=True)
            continue
        directory.mkdir(parents=True, exist_ok=True)
        reference = args.reference_work / directory.name
        comparison = {}
        started = time.time()

        def snapshot(epoch, model):
            """Held-out float32 logits [N,8,24,24]; 200 is an unsaved consistency check."""
            outputs, _ = predict_raw(model, "fomo_bbox50", heldout_samples,
                                     cfg["model"]["input_size"], cfg["training"]["device"])
            if epoch == 200:
                previous = np.load(reference / "outputs_e200.npz")["outputs"]
                previous_names = json.loads((reference / "meta.json").read_text())["test_images_list"]
                assert previous_names == [s.image_path.name for s in heldout_samples]
                comparison.update({"bitwise_equal": bool(np.array_equal(outputs, previous)),
                                   "max_abs_difference": float(np.max(np.abs(outputs - previous)))})
                print("  epoch 200 cache comparison:", comparison, flush=True)
            else:
                np.savez_compressed(directory / f"outputs_e{epoch}.npz", outputs=outputs)

        print(f"[{held_out}] B-box50 -> 300 epochs from original initialization", flush=True)
        _, history, init = train_model(
            "fomo_bbox50", 300, train_samples, cfg, args.init_weights,
            log=lambda message: print(message, flush=True),
            snapshot_epochs=(200, 250, 300), on_snapshot=snapshot,
        )
        meta = {"family": "B-box50", "kind": "fomo_bbox50", "held_out": held_out,
                "epochs": [250, 300], "init": init, "history": history,
                "seconds": time.time() - started,
                "test_images_list": [s.image_path.name for s in heldout_samples],
                "training_mode": "from original initialization; no resume",
                "epoch200_cache_comparison": comparison}
        (directory / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
