"""Task 14 round 2 reference: the untouched task 13 B-box50 path on the keypoint dataset.

The keypoint control/detach runs must not change detection. Because task 13 read
``lab_pool_v2_vis`` and task 14 reads ``lab_pool_v3_kp``, the stored task 13
caches cannot be used for a bitwise cross-dataset comparison; this entry point
retrains the unmodified ``fomo_bbox50`` recipe on the v3 copy so the comparison
is made on identical data. No keypoint code is involved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from fomo_servo.centernet.annotations import split_fold  # noqa: E402
from fomo_servo.centernet.keypoints import load_keypoint_samples  # noqa: E402

EPOCHS = (20, 40, 60, 100, 150, 200, 250)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "dataset-root", "init-weights", "work"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--family", default="B-box50")
    args = parser.parse_args()
    from fomo_servo.centernet.training import predict_raw, train_model

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    cfg["training"]["num_workers"] = 0
    issues = []
    samples = load_keypoint_samples(args.dataset_root, issues=issues)
    if cfg["data"]["frozen_test_session"] in {s.session for s in samples}:
        raise RuntimeError("frozen test session present in training data")
    args.work.mkdir(parents=True, exist_ok=True)
    for held_out in cfg["data"]["held_out_sessions"]:
        train_samples, heldout = split_fold(samples, held_out)
        target = args.work / "{}__{}".format(held_out, args.family)
        if (target / "meta.json").exists():
            print("skip", target.name, flush=True)
            continue
        target.mkdir(parents=True, exist_ok=True)
        started = time.time()

        def snapshot(epoch, model):
            outputs, _ = predict_raw(model, "fomo_bbox50", heldout, cfg["model"]["input_size"],
                                     cfg["training"]["device"])
            np.savez_compressed(target / "outputs_e{}.npz".format(epoch), outputs=outputs)

        print("[{}] plain B-box50 on the keypoint dataset -> 250 epochs".format(held_out), flush=True)
        _, history, report = train_model("fomo_bbox50", max(EPOCHS), train_samples, cfg, args.init_weights,
                                         log=lambda message: print(message, flush=True),
                                         snapshot_epochs=EPOCHS, on_snapshot=snapshot)
        meta = {"family": args.family, "kind": "fomo_bbox50", "held_out": held_out, "epochs": list(EPOCHS),
                "init": report, "history": history, "seconds": time.time() - started,
                "test_images_list": [s.image_path.name for s in heldout],
                "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
                "note": "unmodified task 13 detection recipe on lab_pool_v3_kp; reference for round 2",
                "cache_sha256": {"outputs_e{}.npz".format(e): hashlib.sha256(
                    (target / "outputs_e{}.npz".format(e)).read_bytes()).hexdigest() for e in EPOCHS}}
        (target / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
