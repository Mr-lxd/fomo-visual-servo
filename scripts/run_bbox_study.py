"""Task 13: train B-box and B-box50 to 200 epochs with Task 08 snapshots."""

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

from run_centernet_cv import _load_cfg  # noqa: E402
from fomo_servo.centernet.annotations import load_pool_samples, split_fold  # noqa: E402

FAMILIES = {"B-box": "fomo_bbox", "B-box50": "fomo_bbox50"}
EPOCHS = (20, 40, 60, 100, 150, 200)


def cmd_train(args: argparse.Namespace) -> None:
    """Save held-out float32 logits ``[N,C,G,G]`` at the fixed snapshot epochs."""
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
                kind, max(EPOCHS), train_samples, cfg, args.init_weights,
                log=lambda m: print(m, flush=True), snapshot_epochs=EPOCHS, on_snapshot=snapshot,
            )
            meta = {
                "family": family, "kind": kind, "held_out": held_out,
                "epochs": list(EPOCHS), "init": report,
                "history": history, "seconds": time.time() - started,
                "test_images_list": [s.image_path.name for s in test_samples],
                "overlap_ownership": "keep_first; normalise object_weight over owned cells",
            }
            (target / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--init-weights", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    cmd_train(parser.parse_args())


if __name__ == "__main__":
    main()
