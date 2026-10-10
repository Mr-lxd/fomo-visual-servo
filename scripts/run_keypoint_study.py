"""Task 14: joint B-box50 keypoint training, four folds and fixed 250 epochs."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_centernet_cv import _load_cfg  # noqa: E402
from fomo_servo.centernet.annotations import split_fold  # noqa: E402
from fomo_servo.centernet.keypoints import load_keypoint_samples  # noqa: E402

FAMILY = "B-box50-kp"
KIND = "fomo_keypoint"
EPOCHS = (20, 40, 60, 100, 150, 200, 250)


def cmd_train(args: argparse.Namespace) -> None:
    """Cache logits [N,8,24,24] and offsets [N,4,24,24], both float32."""
    from fomo_servo.centernet.keypoint_training import predict_keypoint_raw, train_keypoint_model

    cfg = _load_cfg(args.config)
    issues = []
    samples = load_keypoint_samples(args.dataset_root, issues=issues)
    args.work.mkdir(parents=True, exist_ok=True)
    (args.work / "{}_annotation_issues.json".format(args.config.stem)).write_text(json.dumps(issues, indent=1), encoding="utf-8")
    for issue in issues:
        print("annotation:", issue, flush=True)
    inputs = []
    for sample in samples:
        for path in (sample.image_path, sample.image_path.with_suffix(".json")):
            if path.is_file():
                inputs.append({"path": str(path.relative_to(args.dataset_root)),
                               "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    provenance = {"config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
                  "dataset_root": str(args.dataset_root.resolve()), "inputs": inputs}
    (args.work / "{}_input_provenance.json".format(args.config.stem)).write_text(json.dumps(provenance, indent=1), encoding="utf-8")
    for held_out in cfg["data"]["held_out_sessions"]:
        train_samples, test_samples = split_fold(samples, held_out)
        target = args.work / "{}__{}".format(held_out, FAMILY)
        if (target / "meta.json").exists():
            print("skip", target.name, flush=True)
            continue
        target.mkdir(parents=True, exist_ok=True)
        started = time.time()

        def snapshot(epoch, model):
            outputs, offsets, _ = predict_keypoint_raw(
                model, test_samples, cfg["model"]["input_size"], cfg["training"]["device"],
            )
            path = target / "outputs_e{}.npz".format(epoch)
            np.savez_compressed(path, outputs=outputs, offsets=offsets)

        print("[{}] {} -> 250 epochs".format(held_out, FAMILY), flush=True)
        _, history, report = train_keypoint_model(
            max(EPOCHS), train_samples, cfg, args.init_weights,
            log=lambda message: print(message, flush=True), snapshot_epochs=EPOCHS, on_snapshot=snapshot,
        )
        meta = {
            "family": FAMILY, "kind": KIND, "held_out": held_out, "epochs": list(EPOCHS),
            "init": report, "history": history, "seconds": time.time() - started,
            "test_images_list": [s.image_path.name for s in test_samples],
            "config_sha256": provenance["config_sha256"],
            "offset_encoding": "signed_log1p_grid_cell_geometric_centre_hx_hy_tx_ty",
            "keypoint_supervision": "ignore regions and all centre collisions masked",
            "cache_sha256": {"outputs_e{}.npz".format(e): hashlib.sha256(
                (target / "outputs_e{}.npz".format(e)).read_bytes()).hexdigest() for e in EPOCHS},
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
