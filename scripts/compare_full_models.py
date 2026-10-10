"""Task 08 section 3: new full B vs d45c3fb3 on the 213 training images and 31 frozen frames.

The 213 images are training data for both models (optimistic numbers). The frozen frames
are only used to COUNT detections; no labels exist and nothing is evaluated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fomo_servo.centernet.annotations import IMAGE_SUFFIXES, load_pool_samples  # noqa: E402
from fomo_servo.centernet.evaluation import Pred, evaluate_threshold  # noqa: E402
from fomo_servo.geometry.letterbox import letterbox_rgb  # noqa: E402
from fomo_servo.postprocess import postprocess_numpy_logits  # noqa: E402

NAMES = ("fish", "jellyfish", "penguin", "puffin", "shark", "starfish", "stingray")


def run_model(session, paths):
    outputs, transforms = [], []
    name = session.get_inputs()[0].name
    for p in paths:
        rgb = cv2.cvtColor(cv2.imread(str(p), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        lb, tf = letterbox_rgb(rgb, 192)
        x = (np.ascontiguousarray(lb.transpose(2, 0, 1), dtype=np.float32) / 255.0)[None]
        outputs.append(session.run(None, {name: x})[0])
        transforms.append(tf)
    return outputs, transforms


def detect(outputs, transforms, thr):
    return [
        postprocess_numpy_logits(o, class_names=NAMES, stride=8, transforms=(t,), confidence_threshold=thr)[0]
        for o, t in zip(outputs, transforms)
    ]


def main() -> None:
    import onnxruntime as ort

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=Path, required=True)
    ap.add_argument("--old-onnx", type=Path, required=True)
    ap.add_argument("--new-onnx", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    samples = load_pool_samples(a.dataset_root, use_visibility=True)
    train_paths = [s.image_path for s in samples]
    gts = [list(s.boxes) for s in samples]
    frozen = sorted(p for p in (a.dataset_root / "images" / "test").iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)

    result: dict = {"train_images": len(train_paths), "frozen_frames": len(frozen)}
    for label, path, thr in (("d45c3fb3", a.old_onnx, 0.40), ("new_e150", a.new_onnx, 0.60)):
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        outs, tfs = run_model(session, train_paths)
        entry = {"onnx_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "own_threshold": thr}
        for t in sorted({thr, 0.40, 0.60}):
            dets = detect(outs, tfs, t)
            preds = [[Pred(d.class_id, d.confidence, d.original_x, d.original_y) for d in img] for img in dets]
            entry["thr_{:.2f}".format(t)] = {
                m: {k: v for k, v in evaluate_threshold(preds, gts, class_agnostic=ag, adjacent_distance=81.0).items()
                    if k in ("tp", "fp", "fn", "precision", "recall", "f1")}
                for m, ag in (("agnostic", True), ("class_aware", False))
            }
        f_outs, f_tfs = run_model(session, frozen)  # counting only
        for t in sorted({thr, 0.40}):
            dets = detect(f_outs, f_tfs, t)
            entry["frozen_thr_{:.2f}".format(t)] = {
                "frames_with_detection": sum(1 for d in dets if d),
                "total_detections": sum(len(d) for d in dets),
            }
        x = np.random.default_rng(0).random((1, 3, 192, 192), dtype=np.float32)
        name = session.get_inputs()[0].name
        for _ in range(50):
            session.run(None, {name: x})
        times = []
        for _ in range(500):
            t0 = time.perf_counter()
            session.run(None, {name: x})
            times.append((time.perf_counter() - t0) * 1000.0)
        entry["ort_cpu_ms"] = {"median": float(np.median(times)), "p95": float(np.percentile(times, 95))}
        result[label] = entry
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
