"""Task 07 section 3: train C60 on all 213 images (v2_vis), export ONNX + sidecar, parity, timing."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fomo_servo.centernet.annotations import load_pool_samples  # noqa: E402
from fomo_servo.centernet.export import export_centernet_onnx  # noqa: E402
from fomo_servo.centernet.training import train_model  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bench(session, x: np.ndarray, warmup: int, runs: int) -> dict:
    name = session.get_inputs()[0].name
    for _ in range(warmup):
        session.run(None, {name: x})
    times = []
    for _ in range(runs):
        t = time.perf_counter()
        session.run(None, {name: x})
        times.append((time.perf_counter() - t) * 1000.0)
    return {"median_ms": float(np.median(times)), "p95_ms": float(np.percentile(times, 95)), "mean_ms": float(np.mean(times))}


def main() -> None:
    import onnxruntime as ort

    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=ROOT / "configs/experiments/centernet_lite_cv.yaml")
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--init-weights", type=Path, required=True)
    p.add_argument("--baseline-onnx", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)

    samples = load_pool_samples(args.dataset_root, use_visibility=True)
    model, history, report = train_model("centernet", cfg["training"]["runs"]["C60"]["epochs"], samples, cfg, args.init_weights)
    weights = args.out / "centernet_lite_c60_all213.pt"
    torch.save(model.state_dict(), weights)
    onnx_path = export_centernet_onnx(model, args.out / "centernet_lite_c60_all213.onnx", opset=17)

    m = cfg["model"]
    grid = m["input_size"] // m["output_stride"]
    classes = ["fish", "jellyfish", "penguin", "puffin", "shark", "starfish", "stingray"]
    sidecar = {
        "head": "centernet_lite_v1",
        "onnx_sha256": sha256(onnx_path),
        "weights_sha256": sha256(weights),
        "init_weights_sha256": m["init_sha256"],
        "input": {"shape": [1, 3, m["input_size"], m["input_size"]], "range": "0..1 RGB, letterbox pad 114"},
        "output": {
            "shape": [1, len(classes) + 4, grid, grid],
            "channels": {
                "heat": {"start": 0, "count": len(classes), "activation": "sigmoid (in graph)"},
                "offset": {"start": len(classes), "count": 2, "activation": "sigmoid (in graph)", "meaning": "dx, dy within the cell, 0..1"},
                "log_size": {"start": len(classes) + 2, "count": 2, "activation": "none", "meaning": "log(w), log(h) in grid cells"},
            },
        },
        "classes": classes,
        "stride": m["output_stride"],
        "decode": {"peak": "3x3 max-pool equality", "score_threshold": None, "score_threshold_note": "TBD: not chosen; round-1/2 CV showed 0.4 is too high for this head"},
        "training": {"epochs": 60, "seed": cfg["training"]["seed"], "images": len(samples),
                     "targets": sum(len(s.boxes) for s in samples), "amp_skipped_steps": report["amp_skipped_steps"]},
    }
    (args.out / "centernet_lite_c60_all213.sidecar.json").write_text(json.dumps(sidecar, indent=1), encoding="utf-8")

    # parity PyTorch vs ORT on 16 real letterboxed frames
    from fomo_servo.centernet.training import _read_rgb
    from fomo_servo.geometry.letterbox import letterbox_rgb

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    baseline = ort.InferenceSession(str(args.baseline_onnx), providers=["CPUExecutionProvider"])
    model.cpu().eval()
    worst = 0.0
    for s in samples[:16]:
        lb, _ = letterbox_rgb(_read_rgb(s.image_path), m["input_size"])
        x = (np.ascontiguousarray(lb.transpose(2, 0, 1), dtype=np.float32) / 255.0)[None]
        with torch.no_grad():
            expected = model(torch.from_numpy(x)).numpy()
        actual = session.run(None, {session.get_inputs()[0].name: x})[0]
        worst = max(worst, float(np.abs(actual - expected).max()))
    x = np.random.default_rng(0).random((1, 3, m["input_size"], m["input_size"]), dtype=np.float32)
    timing = {"centernet_lite": bench(session, x, 50, 500), "baseline_d45c3fb3": bench(baseline, x, 50, 500)}
    result = {
        "onnx_bytes": onnx_path.stat().st_size, "baseline_onnx_bytes": args.baseline_onnx.stat().st_size,
        "baseline_onnx_sha256": sha256(args.baseline_onnx),
        "params": sum(q.numel() for q in model.parameters()),
        "parity_max_abs_diff_16_frames": worst, "ort_cpu_timing_ms": timing,
        "ort": ort.__version__, "history_last": history[-1], "sidecar": sidecar["head"],
    }
    (args.out / "export_report.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
