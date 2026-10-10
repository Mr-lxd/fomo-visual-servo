"""Task 10 A': choose eight unchanged 640x480 training images for screen snapshots.

Uses the deployed ONNX artifact and shared RGB/letterbox preprocessing. Source
boxes and reported peaks are in original-image pixels; no training or tests.
"""
import argparse
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2
import numpy as np
from fomo_servo.inference.ort_predictor import OnnxRuntimePredictor
from fomo_servo.inference.preprocessing import preprocess_rgb_image, prediction_from_numpy_logits


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    predictor = OnnxRuntimePredictor.from_files(a.model, a.report)
    contract = predictor.contract
    candidates = []
    for annotation in sorted((a.dataset_root / "images/train").glob("*.json")):
        if annotation.name.startswith("pool-20260831-005"):
            continue
        payload = json.loads(annotation.read_text(encoding="utf-8"))
        shapes = [s for s in payload["shapes"] if s["label"] in ("tuna", "fish", "jellyfish")
                  and s.get("attributes", {}).get("visibility", s.get("description") or "full") == "full"]
        if not shapes:
            continue
        image_path = annotation.parent / payload["imagePath"]
        bgr = cv2.imread(str(image_path))
        if bgr.shape != (480, 640, 3):
            continue
        prepared = preprocess_rgb_image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), input_size=contract.input_shape[-1])
        logits = predictor.predict_logits(prepared.input_tensor)
        prediction = prediction_from_numpy_logits(
            prepared, logits, class_names=contract.class_names, output_stride=contract.output_stride,
            confidence_threshold=.60, class_thresholds=contract.class_thresholds,
            component_mode=contract.component_mode, confidence_mode=contract.confidence_mode)
        z = logits[0]
        ex = np.exp(z-z.max(axis=0, keepdims=True))
        prob = ex/ex.sum(axis=0, keepdims=True)
        fg = prob[1:].max(axis=0)
        yy, xx = np.indices(fg.shape)
        tr = prepared.transform
        ox = ((xx+.5)*contract.output_stride-tr.pad_left)/tr.scale
        oy = ((yy+.5)*contract.output_stride-tr.pad_top)/tr.scale
        gy_global, gx_global = np.unravel_index(fg.argmax(), fg.shape)
        for si, s in enumerate(shapes):
            x0, x1 = max(0, min(v[0] for v in s["points"])), min(640, max(v[0] for v in s["points"]))
            y0, y1 = max(0, min(v[1] for v in s["points"])), min(480, max(v[1] for v in s["points"]))
            size = ((x1-x0)*(y1-y0))**.5
            source = s["label"]
            cid = 1 if source == "jellyfish" else 0
            mask = (ox>=x0)&(ox<=x1)&(oy>=y0)&(oy<=y1)
            if not mask.any():
                continue
            # Require a high correct-class response and an actual deploy-threshold detection.
            class_map = prob[cid+1]
            gy, gx = np.unravel_index(np.where(mask, class_map, -1).argmax(), fg.shape)
            score = float(class_map[gy,gx])
            hits = [d for d in prediction.detections if d.class_id == cid and x0<=d.original_x<=x1 and y0<=d.original_y<=y1]
            if score < .8 or not hits:
                continue
            if source == "tuna":
                group = "tuna_small" if size < 150 else "tuna_medium" if size < 250 else "tuna_large" if size >= 350 else None
                if group is None:
                    continue
            else:
                group = source
            candidates.append({"group": group, "source": image_path, "source_class": source,
                "gt_size_px": size, "roi_x0": x0, "roi_y0": y0, "roi_x1": x1, "roi_y1": y1,
                "target_score": score, "target_x": float(ox[gy,gx]), "target_y": float(oy[gy,gx]),
                "target_class": contract.class_names[cid], "detection_score_at_0_60": max(d.confidence for d in hits),
                "offline_highest_score": float(fg[gy_global,gx_global]),
                "offline_highest_x": float(ox[gy_global,gx_global]), "offline_highest_y": float(oy[gy_global,gx_global]),
                "offline_highest_class": contract.class_names[int(prob[1:,gy_global,gx_global].argmax())]})
    chosen = []
    used = set()
    for group, n in (("tuna_small",2),("tuna_medium",2),("tuna_large",2),("jellyfish",1),("fish",1)):
        available = sorted((r for r in candidates if r["group"]==group), key=lambda r: (-r["target_score"],str(r["source"])))
        selected = []
        for r in available:
            if r["source"] not in used:
                selected.append(r); used.add(r["source"])
                if len(selected)==n:
                    break
        assert len(selected)==n, f"not enough images for {group}: {len(selected)}/{n}"
        chosen.extend(selected)
    a.output.mkdir(parents=True, exist_ok=True)
    rows=[]
    for i, r in enumerate(chosen,1):
        source = r.pop("source")
        filename = f"{i:02d}_{r['group']}{source.suffix.lower()}"
        shutil.copyfile(source,a.output/filename)
        rows.append({"id": f"{i:02d}", "file": filename, **r, "source_path": str(source),
                     "image_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                     "model_sha256": contract.onnx_sha256})
    with (a.output/"expected.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(json.dumps(rows,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
