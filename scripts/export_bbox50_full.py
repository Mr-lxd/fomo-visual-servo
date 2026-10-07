"""Task 15: original-init B-box50 full training and Task 08 formal FOMO export.

Uses only images/train, with raw FP32 logits [1,8,24,24] from normalized
letterboxed RGB [1,3,192,192]. The fixed recipe is supplied by --config;
deployment metadata is separate. No selection, resume, or test-image reads.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from run_centernet_cv import _load_cfg  # noqa: E402
from export_centernet_full import bench  # noqa: E402
from fomo_servo.centernet.annotations import load_pool_samples  # noqa: E402
from fomo_servo.centernet.training import train_model  # noqa: E402
from fomo_servo.deployment.onnx_export import export_checkpoint_to_onnx  # noqa: E402
from fomo_servo.training.snapshots import sha256_file, write_epoch_snapshot  # noqa: E402


def main() -> None:
    """Train the final epoch, save provenance, export raw logits and record PC parity/timing."""
    import onnxruntime as ort

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "dataset-root", "init-weights", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--deployment-config", type=Path,
                        default=ROOT / "configs/export/bbox50_full250.yaml")
    args = parser.parse_args()
    cfg = _load_cfg(args.config)
    deploy = yaml.safe_load(args.deployment_config.read_text(encoding="utf-8"))
    samples = load_pool_samples(args.dataset_root, use_visibility=True)
    if len(samples) != deploy["train_images"]:
        raise ValueError(f"expected {deploy['train_images']} train images, got {len(samples)}")
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    model, history, initialization = train_model(
        "fomo_bbox50", deploy["epochs"], samples, cfg, args.init_weights,
        log=lambda message: print(message, flush=True),
    )
    training_seconds = time.time() - started

    # Keep the exact CV recipe alongside the formal export source description.
    recipe_path = args.out / "training_recipe.yaml"
    recipe_path.write_bytes(args.config.read_bytes())
    source = yaml.safe_load((args.deployment_config.parent / deploy["source_template"]).read_text(encoding="utf-8"))
    artifact = deploy["artifact_name"]
    source["project"]["name"] = artifact
    source["augmentation"] = cfg["augmentation"]
    source["training"] = dict(cfg["training"])
    source["training"].pop("runs", None)
    source["training"].update({"epochs": deploy["epochs"], "initialize_sha256": cfg["model"]["init_sha256"],
                               "output_dir": artifact, "resume": None,
                               "scheduler": {"name": "none"}, "checkpoint_policy": "fixed_final_epoch"})
    source["postprocess"]["inference_threshold"] = deploy["confidence_threshold"]
    source.pop("evaluation")
    source.pop("experiment")
    source["loss"] = {"type": "bbox_classification", "object_weight": cfg["fomo_loss"]["object_weight"],
                      "background_weight": cfg["fomo_loss"]["background_weight"]}
    source["bbox_training"] = {"kind": "fomo_bbox50", "central_fraction": 0.5,
                               "per_object_positive_weight_normalized": True,
                               "ignore_masked": True, "recipe_file": recipe_path.name,
                               "recipe_sha256": sha256_file(recipe_path)}
    source_path = args.out / f"{artifact}.yaml"
    source_path.write_text(yaml.safe_dump(source, sort_keys=False), encoding="utf-8")
    fingerprint = hashlib.sha256(json.dumps(
        {"recipe": cfg, "kind": "fomo_bbox50", "epochs": deploy["epochs"]},
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    dataset_digest = hashlib.sha256()
    for sample in samples:
        for path in (sample.image_path, sample.image_path.with_suffix(".json")):
            if not path.is_file():
                # Missing LabelMe JSON means an empty-GT image in load_pool_samples.
                dataset_digest.update((path.name + ":absent").encode("utf-8"))
                continue
            dataset_digest.update(path.name.encode("utf-8"))
            dataset_digest.update(bytes.fromhex(sha256_file(path)))
    commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    m = cfg["model"]
    metadata = {
        "backbone_name": "mobilenet_v2_fomo", "width_multiplier": m["width_multiplier"],
        "cut_point": model.cut_point, "cut_point_input_channels": 16,
        "cut_point_output_channels": 96, "output_stride": m["output_stride"],
        "head_channels": m["head_channels"], "pretrained": False,
        "initialization": "weights_only_checkpoint",
        "initialization_checkpoint_sha256": initialization["sha256"],
        "initialization_source_epoch": initialization["source_epoch"],
        "initialization_source_seed": initialization["source_seed"],
        "backbone_parameter_count": sum(p.numel() for p in model.backbone.parameters()),
        "head_parameter_count": sum(p.numel() for p in model.head.parameters()),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "training_kind": "fomo_bbox50",
    }
    checkpoint = write_epoch_snapshot(
        model=model, epoch=deploy["epochs"], output_dir=args.out, model_metadata=metadata,
        config_fingerprint=fingerprint, dataset_content_hash=dataset_digest.hexdigest(),
        git_commit_sha=commit, seed=cfg["training"]["seed"],
        augmentation_preset=cfg["augmentation"]["preset"],
        checkpoint_threshold=deploy["confidence_threshold"],
        loss_metadata={**cfg["fomo_loss"], **source["bbox_training"]},
    )
    with (args.out / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    summary = {"kind": "fomo_bbox50", "epochs": deploy["epochs"], "seed": cfg["training"]["seed"],
               "images": len(samples), "train_images_list": [s.image_path.name for s in samples],
               "nonignore_targets": sum(b.visibility != "ignore" for s in samples for b in s.boxes),
               "initialization": initialization, "training_seconds": training_seconds,
               "recipe_sha256": sha256_file(recipe_path), "config_fingerprint": fingerprint,
               "dataset_content_hash": dataset_digest.hexdigest(), "git_commit_sha": commit,
               "entry_sha256": sha256_file(Path(__file__)), "checkpoint_sha256": sha256_file(checkpoint)}
    (args.out / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    contract = yaml.safe_load((args.deployment_config.parent / deploy["export_template"]).read_text(encoding="utf-8"))
    contract["artifact"].update({"name": artifact, "source_experiment_config": source_path.name,
                                "source_experiment_config_sha256": sha256_file(source_path),
                                "validation_threshold": deploy["confidence_threshold"]})
    contract["checkpoint"].update({"sha256": sha256_file(checkpoint), "epoch": deploy["epochs"],
                                  "seed": cfg["training"]["seed"], "parameter_count": metadata["parameter_count"],
                                  "config_fingerprint": fingerprint})
    contract["postprocess"]["confidence_threshold"] = deploy["confidence_threshold"]
    export_config = args.out / f"{artifact}_onnx.yaml"
    export_config.write_text(yaml.safe_dump(contract, sort_keys=False), encoding="utf-8")
    onnx_path = args.out / f"{artifact}.onnx"
    sidecar_path = args.out / f"{artifact}.onnx.json"
    result = export_checkpoint_to_onnx(config_path=export_config, checkpoint_path=checkpoint,
                                     onnx_path=onnx_path, report_path=sidecar_path)

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    x = np.random.default_rng(0).random(tuple(contract["input"]["shape"]), dtype=np.float32)
    timing = bench(session, x, deploy["benchmark"]["warmup"], deploy["benchmark"]["runs"])
    report = {"onnx_sha256": result["onnx_sha256"], "onnx_bytes": onnx_path.stat().st_size,
              "checkpoint_sha256": sha256_file(checkpoint), "checkpoint_bytes": checkpoint.stat().st_size,
              "parity": result["parity"],
              "ort_cpu_timing_ms": timing, "benchmark": deploy["benchmark"], "ort": ort.__version__}
    (args.out / "export_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
