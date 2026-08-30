"""Re-evaluate the six existing aug03/model01 checkpoints with protocol v2 metrics."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Optional, Sequence

import torch

from fomo_servo.config import ConfigurationError, load_config
from fomo_servo.evaluation import CheckpointSelectionError, collect_split_logits, evaluate_collected_logits
from fomo_servo.inference import InferenceError, load_inference_model
from fomo_servo.models import ModelConfigurationError
from fomo_servo.runtime import RuntimeDeviceError


ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit six legacy FOMO checkpoints with v2 metrics.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "experiments" / "checkpoint_selection_v2",
    )
    parser.add_argument("--tolerance", type=float, default=1e-6)
    return parser


def main(arguments: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(arguments)
    entries = (
        ("aug03", ROOT / "configs/experiments/aug03_underwater_conservative.yaml", ROOT / "outputs/experiments/aug03_underwater_conservative"),
        ("model01", ROOT / "configs/experiments/model01_mobilenet_v2_fomo_aug03.yaml", ROOT / "outputs/experiments/model01_mobilenet_v2_fomo_aug03"),
    )
    rows = []
    started = perf_counter()
    try:
        for experiment, config_path, directory in entries:
            config = load_config(config_path)
            metadata_path = directory / "experiment_metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
            for filename in ("best_centroid_f1.pt", "best_grid_f1.pt", "last.pt"):
                checkpoint = directory / filename
                model, device = load_inference_model(config, checkpoint, args.device)
                collection = collect_split_logits(config, model, device, config.dataset.validation_split)
                payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
                epoch = int(payload.get("epoch", 0)) if isinstance(payload, dict) else 0
                report = evaluate_collected_logits(
                    config,
                    collection,
                    epoch=epoch,
                    source_snapshot=filename,
                    checkpoint_path=checkpoint,
                ).as_dict()
                expected_fixed = payload.get("centroid_f1") if isinstance(payload, dict) else None
                fixed_delta = (
                    report["fixed_centroid_f1"] - float(expected_fixed)
                    if isinstance(expected_fixed, (int, float))
                    else None
                )
                expected_sweep = metadata.get("best_centroid_f1") if filename == "best_centroid_f1.pt" else None
                sweep_delta = (
                    report["sweep_centroid_f1"] - float(expected_sweep)
                    if isinstance(expected_sweep, (int, float))
                    else None
                )
                rows.append({
                    "experiment": experiment,
                    "checkpoint": filename,
                    "legacy_fixed_centroid_f1": expected_fixed,
                    "v2_fixed_centroid_f1": report["fixed_centroid_f1"],
                    "fixed_delta": fixed_delta,
                    "legacy_sweep_centroid_f1": expected_sweep,
                    "v2_sweep_centroid_f1": report["sweep_centroid_f1"],
                    "sweep_delta": sweep_delta,
                    "within_fixed_tolerance": fixed_delta is None or abs(fixed_delta) <= args.tolerance,
                    "within_sweep_tolerance": sweep_delta is None or abs(sweep_delta) <= args.tolerance,
                    **report,
                })
        _write_audit(args.output_dir, rows, perf_counter() - started)
    except (
        ConfigurationError,
        CheckpointSelectionError,
        InferenceError,
        ModelConfigurationError,
        RuntimeDeviceError,
        OSError,
        RuntimeError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        print("Error: {}".format(error), file=sys.stderr)
        return 1
    return 0


def _write_audit(output_dir: Path, rows: list[dict[str, object]], elapsed_seconds: float) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    scalar_keys = [
        "experiment", "checkpoint", "epoch", "legacy_fixed_centroid_f1", "v2_fixed_centroid_f1",
        "fixed_delta", "legacy_sweep_centroid_f1", "v2_sweep_centroid_f1", "sweep_delta",
        "within_fixed_tolerance", "within_sweep_tolerance", "grid_f1", "centroid_pr_auc_macro",
        "centroid_pr_auc_micro", "sweep_threshold", "mean_localization_error_pixels", "count_mae",
    ]
    with (output_dir / "existing_checkpoint_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=scalar_keys)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in scalar_keys} for row in rows)
    (output_dir / "existing_checkpoint_audit.json").write_text(
        json.dumps(
            {
                "checkpoint_selection_protocol": "v2",
                "checkpoint_count": len(rows),
                "elapsed_seconds": elapsed_seconds,
                "rows": rows,
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
