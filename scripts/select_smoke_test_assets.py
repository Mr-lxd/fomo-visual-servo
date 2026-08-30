"""Select deterministic Raspberry Pi deployment smoke-test images from one split.

The script scans one dataset split's ``images`` directory in ascending POSIX
relative-path order with the official ONNX Runtime predictor at the sidecar's
fixed confidence threshold, and records the first ``--max-positive`` images
that produce at least one non-empty detection plus the first
``--max-negative`` images that produce zero detections. Selection is purely
order-based: no threshold tuning, no model or epoch reselection, and no
access to any split other than ``--split``.

The emitted manifest embeds only relative paths and content hashes, so the
same file drives Windows and Raspberry Pi cross-platform regression checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

from fomo_servo.inference import (
    InferenceError,
    OnnxRuntimePredictor,
    OrtPredictorError,
    OutputPathError,
    PreprocessingError,
    read_rgb_image,
    validate_output_paths,
)
from fomo_servo.postprocess import PostprocessError


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select deterministic deployment smoke-test images from one split."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--onnx-report", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--media-output-dir", type=Path)
    parser.add_argument("--max-positive", type=int, default=4)
    parser.add_argument("--max-negative", type=int, default=1)
    return parser


def main(arguments: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(arguments)
    try:
        if args.max_positive < 1:
            raise InferenceError("--max-positive must be at least 1")
        if args.max_negative < 0:
            raise InferenceError("--max-negative must be at least 0")
        outputs = {"output_manifest": args.output_manifest}
        if args.media_output_dir is not None:
            outputs["media_output_dir"] = args.media_output_dir
        validate_output_paths(
            protected_inputs={
                "dataset_root": args.dataset_root,
                "onnx": args.onnx,
                "onnx_report": args.onnx_report,
            },
            outputs=outputs,
        )
        predictor = OnnxRuntimePredictor.from_files(args.onnx, args.onnx_report)
        manifest = _build_manifest(args, predictor)
        _write_manifest(args.output_manifest, manifest)
        if args.media_output_dir is not None:
            _copy_selected_media(
                dataset_root=args.dataset_root,
                media_dir=args.media_output_dir,
                manifest=manifest,
            )
    except (
        InferenceError,
        OrtPredictorError,
        OutputPathError,
        PreprocessingError,
        PostprocessError,
        ValueError,
    ) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    print(
        "selected {} positive and {} negative-control images "
        "(threshold {}); manifest: {}".format(
            manifest["selection"]["positive_selected_count"],
            manifest["selection"]["negative_selected_count"],
            manifest["model_contract"]["confidence_threshold"],
            args.output_manifest,
        )
    )
    return 0


def _build_manifest(
    args: argparse.Namespace, predictor: OnnxRuntimePredictor
) -> dict:
    """Scan ``<split>/images`` in deterministic order and build the manifest."""

    dataset_root = Path(args.dataset_root)
    images_dir = dataset_root / args.split / "images"
    if not images_dir.is_dir():
        raise InferenceError(
            "split images directory does not exist: {}".format(images_dir)
        )
    candidates = sorted(
        (
            path
            for path in images_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda path: path.relative_to(dataset_root).as_posix(),
    )
    if not candidates:
        raise InferenceError(
            "no image files with extensions {} found under {}".format(
                sorted(IMAGE_EXTENSIONS), images_dir
            )
        )
    positives: list[dict] = []
    negatives: list[dict] = []
    scanned = 0
    for image_path in candidates:
        if len(positives) >= args.max_positive and len(negatives) >= args.max_negative:
            break
        image = read_rgb_image(image_path)
        scanned += 1
        prediction = predictor.predict_rgb_image(image)
        if prediction.detections and len(positives) < args.max_positive:
            positives.append(
                _case_record(
                    "positive", dataset_root, args.split, image_path, image, prediction.detections
                )
            )
        elif not prediction.detections and len(negatives) < args.max_negative:
            negatives.append(
                _case_record(
                    "negative_control",
                    dataset_root,
                    args.split,
                    image_path,
                    image,
                    prediction.detections,
                )
            )
    if not positives:
        raise InferenceError(
            "no '{}' image produced a non-empty detection at confidence "
            "threshold {}; keeping the threshold unchanged and stopping "
            "(scanned {} images)".format(
                args.split,
                predictor.contract.confidence_threshold,
                scanned,
            )
        )
    contract = predictor.contract
    return {
        "manifest_version": 1,
        "kind": "fomo_deployment_smoke_test",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_contract": {
            "artifact_name": contract.artifact_name,
            "checkpoint_epoch": contract.checkpoint_epoch,
            "checkpoint_seed": contract.checkpoint_seed,
            "onnx_file": contract.onnx_file,
            "onnx_sha256": contract.onnx_sha256,
            "onnx_size_bytes": contract.onnx_size_bytes,
            "onnx_report_file": Path(args.onnx_report).name,
            "onnx_report_sha256": _sha256_file(Path(args.onnx_report)),
            "opset": contract.opset,
            "input_name": contract.input_name,
            "input_shape": list(contract.input_shape),
            "input_color_order": contract.input_color_order,
            "input_value_range": list(contract.input_value_range),
            "output_name": contract.output_name,
            "output_shape": list(contract.output_shape),
            "output_semantic": contract.output_semantic,
            "output_stride": contract.output_stride,
            "confidence_threshold": contract.confidence_threshold,
            "class_names": list(contract.class_names),
            "component_mode": contract.component_mode,
            "confidence_mode": contract.confidence_mode,
        },
        "selection": {
            "dataset_split": args.split,
            "scan_order": "ascending posix relative path under <split>/images",
            "positive_rule": (
                "first {} scanned images with detection_count >= 1".format(
                    args.max_positive
                )
            ),
            "negative_rule": (
                "first {} scanned images with detection_count == 0".format(
                    args.max_negative
                )
            ),
            "threshold_or_model_reselected": False,
            "train_image_count": len(candidates),
            "scanned_image_count": scanned,
            "positive_selected_count": len(positives),
            "negative_selected_count": len(negatives),
        },
        "cases": sorted(
            positives + negatives, key=lambda case: case["image_path"]
        ),
        "verification": {
            "detection_count": "exact",
            "class_id_and_name": "exact",
            "centroid_original_pixels_tolerance": 2.0,
            "confidence_absolute_tolerance": 0.02,
            "note": (
                "Run scripts/predict_image.py --onnx --onnx-report per case on "
                "each platform and compare the JSON detections against the "
                "case records; cross-platform ORT CPU logits agree within the "
                "sidecar parity tolerance (rtol=1e-4, atol=1e-5), so only "
                "tiny centroid/confidence drift is expected."
            ),
        },
    }


def _case_record(
    role: str,
    dataset_root: Path,
    split: str,
    image_path: Path,
    image: object,
    detections: Sequence,
) -> dict:
    """Record one selected image with its contract-required provenance."""

    height, width = image.shape[:2]  # type: ignore[union-attr]
    return {
        "role": role,
        "split": split,
        "image_path": image_path.relative_to(dataset_root).as_posix(),
        "image_sha256": _sha256_file(image_path),
        "image_width": int(width),
        "image_height": int(height),
        "detection_count": len(detections),
        "detections": [detection.as_dict() for detection in detections],
    }


def _copy_selected_media(
    *, dataset_root: Path, media_dir: Path, manifest: dict
) -> None:
    """Copy exactly the selected case images and re-verify their SHA-256."""

    media_dir.mkdir(parents=True, exist_ok=True)
    for case in manifest["cases"]:
        source = dataset_root / case["image_path"]
        destination = media_dir / Path(case["image_path"]).name
        if destination.exists():
            raise InferenceError(
                "media output name collision: '{}' already exists".format(destination)
            )
        shutil.copyfile(source, destination)
        copied_sha256 = _sha256_file(destination)
        if copied_sha256 != case["image_sha256"]:
            raise InferenceError(
                "copied media SHA-256 mismatch for '{}': expected {}, got {}".format(
                    destination, case["image_sha256"], copied_sha256
                )
            )


def _write_manifest(output_manifest: Path, manifest: dict) -> None:
    output = Path(output_manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
