"""Run the explicit Edge Impulse float32 TFLite parity protocol at threshold 0.5."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import zipfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, Optional, Sequence

import cv2
import numpy as np
import torch

from fomo_servo.config import ConfigurationError, ProjectConfig, load_config
from fomo_servo.datasets import YOLOv5FOMODataset
from fomo_servo.deployment import (
    TFLiteRuntimeError,
    create_tflite_interpreter,
    inspect_tflite_interpreter,
    output_looks_like_probabilities,
    prepare_tflite_input,
)
from fomo_servo.evaluation.edge_impulse import EdgeImpulseDetection, decode_edge_impulse_fomo
from fomo_servo.evaluation.parity_clean import (
    ParityCleanError,
    ParityCleanView,
    verify_parity_clean_view,
)
from fomo_servo.evaluation.parity_reporting import (
    edge_ground_truths_from_local,
    evaluate_local_parity,
    serialize_edge_evaluation,
)
from fomo_servo.geometry import LetterboxTransform, letterbox_rgb
from fomo_servo.metrics import ground_truths_from_boxes
from fomo_servo.postprocess import Detection, postprocess_probabilities


_EI_PARITY_THRESHOLD = 0.5


class EdgeImpulseTFLiteError(RuntimeError):
    """Raised when the evidence-preserving TFLite parity protocol is invalid."""


@dataclass(frozen=True)
class ResolvedTFLiteModel:
    """One selected TFLite file plus provenance from a file or safely extracted ZIP."""

    model_path: Path
    model_relative_path: str
    model_sha256: str
    model_size_bytes: int
    archive_path: Optional[Path]
    archive_sha256: Optional[str]
    extracted_directory: Optional[Path]
    candidate_paths: tuple[Path, ...]
    support_files: tuple[str, ...]
    metadata_hints: Mapping[str, str]


def build_parser() -> argparse.ArgumentParser:
    """Build the fixed-0.5 no-sweep TFLite parity command line."""

    parser = argparse.ArgumentParser(
        description="Evaluate an EI TFLite FOMO model on parity-clean-v1 test data."
    )
    parser.add_argument("--model", type=Path, required=True, help="EI export ZIP or .tflite file")
    parser.add_argument("--model-entry", default=None, help="Exact .tflite member path when a ZIP has multiple candidates")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--cleaning-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--raw-output-cache",
        type=Path,
        default=None,
        help="Prior compatible parity output directory; reuses saved .npy tensors without invoking TFLite.",
    )
    parser.add_argument("--expected-image-count", type=int, default=63)
    parser.add_argument(
        "--float-input-scale",
        type=float,
        default=None,
        help="Required only for a bare .tflite without exported DSP evidence.",
    )
    return parser


def resolve_tflite_model(model: Path, *, model_entry: Optional[str]) -> ResolvedTFLiteModel:
    """Resolve one TFLite model; ZIPs are extracted to a fresh external directory."""

    source = Path(model)
    if not source.is_file():
        raise EdgeImpulseTFLiteError("model input does not exist: {}".format(source))
    suffix = source.suffix.lower()
    if suffix == ".tflite":
        if model_entry is not None:
            raise EdgeImpulseTFLiteError("--model-entry is valid only for a ZIP input")
        return ResolvedTFLiteModel(
            model_path=source,
            model_relative_path=source.name,
            model_sha256=_sha256_file(source),
            model_size_bytes=source.stat().st_size,
            archive_path=None,
            archive_sha256=None,
            extracted_directory=None,
            candidate_paths=(source,),
            support_files=(),
            metadata_hints={},
        )
    if suffix != ".zip":
        raise EdgeImpulseTFLiteError("--model must be a .tflite file or .zip export")
    archive_hash = _sha256_file(source)
    destination = Path(tempfile.mkdtemp(prefix="ei-fomo-fp32-"))
    try:
        with zipfile.ZipFile(source) as archive:
            _safe_extract(archive, destination)
            member_names = tuple(item.filename.replace("\\", "/") for item in archive.infolist() if not item.is_dir())
    except (OSError, zipfile.BadZipFile, RuntimeError) as error:
        raise EdgeImpulseTFLiteError("unable to inspect or extract ZIP '{}': {}".format(source, error)) from error
    candidates = tuple(sorted(destination / Path(name) for name in member_names if name.lower().endswith(".tflite")))
    if not candidates:
        raise EdgeImpulseTFLiteError("ZIP contains no .tflite model")
    selected: Path
    if model_entry is None:
        if len(candidates) != 1:
            raise EdgeImpulseTFLiteError(
                "ZIP contains multiple .tflite models; specify --model-entry exactly: {}".format(
                    ", ".join(path.relative_to(destination).as_posix() for path in candidates)
                )
            )
        selected = candidates[0]
    else:
        normalized_entry = model_entry.replace("\\", "/")
        matches = [path for path in candidates if path.relative_to(destination).as_posix() == normalized_entry]
        if len(matches) != 1:
            raise EdgeImpulseTFLiteError("--model-entry is not a .tflite member of the ZIP: {}".format(model_entry))
        selected = matches[0]
    support_files = tuple(
        name for name in member_names if re.search(r"metadata|variables|labels|trained", Path(name).name, re.IGNORECASE)
    )
    return ResolvedTFLiteModel(
        model_path=selected,
        model_relative_path=selected.relative_to(destination).as_posix(),
        model_sha256=_sha256_file(selected),
        model_size_bytes=selected.stat().st_size,
        archive_path=source,
        archive_sha256=archive_hash,
        extracted_directory=destination,
        candidate_paths=candidates,
        support_files=support_files,
        metadata_hints=_read_metadata_hints(destination),
    )


def run_tflite_parity(
    *,
    config: ProjectConfig,
    dataset_root: Path,
    cleaning_manifest: Mapping[str, object],
    resolved_model: ResolvedTFLiteModel,
    expected_image_count: int,
    output_dir: Path,
    float_input_scale: Optional[float] = None,
    raw_output_cache: Optional[Path] = None,
) -> dict[str, object]:
    """Run raw-RGB, zero-padded EI TFLite inference and both evaluator columns."""

    if expected_image_count <= 0:
        raise EdgeImpulseTFLiteError("expected_image_count must be positive")
    candidate_inspections = []
    for candidate in resolved_model.candidate_paths:
        interpreter, backend = create_tflite_interpreter(candidate)
        candidate_info = inspect_tflite_interpreter(interpreter, model_sha256=_sha256_file(candidate))
        candidate_info["relative_path"] = (
            candidate.relative_to(resolved_model.extracted_directory).as_posix()
            if resolved_model.extracted_directory is not None
            else candidate.name
        )
        candidate_info["backend"] = backend
        candidate_inspections.append(candidate_info)

    interpreter, backend = create_tflite_interpreter(resolved_model.model_path)
    tensor_info = inspect_tflite_interpreter(interpreter, model_sha256=resolved_model.model_sha256)
    tensor_info["backend"] = backend
    _validate_tflite_contract(tensor_info, config)
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    resolved_float_input_scale = _resolve_float_input_scale(
        resolved_model.metadata_hints, float_input_scale
    )

    view = ParityCleanView(dataset_root, cleaning_manifest, len(config.dataset.class_names))
    cleaning_hashes = verify_parity_clean_view(
        dataset_root, cleaning_manifest, len(config.dataset.class_names)
    )
    cached_outputs = _load_raw_output_cache(
        raw_output_cache,
        model_sha256=resolved_model.model_sha256,
        cleaning_hashes=cleaning_hashes,
    )
    dataset = YOLOv5FOMODataset(
        root=dataset_root,
        split="test",
        input_size=config.model.input_size,
        stride=config.model.output_stride,
        class_mode=config.dataset.class_mode,
        merged_class_name=config.dataset.merged_class_name,
        collision_policy=config.dataset.collision_policy,
        augmentation=None,
        train_split=config.dataset.train_split,
        augmentation_seed=config.training.seed,
        label_loader=view.parse_label_file,
    )
    if len(dataset) != expected_image_count:
        raise EdgeImpulseTFLiteError(
            "cleaned test image count is {}; expected {}".format(len(dataset), expected_image_count)
        )
    if cached_outputs is not None:
        expected_cache_paths = {
            sample.image_path.relative_to(dataset_root).as_posix() for sample in dataset
        }
        if set(cached_outputs) != expected_cache_paths:
            raise EdgeImpulseTFLiteError(
                "raw-output cache image set does not exactly match the cleaned test split"
            )
    output_dir = prepare_output_dir(output_dir)
    raw_dir = output_dir / "raw_outputs"
    preprocessed_dir = output_dir / "preprocessed_images"
    raw_dir.mkdir()
    preprocessed_dir.mkdir()

    local_predictions: list[tuple[Detection, ...]] = []
    edge_predictions: list[tuple[EdgeImpulseDetection, ...]] = []
    ground_truths = []
    image_sizes: list[tuple[int, int]] = []
    images: list[dict[str, object]] = []
    global_input_min = float("inf")
    global_input_max = float("-inf")
    probability_checks = []
    for index, sample in enumerate(dataset):
        # EI protocol is RGB, Fit-longest-axis, and zero padding. The local
        # evaluator's own 114-padding preprocessing is not reused here.
        preprocessed, transform = letterbox_rgb(sample.original_image, config.model.input_size, pad_value=0)
        input_tensor = prepare_tflite_input(
            preprocessed, input_detail, float_scale=resolved_float_input_scale
        )
        global_input_min = min(global_input_min, float(input_tensor.min()))
        global_input_max = max(global_input_max, float(input_tensor.max()))
        relative_image_path = sample.image_path.relative_to(dataset_root).as_posix()
        cached_path = cached_outputs.get(relative_image_path) if cached_outputs is not None else None
        if cached_path is not None:
            try:
                output = np.load(cached_path, allow_pickle=False)
            except (OSError, ValueError) as error:
                raise EdgeImpulseTFLiteError(
                    "unable to load cached TFLite output '{}': {}".format(cached_path, error)
                ) from error
            output_source = "cache"
        else:
            try:
                interpreter.set_tensor(int(input_detail["index"]), input_tensor)
                interpreter.invoke()
                output = np.asarray(interpreter.get_tensor(int(output_detail["index"]))).copy()
            except (AttributeError, RuntimeError, ValueError) as error:
                raise EdgeImpulseTFLiteError(
                    "TFLite inference failed for '{}': {}".format(sample.image_path, error)
                ) from error
            output_source = "inference"
        is_probability = output_looks_like_probabilities(output)
        probability_checks.append(is_probability)
        if not is_probability:
            raise EdgeImpulseTFLiteError(
                "TFLite output is not normalized probabilities; refusing to add an unverified softmax"
            )
        nhwc_probability = _validate_output_shape(output, config)
        nchw_probability = torch.from_numpy(nhwc_probability).permute(0, 3, 1, 2).contiguous()
        local = postprocess_probabilities(
            nchw_probability,
            class_names=config.dataset.class_names,
            stride=config.model.output_stride,
            transforms=(transform,),
            confidence_threshold=_EI_PARITY_THRESHOLD,
            class_thresholds=None,
            component_mode=config.postprocess.component_mode,
            confidence_mode=config.postprocess.confidence_mode,
        )[0]
        edge = _inverse_edge_detections(
            decode_edge_impulse_fomo(
                nhwc_probability[0],
                class_names=config.dataset.class_names,
                input_size=(config.model.input_size, config.model.input_size),
                threshold=_EI_PARITY_THRESHOLD,
            ),
            transform,
        )
        prefix = "{:03d}_{}".format(index, sample.image_path.stem)
        output_file = raw_dir / "{}.npy".format(prefix)
        image_file = preprocessed_dir / "{}.png".format(prefix)
        np.save(output_file, output)
        if not cv2.imwrite(str(image_file), cv2.cvtColor(preprocessed, cv2.COLOR_RGB2BGR)):
            raise EdgeImpulseTFLiteError("unable to save preprocessed image: {}".format(image_file))
        targets = tuple(ground_truths_from_boxes(sample.original_boxes, config.dataset.class_names))
        local_predictions.append(local)
        edge_predictions.append(edge)
        ground_truths.append(targets)
        image_sizes.append((transform.original_width, transform.original_height))
        images.append(
            {
                "image_path": relative_image_path,
                "label_path": sample.label_path.relative_to(dataset_root).as_posix(),
                "original_size": [transform.original_width, transform.original_height],
                "preprocessed_image": image_file.relative_to(output_dir).as_posix(),
                "raw_output_tensor": output_file.relative_to(output_dir).as_posix(),
                "raw_output_source": output_source,
                "output_shape": list(output.shape),
                "output_looks_like_probabilities": is_probability,
                "letterbox": _letterbox_dict(transform),
                "active_cells": _active_cells(nhwc_probability[0], config.dataset.class_names),
                "edge_impulse_fused_detections": [_edge_detection_dict(item) for item in edge],
                "local_predictions": [item.as_dict() for item in local],
                "ground_truths": [_ground_truth_dict(item) for item in targets],
            }
        )

    local_report = evaluate_local_parity(
        predictions=local_predictions,
        ground_truths=ground_truths,
        class_names=config.dataset.class_names,
        matching_mode=config.evaluation.matching_mode,
        max_distance_pixels=config.evaluation.max_distance_pixels,
    )
    edge_ground_truths = tuple(edge_ground_truths_from_local(items) for items in ground_truths)
    legacy_report = serialize_edge_evaluation(
        predictions=edge_predictions,
        ground_truths=edge_ground_truths,
        image_sizes=image_sizes,
        class_names=config.dataset.class_names,
        mode="edge_impulse_legacy",
    )
    strict_report = serialize_edge_evaluation(
        predictions=edge_predictions,
        ground_truths=edge_ground_truths,
        image_sizes=image_sizes,
        class_names=config.dataset.class_names,
        mode="strict_one_to_one",
    )
    for index, image in enumerate(images):
        image["local_current"] = local_report["image_results"][index]
        image["edge_impulse_legacy"] = legacy_report["image_results"][index]
        image["strict_one_to_one"] = strict_report["image_results"][index]

    return {
        "protocol": "edge-impulse-tflite-parity-v1",
        "split": "test",
        "image_count": len(dataset),
        "threshold": _EI_PARITY_THRESHOLD,
        "model": {
            "path": str(resolved_model.model_path),
            "relative_path_in_zip": resolved_model.model_relative_path,
            "size_bytes": resolved_model.model_size_bytes,
            "sha256": resolved_model.model_sha256,
            "zip_path": str(resolved_model.archive_path) if resolved_model.archive_path else None,
            "zip_sha256": resolved_model.archive_sha256,
            "extracted_directory": str(resolved_model.extracted_directory) if resolved_model.extracted_directory else None,
            "support_files": list(resolved_model.support_files),
            "metadata_hints": dict(resolved_model.metadata_hints),
            "candidate_tensor_inspections": candidate_inspections,
        },
        "tensor_inspection": tensor_info,
        "cleaning_manifest": {
            "protocol": cleaning_manifest.get("protocol"),
            "cleaning_view_hash": cleaning_hashes["cleaning_view_hash"],
            "cleaned_test_view_hash": cleaning_hashes["cleaned_test_view_hash"],
        },
        "preprocessing": {
            "color_order": "RGB",
            "resize": "fit_longest_axis",
            "padding_value": 0,
            "padding": "letterbox_no_crop",
            "input_layout": "NHWC",
            "float_input_scale": resolved_float_input_scale,
            "float_input_range_observed": [global_input_min, global_input_max],
            "normalization": "RGB multiplied by {}".format(resolved_float_input_scale),
            "evidence": "ZIP DSP extraction source plus actual input dtype/quantization inspection",
        },
        "output": {
            "softmax_in_graph": tensor_info["graph_has_softmax"],
            "all_image_outputs_look_like_probabilities": all(probability_checks),
            "softmax_applied_by_evaluator": False,
        },
        "raw_output_cache": (
            {"enabled": False}
            if raw_output_cache is None
            else {"enabled": True, "source_directory": str(raw_output_cache)}
        ),
        "local_current_evaluator": local_report,
        "edge_impulse_legacy_evaluator": legacy_report,
        "strict_one_to_one_evaluator": strict_report,
        "images": images,
    }


def prepare_output_dir(path: Path) -> Path:
    """Create a fresh output directory; prior audit records are immutable."""

    destination = Path(path)
    if destination.exists():
        raise EdgeImpulseTFLiteError("refusing to overwrite existing output directory: {}".format(destination))
    try:
        destination.mkdir(parents=True, exist_ok=False)
    except OSError as error:
        raise EdgeImpulseTFLiteError("unable to create output directory '{}': {}".format(destination, error)) from error
    return destination


def _safe_extract(archive: zipfile.ZipFile, destination: Path) -> None:
    root = destination.resolve()
    for member in archive.infolist():
        target = (destination / member.filename).resolve()
        try:
            target.relative_to(root)
        except ValueError as error:
            raise EdgeImpulseTFLiteError("ZIP member path escapes extraction directory: {}".format(member.filename)) from error
    archive.extractall(destination)


def _read_metadata_hints(root: Path) -> dict[str, str]:
    metadata_files = tuple(root.rglob("model_metadata.h"))
    if not metadata_files:
        return {"availability": "not found in model input"}
    content = metadata_files[0].read_text(encoding="utf-8", errors="replace")
    keys = (
        "EI_CLASSIFIER_INPUT_WIDTH",
        "EI_CLASSIFIER_INPUT_HEIGHT",
        "EI_CLASSIFIER_RESIZE_MODE",
        "EI_CLASSIFIER_TFLITE_INPUT_DATATYPE",
        "EI_CLASSIFIER_TFLITE_OUTPUT_DATATYPE",
        "EI_CLASSIFIER_QUANTIZATION_ENABLED",
        "EI_CLASSIFIER_HAS_DATA_NORMALIZATION",
        "EI_CLASSIFIER_LOAD_IMAGE_SCALING",
    )
    hints = {"source": metadata_files[0].relative_to(root).as_posix()}
    for key in keys:
        match = re.search(r"^#define\s+{}\s+(.+?)\s*$".format(re.escape(key)), content, re.MULTILINE)
        if match:
            hints[key] = match.group(1)
    dsp_path = root / "edge-impulse-sdk" / "classifier" / "ei_run_dsp.h"
    if dsp_path.is_file():
        dsp_content = dsp_path.read_text(encoding="utf-8", errors="replace")
        if "// rgb to 0..1" in dsp_content and "/ 255.0f" in dsp_content:
            hints["EI_DSP_EXTRACT_IMAGE_FEATURES_RGB_SCALE"] = "1/255"
            hints["EI_DSP_EXTRACT_IMAGE_FEATURES_SOURCE"] = dsp_path.relative_to(root).as_posix()
    return hints


def _resolve_float_input_scale(
    metadata_hints: Mapping[str, str], explicit_scale: Optional[float]
) -> float:
    if explicit_scale is not None:
        if not np.isfinite(explicit_scale) or explicit_scale <= 0.0:
            raise EdgeImpulseTFLiteError("--float-input-scale must be finite and positive")
        return float(explicit_scale)
    if metadata_hints.get("EI_DSP_EXTRACT_IMAGE_FEATURES_RGB_SCALE") == "1/255":
        return 1.0 / 255.0
    raise EdgeImpulseTFLiteError(
        "bare TFLite preprocessing is not inferable from tensor metadata; provide "
        "--float-input-scale or use the complete EI ZIP with DSP source"
    )


def _load_raw_output_cache(
    cache_directory: Optional[Path],
    *,
    model_sha256: str,
    cleaning_hashes: Mapping[str, str],
) -> Optional[dict[str, Path]]:
    """Validate and index a previous compatible per-image raw-output cache."""

    if cache_directory is None:
        return None
    root = Path(cache_directory)
    report_path = root / "parity_report.json"
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EdgeImpulseTFLiteError("unable to read raw-output cache report '{}': {}".format(report_path, error)) from error
    if not isinstance(report, Mapping):
        raise EdgeImpulseTFLiteError("raw-output cache report root must be a JSON object")
    model = report.get("model")
    clean = report.get("cleaning_manifest")
    images = report.get("images")
    if not isinstance(model, Mapping) or str(model.get("sha256", "")).lower() != model_sha256.lower():
        raise EdgeImpulseTFLiteError("raw-output cache model SHA-256 does not match selected TFLite model")
    if not isinstance(clean, Mapping) or clean.get("cleaning_view_hash") != cleaning_hashes["cleaning_view_hash"] or clean.get("cleaned_test_view_hash") != cleaning_hashes["cleaned_test_view_hash"]:
        raise EdgeImpulseTFLiteError("raw-output cache cleaning hashes do not match current manifest")
    if report.get("threshold") != _EI_PARITY_THRESHOLD:
        raise EdgeImpulseTFLiteError("raw-output cache threshold is not the fixed EI value 0.5")
    if not isinstance(images, list):
        raise EdgeImpulseTFLiteError("raw-output cache has no image records")
    indexed: dict[str, Path] = {}
    for image in images:
        if not isinstance(image, Mapping):
            raise EdgeImpulseTFLiteError("raw-output cache image record must be an object")
        image_path = image.get("image_path")
        tensor_path = image.get("raw_output_tensor")
        if not isinstance(image_path, str) or not isinstance(tensor_path, str):
            raise EdgeImpulseTFLiteError("raw-output cache image record lacks image_path or raw_output_tensor")
        if image_path in indexed:
            raise EdgeImpulseTFLiteError("raw-output cache has duplicate image_path: {}".format(image_path))
        candidate = (root / tensor_path).resolve()
        try:
            candidate.relative_to(root.resolve())
        except ValueError as error:
            raise EdgeImpulseTFLiteError("raw-output cache tensor path escapes cache directory") from error
        if not candidate.is_file():
            raise EdgeImpulseTFLiteError("raw-output cache tensor does not exist: {}".format(candidate))
        indexed[image_path] = candidate
    return indexed


def _validate_tflite_contract(info: Mapping[str, object], config: ProjectConfig) -> None:
    input_info = info.get("input")
    output_info = info.get("output")
    if not isinstance(input_info, Mapping) or not isinstance(output_info, Mapping):
        raise EdgeImpulseTFLiteError("TFLite tensor inspection is incomplete")
    expected_input = [1, config.model.input_size, config.model.input_size, 3]
    expected_output = [1, config.grid_size, config.grid_size, config.output_channels]
    if input_info.get("shape") != expected_input:
        raise EdgeImpulseTFLiteError("TFLite input shape {} does not match {}".format(input_info.get("shape"), expected_input))
    if output_info.get("shape") != expected_output:
        raise EdgeImpulseTFLiteError("TFLite output shape {} does not match {}".format(output_info.get("shape"), expected_output))
    if input_info.get("dtype") != "float32" or output_info.get("dtype") != "float32":
        raise EdgeImpulseTFLiteError("this parity protocol requires measured float32 input and output tensors")


def _validate_output_shape(output: np.ndarray, config: ProjectConfig) -> np.ndarray:
    expected = (1, config.grid_size, config.grid_size, config.output_channels)
    if output.shape != expected:
        raise EdgeImpulseTFLiteError("runtime output shape {} does not match {}".format(tuple(output.shape), expected))
    if output.dtype != np.float32:
        raise EdgeImpulseTFLiteError("runtime output dtype must be float32, got {}".format(output.dtype))
    return output


def _inverse_edge_detections(
    detections: Sequence[EdgeImpulseDetection], transform: LetterboxTransform
) -> tuple[EdgeImpulseDetection, ...]:
    output = []
    for detection in detections:
        original_x, original_y = transform.inverse_point(*detection.input_centroid)
        original_x = float(np.clip(original_x, 0.0, transform.original_width - 1.0))
        original_y = float(np.clip(original_y, 0.0, transform.original_height - 1.0))
        output.append(replace(detection, original_centroid=(original_x, original_y)))
    return tuple(output)


def _active_cells(probabilities: np.ndarray, class_names: Sequence[str]) -> list[dict[str, object]]:
    cells = []
    for grid_y in range(probabilities.shape[0]):
        for grid_x in range(probabilities.shape[1]):
            for class_id, class_name in enumerate(class_names):
                value = float(probabilities[grid_y, grid_x, class_id + 1])
                if value >= _EI_PARITY_THRESHOLD:
                    cells.append(
                        {
                            "grid_x": grid_x,
                            "grid_y": grid_y,
                            "class_id": class_id,
                            "class_name": class_name,
                            "probability": value,
                        }
                    )
    return cells


def _letterbox_dict(transform: LetterboxTransform) -> dict[str, object]:
    return {
        "input_size": transform.input_size,
        "scale": transform.scale,
        "pad_left": transform.pad_left,
        "pad_top": transform.pad_top,
        "pad_right": transform.pad_right,
        "pad_bottom": transform.pad_bottom,
    }


def _ground_truth_dict(value: object) -> dict[str, object]:
    from fomo_servo.metrics import GroundTruthCentroid

    if not isinstance(value, GroundTruthCentroid):
        raise EdgeImpulseTFLiteError("unexpected ground-truth centroid type")
    return {
        "class_id": value.class_id,
        "class_name": value.class_name,
        "original_centroid": [value.original_x, value.original_y],
        "original_bbox": [value.x_min, value.y_min, value.x_max, value.y_max],
    }


def _edge_detection_dict(value: EdgeImpulseDetection) -> dict[str, object]:
    return {
        "class_id": value.class_id,
        "class_name": value.class_name,
        "confidence": value.confidence,
        "input_bbox": list(value.input_bbox),
        "input_centroid": list(value.input_centroid),
        "original_centroid": list(value.original_centroid),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise EdgeImpulseTFLiteError("unable to hash '{}': {}".format(path, error)) from error
    return digest.hexdigest()


def _load_manifest(path: Path) -> Mapping[str, object]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EdgeImpulseTFLiteError("unable to read cleaning manifest '{}': {}".format(path, error)) from error
    if not isinstance(payload, Mapping):
        raise EdgeImpulseTFLiteError("cleaning manifest root must be a JSON object")
    return payload


def main(arguments: Optional[Sequence[str]] = None) -> int:
    """Run one fixed-0.5 Edge Impulse TFLite evaluation and persist evidence."""

    args = build_parser().parse_args(arguments)
    try:
        if not args.dataset_root.is_dir():
            raise EdgeImpulseTFLiteError("dataset root does not exist: {}".format(args.dataset_root))
        os.environ["FOMO_DATASET_ROOT"] = str(args.dataset_root)
        config = load_config(args.config)
        resolved = resolve_tflite_model(args.model, model_entry=args.model_entry)
        payload = run_tflite_parity(
            config=config,
            dataset_root=args.dataset_root,
            cleaning_manifest=_load_manifest(args.cleaning_manifest),
            resolved_model=resolved,
            expected_image_count=args.expected_image_count,
            output_dir=args.output_dir,
            float_input_scale=args.float_input_scale,
            raw_output_cache=args.raw_output_cache,
        )
        report_path = args.output_dir / "parity_report.json"
        report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"report": str(report_path), "summary": {
            "local_f1": payload["local_current_evaluator"]["f1"],
            "edge_impulse_legacy_f1": payload["edge_impulse_legacy_evaluator"]["f1"],
            "strict_one_to_one_f1": payload["strict_one_to_one_evaluator"]["f1"],
        }}, ensure_ascii=False))
    except (ConfigurationError, ParityCleanError, TFLiteRuntimeError, EdgeImpulseTFLiteError, OSError, RuntimeError, ValueError) as error:
        print("Error: {}".format(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
