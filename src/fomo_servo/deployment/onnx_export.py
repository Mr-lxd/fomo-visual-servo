"""Validated fixed-shape checkpoint export and PyTorch/ONNX Runtime parity."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from math import isfinite
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
import yaml
from torch import Tensor, nn

from fomo_servo.models import MobileNetV2FOMONet, count_trainable_parameters
from fomo_servo.inference.path_safety import OutputPathError, validate_output_paths
from fomo_servo.training.snapshots import SnapshotError, load_epoch_snapshot


class OnnxExportError(RuntimeError):
    """Raised when an export contract, checkpoint, ONNX graph, or parity check fails."""


class _FixedOutputShapeWrapper(nn.Module):
    """Expose fixed raw-logit metadata without changing model tensor values."""

    def __init__(self, model: nn.Module, output_shape: tuple[int, int, int, int]) -> None:
        super().__init__()
        self.model = model
        self.output_shape = output_shape

    def forward(self, images: Tensor) -> Tensor:
        """Map NCHW RGB float32 input to the configured fixed raw-logit shape."""

        return self.model(images).reshape(self.output_shape)


@dataclass(frozen=True)
class OnnxExportContract:
    """Locked YAML contract for one fixed CPU-deployable FOMO artifact.

    ``input_shape`` is batch-one NCHW RGB float32 in the normalized 0..1 range.
    ``output_shape`` is batch-one raw float32 logits in stride-8 heatmap space.
    ``validation_threshold`` is provenance only and is never applied during export.
    """

    config_path: Path
    artifact_name: str
    source_experiment_config: str
    source_experiment_config_path: Path
    source_experiment_config_sha256: str
    validation_threshold: float
    checkpoint_sha256: str
    checkpoint_epoch: int
    checkpoint_seed: int
    checkpoint_parameter_count: int
    checkpoint_config_fingerprint: str
    backbone: str
    width_multiplier: float
    cut_point: str
    head_channels: int
    output_stride: int
    class_names: tuple[str, ...]
    pretrained: bool
    initialization: str
    pretrained_sha256: str | None
    initialization_checkpoint_sha256: str | None
    initialization_source_epoch: int | None
    initialization_source_seed: int | None
    input_name: str
    input_shape: tuple[int, int, int, int]
    input_dtype: str
    input_color_order: str
    input_value_range: tuple[float, float]
    output_name: str
    output_shape: tuple[int, int, int, int]
    output_dtype: str
    output_semantic: str
    opset: int
    parity_seed: int
    parity_rtol: float
    parity_atol: float
    confidence_threshold: float
    class_thresholds: Any
    component_mode: str
    confidence_mode: str
    selection_strategy: str
    max_match_distance_pixels: float
    max_lost_frames: int
    allowed_class_ids: tuple[int, ...] | None

    @property
    def input_size(self) -> int:
        """Return the fixed square input side in pixels."""

        return self.input_shape[-1]


def sha256_file(path: Path) -> str:
    """Return the lowercase SHA-256 of a file without loading it all into memory."""

    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise OnnxExportError("unable to hash '{}': {}".format(path, error)) from error
    return digest.hexdigest()


def load_export_contract(path: Path) -> OnnxExportContract:
    """Load a dataset-independent YAML contract for fixed NCHW/logits export.

    The loader reads only the supplied YAML. It neither expands dataset paths nor
    opens train, validation, or test data.
    """

    source = Path(path)
    try:
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    except OSError as error:
        raise OnnxExportError("unable to read export config '{}': {}".format(source, error)) from error
    except yaml.YAMLError as error:
        raise OnnxExportError("unable to parse export config '{}': {}".format(source, error)) from error
    root = _mapping(payload, "configuration root")
    artifact = _section(root, "artifact")
    checkpoint = _section(root, "checkpoint")
    model = _section(root, "model")
    input_config = _section(root, "input")
    output_config = _section(root, "output")
    onnx_config = _section(root, "onnx")
    parity = _section(root, "parity")
    postprocess = _section(root, "postprocess")

    input_shape = _shape(input_config.get("shape"), "input.shape")
    output_shape = _shape(output_config.get("shape"), "output.shape")
    output_stride = _positive_int(model.get("output_stride"), "model.output_stride")
    class_names = _class_names(model.get("classes"))
    if input_shape[0] != 1 or input_shape[1] != 3:
        raise OnnxExportError("input.shape must be fixed batch-one NCHW RGB [1,3,S,S]")
    if input_shape[2] != input_shape[3] or input_shape[2] % output_stride != 0:
        raise OnnxExportError("input.shape must be square and divisible by model.output_stride")
    expected_output_shape = (
        1,
        len(class_names) + 1,
        input_shape[2] // output_stride,
        input_shape[3] // output_stride,
    )
    if output_shape != expected_output_shape:
        raise OnnxExportError(
            "output.shape must be {} for background plus {} classes at stride {}".format(
                list(expected_output_shape), len(class_names), output_stride
            )
        )
    backbone = _text(model.get("backbone"), "model.backbone")
    cut_point = _text(model.get("cut_point"), "model.cut_point")
    if backbone != "mobilenet_v2_fomo":
        raise OnnxExportError("model.backbone must be 'mobilenet_v2_fomo' for this exporter")
    if cut_point != "block_6_expand_relu":
        raise OnnxExportError("model.cut_point must be 'block_6_expand_relu'")
    pretrained = _boolean(model.get("pretrained"), "model.pretrained")
    initialization = _text(model.get("initialization"), "model.initialization")
    if pretrained:
        pretrained_sha256: str | None = _sha256(
            model.get("pretrained_sha256"), "model.pretrained_sha256"
        )
        initialization_checkpoint_sha256: str | None = None
        initialization_source_epoch: int | None = None
        initialization_source_seed: int | None = None
    else:
        if initialization != "weights_only_checkpoint":
            raise OnnxExportError(
                "model.pretrained=false requires initialization='weights_only_checkpoint'"
            )
        pretrained_sha256 = None
        initialization_checkpoint_sha256 = _sha256(
            model.get("initialization_checkpoint_sha256"),
            "model.initialization_checkpoint_sha256",
        )
        initialization_source_epoch = _positive_int(
            model.get("initialization_source_epoch"),
            "model.initialization_source_epoch",
        )
        initialization_source_seed = _nonnegative_int(
            model.get("initialization_source_seed"),
            "model.initialization_source_seed",
        )
    opset = _positive_int(onnx_config.get("opset"), "onnx.opset")
    if opset != 17:
        raise OnnxExportError("onnx.opset must be 17 for the locked deployment contract")
    input_dtype = _text(input_config.get("dtype"), "input.dtype")
    input_color_order = _text(input_config.get("color_order"), "input.color_order")
    input_value_range = _float_pair(input_config.get("value_range"), "input.value_range")
    if input_dtype != "float32" or input_color_order != "RGB":
        raise OnnxExportError("input must use NCHW RGB float32")
    if input_value_range != (0.0, 1.0):
        raise OnnxExportError("input.value_range must be [0.0, 1.0]")
    output_dtype = _text(output_config.get("dtype"), "output.dtype")
    output_semantic = _text(output_config.get("semantic"), "output.semantic")
    if output_dtype != "float32" or output_semantic != "raw_logits":
        raise OnnxExportError("output must be float32 raw_logits")

    source_experiment_config = _text(
        artifact.get("source_experiment_config"), "artifact.source_experiment_config"
    )
    source_experiment_config_path = source.parent / source_experiment_config
    validation_threshold = _probability(
        artifact.get("validation_threshold"), "artifact.validation_threshold"
    )
    confidence_threshold = _probability(
        postprocess.get("confidence_threshold"), "postprocess.confidence_threshold"
    )
    if confidence_threshold != validation_threshold:
        raise OnnxExportError(
            "postprocess.confidence_threshold must equal artifact.validation_threshold"
        )
    component_mode = _text(postprocess.get("component_mode"), "postprocess.component_mode")
    if component_mode != "connected_components":
        raise OnnxExportError("postprocess.component_mode must be 'connected_components'")
    confidence_mode = _text(postprocess.get("confidence_mode"), "postprocess.confidence_mode")
    if confidence_mode not in {"max", "mean"}:
        raise OnnxExportError("postprocess.confidence_mode must be 'max' or 'mean'")
    selection_strategy = _text(
        postprocess.get("selection_strategy"), "postprocess.selection_strategy"
    )
    if selection_strategy not in {
        "highest_confidence",
        "largest_component",
        "nearest_previous",
    }:
        raise OnnxExportError("postprocess.selection_strategy is unsupported")
    class_thresholds = _class_thresholds(
        postprocess.get("class_thresholds"), class_names
    )
    allowed_class_ids = _allowed_class_ids(
        postprocess.get("allowed_class_ids"), len(class_names)
    )
    contract = OnnxExportContract(
        config_path=source,
        artifact_name=_text(artifact.get("name"), "artifact.name"),
        source_experiment_config=source_experiment_config,
        source_experiment_config_path=source_experiment_config_path,
        source_experiment_config_sha256=_sha256(
            artifact.get("source_experiment_config_sha256"),
            "artifact.source_experiment_config_sha256",
        ),
        validation_threshold=validation_threshold,
        checkpoint_sha256=_sha256(checkpoint.get("sha256"), "checkpoint.sha256"),
        checkpoint_epoch=_positive_int(checkpoint.get("epoch"), "checkpoint.epoch"),
        checkpoint_seed=_nonnegative_int(checkpoint.get("seed"), "checkpoint.seed"),
        checkpoint_parameter_count=_positive_int(
            checkpoint.get("parameter_count"), "checkpoint.parameter_count"
        ),
        checkpoint_config_fingerprint=_sha256(
            checkpoint.get("config_fingerprint"), "checkpoint.config_fingerprint"
        ),
        backbone=backbone,
        width_multiplier=_positive_float(
            model.get("width_multiplier"), "model.width_multiplier"
        ),
        cut_point=cut_point,
        head_channels=_positive_int(model.get("head_channels"), "model.head_channels"),
        output_stride=output_stride,
        class_names=class_names,
        pretrained=pretrained,
        initialization=initialization,
        pretrained_sha256=pretrained_sha256,
        initialization_checkpoint_sha256=initialization_checkpoint_sha256,
        initialization_source_epoch=initialization_source_epoch,
        initialization_source_seed=initialization_source_seed,
        input_name=_text(input_config.get("name"), "input.name"),
        input_shape=input_shape,
        input_dtype=input_dtype,
        input_color_order=input_color_order,
        input_value_range=input_value_range,
        output_name=_text(output_config.get("name"), "output.name"),
        output_shape=output_shape,
        output_dtype=output_dtype,
        output_semantic=output_semantic,
        opset=opset,
        parity_seed=_nonnegative_int(parity.get("seed"), "parity.seed"),
        parity_rtol=_nonnegative_float(parity.get("rtol"), "parity.rtol"),
        parity_atol=_nonnegative_float(parity.get("atol"), "parity.atol"),
        confidence_threshold=confidence_threshold,
        class_thresholds=class_thresholds,
        component_mode=component_mode,
        confidence_mode=confidence_mode,
        selection_strategy=selection_strategy,
        max_match_distance_pixels=_positive_float(
            postprocess.get("max_match_distance_pixels"),
            "postprocess.max_match_distance_pixels",
        ),
        max_lost_frames=_nonnegative_int(
            postprocess.get("max_lost_frames"), "postprocess.max_lost_frames"
        ),
        allowed_class_ids=allowed_class_ids,
    )
    _validate_source_experiment(contract)
    return contract


def load_checkpoint_model(
    contract: OnnxExportContract, checkpoint_path: Path
) -> tuple[nn.Module, dict[str, object]]:
    """Validate and restore one formal snapshot into a CPU eval model.

    The restored module consumes RGB float32 ``[1,3,S,S]`` and returns raw
    float32 logits ``[1,1+N,S/8,S/8]``. Initial pretrained H5 weights are not
    accessed because the snapshot state is complete and loaded strictly.
    """

    if not isinstance(contract, OnnxExportContract):
        raise OnnxExportError("contract must be an OnnxExportContract")
    source = Path(checkpoint_path)
    actual_sha256 = sha256_file(source)
    if actual_sha256 != contract.checkpoint_sha256:
        raise OnnxExportError(
            "checkpoint SHA-256 mismatch: expected {}, got {}".format(
                contract.checkpoint_sha256, actual_sha256
            )
        )
    try:
        payload = load_epoch_snapshot(source)
    except SnapshotError as error:
        raise OnnxExportError("invalid formal checkpoint '{}': {}".format(source, error)) from error
    _require_equal(payload, "epoch", contract.checkpoint_epoch)
    _require_equal(payload, "seed", contract.checkpoint_seed)
    _require_equal(payload, "parameter_count", contract.checkpoint_parameter_count)
    _require_equal(
        payload, "config_fingerprint", contract.checkpoint_config_fingerprint
    )

    metadata = _mapping(payload.get("model_metadata"), "checkpoint model metadata")
    expected_metadata = {
        "backbone_name": contract.backbone,
        "width_multiplier": contract.width_multiplier,
        "cut_point": contract.cut_point,
        "cut_point_input_channels": 16,
        "cut_point_output_channels": 96,
        "output_stride": contract.output_stride,
        "head_channels": contract.head_channels,
        "pretrained": contract.pretrained,
        "initialization": contract.initialization,
        "backbone_parameter_count": 15_840,
        "head_parameter_count": 3_368,
        "parameter_count": contract.checkpoint_parameter_count,
    }
    mismatches = {
        key: {"expected": expected, "actual": metadata.get(key)}
        for key, expected in expected_metadata.items()
        if metadata.get(key) != expected
    }
    if contract.pretrained:
        load_report = metadata.get("pretrained_load_report")
        if not isinstance(load_report, Mapping):
            mismatches["pretrained_load_report"] = {
                "expected": "mapping",
                "actual": type(load_report).__name__,
            }
        elif load_report.get("sha256") != contract.pretrained_sha256:
            mismatches["pretrained_load_report.sha256"] = {
                "expected": contract.pretrained_sha256,
                "actual": load_report.get("sha256"),
            }
    else:
        expected_initialization = {
            "initialization_checkpoint_sha256": contract.initialization_checkpoint_sha256,
            "initialization_source_epoch": contract.initialization_source_epoch,
            "initialization_source_seed": contract.initialization_source_seed,
        }
        mismatches.update(
            {
                key: {"expected": expected, "actual": metadata.get(key)}
                for key, expected in expected_initialization.items()
                if metadata.get(key) != expected
            }
        )
    if mismatches:
        raise OnnxExportError(
            "checkpoint model metadata mismatch: {}".format(
                json.dumps(mismatches, sort_keys=True)
            )
        )

    model = MobileNetV2FOMONet(
        num_classes=len(contract.class_names),
        input_size=contract.input_size,
        width_multiplier=contract.width_multiplier,
        head_channels=contract.head_channels,
        output_stride=contract.output_stride,
        cut_point=contract.cut_point,
        pretrained=False,
    ).to(device="cpu", dtype=torch.float32)
    if count_trainable_parameters(model) != contract.checkpoint_parameter_count:
        raise OnnxExportError(
            "configured model parameter count mismatch: expected {}, got {}".format(
                contract.checkpoint_parameter_count, count_trainable_parameters(model)
            )
        )
    state_dict = payload.get("model_state")
    if not isinstance(state_dict, Mapping):
        raise OnnxExportError("checkpoint state dict must be a mapping")
    _validate_state_dict(model, state_dict)
    try:
        model.load_state_dict(state_dict, strict=True)
    except (RuntimeError, TypeError, ValueError) as error:
        raise OnnxExportError("checkpoint state dict is incompatible: {}".format(error)) from error
    model.eval()
    return model, {
        "checkpoint_file": source.name,
        "checkpoint_sha256": actual_sha256,
        "epoch": payload["epoch"],
        "seed": payload["seed"],
        "parameter_count": payload["parameter_count"],
        "config_fingerprint": payload["config_fingerprint"],
        "pretrained": contract.pretrained,
        "initialization": contract.initialization,
        "initialization_checkpoint_sha256": (
            contract.initialization_checkpoint_sha256
        ),
        "initialization_source_epoch": contract.initialization_source_epoch,
        "initialization_source_seed": contract.initialization_source_seed,
    }


def export_checkpoint_to_onnx(
    *,
    config_path: Path,
    checkpoint_path: Path,
    onnx_path: Path,
    report_path: Path,
) -> dict[str, object]:
    """Export fixed batch-one NCHW RGB FP32 input to raw FP32 logits and verify parity.

    A deterministic normalized input ``[1,3,S,S]`` is evaluated by both PyTorch
    and ONNX Runtime CPU. The ONNX file is published only after checker, static
    I/O contract, and ``allclose`` parity validation succeed.
    """

    _validate_export_paths(
        protected_inputs={
            "config": Path(config_path),
            "checkpoint": Path(checkpoint_path),
        },
        outputs={
            "output": Path(onnx_path),
            "report": Path(report_path),
        },
    )
    try:
        import onnx
        import onnxruntime as ort
    except ImportError as error:
        raise OnnxExportError(
            "ONNX export requires the 'onnx' and 'onnxruntime' packages"
        ) from error

    contract = load_export_contract(config_path)
    _validate_export_paths(
        protected_inputs={
            "config": Path(config_path),
            "source experiment config": contract.source_experiment_config_path,
            "checkpoint": Path(checkpoint_path),
        },
        outputs={
            "output": Path(onnx_path),
            "report": Path(report_path),
        },
    )
    model, checkpoint_provenance = load_checkpoint_model(contract, checkpoint_path)
    generator = torch.Generator(device="cpu").manual_seed(contract.parity_seed)
    images = torch.rand(contract.input_shape, generator=generator, dtype=torch.float32)
    with torch.inference_mode():
        pytorch_logits = model(images)
    if tuple(pytorch_logits.shape) != contract.output_shape:
        raise OnnxExportError(
            "PyTorch output shape mismatch: expected {}, got {}".format(
                list(contract.output_shape), list(pytorch_logits.shape)
            )
        )
    if pytorch_logits.dtype != torch.float32:
        raise OnnxExportError("PyTorch output dtype must be float32 raw logits")

    destination = Path(onnx_path)
    report_destination = Path(report_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    report_destination.parent.mkdir(parents=True, exist_ok=True)
    staged_onnx: Path | None = None
    staged_report: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=".onnx", prefix=destination.stem + ".", dir=destination.parent, delete=False
        ) as handle:
            staged_onnx = Path(handle.name)
        export_model = _FixedOutputShapeWrapper(model, contract.output_shape).eval()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
            warnings.filterwarnings(
                "ignore",
                message=r"Constant folding - Only steps=1 can be constant folded.*",
                category=UserWarning,
            )
            try:
                torch.onnx.export(
                    export_model,
                    images,
                    staged_onnx,
                    input_names=[contract.input_name],
                    output_names=[contract.output_name],
                    opset_version=contract.opset,
                    dynamic_axes=None,
                    do_constant_folding=True,
                )
            except Exception as error:
                raise OnnxExportError("PyTorch ONNX export failed: {}".format(error)) from error
        try:
            graph = onnx.load(str(staged_onnx))
        except Exception as error:
            raise OnnxExportError("unable to load exported ONNX: {}".format(error)) from error
        try:
            onnx.checker.check_model(graph)
        except Exception as error:
            raise OnnxExportError("ONNX checker failed: {}".format(error)) from error
        _validate_onnx_graph_contract(graph, contract)
        try:
            session = ort.InferenceSession(
                str(staged_onnx), providers=["CPUExecutionProvider"]
            )
        except Exception as error:
            raise OnnxExportError(
                "ONNX Runtime session creation failed: {}".format(error)
            ) from error
        _validate_ort_contract(session, contract)
        try:
            ort_logits = session.run(
                [contract.output_name], {contract.input_name: images.numpy()}
            )[0]
        except Exception as error:
            raise OnnxExportError("ONNX Runtime inference failed: {}".format(error)) from error
        pytorch_array = pytorch_logits.detach().cpu().numpy()
        difference = np.abs(pytorch_array - ort_logits)
        if not np.allclose(
            pytorch_array,
            ort_logits,
            rtol=contract.parity_rtol,
            atol=contract.parity_atol,
        ):
            raise OnnxExportError(
                "PyTorch/ONNX Runtime logits parity failed at rtol={}, atol={}; "
                "max_absolute_error={:.8g}".format(
                    contract.parity_rtol,
                    contract.parity_atol,
                    float(difference.max()),
                )
            )
        report: dict[str, object] = {
            "artifact_name": contract.artifact_name,
            "source_experiment_config": contract.source_experiment_config,
            "source_experiment_config_sha256": contract.source_experiment_config_sha256,
            "export_config_file": contract.config_path.name,
            "export_config_sha256": sha256_file(contract.config_path),
            **checkpoint_provenance,
            "validation_threshold": contract.validation_threshold,
            "validation_threshold_usage": "provenance_only_raw_logits_export",
            "onnx_file": destination.name,
            "onnx_sha256": sha256_file(staged_onnx),
            "onnx_size_bytes": staged_onnx.stat().st_size,
            "onnx_opset": contract.opset,
            "onnx_checker": "passed",
            "input_name": contract.input_name,
            "input_shape": list(contract.input_shape),
            "input_dtype": contract.input_dtype,
            "input_color_order": contract.input_color_order,
            "input_value_range": list(contract.input_value_range),
            "output_name": contract.output_name,
            "output_shape": list(contract.output_shape),
            "output_dtype": contract.output_dtype,
            "output_semantic": contract.output_semantic,
            "output_stride": contract.output_stride,
            "class_names": list(contract.class_names),
            "postprocess": {
                "confidence_threshold": contract.confidence_threshold,
                "class_thresholds": contract.class_thresholds,
                "component_mode": contract.component_mode,
                "confidence_mode": contract.confidence_mode,
                "selection_strategy": contract.selection_strategy,
                "max_match_distance_pixels": contract.max_match_distance_pixels,
                "max_lost_frames": contract.max_lost_frames,
                "allowed_class_ids": (
                    list(contract.allowed_class_ids)
                    if contract.allowed_class_ids is not None
                    else None
                ),
            },
            "pytorch_version": torch.__version__,
            "onnx_version": onnx.__version__,
            "onnxruntime_version": ort.__version__,
            "exported_at_utc": datetime.now(timezone.utc).isoformat(),
            "parity": {
                "passed": True,
                "input_seed": contract.parity_seed,
                "rtol": contract.parity_rtol,
                "atol": contract.parity_atol,
                "max_absolute_error": float(difference.max()),
                "mean_absolute_error": float(difference.mean()),
            },
        }
        staged_report = _write_json_staged(report, report_destination)
        _validate_staged_artifact_pair(
            onnx_path=staged_onnx,
            report_path=staged_report,
            final_onnx_name=destination.name,
            expected_report=report,
        )
        _publish_staged_artifact_pair(
            staged_onnx=staged_onnx,
            staged_report=staged_report,
            onnx_path=destination,
            report_path=report_destination,
            verify_published=lambda: _validate_staged_artifact_pair(
                onnx_path=destination,
                report_path=report_destination,
                final_onnx_name=destination.name,
                expected_report=report,
            ),
        )
    except OnnxExportError:
        raise
    except (OSError, RuntimeError, ValueError) as error:
        raise OnnxExportError("unable to export and verify ONNX: {}".format(error)) from error
    finally:
        for staged_path in (staged_onnx, staged_report):
            if staged_path is None:
                continue
            try:
                staged_path.unlink(missing_ok=True)
            except OSError:
                pass
    return report


def _validate_state_dict(model: nn.Module, state_dict: Mapping[str, Any]) -> None:
    expected = model.state_dict()
    missing = sorted(set(expected).difference(state_dict))
    unexpected = sorted(set(state_dict).difference(expected))
    if missing or unexpected:
        raise OnnxExportError(
            "checkpoint state dict keys mismatch: missing={}, unexpected={}".format(
                missing, unexpected
            )
        )
    for name, expected_tensor in expected.items():
        actual = state_dict[name]
        if not isinstance(actual, Tensor):
            raise OnnxExportError("checkpoint state dict entry '{}' is not a tensor".format(name))
        if tuple(actual.shape) != tuple(expected_tensor.shape):
            raise OnnxExportError(
                "checkpoint state dict shape mismatch for '{}': expected {}, got {}".format(
                    name, tuple(expected_tensor.shape), tuple(actual.shape)
                )
            )
        if actual.dtype != expected_tensor.dtype:
            raise OnnxExportError(
                "checkpoint state dict dtype mismatch for '{}': expected {}, got {}".format(
                    name, expected_tensor.dtype, actual.dtype
                )
            )


def _validate_source_experiment(contract: OnnxExportContract) -> None:
    actual_sha256 = sha256_file(contract.source_experiment_config_path)
    if actual_sha256 != contract.source_experiment_config_sha256:
        raise OnnxExportError(
            "source experiment config SHA-256 mismatch: expected {}, got {}".format(
                contract.source_experiment_config_sha256, actual_sha256
            )
        )
    try:
        payload = yaml.safe_load(
            contract.source_experiment_config_path.read_text(encoding="utf-8")
        )
    except OSError as error:
        raise OnnxExportError(
            "unable to read source experiment config '{}': {}".format(
                contract.source_experiment_config, error
            )
        ) from error
    except yaml.YAMLError as error:
        raise OnnxExportError(
            "unable to parse source experiment config '{}': {}".format(
                contract.source_experiment_config, error
            )
        ) from error
    root = _mapping(payload, "source experiment configuration root")
    dataset = _section(root, "dataset")
    model = _section(root, "model")
    training = _section(root, "training")
    export = _section(root, "export")
    postprocess = _section(root, "postprocess")
    source_classes = _class_names(dataset.get("classes"))
    if source_classes != contract.class_names:
        raise OnnxExportError(
            "source experiment class mapping mismatch: expected {}, got {}".format(
                list(source_classes), list(contract.class_names)
            )
        )
    expected_values = {
        "model.backbone": (model.get("backbone"), contract.backbone),
        "model.width_multiplier": (
            model.get("width_multiplier"),
            contract.width_multiplier,
        ),
        "model.head_channels": (model.get("head_channels"), contract.head_channels),
        "model.input_size": (model.get("input_size"), contract.input_size),
        "model.output_stride": (model.get("output_stride"), contract.output_stride),
        "model.pretrained": (model.get("pretrained"), contract.pretrained),
        "model.pretrained_sha256": (
            model.get("pretrained_sha256"),
            contract.pretrained_sha256,
        ),
        "training.seed": (training.get("seed"), contract.checkpoint_seed),
        "export.onnx_opset": (export.get("onnx_opset"), contract.opset),
        "export.input_size": (export.get("input_size"), contract.input_size),
        "postprocess.class_thresholds": (
            postprocess.get("class_thresholds"),
            contract.class_thresholds,
        ),
        "postprocess.component_mode": (
            postprocess.get("component_mode"),
            contract.component_mode,
        ),
        "postprocess.confidence_mode": (
            postprocess.get("confidence_mode"),
            contract.confidence_mode,
        ),
        "postprocess.selection_strategy": (
            postprocess.get("selection_strategy"),
            contract.selection_strategy,
        ),
        "postprocess.max_match_distance_pixels": (
            postprocess.get("max_match_distance_pixels"),
            contract.max_match_distance_pixels,
        ),
        "postprocess.max_lost_frames": (
            postprocess.get("max_lost_frames"),
            contract.max_lost_frames,
        ),
        "postprocess.allowed_class_ids": (
            postprocess.get("allowed_class_ids"),
            list(contract.allowed_class_ids)
            if contract.allowed_class_ids is not None
            else None,
        ),
    }
    if not contract.pretrained:
        expected_values["training.initialize_sha256"] = (
            training.get("initialize_sha256"),
            contract.initialization_checkpoint_sha256,
        )
    mismatches = {
        name: {"source": actual, "export_contract": expected}
        for name, (actual, expected) in expected_values.items()
        if actual != expected
    }
    if mismatches:
        raise OnnxExportError(
            "source experiment config mismatch: {}".format(
                json.dumps(mismatches, sort_keys=True)
            )
        )


def _validate_export_paths(
    *, protected_inputs: Mapping[str, Path], outputs: Mapping[str, Path]
) -> None:
    try:
        validate_output_paths(
            protected_inputs=protected_inputs,
            outputs=outputs,
        )
    except OutputPathError as error:
        raise OnnxExportError(
            "export paths must use distinct paths and safe output names: {}".format(error)
        ) from error


def _publish_staged_artifact_pair(
    *,
    staged_onnx: Path,
    staged_report: Path,
    onnx_path: Path,
    report_path: Path,
    verify_published: Callable[[], None] | None = None,
) -> None:
    """Publish two staged files with exception-safe backup and rollback semantics."""

    staged_paths = (Path(staged_onnx), Path(staged_report))
    destinations = (Path(onnx_path), Path(report_path))
    backups: dict[Path, Path] = {}
    allocated_backups: list[Path] = []
    published: set[Path] = set()
    try:
        for destination in destinations:
            if not destination.exists():
                continue
            backup = _allocate_sibling_temporary_path(destination, suffix=".backup")
            allocated_backups.append(backup)
            os.replace(destination, backup)
            backups[destination] = backup
        for staged_path, destination in zip(staged_paths, destinations):
            os.replace(staged_path, destination)
            published.add(destination)
        if verify_published is not None:
            verify_published()
    except Exception as error:
        rollback_errors: list[str] = []
        for destination in reversed(destinations):
            backup = backups.get(destination)
            try:
                if backup is not None:
                    os.replace(backup, destination)
                elif destination in published:
                    destination.unlink(missing_ok=True)
            except OSError as rollback_error:
                rollback_errors.append(
                    "{}: {}".format(destination, rollback_error)
                )
        for backup in allocated_backups:
            try:
                backup.unlink(missing_ok=True)
            except OSError as cleanup_error:
                rollback_errors.append("{}: {}".format(backup, cleanup_error))
        detail = ""
        if rollback_errors:
            detail = "; rollback errors: {}".format(", ".join(rollback_errors))
        raise OnnxExportError(
            "unable to publish ONNX artifact pair: {}{}".format(error, detail)
        ) from error

    cleanup_errors: list[str] = []
    for backup in allocated_backups:
        try:
            backup.unlink(missing_ok=True)
        except OSError as error:
            cleanup_errors.append("{}: {}".format(backup, error))
    if cleanup_errors:
        raise OnnxExportError(
            "ONNX artifact pair was published but backup cleanup failed: {}".format(
                ", ".join(cleanup_errors)
            )
        )


def _allocate_sibling_temporary_path(destination: Path, *, suffix: str) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=destination.name + ".",
        suffix=suffix,
        dir=destination.parent,
        delete=False,
    ) as handle:
        return Path(handle.name)


def _validate_onnx_graph_contract(graph: Any, contract: OnnxExportContract) -> None:
    opsets = {
        imported.domain: imported.version for imported in graph.opset_import
    }
    default_opset = opsets.get("", opsets.get("ai.onnx"))
    if default_opset != contract.opset:
        raise OnnxExportError(
            "ONNX opset mismatch: expected {}, got {}".format(contract.opset, default_opset)
        )
    if len(graph.graph.input) != 1 or graph.graph.input[0].name != contract.input_name:
        raise OnnxExportError("ONNX graph input name/count violates the export contract")
    if len(graph.graph.output) != 1 or graph.graph.output[0].name != contract.output_name:
        raise OnnxExportError("ONNX graph output name/count violates the export contract")
    input_shape = _onnx_value_shape(graph.graph.input[0])
    output_shape = _onnx_value_shape(graph.graph.output[0])
    if input_shape != contract.input_shape or output_shape != contract.output_shape:
        raise OnnxExportError(
            "ONNX graph shape mismatch: input={}, output={}".format(
                list(input_shape), list(output_shape)
            )
        )


def _validate_ort_contract(session: Any, contract: OnnxExportContract) -> None:
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if (
        len(inputs) != 1
        or inputs[0].name != contract.input_name
        or tuple(inputs[0].shape) != contract.input_shape
        or inputs[0].type != "tensor(float)"
    ):
        raise OnnxExportError("ONNX Runtime input contract mismatch")
    if (
        len(outputs) != 1
        or outputs[0].name != contract.output_name
        or tuple(outputs[0].shape) != contract.output_shape
        or outputs[0].type != "tensor(float)"
    ):
        raise OnnxExportError("ONNX Runtime output contract mismatch")


def _onnx_value_shape(value: Any) -> tuple[int, ...]:
    return tuple(
        int(dimension.dim_value)
        for dimension in value.type.tensor_type.shape.dim
    )


def _write_json_staged(payload: Mapping[str, object], path: Path) -> Path:
    """Write and fsync a JSON sidecar staging file beside its final destination."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n"
    temporary_path: Path | None = None
    completed = False
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            suffix=".tmp",
            prefix=destination.name + ".",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        completed = True
        return temporary_path
    except OSError as error:
        raise OnnxExportError(
            "unable to stage report '{}': {}".format(destination, error)
        ) from error
    finally:
        if temporary_path is not None and not completed:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass


def _validate_staged_artifact_pair(
    *,
    onnx_path: Path,
    report_path: Path,
    final_onnx_name: str,
    expected_report: Mapping[str, object],
) -> None:
    """Read back a staged pair and cross-check its contract, SHA, size, and provenance."""

    try:
        payload = json.loads(Path(report_path).read_text(encoding="utf-8"))
    except OSError as error:
        raise OnnxExportError(
            "unable to read back staged ONNX report '{}': {}".format(
                report_path, error
            )
        ) from error
    except json.JSONDecodeError as error:
        raise OnnxExportError(
            "unable to parse staged ONNX report '{}': {}".format(
                report_path, error
            )
        ) from error
    expected_payload = json.loads(
        json.dumps(dict(expected_report), ensure_ascii=False)
    )
    if payload != expected_payload:
        raise OnnxExportError(
            "staged ONNX report does not match the validated export provenance"
        )
    actual_sha256 = sha256_file(Path(onnx_path))
    try:
        actual_size = Path(onnx_path).stat().st_size
    except OSError as error:
        raise OnnxExportError(
            "unable to inspect staged ONNX '{}': {}".format(onnx_path, error)
        ) from error
    if payload.get("onnx_file") != final_onnx_name:
        raise OnnxExportError("staged ONNX report filename does not match destination")
    if payload.get("onnx_sha256") != actual_sha256:
        raise OnnxExportError("staged ONNX report SHA-256 does not match staged model")
    if payload.get("onnx_size_bytes") != actual_size:
        raise OnnxExportError("staged ONNX report size does not match staged model")


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise OnnxExportError("{} must be a mapping".format(name))
    return value


def _section(payload: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    return _mapping(payload.get(name), "{} section".format(name))


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise OnnxExportError("{} must be a non-empty string".format(name))
    return value


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise OnnxExportError("{} must be a boolean".format(name))
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise OnnxExportError("{} must be a positive integer".format(name))
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise OnnxExportError("{} must be a non-negative integer".format(name))
    return value


def _positive_float(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(float(value))
        or float(value) <= 0
    ):
        raise OnnxExportError("{} must be a positive number".format(name))
    return float(value)


def _nonnegative_float(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(float(value))
        or float(value) < 0
    ):
        raise OnnxExportError("{} must be a non-negative number".format(name))
    return float(value)


def _probability(value: Any, name: str) -> float:
    result = _nonnegative_float(value, name)
    if result > 1.0:
        raise OnnxExportError("{} must be between 0 and 1".format(name))
    return result


def _sha256(value: Any, name: str) -> str:
    text = _text(value, name).lower()
    if re.fullmatch(r"[0-9a-f]{64}", text) is None:
        raise OnnxExportError("{} must be a 64-character hexadecimal SHA-256".format(name))
    return text


def _shape(value: Any, name: str) -> tuple[int, int, int, int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 4:
        raise OnnxExportError("{} must contain four positive integers".format(name))
    result = tuple(_positive_int(item, name) for item in value)
    return result  # type: ignore[return-value]


def _float_pair(value: Any, name: str) -> tuple[float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise OnnxExportError("{} must contain two numbers".format(name))
    result = tuple(_finite_float(item, name) for item in value)
    return result  # type: ignore[return-value]


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OnnxExportError("{} must contain finite numbers".format(name))
    result = float(value)
    if not isfinite(result):
        raise OnnxExportError("{} must contain finite numbers".format(name))
    return result


def _class_names(value: Any) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise OnnxExportError("model.classes must be a non-empty sequence")
    names = tuple(_text(item, "model.classes") for item in value)
    if len(set(names)) != len(names):
        raise OnnxExportError("model.classes must not contain duplicates")
    return names


def _class_thresholds(value: Any, class_names: tuple[str, ...]) -> Any:
    if value is None:
        return None
    if isinstance(value, Mapping):
        result: dict[int | str, float] = {}
        for key, threshold in value.items():
            if isinstance(key, bool) or not isinstance(key, (int, str)):
                raise OnnxExportError(
                    "postprocess.class_thresholds keys must be class IDs or names"
                )
            if isinstance(key, int) and not 0 <= key < len(class_names):
                raise OnnxExportError("postprocess.class_thresholds class ID is invalid")
            if isinstance(key, str) and key not in class_names:
                raise OnnxExportError("postprocess.class_thresholds class name is invalid")
            result[key] = _probability(
                threshold, "postprocess.class_thresholds[{}]".format(key)
            )
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != len(class_names):
            raise OnnxExportError(
                "postprocess.class_thresholds length must equal class count"
            )
        return [
            _probability(item, "postprocess.class_thresholds[{}]".format(index))
            for index, item in enumerate(value)
        ]
    raise OnnxExportError(
        "postprocess.class_thresholds must be null, a sequence, or mapping"
    )


def _allowed_class_ids(value: Any, class_count: int) -> tuple[int, ...] | None:
    if value is None:
        return None
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise OnnxExportError("postprocess.allowed_class_ids must be null or a sequence")
    result = tuple(
        _nonnegative_int(item, "postprocess.allowed_class_ids") for item in value
    )
    if len(set(result)) != len(result) or any(item >= class_count for item in result):
        raise OnnxExportError("postprocess.allowed_class_ids contains invalid class IDs")
    return result


def _require_equal(payload: Mapping[str, Any], key: str, expected: Any) -> None:
    actual = payload.get(key)
    if actual != expected:
        raise OnnxExportError(
            "checkpoint {} mismatch: expected {!r}, got {!r}".format(key, expected, actual)
        )


__all__ = [
    "OnnxExportContract",
    "OnnxExportError",
    "export_checkpoint_to_onnx",
    "load_checkpoint_model",
    "load_export_contract",
    "sha256_file",
]
