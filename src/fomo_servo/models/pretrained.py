"""Strict loading of the verified Edge Impulse Keras MobileNetV2 H5 backbone."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Literal

import numpy as np
import torch
from torch import Tensor, nn


class PretrainedWeightsError(ValueError):
    """Raised when a verified EI backbone cannot be loaded without ambiguity."""


@dataclass(frozen=True)
class PretrainedLoadReport:
    """Provenance and strict coverage for one imported H5 backbone."""

    source: str
    sha256: str
    loaded_tensor_count: int
    skipped_tensor_count: int
    loaded_keys: tuple[str, ...]
    skipped_keys: tuple[str, ...]
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]
    initialization_policy: str = (
        "load EI Keras MobileNetV2 backbone through block_6_expand_relu; "
        "initialize FOMO head and classifier with PyTorch defaults"
    )

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-safe metadata for snapshots and experiment reports."""

        return {
            "source": self.source,
            "sha256": self.sha256,
            "loaded_tensor_count": self.loaded_tensor_count,
            "skipped_tensor_count": self.skipped_tensor_count,
            "loaded_keys": list(self.loaded_keys),
            "skipped_keys": list(self.skipped_keys),
            "missing_keys": list(self.missing_keys),
            "unexpected_keys": list(self.unexpected_keys),
            "initialization_policy": self.initialization_policy,
        }


@dataclass(frozen=True)
class _TensorMapping:
    source_path: str
    target: Tensor
    target_name: str
    layout: Literal["conv", "depthwise", "vector"]


def load_ei_mobilenet_v2_backbone(
    model: nn.Module,
    source: str | Path,
    expected_sha256: str,
) -> PretrainedLoadReport:
    """Load verified EI Keras weights into a local FOMO backbone.

    Args:
        model: A ``MobileNetV2FOMONet``-compatible module whose backbone has
            tensors in PyTorch NCHW/OIHW layout.
        source: Local H5 path supplied by YAML or an environment variable.
        expected_sha256: Exact SHA-256 for the source artifact.

    Returns:
        A strict report. The model's backbone tensors are updated in place;
        the FOMO head and classifier are untouched.

    Raises:
        PretrainedWeightsError: If the file, hash, H5 dependency, source
            tensor set, or tensor shapes are not exactly compatible.
    """

    source_path = Path(source)
    _validate_sha256(expected_sha256)
    if not source_path.is_file():
        raise PretrainedWeightsError(
            "pretrained H5 source does not exist: {}".format(source_path)
        )
    actual_sha256 = _sha256(source_path)
    expected_sha256 = expected_sha256.lower()
    if actual_sha256 != expected_sha256:
        raise PretrainedWeightsError(
            "SHA-256 mismatch for pretrained H5 source '{}': expected {}, got {}".format(
                source_path, expected_sha256, actual_sha256
            )
        )

    h5py = _require_h5py()
    mappings = tuple(_build_mappings(model))
    required_paths = {mapping.source_path for mapping in mappings}
    group_prefixes = {path.rsplit("/", 1)[0] for path in required_paths}

    try:
        with h5py.File(source_path, "r") as h5_file:
            dataset_paths = tuple(_dataset_paths(h5_file, h5py.Dataset))
            missing = tuple(
                sorted(path for path in required_paths if path not in h5_file)
            )
            if missing:
                raise PretrainedWeightsError(
                    "pretrained H5 is missing required tensors: {}".format(
                        ", ".join(missing)
                    )
                )
            unexpected = tuple(
                sorted(
                    path
                    for path in dataset_paths
                    if path not in required_paths
                    and any(path.startswith(prefix + "/") for prefix in group_prefixes)
                )
            )
            if unexpected:
                raise PretrainedWeightsError(
                    "pretrained H5 has unexpected tensors in required layers: {}".format(
                        ", ".join(unexpected)
                    )
                )

            converted: list[tuple[_TensorMapping, Tensor]] = []
            for mapping in mappings:
                array = np.asarray(h5_file[mapping.source_path], dtype=np.float32)
                transformed = _transpose(array, mapping.layout)
                if tuple(transformed.shape) != tuple(mapping.target.shape):
                    raise PretrainedWeightsError(
                        "shape mismatch for '{}': source converts to {}, target '{}' expects {}".format(
                            mapping.source_path,
                            tuple(transformed.shape),
                            mapping.target_name,
                            tuple(mapping.target.shape),
                        )
                    )
                converted.append(
                    (
                        mapping,
                        torch.as_tensor(
                            transformed,
                            dtype=mapping.target.dtype,
                            device=mapping.target.device,
                        ),
                    )
                )
    except PretrainedWeightsError:
        raise
    except (OSError, ValueError, RuntimeError) as error:
        raise PretrainedWeightsError(
            "unable to read pretrained H5 '{}': {}".format(source_path, error)
        ) from error

    with torch.no_grad():
        for mapping, tensor in converted:
            mapping.target.copy_(tensor)

    loaded_keys = tuple(
        "{} -> {}".format(mapping.source_path, mapping.target_name)
        for mapping in mappings
    )
    skipped_keys = tuple(sorted(path for path in dataset_paths if path not in required_paths))
    return PretrainedLoadReport(
        source=str(source_path),
        sha256=actual_sha256,
        loaded_tensor_count=len(loaded_keys),
        skipped_tensor_count=len(skipped_keys),
        loaded_keys=loaded_keys,
        skipped_keys=skipped_keys,
        missing_keys=(),
        unexpected_keys=(),
    )


def _build_mappings(model: nn.Module) -> Iterable[_TensorMapping]:
    backbone = getattr(model, "backbone", None)
    if not isinstance(backbone, nn.Module):
        raise PretrainedWeightsError("model must expose a MobileNetV2 backbone module")
    stem = getattr(backbone, "stem", None)
    blocks = getattr(backbone, "blocks_0_to_5", None)
    block_6 = getattr(backbone, "block_6_expansion", None)
    if not isinstance(stem, nn.Sequential) or not isinstance(blocks, nn.Sequential):
        raise PretrainedWeightsError(
            "model backbone is not the expected mobilenet_v2_fomo topology"
        )
    if not isinstance(block_6, nn.Sequential) or len(blocks) != 6:
        raise PretrainedWeightsError(
            "model backbone must contain blocks 0-5 and block_6_expansion"
        )

    mappings: list[_TensorMapping] = []

    def add_conv(source_path: str, target: nn.Conv2d, target_name: str, layout: Literal["conv", "depthwise"]) -> None:
        mappings.append(_TensorMapping(source_path, target.weight, target_name, layout))

    def add_bn(source_prefix: str, target: nn.BatchNorm2d, target_prefix: str) -> None:
        for source_name, target_name, target_tensor in (
            ("gamma:0", "weight", target.weight),
            ("beta:0", "bias", target.bias),
            ("moving_mean:0", "running_mean", target.running_mean),
            ("moving_variance:0", "running_var", target.running_var),
        ):
            mappings.append(
                _TensorMapping(
                    f"{source_prefix}/{source_name}",
                    target_tensor,
                    f"{target_prefix}.{target_name}",
                    "vector",
                )
            )

    add_conv("Conv1/Conv1/kernel:0", stem[0], "backbone.stem.0.weight", "conv")
    add_bn("bn_Conv1/bn_Conv1", stem[1], "backbone.stem.1")

    for block_index in range(6):
        block = blocks[block_index].block
        if block_index == 0:
            depthwise = block[0].conv
            project = block[1]
            add_conv(
                f"mobl{block_index}_conv_{block_index}_depthwise/mobl{block_index}_conv_{block_index}_depthwise/depthwise_kernel:0",
                depthwise,
                f"backbone.blocks_0_to_5.{block_index}.block.0.0.weight",
                "depthwise",
            )
            add_bn(
                f"bn{block_index}_conv_{block_index}_bn_depthwise/bn{block_index}_conv_{block_index}_bn_depthwise",
                block[0][1],
                f"backbone.blocks_0_to_5.{block_index}.block.0.1",
            )
            add_conv(
                f"mobl{block_index}_conv_{block_index}_project/mobl{block_index}_conv_{block_index}_project/kernel:0",
                project,
                f"backbone.blocks_0_to_5.{block_index}.block.1.weight",
                "conv",
            )
            add_bn(
                f"bn{block_index}_conv_{block_index}_bn_project/bn{block_index}_conv_{block_index}_bn_project",
                block[2],
                f"backbone.blocks_0_to_5.{block_index}.block.2",
            )
            continue

        expand = block[0].conv
        depthwise = block[1].conv
        project = block[2]
        add_conv(
            f"mobl{block_index}_conv_{block_index}_expand/mobl{block_index}_conv_{block_index}_expand/kernel:0",
            expand,
            f"backbone.blocks_0_to_5.{block_index}.block.0.0.weight",
            "conv",
        )
        add_bn(
            f"bn{block_index}_conv_{block_index}_bn_expand/bn{block_index}_conv_{block_index}_bn_expand",
            block[0][1],
            f"backbone.blocks_0_to_5.{block_index}.block.0.1",
        )
        add_conv(
            f"mobl{block_index}_conv_{block_index}_depthwise/mobl{block_index}_conv_{block_index}_depthwise/depthwise_kernel:0",
            depthwise,
            f"backbone.blocks_0_to_5.{block_index}.block.1.0.weight",
            "depthwise",
        )
        add_bn(
            f"bn{block_index}_conv_{block_index}_bn_depthwise/bn{block_index}_conv_{block_index}_bn_depthwise",
            block[1][1],
            f"backbone.blocks_0_to_5.{block_index}.block.1.1",
        )
        add_conv(
            f"mobl{block_index}_conv_{block_index}_project/mobl{block_index}_conv_{block_index}_project/kernel:0",
            project,
            f"backbone.blocks_0_to_5.{block_index}.block.2.weight",
            "conv",
        )
        add_bn(
            f"bn{block_index}_conv_{block_index}_bn_project/bn{block_index}_conv_{block_index}_bn_project",
            block[3],
            f"backbone.blocks_0_to_5.{block_index}.block.3",
        )

    add_conv(
        "mobl6_conv_6_expand/mobl6_conv_6_expand/kernel:0",
        block_6[0],
        "backbone.block_6_expansion.0.weight",
        "conv",
    )
    add_bn(
        "bn6_conv_6_bn_expand/bn6_conv_6_bn_expand",
        block_6[1],
        "backbone.block_6_expansion.1",
    )
    return mappings


def _transpose(array: np.ndarray, layout: str) -> np.ndarray:
    if layout == "conv":
        if array.ndim != 4:
            raise PretrainedWeightsError(
                "Keras Conv2D tensor must have rank 4, got {}".format(array.ndim)
            )
        return np.transpose(array, (3, 2, 0, 1))
    if layout == "depthwise":
        if array.ndim != 4:
            raise PretrainedWeightsError(
                "Keras DepthwiseConv2D tensor must have rank 4, got {}".format(array.ndim)
            )
        return np.transpose(array, (2, 3, 0, 1))
    if array.ndim != 1:
        raise PretrainedWeightsError(
            "Keras BatchNorm tensor must have rank 1, got {}".format(array.ndim)
        )
    return array


def _dataset_paths(h5_file: Any, dataset_type: type) -> Iterable[str]:
    paths: list[str] = []

    def visit(name: str, node: Any) -> None:
        if isinstance(node, dataset_type):
            paths.append(name)

    h5_file.visititems(visit)
    return paths


def _require_h5py() -> Any:
    try:
        import h5py
    except ModuleNotFoundError as error:
        raise PretrainedWeightsError(
            "loading EI H5 weights requires h5py; install the project's training extra"
        ) from error
    return h5py


def _validate_sha256(value: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-fA-F]{64}", value) is None:
        raise PretrainedWeightsError(
            "expected_sha256 must be a 64-character hexadecimal SHA-256 string"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "PretrainedLoadReport",
    "PretrainedWeightsError",
    "load_ei_mobilenet_v2_backbone",
]
