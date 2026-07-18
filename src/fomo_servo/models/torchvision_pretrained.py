"""Strict local torchvision ImageNet weight loading for Stage E encoders."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping

import torch
import torchvision
from torch import nn
from torchvision.models import mobilenet_v3_small, squeezenet1_1


class TorchvisionPretrainedWeightsError(ValueError):
    """Raised when a local torchvision source cannot be proven compatible."""


@dataclass(frozen=True)
class TorchvisionPretrainedLoadReport:
    """Immutable provenance and strict-prefix coverage for one local source file."""

    source_sha256: str
    torchvision_version: str
    weights_enum: str
    url: str
    source_tensor_count: int
    loaded_tensor_count: int
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        """Return JSON-serializable checkpoint metadata."""

        return asdict(self)


_SPECS: dict[str, tuple[Callable[[], nn.Module], str, str]] = {
    "mobilenet_v3_small_fomo": (
        lambda: mobilenet_v3_small(weights=None),
        "MobileNet_V3_Small_Weights.IMAGENET1K_V1",
        "https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth",
    ),
    "squeezenet1_1_fomo": (
        lambda: squeezenet1_1(weights=None),
        "SqueezeNet1_1_Weights.IMAGENET1K_V1",
        "https://download.pytorch.org/models/squeezenet1_1-b8a52dc0.pth",
    ),
}


def load_torchvision_backbone_weights(
    backbone: nn.Module,
    *,
    backbone_name: str,
    source: Path,
    expected_sha256: str,
    expected_torchvision_version: str,
    expected_weights_enum: str,
    expected_url: str,
) -> TorchvisionPretrainedLoadReport:
    """Strictly load a verified torchvision `features.*` prefix into a FOMO encoder."""

    try:
        factory, weights_enum, url = _SPECS[backbone_name]
    except KeyError as error:
        raise TorchvisionPretrainedWeightsError(
            f"unsupported torchvision FOMO backbone '{backbone_name}'"
        ) from error
    if expected_torchvision_version != torchvision.__version__:
        raise TorchvisionPretrainedWeightsError(
            "torchvision version mismatch: expected {}, got {}".format(
                expected_torchvision_version, torchvision.__version__
            )
        )
    if expected_weights_enum != weights_enum:
        raise TorchvisionPretrainedWeightsError(
            f"weights enum mismatch: expected {weights_enum}, got {expected_weights_enum}"
        )
    if expected_url != url:
        raise TorchvisionPretrainedWeightsError(
            f"weights URL mismatch: expected {url}, got {expected_url}"
        )
    source = Path(source)
    if not source.is_file():
        raise TorchvisionPretrainedWeightsError(f"pretrained source does not exist: {source}")
    actual_sha256 = _sha256_file(source)
    if actual_sha256 != expected_sha256.lower():
        raise TorchvisionPretrainedWeightsError(
            "SHA-256 mismatch for pretrained source '{}': expected {}, got {}".format(
                source, expected_sha256, actual_sha256
            )
        )
    try:
        payload = torch.load(source, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError) as error:
        raise TorchvisionPretrainedWeightsError(
            f"unable to read pretrained source '{source}': {error}"
        ) from error
    if not isinstance(payload, Mapping) or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in payload.items()
    ):
        raise TorchvisionPretrainedWeightsError(
            "pretrained source must be a tensor state-dict mapping"
        )
    full_state = dict(payload)
    reference = factory()
    expected_full = reference.state_dict()
    missing_full = tuple(sorted(set(expected_full).difference(full_state)))
    unexpected_full = tuple(sorted(set(full_state).difference(expected_full)))
    if missing_full:
        raise TorchvisionPretrainedWeightsError(
            "pretrained source is missing required tensors: {}".format(missing_full)
        )
    if unexpected_full:
        raise TorchvisionPretrainedWeightsError(
            "pretrained source has unexpected tensors: {}".format(unexpected_full)
        )
    try:
        reference.load_state_dict(full_state, strict=True)
    except RuntimeError as error:
        raise TorchvisionPretrainedWeightsError(
            f"pretrained source does not strictly match torchvision architecture: {error}"
        ) from error
    expected_prefix = backbone.state_dict()
    prefix_state = {key: full_state[key] for key in expected_prefix}
    try:
        incompatible = backbone.load_state_dict(prefix_state, strict=True)
    except RuntimeError as error:
        raise TorchvisionPretrainedWeightsError(
            f"pretrained feature prefix does not strictly match FOMO backbone: {error}"
        ) from error
    return TorchvisionPretrainedLoadReport(
        source_sha256=actual_sha256,
        torchvision_version=torchvision.__version__,
        weights_enum=weights_enum,
        url=url,
        source_tensor_count=len(full_state),
        loaded_tensor_count=len(prefix_state),
        missing_keys=tuple(incompatible.missing_keys),
        unexpected_keys=tuple(incompatible.unexpected_keys),
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = [
    "TorchvisionPretrainedLoadReport",
    "TorchvisionPretrainedWeightsError",
    "load_torchvision_backbone_weights",
]
