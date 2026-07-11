"""Tests for weighted cross entropy and focal FOMO classification losses."""

from __future__ import annotations

import importlib
from typing import Any, Callable

import pytest
import torch
import torch.nn.functional as functional


def _loss_api() -> tuple[type[Any] | None, Callable[..., torch.nn.Module] | None, type[Exception] | None]:
    """Return optional loss APIs so missing implementation yields clear assertions."""

    config_module = importlib.import_module("fomo_servo.config")
    loss_module = importlib.import_module("fomo_servo.losses")
    return (
        getattr(config_module, "LossConfig", None),
        getattr(loss_module, "build_classification_loss", None),
        getattr(loss_module, "LossConfigurationError", None),
    )


def _loss_config(name: str, gamma: float, weights: tuple[float, ...]) -> Any:
    """Construct a loss config only after its public dataclass is available."""

    config_type, _, _ = _loss_api()
    fields = getattr(config_type, "__dataclass_fields__", {})
    assert {"name", "gamma", "class_weights"}.issubset(fields), (
        "fomo_servo.config.LossConfig must expose name, gamma, and class_weights"
    )
    return config_type(name=name, gamma=gamma, class_weights=weights)


def _logits_and_targets() -> tuple[torch.Tensor, torch.Tensor]:
    """Return deterministic logits [1,2,2,2] and targets int64 [1,2,2]."""

    logits = torch.tensor(
        [[[[2.0, -1.0], [0.5, 1.5]], [[-0.5, 2.0], [1.0, -1.0]]]],
        dtype=torch.float32,
    )
    targets = torch.tensor([[[0, 1], [1, 0]]], dtype=torch.int64)
    return logits, targets


def test_weighted_cross_entropy_matches_torch_reference() -> None:
    """Weighted logits [B,C,G,G] loss must equal PyTorch cross entropy."""

    _, builder, _ = _loss_api()
    assert callable(builder), "fomo_servo.losses.build_classification_loss must exist"
    logits, targets = _logits_and_targets()
    weights = (1.0, 3.0)

    criterion = builder(_loss_config("weighted_cross_entropy", 2.0, weights))
    actual = criterion(logits, targets)
    expected = functional.cross_entropy(logits, targets, weight=torch.tensor(weights))

    torch.testing.assert_close(actual, expected)


def test_focal_gamma_zero_matches_weighted_cross_entropy() -> None:
    """Focal gamma=0 must retain exactly the weighted CE contract."""

    _, builder, _ = _loss_api()
    assert callable(builder), "fomo_servo.losses.build_classification_loss must exist"
    logits, targets = _logits_and_targets()
    weights = (1.0, 2.0)

    criterion = builder(_loss_config("focal_cross_entropy", 0.0, weights))
    actual = criterion(logits, targets)
    expected = functional.cross_entropy(logits, targets, weight=torch.tensor(weights))

    torch.testing.assert_close(actual, expected)


def test_loss_rejects_class_weight_count_mismatch() -> None:
    """Weights must cover background plus every model foreground class."""

    _, builder, error_type = _loss_api()
    assert callable(builder), "fomo_servo.losses.build_classification_loss must exist"
    assert isinstance(error_type, type), "LossConfigurationError must exist"

    criterion = builder(_loss_config("weighted_cross_entropy", 0.0, (1.0,)))

    with pytest.raises(error_type, match="class_weights length"):
        criterion(torch.randn(1, 2, 2, 2), torch.zeros(1, 2, 2, dtype=torch.int64))
