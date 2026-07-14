"""Tests for the immutable Stage D2 cleaned-test protocol."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _script_module():
    path = Path(__file__).parents[1] / "scripts" / "evaluate_d2_locked_test.py"
    spec = importlib.util.spec_from_file_location("evaluate_d2_locked_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_locked_protocol_rejects_all_runtime_selection_overrides() -> None:
    module = _script_module()

    with pytest.raises(module.D2LockedTestError, match="threshold"):
        module.validate_cli_invariants(
            threshold_override=0.35,
            checkpoint_override=None,
            split_override=None,
            sweep_requested=False,
        )
    with pytest.raises(module.D2LockedTestError, match="checkpoint"):
        module.validate_cli_invariants(
            threshold_override=None,
            checkpoint_override=Path("other.pt"),
            split_override=None,
            sweep_requested=False,
        )
    with pytest.raises(module.D2LockedTestError, match="split"):
        module.validate_cli_invariants(
            threshold_override=None,
            checkpoint_override=None,
            split_override="val",
            sweep_requested=False,
        )
    with pytest.raises(module.D2LockedTestError, match="sweep"):
        module.validate_cli_invariants(
            threshold_override=None,
            checkpoint_override=None,
            split_override=None,
            sweep_requested=True,
        )


def test_confidence_summary_is_deterministic() -> None:
    module = _script_module()

    summary = module.summarize_confidences([0.2, 0.5, 0.9])

    assert summary == {
        "count": 3,
        "min": 0.2,
        "max": 0.9,
        "mean": 0.5333333333333333,
        "p50": 0.5,
        "p90": 0.82,
    }


def test_metric_rows_include_all_fixed_evaluators() -> None:
    module = _script_module()
    report = {
        "local_current_evaluator": {
            "evaluator": "local_current",
            "per_class": {},
            "true_positives": 1,
            "false_positives": 2,
            "false_negatives": 3,
            "precision": 0.25,
            "recall": 0.5,
            "f1": 1 / 3,
            "macro_f1": 0.4,
            "prediction_count": 3,
            "ground_truth_count": 4,
            "mean_absolute_count_error": 1.0,
            "mean_count_bias": -1.0,
            "mean_localization_error_pixels": 2.0,
            "median_localization_error_pixels": 1.5,
        },
        "edge_impulse_legacy_evaluator": {
            "evaluator": "edge_impulse_legacy",
            "per_class": {},
            "true_positives": 2,
            "false_positives": 1,
            "false_negatives": 2,
            "precision": 2 / 3,
            "recall": 0.5,
            "f1": 4 / 7,
            "macro_f1": 0.45,
            "prediction_count": 3,
            "ground_truth_count": 4,
            "mean_absolute_count_error": 0.5,
            "mean_count_bias": -0.5,
            "mean_localization_error_pixels": 1.0,
            "median_localization_error_pixels": 1.0,
        },
        "strict_one_to_one_evaluator": {
            "evaluator": "strict_one_to_one",
            "per_class": {},
            "true_positives": 1,
            "false_positives": 2,
            "false_negatives": 3,
            "precision": 0.25,
            "recall": 0.5,
            "f1": 1 / 3,
            "macro_f1": 0.4,
            "prediction_count": 3,
            "ground_truth_count": 4,
            "mean_absolute_count_error": 1.0,
            "mean_count_bias": -1.0,
            "mean_localization_error_pixels": 2.0,
            "median_localization_error_pixels": 1.5,
        },
    }

    rows = module.metric_rows(report)

    assert [row["evaluator"] for row in rows] == [
        "local_current",
        "edge_impulse_legacy",
        "strict_one_to_one",
    ]
    assert rows[0]["true_positives"] == 1
