"""Tests for the validation-only Stage C.1 focal threshold selector."""

import pytest

from scripts.tune_focal_stage_c1 import StageC1FocalError, select_best_threshold


def _result(threshold: float, f1: float) -> dict[str, object]:
    return {"threshold": threshold, "strict": {"f1": f1}}


def test_select_best_threshold_prefers_strict_f1() -> None:
    selected = select_best_threshold((_result(0.05, 0.2), _result(0.10, 0.3)))

    assert selected["threshold"] == pytest.approx(0.10)


def test_select_best_threshold_breaks_exact_ties_by_lower_threshold() -> None:
    selected = select_best_threshold((_result(0.20, 0.3), _result(0.10, 0.3)))

    assert selected["threshold"] == pytest.approx(0.10)


def test_select_best_threshold_rejects_empty_results() -> None:
    with pytest.raises(StageC1FocalError, match="must not be empty"):
        select_best_threshold(())
