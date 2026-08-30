"""Contract tests for checkpoint selection protocol v2."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest
import torch
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = ROOT / "tests" / "fixtures" / "yolo_micro"


def _fixture_config(tmp_path: Path, *, calibration_enabled: bool = False):
    """Load a tiny real-data config used by FP32/offline evaluator tests."""

    from fomo_servo.config import load_config

    path = tmp_path / "evaluation.yaml"
    path.write_text(
        (
            "dataset:\n"
            "  root: \"{}\"\n"
            "  train_split: train\n"
            "  validation_split: val\n"
            "  classes: [creature]\n"
            "  class_mode: merge_single\n"
            "  collision_policy: keep_first\n"
            "model:\n"
            "  backbone: mobilenet_v2_lite\n"
            "  width_multiplier: 0.35\n"
            "  head_channels: 32\n"
            "  input_size: 96\n"
            "  output_stride: 8\n"
            "loss:\n"
            "  name: focal_cross_entropy\n"
            "  gamma: 2.0\n"
            "  class_weights: [1.0, 3.0]\n"
            "evaluation:\n"
            "  threshold_calibration:\n"
            "    enabled: {}\n"
            "    split: calibration\n"
        ).format(FIXTURE_ROOT.as_posix(), str(calibration_enabled).lower()),
        encoding="utf-8",
    )
    return load_config(path)


def _snapshot_api():
    module = importlib.import_module("fomo_servo.training.snapshots")
    return (
        module.write_epoch_snapshot,
        module.write_inference_candidate,
        module.sha256_file,
    )


def test_checkpoint_selection_v2_public_modules_exist() -> None:
    """The protocol exposes separate snapshot and PR-AUC implementation modules."""

    snapshots = importlib.import_module("fomo_servo.training.snapshots")
    pr_auc = importlib.import_module("fomo_servo.metrics.pr_auc")

    assert callable(getattr(snapshots, "write_epoch_snapshot"))
    assert callable(getattr(snapshots, "write_inference_candidate"))
    assert callable(getattr(pr_auc, "centroid_pr_auc"))


def test_weights_only_snapshot_and_candidate_preserve_model_state(tmp_path: Path) -> None:
    """Candidates retain selected snapshot weights but cannot be mistaken for resume files."""

    write_snapshot, write_candidate, sha256_file = _snapshot_api()
    model = torch.nn.Conv2d(3, 2, kernel_size=1)
    snapshot = write_snapshot(
        model=model,
        epoch=3,
        output_dir=tmp_path,
        model_metadata={"backbone_name": "test"},
        config_fingerprint="config-sha",
        dataset_content_hash="dataset-sha",
        git_commit_sha="a" * 40,
        seed=123,
        augmentation_preset="aug03",
        checkpoint_threshold=0.5,
    )
    assert snapshot.name == "epoch_003_weights.pt"
    payload = torch.load(snapshot, map_location="cpu", weights_only=False)
    assert payload["checkpoint_kind"] == "epoch_snapshot"
    assert payload["weights_only"] is True
    assert payload["resumable"] is False
    assert "optimizer_state" not in payload
    assert "scheduler_state" not in payload
    assert "scaler_state" not in payload
    assert "rng_state" not in payload

    candidate = write_candidate(
        source_snapshot=snapshot,
        destination=tmp_path / "best_centroid_pr_auc_macro.pt",
        selection_metric="centroid_pr_auc_macro",
        selection_metric_value=0.75,
        selection_split="val",
        selection_details={
            "threshold_grid": [0.05, 0.5, 0.95],
            "integration": "trapezoidal_observed_recall_no_envelope",
            "macro_effective_class_count": 1,
        },
    )
    candidate_payload = torch.load(candidate, map_location="cpu", weights_only=False)
    assert candidate_payload["checkpoint_kind"] == "inference_candidate"
    assert candidate_payload["weights_only"] is True
    assert candidate_payload["resumable"] is False
    assert candidate_payload["source_snapshot"] == snapshot.name
    assert candidate_payload["source_snapshot_sha256"] == sha256_file(snapshot)
    assert candidate_payload["selected_epoch"] == 3
    assert candidate_payload["selection_dtype"] == "float32"
    assert candidate_payload["pr_auc_threshold_grid"] == [0.05, 0.5, 0.95]
    assert candidate_payload["pr_auc_integration"] == "trapezoidal_observed_recall_no_envelope"
    assert candidate_payload["pr_auc_macro_effective_class_count"] == 1
    assert candidate_payload["model_metadata"] == {"backbone_name": "test"}
    for name, tensor in payload["model_state"].items():
        assert torch.equal(tensor, candidate_payload["model_state"][name])


def test_candidate_publication_uses_atomic_replace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Candidate destination is published through os.replace after a complete temp write."""

    write_snapshot, write_candidate, _ = _snapshot_api()
    model = torch.nn.Conv2d(1, 1, kernel_size=1)
    snapshot = write_snapshot(
        model=model,
        epoch=1,
        output_dir=tmp_path,
        model_metadata={},
        config_fingerprint="c",
        dataset_content_hash="d",
        git_commit_sha="b" * 40,
        seed=1,
        augmentation_preset=None,
        checkpoint_threshold=0.5,
    )
    snapshots_module = importlib.import_module("fomo_servo.training.snapshots")
    calls: list[tuple[Path, Path]] = []
    original_replace = snapshots_module.os.replace

    def recording_replace(source: object, destination: object) -> None:
        source_path, destination_path = Path(source), Path(destination)
        assert source_path.is_file()
        calls.append((source_path, destination_path))
        original_replace(source, destination)

    monkeypatch.setattr(snapshots_module.os, "replace", recording_replace)
    destination = tmp_path / "candidate.pt"
    write_candidate(
        source_snapshot=snapshot,
        destination=destination,
        selection_metric="centroid_pr_auc_macro",
        selection_metric_value=0.1,
        selection_split="val",
        selection_details={"threshold_grid": [], "integration": "test", "macro_effective_class_count": 0},
    )
    assert calls and calls[-1][1] == destination
    assert destination.is_file()


def test_resume_rejects_inference_candidate_with_actionable_error(tmp_path: Path) -> None:
    """Resume never guesses missing optimizer/scheduler/scaler state from a candidate."""

    write_snapshot, write_candidate, _ = _snapshot_api()
    model = torch.nn.Linear(2, 2)
    snapshot = write_snapshot(
        model=model,
        epoch=1,
        output_dir=tmp_path,
        model_metadata={},
        config_fingerprint="c",
        dataset_content_hash="d",
        git_commit_sha="c" * 40,
        seed=1,
        augmentation_preset=None,
        checkpoint_threshold=0.5,
    )
    candidate = write_candidate(
        source_snapshot=snapshot,
        destination=tmp_path / "candidate.pt",
        selection_metric="centroid_pr_auc_macro",
        selection_metric_value=0.1,
        selection_split="val",
        selection_details={"threshold_grid": [], "integration": "test", "macro_effective_class_count": 0},
    )
    from fomo_servo.training.engine import TrainingError, _restore_checkpoint

    optimizer = torch.optim.AdamW(model.parameters())
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    with pytest.raises(TrainingError, match="inference/evaluation candidate.*optimizer.*scheduler.*scaler"):
        _restore_checkpoint(
            candidate,
            model=model,
            optimizer=optimizer,
            scheduler=None,
            scaler=scaler,
            device=torch.device("cpu"),
            checkpoint_criterion="grid_f1",
        )


def _centroid_evaluation(
    *,
    precision: float,
    recall: float,
    class_metrics: dict[str, dict[str, float]],
    true_positives: int = 0,
    false_positives: int = 0,
    false_negatives: int = 0,
):
    from fomo_servo.metrics import CentroidEvaluation

    return CentroidEvaluation(
        centroid_precision=precision,
        centroid_recall=recall,
        centroid_f1=0.0,
        per_class_precision_recall_f1=class_metrics,
        confusion_matrix=np.zeros((len(class_metrics), len(class_metrics)), dtype=np.int64),
        mean_localization_error_pixels=0.0,
        median_localization_error_pixels=0.0,
        count_error_per_image=(),
        mean_count_bias=0.0,
        mean_absolute_count_error=0.0,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
    )


def test_centroid_pr_auc_is_order_invariant_and_excludes_no_ground_truth_class() -> None:
    """Observed-grid trapezoids use recall order and macro averages only GT classes."""

    from fomo_servo.metrics import centroid_pr_auc

    low = _centroid_evaluation(
        precision=0.5,
        recall=1.0,
        class_metrics={
            "fish": {"precision": 0.5, "recall": 1.0, "true_positives": 2, "false_negatives": 0},
            "crab": {"precision": 0.0, "recall": 0.0, "true_positives": 0, "false_negatives": 0},
        },
        true_positives=2,
    )
    high = _centroid_evaluation(
        precision=1.0,
        recall=0.5,
        class_metrics={
            "fish": {"precision": 1.0, "recall": 0.5, "true_positives": 1, "false_negatives": 1},
            "crab": {"precision": 0.0, "recall": 0.0, "true_positives": 0, "false_negatives": 0},
        },
        true_positives=1,
        false_negatives=1,
    )

    forward = centroid_pr_auc({0.05: low, 0.95: high}, ("fish", "crab"))
    reverse = centroid_pr_auc({0.95: high, 0.05: low}, ("fish", "crab"))

    assert forward.macro_auc == pytest.approx(0.375)
    assert forward.macro_auc == reverse.macro_auc
    assert forward.micro_auc == pytest.approx(0.375)
    assert forward.macro_effective_class_count == 1
    assert forward.per_class["crab"].auc is None


def test_centroid_pr_auc_with_no_predictions_has_zero_area() -> None:
    """A GT class with only zero-recall samples gets a defined zero area."""

    from fomo_servo.metrics import centroid_pr_auc

    result = _centroid_evaluation(
        precision=0.0,
        recall=0.0,
        class_metrics={
            "fish": {"precision": 0.0, "recall": 0.0, "true_positives": 0, "false_negatives": 3},
        },
        false_negatives=3,
    )
    report = centroid_pr_auc({0.05: result, 0.95: result}, ("fish",))

    assert report.macro_auc == pytest.approx(0.0)
    assert report.micro_auc == pytest.approx(0.0)


def test_selection_tie_chooses_earlier_epoch_deterministically() -> None:
    """Equal selection metrics never depend on filesystem ordering."""

    module = importlib.import_module("fomo_servo.evaluation.epoch_snapshots")
    select = getattr(module, "select_best_epoch_report")
    reports = [
        {"epoch": 8, "source_snapshot": "epoch_008_weights.pt", "centroid_pr_auc_macro": 0.4},
        {"epoch": 3, "source_snapshot": "epoch_003_weights.pt", "centroid_pr_auc_macro": 0.4},
    ]

    best = select(reports, metric="centroid_pr_auc_macro")

    assert best["epoch"] == 3


def test_inference_loader_accepts_weights_only_candidate(tmp_path: Path) -> None:
    """Inference uses model_state only, so a v2 candidate remains deployable."""

    from fomo_servo.config import load_config
    from fomo_servo.inference import load_inference_model
    from fomo_servo.models import build_fomo_model

    config_path = tmp_path / "inference.yaml"
    config_path.write_text(
        """
dataset:
  root: data/unused
  classes: [creature]
model:
  backbone: mobilenet_v2_lite
  width_multiplier: 0.35
  head_channels: 32
  input_size: 96
  output_stride: 8
""".lstrip(),
        encoding="utf-8",
    )
    config = load_config(config_path)
    write_snapshot, write_candidate, _ = _snapshot_api()
    snapshot = write_snapshot(
        model=build_fomo_model(config),
        epoch=1,
        output_dir=tmp_path,
        model_metadata={"backbone_name": "mobilenet_v2_lite"},
        config_fingerprint="c",
        dataset_content_hash="d",
        git_commit_sha="d" * 40,
        seed=1,
        augmentation_preset=None,
        checkpoint_threshold=0.5,
    )
    candidate = write_candidate(
        source_snapshot=snapshot,
        destination=tmp_path / "candidate.pt",
        selection_metric="centroid_pr_auc_macro",
        selection_metric_value=0.1,
        selection_split="val",
        selection_details={"threshold_grid": [0.5], "integration": "test", "macro_effective_class_count": 1},
    )

    model, device = load_inference_model(config, candidate, "cpu")

    assert device.type == "cpu"
    assert model(torch.zeros((1, 3, 96, 96))).shape == (1, 2, 12, 12)


def test_calibration_same_split_requires_explicit_opt_in() -> None:
    """The selection split cannot silently become an optimistic calibration split."""

    from dataclasses import replace
    from fomo_servo.config import (
        CheckpointSelectionConfig,
        EvaluationConfig,
        ProjectConfig,
        ThresholdCalibrationConfig,
    )
    from fomo_servo.evaluation import CheckpointSelectionError, validate_calibration_request

    # The helper only consumes evaluation fields, so a minimal proxy is sufficient.
    evaluation = EvaluationConfig(
        checkpoint_selection=CheckpointSelectionConfig(split="val"),
        threshold_calibration=ThresholdCalibrationConfig(enabled=True, split="val"),
    )
    proxy = type("ConfigProxy", (), {"evaluation": evaluation})()
    with pytest.raises(CheckpointSelectionError, match="allow_selection_split"):
        validate_calibration_request(proxy, selection_split="val")
    permitted = replace(
        evaluation,
        threshold_calibration=replace(
            evaluation.threshold_calibration, allow_selection_split=True
        ),
    )
    assert validate_calibration_request(
        type("ConfigProxy", (), {"evaluation": permitted})(), selection_split="val"
    ) is True


def test_offline_collection_is_fp32_and_missing_calibration_split_is_explicit(
    tmp_path: Path,
) -> None:
    """Offline selection has no autocast and never silently falls back to validation."""

    from fomo_servo.evaluation import CheckpointSelectionError, collect_split_logits

    class RecordingModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.dtypes: list[torch.dtype] = []
            self.autocast_flags: list[bool] = []

        def forward(self, image: torch.Tensor) -> torch.Tensor:
            self.dtypes.append(image.dtype)
            self.autocast_flags.append(torch.is_autocast_enabled("cpu"))
            return torch.zeros(
                (image.shape[0], 2, 12, 12), dtype=torch.float32, device=image.device
            )

    config = _fixture_config(tmp_path, calibration_enabled=True)
    model = RecordingModel()
    collected = collect_split_logits(config, model, torch.device("cpu"), "val")
    assert collected.logits and all(item.dtype == torch.float32 for item in collected.logits)
    assert model.dtypes == [torch.float32] * len(model.dtypes)
    assert not any(model.autocast_flags)
    with pytest.raises(CheckpointSelectionError, match="calibration.*unavailable"):
        collect_split_logits(config, model, torch.device("cpu"), "calibration")
