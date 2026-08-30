from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from fomo_servo.evaluation.stage_b import (
    StageBProtocolError,
    build_locked_test_manifest,
    build_threshold_tuning_artifact,
    run_locked_test,
    tune_validation_threshold,
    validate_locked_manifest,
    write_final_test_artifacts,
    write_json_artifact,
)
from fomo_servo.geometry import LetterboxTransform
from fomo_servo.metrics import GroundTruthCentroid
from fomo_servo.metrics.centroid import sweep_confidence_thresholds
from fomo_servo.training.snapshots import sha256_file


def _truth() -> GroundTruthCentroid:
    return GroundTruthCentroid(
        class_id=0,
        class_name="creature",
        original_x=4.0,
        original_y=4.0,
        x_min=0.0,
        y_min=0.0,
        x_max=8.0,
        y_max=8.0,
    )


def _logits(with_false_positive: bool = True) -> tuple[torch.Tensor, ...]:
    background = torch.full((3, 3), 4.0, dtype=torch.float32)
    foreground = torch.zeros((3, 3), dtype=torch.float32)
    background[0, 0] = 0.0
    foreground[0, 0] = 2.0
    if with_false_positive:
        background[0, 2] = 0.0
        foreground[0, 2] = 1.0
    return (torch.stack((background, foreground)).unsqueeze(0),)


def _sweep(with_false_positive: bool = True):
    return sweep_confidence_thresholds(
        logits=_logits(with_false_positive),
        transforms=(LetterboxTransform.from_image_size(24, 24, 24),),
        ground_truths=((_truth(),),),
        class_names=("creature",),
        stride=8,
        thresholds=(0.5, 0.8),
        matching_mode="max_distance_pixels",
        max_distance_pixels=5.0,
    )


def test_validation_threshold_tuning_selects_best_threshold() -> None:
    result = tune_validation_threshold(_sweep(), threshold_grid=(0.5, 0.8))

    assert result.selected_threshold == pytest.approx(0.8)
    assert result.selected_metrics.centroid_f1 == pytest.approx(1.0)
    assert [item["threshold"] for item in result.threshold_results] == [0.5, 0.8]


def test_validation_threshold_tie_selects_lower_threshold() -> None:
    result = tune_validation_threshold(_sweep(False), threshold_grid=(0.5, 0.8))

    assert result.selected_threshold == pytest.approx(0.5)
    assert result.selected_metrics.centroid_f1 == pytest.approx(1.0)


def test_threshold_artifact_contains_complete_provenance(tmp_path: Path) -> None:
    candidate = tmp_path / "best_centroid_pr_auc_macro.pt"
    source = tmp_path / "epoch_058_weights.pt"
    candidate.write_bytes(b"candidate")
    source.write_bytes(b"source")
    artifact = build_threshold_tuning_artifact(
        candidate_path=candidate,
        source_snapshot_path=source,
        source_epoch=58,
        selection_metric="centroid_pr_auc_macro",
        selection_metric_value=0.1129353537,
        selection_split="val",
        tuning_split="val",
        threshold_grid=(0.5, 0.8),
        sweep=_sweep(),
        config_fingerprint="config-sha",
        dataset_content_hash="dataset-sha",
        git_commit_sha="a" * 40,
        device="cpu",
    )
    required = {
        "protocol_version",
        "candidate_checkpoint_path",
        "candidate_checkpoint_sha256",
        "source_snapshot_path",
        "source_snapshot_sha256",
        "source_epoch",
        "checkpoint_selection",
        "threshold_tuning_split",
        "objective",
        "threshold_grid",
        "threshold_results",
        "selected_threshold",
        "selected_objective_value",
        "tie_breaking_rule",
        "device",
        "dtype",
        "config_fingerprint",
        "dataset_content_hash",
        "git_commit_sha",
        "selection_and_threshold_tuning_shared",
        "threshold_tuning_independent",
    }
    assert required.issubset(artifact)
    assert artifact["dtype"] == "float32"
    assert artifact["selection_and_threshold_tuning_shared"] is True
    assert artifact["threshold_tuning_independent"] is False


def _locked_fixture(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    source = tmp_path / "epoch_snapshots" / "epoch_001_weights.pt"
    source.parent.mkdir()
    source_payload = {
        "checkpoint_kind": "epoch_snapshot",
        "weights_only": True,
        "resumable": False,
        "model_state": {"weight": torch.ones(1)},
        "epoch": 1,
        "model_metadata": {"name": "fixture"},
        "parameter_count": 1,
        "config_fingerprint": "config-sha",
        "dataset_content_hash": "dataset-sha",
        "git_commit_sha": "a" * 40,
        "seed": 42,
        "augmentation_preset": "none",
        "checkpoint_threshold": 0.5,
    }
    torch.save(source_payload, source)
    candidate = tmp_path / "best_centroid_pr_auc_macro.pt"
    candidate_payload = dict(source_payload)
    candidate_payload.update(
        {
            "checkpoint_kind": "inference_candidate",
            "source_snapshot": source.name,
            "source_snapshot_sha256": sha256_file(source),
            "selected_epoch": 1,
            "selection_metric": "centroid_pr_auc_macro",
            "selection_metric_value": 0.1,
            "selection_split": "val",
            "selection_dtype": "float32",
            "selection_details": {
                "threshold_grid": [0.5, 0.8],
                "integration": "fixture",
                "macro_effective_class_count": 1,
            },
        }
    )
    torch.save(candidate_payload, candidate)
    artifact_path = tmp_path / "threshold_tuning.json"
    artifact = build_threshold_tuning_artifact(
        candidate_path=candidate,
        source_snapshot_path=source,
        source_epoch=1,
        selection_metric="centroid_pr_auc_macro",
        selection_metric_value=0.1,
        selection_split="val",
        tuning_split="val",
        threshold_grid=(0.5, 0.8),
        sweep=_sweep(False),
        config_fingerprint="config-sha",
        dataset_content_hash="dataset-sha",
        git_commit_sha="a" * 40,
        device="cpu",
    )
    write_json_artifact(artifact_path, artifact)
    manifest = build_locked_test_manifest(
        candidate_path=candidate,
        source_snapshot_path=source,
        selected_epoch=1,
        selected_threshold=0.5,
        selection_metric="centroid_pr_auc_macro",
        selection_metric_value=0.1,
        selection_split="val",
        threshold_tuning_artifact_path=artifact_path,
        test_split="test",
        dataset_content_hash="dataset-sha",
        config_fingerprint="config-sha",
        git_commit_sha="a" * 40,
    )
    manifest_path = tmp_path / "locked_test_protocol.json"
    write_json_artifact(manifest_path, manifest)
    return manifest_path, manifest


def test_matching_locked_manifest_allows_test(tmp_path: Path) -> None:
    manifest_path, _ = _locked_fixture(tmp_path)

    resolved = validate_locked_manifest(
        manifest_path,
        expected_config_fingerprint="config-sha",
        expected_dataset_content_hash="dataset-sha",
        expected_git_commit_sha="a" * 40,
    )

    assert resolved.selected_epoch == 1
    assert resolved.selected_threshold == pytest.approx(0.5)


def test_checkpoint_hash_mismatch_rejects(tmp_path: Path) -> None:
    manifest_path, _ = _locked_fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["checkpoint_sha256"] = "0" * 64
    write_json_artifact(manifest_path, manifest)

    with pytest.raises(StageBProtocolError, match="checkpoint SHA-256"):
        validate_locked_manifest(manifest_path)


def test_source_snapshot_metadata_mismatch_rejects(tmp_path: Path) -> None:
    manifest_path, manifest = _locked_fixture(tmp_path)
    source = Path(manifest["source_snapshot_path"])
    payload = torch.load(source, map_location="cpu", weights_only=False)
    payload["config_fingerprint"] = "different"
    torch.save(payload, source)

    with pytest.raises(StageBProtocolError, match="source snapshot"):
        validate_locked_manifest(manifest_path)


def test_locked_test_rejects_sweep_request(tmp_path: Path) -> None:
    manifest_path, _ = _locked_fixture(tmp_path)

    with pytest.raises(StageBProtocolError, match="threshold sweep"):
        run_locked_test(None, manifest_path, threshold_sweep_requested=True)


def test_locked_test_rejects_missing_threshold(tmp_path: Path) -> None:
    manifest_path, manifest = _locked_fixture(tmp_path)
    manifest.pop("selected_threshold")
    write_json_artifact(manifest_path, manifest)

    with pytest.raises(StageBProtocolError, match="selected_threshold"):
        validate_locked_manifest(manifest_path)


def test_locked_test_rejects_multiple_candidates(tmp_path: Path) -> None:
    manifest_path, manifest = _locked_fixture(tmp_path)
    manifest["checkpoint_path"] = [manifest["checkpoint_path"], "another.pt"]
    write_json_artifact(manifest_path, manifest)

    with pytest.raises(StageBProtocolError, match="one checkpoint"):
        validate_locked_manifest(manifest_path)


def test_threshold_tuning_and_test_splits_are_different(tmp_path: Path) -> None:
    manifest_path, manifest = _locked_fixture(tmp_path)
    manifest["test_split"] = "val"
    write_json_artifact(manifest_path, manifest)

    with pytest.raises(StageBProtocolError, match="different split"):
        validate_locked_manifest(manifest_path)


def test_final_test_artifacts_do_not_write_back_selection(tmp_path: Path) -> None:
    selection_path = tmp_path / "checkpoint_selection_summary.json"
    selection_path.write_text('{"selected_epoch": 58}\n', encoding="utf-8")
    before = selection_path.read_bytes()

    write_final_test_artifacts(
        tmp_path,
        {
            "selected_epoch": 58,
            "threshold": 0.5,
            "grid_precision": 0.0,
            "grid_recall": 0.0,
            "grid_f1": 0.0,
            "centroid_precision": 0.0,
            "centroid_recall": 0.0,
            "centroid_f1": 0.0,
            "macro_f1": 0.0,
            "mean_localization_error_pixels": 0.0,
            "median_localization_error_pixels": 0.0,
            "mean_count_bias": 0.0,
            "count_mae": 0.0,
            "detection_count": 0,
        },
    )

    assert selection_path.read_bytes() == before
    assert (tmp_path / "final_test_metrics.json").is_file()


def test_old_evaluator_and_checkpoint_loader_remain_importable() -> None:
    from fomo_servo.evaluation import evaluate_logit_collection
    from fomo_servo.inference import load_inference_model
    from fomo_servo.training.snapshots import load_epoch_snapshot

    assert callable(evaluate_logit_collection)
    assert callable(load_inference_model)
    assert callable(load_epoch_snapshot)
