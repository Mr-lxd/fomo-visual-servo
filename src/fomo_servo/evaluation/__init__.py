"""Validation orchestration for grid and centroid FOMO metrics."""

from .validation import (
    ValidationReport,
    evaluate_logit_collection,
    evaluate_logit_collection_at_threshold,
    evaluate_validation_dataset,
)
from .stage_b import (
    LockedTestManifest,
    StageBProtocolError,
    ThresholdTuningResult,
    build_locked_test_manifest,
    build_threshold_tuning_artifact,
    run_locked_test,
    tune_validation_threshold,
    validate_locked_manifest,
    write_final_test_artifacts,
    write_json_artifact,
)
from .epoch_snapshots import (
    CheckpointSelectionError,
    EpochSelectionReport,
    collect_split_logits,
    evaluate_collected_logits,
    select_best_epoch_report,
    validate_calibration_request,
)
from .edge_impulse import (
    EdgeImpulseCentroidEvaluator,
    EdgeImpulseDetection,
    EdgeImpulseEvaluation,
    EdgeImpulseGroundTruth,
    EdgeImpulseMetricError,
    decode_edge_impulse_fomo,
    normalized_centroid_distance,
    probabilities_from_logits,
)
from .parity_reporting import (
    edge_detections_from_local,
    edge_ground_truths_from_local,
    evaluate_local_parity,
    serialize_edge_evaluation,
)
from .parity_clean import ParityCleanError, ParityCleanView, verify_parity_clean_view

__all__ = [
    "CheckpointSelectionError",
    "EpochSelectionReport",
    "EdgeImpulseCentroidEvaluator",
    "EdgeImpulseDetection",
    "EdgeImpulseEvaluation",
    "EdgeImpulseGroundTruth",
    "EdgeImpulseMetricError",
    "ParityCleanError",
    "ParityCleanView",
    "ValidationReport",
    "LockedTestManifest",
    "StageBProtocolError",
    "ThresholdTuningResult",
    "build_locked_test_manifest",
    "build_threshold_tuning_artifact",
    "collect_split_logits",
    "decode_edge_impulse_fomo",
    "edge_detections_from_local",
    "edge_ground_truths_from_local",
    "evaluate_collected_logits",
    "evaluate_logit_collection",
    "evaluate_logit_collection_at_threshold",
    "evaluate_local_parity",
    "evaluate_validation_dataset",
    "run_locked_test",
    "normalized_centroid_distance",
    "probabilities_from_logits",
    "select_best_epoch_report",
    "serialize_edge_evaluation",
    "tune_validation_threshold",
    "validate_calibration_request",
    "verify_parity_clean_view",
    "validate_locked_manifest",
    "write_final_test_artifacts",
    "write_json_artifact",
]
