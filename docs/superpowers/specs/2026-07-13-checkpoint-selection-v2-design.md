# FOMO checkpoint selection protocol v2 — design

## Scope and compatibility

Protocol v2 adds reproducible, offline selection of an epoch's model weights.  It does not alter the training recipe, model topology, augmentation, optimizer, scheduler, seed, thresholds, or existing full-checkpoint protocol.  `last.pt`, `best_val_f1.pt`, `best_grid_f1.pt`, and `best_centroid_f1.pt` remain full, resumable checkpoints with their existing filenames and payload contract.  For newly written `best_centroid_f1.pt`, the metadata spelling is made explicit as `fixed_centroid_f1` at the fixed `0.5` threshold; its selection behaviour remains unchanged.

The protocol is deliberately independent of pretrained weights and requires no network access.  It is implemented and evaluated in FP32, with `model.eval()` and `torch.inference_mode()`; autocast is not used.

## Configuration

`training.epoch_snapshots` defaults to disabled, preserving every existing configuration:

```yaml
training:
  epoch_snapshots:
    enabled: true
    format: weights_only
    interval: 1
    keep_last: null
evaluation:
  checkpoint_selection:
    metric: centroid_pr_auc_macro
    split: validation
    threshold_grid: {minimum: 0.05, maximum: 0.95, step: 0.05}
  threshold_calibration:
    enabled: false
    split: calibration
    objective: centroid_f1
    fallback_threshold: 0.5
    allow_selection_split: false
```

The only supported snapshot format is `weights_only`; `interval` must be positive and `keep_last` is null or positive.  The selection metric may be `centroid_pr_auc_macro` or `max_centroid_f1_over_thresholds`.  Calibration is an offline result, never a mutation of model weights.  A missing requested calibration split is an error.  Reusing the selection split requires explicit `allow_selection_split: true` and writes `calibration_is_optimistic: true`.

## Snapshot and candidate payloads

Each eligible epoch produces `epoch_snapshots/epoch_XXX_weights.pt`.  It is portable CPU `model_state` plus only identity metadata: epoch, model identity and parameter count, sanitized config fingerprint, dataset content hash, Git commit, seed, augmentation preset, checkpoint threshold, and snapshot format/kind.  It has no optimizer, scheduler, GradScaler, RNG state, or history, and is non-resumable.

The selected files `best_centroid_pr_auc_macro.pt` and `best_sweep_centroid_f1.pt` are **weights-only inference/evaluation candidates**, never training checkpoints.  Their payload retains the selected snapshot's model state and all its identity metadata and adds:

```text
checkpoint_kind: inference_candidate
weights_only: true
resumable: false
source_snapshot: epoch_XXX_weights.pt
source_snapshot_sha256: <SHA-256 of published source file>
selected_epoch: <int>
selection_metric: centroid_pr_auc_macro | max_centroid_f1_over_thresholds
selection_metric_value: <float>
selection_split: <str>
selection_dtype: float32
selection_details: {threshold_grid, integration, macro_effective_class_count, ...}
```

Candidates are built by loading and validating their source snapshot—not by byte copying.  Both snapshots and candidates use a temporary file in the destination directory, `flush`, `fsync`, and `os.replace()` publication.  A resume attempt detects `resumable: false`/candidate kind before trying to restore state and explains that optimizer, scheduler, scaler, and full training state are absent.  Inference and offline evaluation continue to load their `model_state` normally.

## Evaluation and PR-AUC definition

The offline evaluator reads one snapshot/checkpoint at a time, reconstructs the configured model, evaluates the requested split without random augmentation, and stores per-epoch metrics as CSV and JSON.  It records validation loss, grid metrics, centroid metrics at 0.5, a centroid threshold sweep, macro/per-class values, localization error, count statistics, detection count, and target-cell foreground-confidence summaries.

`centroid_pr_auc_macro` is **not COCO mAP and not interpolated AP**.  For every foreground class with at least one ground-truth object on the evaluated split, centroid matching is recomputed at every configured postprocess confidence threshold.  The raw `(threshold, precision, recall)` points are retained.  For integration only, points are sorted by ascending recall; duplicate recall values retain the maximum precision.  The value is the trapezoidal integral of precision over these observed recall coordinates, with no precision envelope and no artificial `(0, 1)` or `(1, 0)` endpoints.  A class with ground truth but fewer than two distinct recall coordinates has AUC `0.0`; a class without ground truth has AUC `null` and is excluded from macro averaging.  The macro value is the arithmetic mean over the recorded effective-class count.  Micro PR-AUC uses the same procedure over aggregate matches when aggregate ground truth exists.

The threshold order supplied by configuration cannot change this value.  A selected epoch is the highest finite metric value; an exact tie chooses the earlier epoch, then lexicographically smaller source filename.  Existing sweep threshold ties continue to choose the lower threshold.

## Outputs and migration

`scripts/evaluate_epoch_snapshots.py` emits a metrics CSV, JSON with raw curves, and selection summary.  It can create candidates atomically after selection.  `scripts/audit_checkpoint_selection_v2.py` evaluates the six existing aug03/model01 legacy checkpoints and writes the ignored regression audit under `outputs/experiments/checkpoint_selection_v2/`.  It verifies legacy fixed/sweep values within tolerance while supplying the new PR-AUC measurement.

Training summary v2 fields are additive: legacy best epochs, PR-AUC/sweep candidate epochs, selection metric/split, and calibration result fields.  Old checkpoints and old summaries remain readable because absent v2 keys use explicit defaults.
