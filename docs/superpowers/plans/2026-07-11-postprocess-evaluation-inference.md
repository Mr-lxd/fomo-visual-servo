# FOMO Postprocess, Evaluation, and Inference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add CPU-friendly FOMO heatmap postprocessing, centroid evaluation, target selection/tracking, image/video inference, and validation reporting without changing the model, label generation, loss, optimizer, or existing checkpoint tensor state.

**Architecture:** `postprocess` is stateless and consumes detached logits plus per-image `LetterboxTransform` metadata. `metrics.centroid` consumes postprocess `Detection` objects and dataset `AbsoluteBox` ground truth, using deterministic greedy one-to-one matching. `inference` owns checkpoint loading, letterbox preprocessing, latest-frame buffering, and sequence statistics; CLI scripts only orchestrate I/O and serialization. Training validation receives immutable metadata through `FOMOBatch` and appends grid and centroid metrics while retaining legacy grid aliases for compatibility.

**Tech Stack:** Python 3.10/3.11, PyTorch, NumPy, OpenCV, pytest; no SciPy or other large dependency.

---

### Task 1: Add failing tests for postprocessing and selection

**Files:**
- Create: `tests/test_postprocess.py`
- Create: `tests/test_selection.py`
- Create: `tests/test_tracking.py`

- [x] Test background exclusion, global/per-class thresholds, 8-connected merging, separated components, class names, probability-weighted grid centroid, inverse letterbox coordinates, and empty output.
- [x] Test highest-confidence, largest-component, nearest-previous distance/class filtering and deterministic ties.
- [x] Test `idle -> detected -> detected -> lost -> reacquired`, lost frame limits, and no robot-control dependency.
- [x] Run the focused tests and confirm they fail because the public APIs are absent.

### Task 2: Implement independent postprocessing

**Files:**
- Create: `src/fomo_servo/postprocess/connected_components.py`
- Create: `src/fomo_servo/postprocess/detections.py`
- Create: `src/fomo_servo/postprocess/selection.py`
- Modify: `src/fomo_servo/postprocess/__init__.py`

- [x] Validate logits `[B,1+N,G,G]`, class names, stride, transforms, and threshold specs.
- [x] Apply `torch.softmax(logits.float(), dim=1)` outside the model, then use foreground probability maps and configured thresholds.
- [x] Implement deterministic 8-neighbor connected components and probability-weighted centroids on `(gx+0.5, gy+0.5)`.
- [x] Emit `Detection` with class ID/name, max and mean confidence, component area, heatmap/input/original coordinates, clipping original coordinates to image bounds.
- [x] Keep `local_peaks` as an explicit unsupported extension error rather than silently changing behavior.

### Task 3: Implement target selection, tracker, and sequence statistics

**Files:**
- Create: `src/fomo_servo/postprocess/tracking.py`
- Create: `src/fomo_servo/metrics/sequence.py`
- Modify: package `__init__.py` files
- Create: `tests/test_sequence_metrics.py`

- [x] Implement the three deterministic selection strategies and `TargetTracker` states with `max_lost_frames`.
- [x] Implement normalized coordinates using the project’s `[-1,1]` convention.
- [x] Implement jitter pixel/normalized mean, standard deviation, RMS, loss rate, availability, and reacquisition count; document that these are stability statistics, not MOT metrics.

### Task 4: Implement centroid evaluation and threshold sweep

**Files:**
- Create: `src/fomo_servo/metrics/centroid.py`
- Modify: `src/fomo_servo/metrics/classification.py`
- Modify: `src/fomo_servo/metrics/__init__.py`
- Create: `tests/test_centroid_metrics.py`

- [x] Rename the public grid result to `grid_precision`, `grid_recall`, `grid_f1` while retaining read-only legacy aliases for existing callers.
- [x] Implement `GroundTruthCentroid`, deterministic distance-sorted greedy one-to-one matching for `centroid_in_bbox` and `max_distance_pixels`, wrong-class rejection, per-class scores, confusion matrix, localization errors, and per-image count error.
- [x] Implement threshold sweep over supplied thresholds, selecting the best validation centroid F1 without any test-set access.

### Task 5: Extend configuration and training validation reporting

**Files:**
- Modify: `src/fomo_servo/config.py`
- Modify: `src/fomo_servo/datasets/collate.py`
- Modify: `src/fomo_servo/training/engine.py`
- Modify: `src/fomo_servo/training/__init__.py`
- Modify: `configs/aquarium_pretrain_192.yaml`
- Create/update: configuration, collate, training tests

- [x] Add YAML-only postprocess/evaluation/selection/tracker settings and validate choices/ranges.
- [x] Add transforms and original boxes as non-tensor batch metadata; keep images `[B,3,S,S]` and targets `[B,G,G]` unchanged.
- [x] Run the exact postprocessor during validation, append grid and centroid metrics to `history.csv`, and select checkpoint improvement by configured `grid_f1` or `centroid_f1`; default remains `grid_f1`.
- [x] Keep existing checkpoint keys/load behavior compatible; do not rewrite any existing checkpoint file.

### Task 6: Implement shared inference and CLI scripts

**Files:**
- Create: `src/fomo_servo/inference/predictor.py`
- Create: `src/fomo_servo/inference/video.py`
- Modify: `src/fomo_servo/inference/__init__.py`
- Modify: `scripts/predict_image.py`
- Modify: `scripts/predict_video.py`
- Modify: `scripts/evaluate.py`
- Create/update: CLI/schema tests

- [x] Load the existing checkpoint state into the configured model without changing architecture or weights.
- [x] Implement image prediction overlays, selected target and normalized coordinates, JSON output, and video CSV/JSONL schemas.
- [x] Use a bounded latest-frame buffer for video capture, preserving frame indices and timestamps while dropping stale frames.
- [x] Implement validation checkpoint evaluation, threshold sweep reporting, per-class F1, localization errors, and count error.

### Task 7: Verify with tests and real validation checkpoint

**Files:**
- No new production files unless verification exposes a tested defect.

- [x] Run all pytest tests and record skips/failures explicitly.
- [x] Run `scripts/evaluate.py` on `outputs/aquarium_pretrain_7class_192/best_val_f1.pt` and the complete validation split using `FOMO_DATASET_ROOT`.
- [x] Report grid metrics, centroid metrics, best validation threshold, per-class centroid F1, localization errors, and average per-image count error.
- [x] Confirm no backbone, label generation, loss, optimizer, or checkpoint file was modified.
