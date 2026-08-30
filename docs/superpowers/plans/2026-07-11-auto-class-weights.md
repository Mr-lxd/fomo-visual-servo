# Auto Class Weights and Training Statistics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Support backward-compatible manual loss weights and training-dataset-derived automatic class weights, then persist the resolved values and richer count statistics through training and validation artifacts.

**Architecture:** Configuration distinguishes legacy/manual lists from an explicit auto mapping. A new training-only statistics module iterates existing `YOLOv5FOMODataset` samples and counts encoded heatmap cells plus collisions without altering data loading or label generation. The engine resolves one final foreground/background tuple before building the unchanged classification loss, emits it to console/checkpoints/summary/report, and maintains independent grid/centroid checkpoint bests.

**Tech Stack:** Python 3.10/3.11, PyTorch, NumPy, YAML, pytest; no new dependencies.

---

### Task 1: Add failing unit tests

**Files:**
- Create: `tests/test_class_weights.py`
- Modify: `tests/test_config.py`
- Modify: `tests/test_training_engine.py`
- Modify: `tests/test_centroid_metrics.py`

- [x] Cover balanced and imbalanced encoded-cell counts, zero foreground count error, min/max clipping, and legacy manual list compatibility.
- [x] Cover count bias and mean absolute count error from signed per-image count errors.
- [x] Cover history/checkpoint/summary presence of final resolved class weights and independent best checkpoint names.
- [x] Run targeted tests and confirm missing APIs fail before implementation.

### Task 2: Implement training-dataset statistics and weight resolution

**Files:**
- Create: `src/fomo_servo/training/class_weights.py`
- Modify: `src/fomo_servo/training/__init__.py`

- [x] Count images-with-class, bbox count, encoded heatmap cells, same-class collisions, and different-class collisions per foreground class.
- [x] Mirror existing heatmap collision ordering for per-class collision attribution without changing target generation.
- [x] Resolve `manual` lists directly and `auto` using sqrt inverse frequency, median foreground count, configured base/background/min/max, and explicit zero-count failures.

### Task 3: Extend YAML schema and add weighted configuration

**Files:**
- Modify: `src/fomo_servo/config.py`
- Create: `configs/aquarium_pretrain_7class_192_weighted.yaml`
- Modify: tests for configuration

- [x] Parse legacy `loss.class_weights: [..]`, explicit `manual`, and explicit `auto` mappings.
- [x] Validate only supported auto balance mode and finite positive bounds.
- [x] Add the requested seven-class auto configuration with base 25, gamma 2, and independent output directory/checkpoint criterion `centroid_f1`.

### Task 4: Persist metrics and checkpoints

**Files:**
- Modify: `src/fomo_servo/metrics/centroid.py`
- Modify: `src/fomo_servo/evaluation/validation.py`
- Modify: `src/fomo_servo/training/engine.py`
- Modify: `scripts/evaluate.py`

- [x] Add `mean_count_bias` and `mean_absolute_count_error` to centroid/validation results and history.
- [x] Save `best_grid_f1.pt`, `best_centroid_f1.pt`, `last.pt`, and criterion-compatible `best_val_f1.pt`.
- [x] Store resolved weights/statistics in new checkpoint payloads, `training_summary.json`, and validation report output.
- [x] Preserve loading compatibility with existing checkpoints that lack new metadata.

### Task 5: Verify and run the requested training

**Files:**
- No new production files unless a tested defect is found.

- [x] Run complete pytest.
- [x] Set `FOMO_DATASET_ROOT` only in the command environment and run CUDA training using the new weighted YAML.
- [x] Verify summary, checkpoints, history fields, and validation report contain the final weights/statistics.
