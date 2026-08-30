# Augmentation Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Add a reproducible `aug00_none` experiment baseline with no augmentation and append-only experiment-level metrics.

**Architecture:** Extend the existing YAML schema with an optional experiment record section. The training engine keeps the existing model, dataset, loss, optimizer, scheduler, postprocess, and threshold sweep behavior; after the primary checkpoint is selected, it copies the complete config, records Git/data-list/seed/timing metadata, evaluates the selected checkpoint with the existing validation sweep, and appends one CSV row under `outputs/experiments`.

**Tech Stack:** Python 3.10, existing PyTorch/NumPy/YAML/CSV/JSON standard-library tooling, pytest; no augmentation implementation and no new dependency.

---

### Task 1: Add failing configuration and metadata tests

**Files:**
- Modify: `tests/test_config.py`
- Modify: `tests/test_training_engine.py`
- Create: `tests/test_experiment_metadata.py`

- [x] Test that `configs/experiments/aug00_none.yaml` loads with manual weights `[1,4,4,4,4,4,4,4]`, 60 epochs, and disabled early stopping.
- [x] Test deterministic dataset file-list hashing and Git SHA metadata with temporary fixtures/mocks.
- [x] Test append-only CSV headers/rows and complete config-copy creation.
- [x] Run the focused tests and verify they fail before implementation.

### Task 2: Extend configuration and add experiment metadata helpers

**Files:**
- Modify: `src/fomo_servo/config.py`
- Create: `src/fomo_servo/experiments.py`
- Create: `tests/test_experiment_metadata.py`

- [x] Parse optional `experiment.name` and `experiment.summary_csv` paths without resolving machine-specific paths.
- [x] Implement sorted relative dataset-file-list hashing, Git SHA retrieval with diagnostic errors, config copying, and append-only CSV writing using standard library APIs.
- [x] Keep all existing configurations valid by making experiment recording opt-in.

### Task 3: Persist complete baseline records without changing training behavior

**Files:**
- Modify: `src/fomo_servo/training/engine.py`
- Modify: `scripts/train.py`

- [x] Track elapsed training time, primary best epoch, best validation precision/recall/threshold/localization/count metrics, and existing grid/centroid best F1 values.
- [x] Preserve the existing optimizer/scheduler/model/data/postprocess calls exactly; use the existing validation threshold sweep for experiment metrics.
- [x] Save the complete config copy, `experiment_metadata.json`, and append one row to `experiments_summary.csv` after training.
- [x] Leave non-experiment runs unchanged apart from backward-compatible summary fields.

### Task 4: Add the no-augmentation configuration

**Files:**
- Create: `configs/experiments/aug00_none.yaml`

- [x] Copy the current seven-class 192 configuration semantics.
- [x] Set manual weights to background `1.0` and every foreground class to `4.0`.
- [x] Set epochs to `60`, early stopping patience to `0`, output to `outputs/experiments/aug00_none`, and keep model/data/training/evaluation/postprocess fields unchanged.
- [x] Do not add an augmentation section or augmentation code.

### Task 5: Verify and run the real CUDA experiment

- [x] Run complete pytest.
- [x] Run `aug00_none` with `--device cuda` and the configured dataset environment variable.
- [x] Verify config copy, metadata, checkpoints, and append-only summary row.
