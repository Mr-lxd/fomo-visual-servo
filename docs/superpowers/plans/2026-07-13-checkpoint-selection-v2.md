# Checkpoint Selection v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add reproducible weights-only epoch snapshots, FP32 offline checkpoint selection by centroid PR-AUC, and safe inference-only candidates without changing legacy training checkpoint semantics.

**Architecture:** Keep full training state in the existing engine.  Put portable snapshot/candidate I/O and atomic publication in a small training module; put metric computation and offline evaluation in evaluation/metrics modules; keep scripts as thin CLI adapters.  New configuration defaults are inert so existing locked experiments retain their protocol.

**Tech Stack:** Python 3.10, PyTorch, pytest, CSV/JSON, standard-library hashing and atomic file replacement.

---

### Task 1: Specify and validate v2 configuration

**Files:**
- Modify: `src/fomo_servo/config.py`
- Test: `tests/test_config.py`

- [ ] Write failing tests for default-disabled snapshots, invalid intervals/formats, threshold-grid validation, and calibration split validation.
- [ ] Run `python -m pytest tests/test_config.py -q` and confirm the tests fail because v2 fields do not exist.
- [ ] Add frozen config dataclasses and strict YAML parsing with inert defaults; reject unknown selection metrics and invalid values with `ConfigError`.
- [ ] Re-run the focused tests and then the config suite.

### Task 2: Add atomic weights-only snapshot and candidate protocol

**Files:**
- Create: `src/fomo_servo/training/snapshots.py`
- Modify: `src/fomo_servo/training/engine.py`
- Modify: `src/fomo_servo/inference/predictor.py`
- Test: `tests/test_checkpoint_selection_v2.py`

- [ ] Write failing tests asserting snapshot payload omissions, filename/interval/retention, atomic replacement, source SHA-256, candidate schema, unchanged state dict, inference loading, deterministic ties, and explicit resume rejection.
- [ ] Run `python -m pytest tests/test_checkpoint_selection_v2.py -q` and confirm import/behaviour failures.
- [ ] Implement CPU state extraction, sanitized fingerprinting, atomic `torch.save`, snapshot validation, candidate creation, and candidate-aware resume errors; invoke snapshots only after each completed configured epoch.
- [ ] Re-run focused tests, including legacy checkpoint loading tests.

### Task 3: Add centroid PR-AUC and offline FP32 evaluator

**Files:**
- Create: `src/fomo_servo/metrics/pr_auc.py`
- Modify: `src/fomo_servo/evaluation/validation.py`
- Create: `src/fomo_servo/evaluation/epoch_snapshots.py`
- Test: `tests/test_centroid_metrics.py`
- Test: `tests/test_checkpoint_selection_v2.py`

- [ ] Write failing pure-metric tests for known trapezoidal AUC, threshold-order invariance, no-GT/no-prediction cases, GT-only macro membership, and deterministic selection.
- [ ] Run the focused tests and confirm they fail because PR-AUC is absent.
- [ ] Implement raw PR curves, documented trapezoidal integration, shared FP32 logit collection, per-epoch evaluator records, calibration guards, and candidate selection.
- [ ] Re-run focused metrics/evaluation tests and verify validation/test data paths remain augmentation-free.

### Task 4: Add CLI entry points and legacy regression audit

**Files:**
- Create: `scripts/evaluate_epoch_snapshots.py`
- Create: `scripts/audit_checkpoint_selection_v2.py`
- Test: `tests/test_checkpoint_selection_v2.py`
- Modify: `docs/threshold_protocol.md`

- [ ] Write failing CLI/helper tests for FP32/no-autocast invocation, missing calibration split, explicit optimistic calibration, and CSV/JSON output schema.
- [ ] Run focused tests and confirm the absent scripts/helpers fail.
- [ ] Implement argparse adapters around the evaluator and audit six supplied legacy checkpoints without altering their files; document metric semantics and candidate non-resumability.
- [ ] Run the focused CLI tests and the legacy-audit dry run against available output files.

### Task 5: Verify the protocol without new formal training

**Files:**
- Modify: `docs/superpowers/specs/2026-07-13-checkpoint-selection-v2-design.md`
- Modify: `docs/superpowers/plans/2026-07-13-checkpoint-selection-v2.md`

- [ ] Run `conda run --no-capture-output -n fomo-servo-train python -m pytest -q`.
- [ ] Run `conda run --no-capture-output -n fomo-servo-train python -m compileall src scripts`.
- [ ] Run `git -c safe.directory=D:/DL_Project/fomo-visual-servo diff --check` and inspect `git diff --stat`.
- [ ] Run the six-checkpoint regression audit, estimate a 60-epoch scan from observed per-checkpoint evaluator time, and report results without training, committing, or pushing.
