# AMP Gradient Overflow Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Prevent the first AMP-scaled FOMO training batch from aborting because the default GradScaler scale overflows the dense heatmap head bias gradient.

**Architecture:** Expose AMP's initial loss scale as a validated YAML training parameter. The training engine passes that value to `torch.amp.GradScaler` while retaining the existing non-finite-gradient guard. The real seven-class configuration uses a conservative initial scale of 256, confirmed by a controlled CUDA reproduction.

**Tech Stack:** Python 3.10, PyTorch AMP/GradScaler, pytest, YAML configuration.

---

### Task 1: Add a configuration regression test

**Files:**
- Modify: `tests/test_config.py`

- [x] Add a test that loads a minimal CUDA/AMP training YAML containing `amp_initial_scale: 256.0` and asserts the parsed `TrainingConfig` preserves the value.
- [x] Run `conda run -n fomo-servo-train python -m pytest tests/test_config.py -q`; it failed as expected before the field was added to `TrainingConfig`.

### Task 2: Implement the YAML field and GradScaler wiring

**Files:**
- Modify: `src/fomo_servo/config.py`
- Modify: `src/fomo_servo/training/engine.py`
- Modify: `configs/aquarium_pretrain_192.yaml`

- [x] Add `amp_initial_scale: float = 256.0` to `TrainingConfig` and validate it as a finite positive number under `training.amp_initial_scale`.
- [x] Construct `torch.amp.GradScaler("cuda", init_scale=config.training.amp_initial_scale, enabled=runtime.amp_enabled)`.
- [x] Set `training.amp_initial_scale: 256.0` in the real seven-class configuration.

### Task 3: Verify the fix

**Files:**
- No additional production files.

- [x] Run the targeted configuration and training tests.
- [x] Run the complete pytest suite.
- [x] Start the requested full training command with the real configuration on CUDA.
- [x] Verify `last.pt`, `best_val_f1.pt`, and `history.csv` exist and contain records from the completed training.
