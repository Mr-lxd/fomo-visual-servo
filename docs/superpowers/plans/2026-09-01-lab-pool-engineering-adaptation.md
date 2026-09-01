# Lab Pool Engineering Adaptation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic D2-compatible lab-pool training view, add formal train-only and strict weights initialization support, train the fixed 20-epoch engineering model, freeze/export it, and only then run held-out hardware validation.

**Architecture:** A focused dataset conversion module owns LabelMe-to-D2 transformation and provenance. Existing configuration and training modules gain nullable validation and weights-only initialization while preserving the current validation path. Existing deployment and video inference entry points consume separately named frozen engineering artifacts.

**Tech Stack:** Python 3.10, pathlib/json/hashlib, PyYAML, PyTorch, pytest, ONNX, ONNX Runtime, OpenCV, Git.

---

### Task 1: Deterministic training-view converter

**Files:**
- Create: `src/fomo_servo/datasets/lab_pool_view.py`
- Create: `scripts/build_lab_pool_training_view.py`
- Create: `tests/test_lab_pool_training_view.py`

- [ ] Write failing tests for the five approved mappings, four-point LabelMe rectangles, deterministic YOLO formatting, empty/background labels, source/generated SHA fields, and idempotent output.
- [ ] Run `python -m pytest -q tests/test_lab_pool_training_view.py` and verify failures are caused by the missing API.
- [ ] Implement typed conversion records, SHA helpers, strict LabelMe validation, epsilon-only normalized clamp, atomic staging publication, hardlink-with-copy-fallback image materialization, `data.yaml`, labels and manifest.
- [ ] Add failing tests proving a `5e-7` boundary excess is clamped and a larger violation aborts without publishing a partial view.
- [ ] Implement the minimal clamp/error behavior and rerun the targeted file to green.
- [ ] Run the converter against `lab_pool_v1`; audit 213 images, 318 foreground targets, 53 empty labels, zero test assets and deterministic manifest hash.

### Task 2: Nullable validation configuration

**Files:**
- Modify: `src/fomo_servo/config.py`
- Modify: `tests/test_config.py`

- [ ] Add failing config tests for `validation_split: null`, train-only early-stopping rejection, fixed-final-epoch policy, initialization path/SHA validation and initialize/resume mutual exclusion.
- [ ] Run the new config tests and verify RED.
- [ ] Change `DatasetConfig.validation_split` to optional and add only the required training initialization/checkpoint-policy fields.
- [ ] Preserve all defaults for existing YAML files and rerun `tests/test_config.py` to green.

### Task 3: Strict weights-only initialization

**Files:**
- Modify: `src/fomo_servo/training/engine.py`
- Modify: `tests/test_training_engine.py`

- [ ] Add a failing test that initializes a model from a weights-only snapshot, starts at epoch 1, records source SHA, and does not restore optimizer/RNG/history.
- [ ] Add failing tests for SHA mismatch, missing/unexpected state keys, malformed payload and initialize/resume conflict.
- [ ] Run the initialization tests and verify RED.
- [ ] Implement SHA verification and strict `model_state` loading before optimizer construction without changing resume behavior.
- [ ] Rerun initialization and existing resume tests to green.

### Task 4: Train-only engine path

**Files:**
- Modify: `src/fomo_servo/training/engine.py`
- Modify: `scripts/train.py`
- Modify: `tests/test_training_engine.py`

- [ ] Add failing tests proving null validation constructs only the train dataset and never calls validation/evaluation functions.
- [ ] Add failing tests proving no best checkpoint/alias or fabricated validation metrics are emitted, while `last.pt`, history, summary and final snapshot remain.
- [ ] Run these tests and verify RED.
- [ ] Implement optional validation loader and a separate train-only epoch/reporting branch with nullable summary metrics.
- [ ] Update CLI completion output for train-only runs.
- [ ] Rerun train-only tests and the existing two-epoch train+validation/resume regression tests to green.

### Task 5: Locked engineering configuration

**Files:**
- Create: `configs/engineering/lab_pool_adaptation_seed42_e20.yaml`
- Test: `tests/test_config.py`

- [ ] Add a failing test that asserts every locked protocol field, seven-class order, epoch20 snapshot selection, threshold 0.40 and disabled validation/sweeps.
- [ ] Run the test and verify RED because the config is absent.
- [ ] Add the approved config using environment-provided dataset and initialization paths.
- [ ] Rerun the locked-config test to green and compute its SHA-256.

### Task 6: Pre-training quality gate

- [ ] Run conversion, config, train-only, initialization and validation regression tests.
- [ ] Run `git diff --check`.
- [ ] Run `python -m compileall -q src scripts`.
- [ ] Run full `python -m pytest -q` and require every collected test to pass.
- [ ] Confirm held-out source SHA only; do not decode it or read extracted test frames.

### Task 7: Fixed CUDA training and freeze

- [ ] Run exactly `python scripts/train.py --config configs/engineering/lab_pool_adaptation_seed42_e20.yaml --device cuda` with required environment variables.
- [ ] Confirm exactly 20 completed epochs and 540 optimizer updates; do not choose by loss.
- [ ] Fix `epoch_020_weights.pt` as the selected checkpoint and record final train loss only as telemetry.
- [ ] Write a provenance record containing config SHA, training-view manifest SHA, source checkpoint SHA, final checkpoint SHA, seed, epoch and threshold 0.40.
- [ ] Recheck held-out source SHA without decoding.

### Task 8: Separate ONNX export and parity

- [ ] Inspect the existing formal export CLI requirements and create a lab-pool-specific export config/sidecar path without overwriting baseline artifacts.
- [ ] Export `lab_pool_d2_seed42_e20.onnx` from the fixed epoch20 checkpoint.
- [ ] Run ONNX checker and the existing PyTorch/ORT parity pipeline on non-held-out parity assets.
- [ ] Record ONNX SHA-256 and parity tolerances/results in provenance.

### Task 9: First held-out hardware-validation inference

- [ ] Verify the freeze record includes all required hashes and successful parity before decoding the held-out video.
- [ ] Run existing video inference once on `pool-20260831-005/raw.avi` with threshold 0.40 and the lab-pool ONNX/sidecar.
- [ ] Save a distinctly named annotated video and per-frame telemetry; report no quantitative metrics.
- [ ] Preserve the held-out source and verify its SHA again after inference.

### Task 10: Final verification and experiment branch publication

- [ ] Run targeted regression tests, compileall, `git diff --check` and full pytest after all code/config changes.
- [ ] Audit Git diff for absence of raw/processed datasets, videos, labels, generated outputs and the unrelated handoff.
- [ ] Commit logical experiment-branch changes and push `experiment/lab-pool-adaptation-v1` without merging main or creating a stable tag.
- [ ] Report capture commit, branch/HEAD, conversion counts, tests, training loss, checkpoint/ONNX hashes, parity, held-out ordering/output and Pi deployment recommendation.
