# FOMO Augmentation Suite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复 epoch-aware deterministic RNG，并将当前在线增强扩展为可配置 preset、blur、noise、mild affine、统计和可视化套件。

**Architecture:** 保留 `YOLOv5FOMODataset` 的数据/heatmap边界，新增独立稳定 seed 与 preset resolver；augmentation pipeline 只负责原图和 bbox 变换，训练 engine 负责 epoch 状态和统计聚合。所有几何变换在 letterbox 前完成，validation/test 使用禁用 pipeline。

**Tech Stack:** Python 3.10、NumPy、OpenCV、PyTorch DataLoader、pytest、YAML。

---

### Task 1: Lock the public design and baseline invariants

**Files:**
- Read: `AGENTS.md`, `docs/handoffs/2026-07-12-fomo-augmentation-handoff.md`, `configs/experiments/aug00_none_locked.yaml`, `configs/experiments/aug01_color.yaml`, `configs/experiments/aug02_color_hflip.yaml`
- Create: `docs/superpowers/specs/2026-07-12-fomo-augmentation-suite-design.md`

- [x] **Step 1: Record the approved design.**

The design file fixes RNG, preset, operation order, geometry, no-op and metadata semantics. No model or training hyperparameter is changed.

- [x] **Step 2: Confirm branch and clean baseline.**

Run:

```powershell
git -c safe.directory=D:/DL_Project/fomo-visual-servo branch --show-current
git -c safe.directory=D:/DL_Project/fomo-visual-servo status --porcelain
```

Expected branch: `feature/fomo-augmentation-suite`; expected status: empty.

### Task 2: Add failing RNG and preset contract tests

**Files:**
- Create: `tests/test_augmentation_suite.py`
- Modify: `tests/test_aug01_color.py`
- Modify: `tests/test_aug02_color_hflip.py`
- Test fixtures: existing `tests/fixtures`

- [ ] **Step 1: Test epoch-aware sample seed semantics.**

Assert that `set_epoch(0)` and repeated access of one index are equal, epoch 1 differs for a non-neutral preset, stable seed recreation is equal, different base seeds differ, and worker-count 0/2/4 per-index signatures match.

- [ ] **Step 2: Test resume epoch propagation.**

Use a short training configuration and a spy/real dataset to assert the first resumed epoch calls `set_epoch(checkpoint_epoch + 1)` before loading train samples.

- [ ] **Step 3: Test preset contracts.**

Assert exact expansion for `none`, `photometric`, `underwater_conservative`, and `custom`; reject unknown preset and unknown override paths; assert `enabled=false` forces no-op; assert legacy aug01/aug02 fields resolve without changing their existing values and emit only the documented deprecation warning.

- [ ] **Step 4: Test new transform contracts.**

Add tests for blur/noise shape, dtype, range, deterministic sampling and geometry preservation; add affine identity, translation, scale, rotation, clipping, visibility drop, count and collision tests.

- [ ] **Step 5: Run the new tests before implementation.**

Run:

```powershell
conda run --no-capture-output -n fomo-servo-train python -m pytest tests/test_augmentation_suite.py -q
```

Expected: FAIL because the new resolver, metadata, operations and epoch API are not implemented yet.

### Task 3: Implement stable epoch-aware RNG

**Files:**
- Create: `src/fomo_servo/datasets/rng.py`
- Modify: `src/fomo_servo/datasets/yolo.py`
- Modify: `src/fomo_servo/training/engine.py`
- Modify: `src/fomo_servo/datasets/__init__.py`

- [ ] **Step 1: Add stable seed derivation.**

Implement a public helper that packs non-negative 64-bit `base_seed`, `epoch`, and `sample_index` with `struct`, hashes the bytes with SHA-256, takes the first eight bytes as an unsigned integer, and returns `np.random.default_rng(sample_seed)`. Reject booleans, negatives and values outside uint64 with diagnostic errors.

- [ ] **Step 2: Add Dataset epoch state.**

Initialize `current_epoch=0`, add `set_epoch(epoch)` with non-negative integer validation, and make `_sample_rng(index)` use only the base seed, current epoch and index. Remove worker id from sample seed. Keep explicit `get_sample(index, rng=...)` support for visualization/tests.

- [ ] **Step 3: Propagate epoch from training.**

Before each `train_one_epoch` call in the existing `for epoch in range(...)` loop, call `train_dataset.set_epoch(epoch)`. On resume, the existing restored start epoch is already `checkpoint_epoch + 1`, so the first resumed call must use that value. Set `persistent_workers=False` explicitly in both DataLoaders and document the invariant.

- [ ] **Step 4: Run RNG tests.**

Run the focused RNG tests and confirm the requested same-epoch/different-epoch/recreation/worker behavior passes.

### Task 4: Implement preset resolver and expanded configuration

**Files:**
- Create: `src/fomo_servo/datasets/presets.py`
- Modify: `src/fomo_servo/config.py`
- Modify: `src/fomo_servo/experiments.py`
- Create: `configs/experiments/augmentation_suite.yaml`
- Modify: `tests/test_config.py`

- [ ] **Step 1: Add canonical preset constants and resolver.**

Keep one immutable mapping for all four presets. Deep-merge only known dotted override paths; reject unknown keys, wrong types, invalid ranges and unknown preset names. Return a fully expanded `AugmentationConfig` plus a JSON-safe resolved mapping.

- [ ] **Step 2: Extend dataclasses without changing locked fields.**

Add preset name/override provenance and blur kernel/sigma, noise std, and affine fields. Preserve existing constructor fields and old YAML semantics. Emit `DeprecationWarning` only when a legacy inline augmentation mapping is used without a preset.

- [ ] **Step 3: Persist resolved augmentation.**

Add the preset name and JSON-safe resolved mapping to checkpoint payload, training summary, and experiment metadata. Do not add absolute dataset paths.

- [ ] **Step 4: Add a suite config.**

Create a config based on the locked baseline with `augmentation.enabled: true`, `preset: underwater_conservative`, empty overrides, and a new output directory. All model/training/evaluation fields remain byte-for-byte equivalent after normalization.

- [ ] **Step 5: Run config tests.**

Run:

```powershell
conda run --no-capture-output -n fomo-servo-train python -m pytest tests/test_config.py tests/test_augmentation_suite.py -q
```

### Task 5: Implement the online augmentation operations

**Files:**
- Modify: `src/fomo_servo/datasets/augmentation.py`
- Modify: `src/fomo_servo/datasets/yolo.py`
- Modify: `src/fomo_servo/datasets/heatmap.py` only if collision statistics need a public helper
- Modify: `src/fomo_servo/datasets/__init__.py`

- [ ] **Step 1: Replace color-only metadata with backward-compatible suite metadata.**

Retain existing factor properties and `horizontal_flip_applied`; add epoch/index/seed, operation flags, sampled parameters, clipping/drop counts, object counts and collision counts. Ensure collate does not put metadata into image/target tensors.

- [ ] **Step 2: Implement strict pipeline ordering.**

Apply hflip, affine, color jitter, blur, then noise using the sample RNG. Disabled or probability-zero operations must return elementwise no-op image and unchanged geometry. Validation/test must bypass the full pipeline before any RNG is required.

- [ ] **Step 3: Implement Gaussian blur.**

Validate positive odd kernels, sample one configured kernel and sigma, call OpenCV GaussianBlur in RGB order, preserve uint8 shape/range, and record the chosen values.

- [ ] **Step 4: Implement Gaussian noise.**

Sample uint8-scale std, add `rng.normal(0, std, image.shape)` in float32, round/clip to `[0,255]`, convert back to uint8 and record std.

- [ ] **Step 5: Implement affine geometry.**

Sample scale/translation/rotation, transform image and all bbox corners with one 2x3 matrix, rebuild and clip axis-aligned boxes, calculate visibility, drop invalid boxes, recompute centers and collision counts, and record affine statistics.

- [ ] **Step 6: Run operation tests.**

Run:

```powershell
conda run --no-capture-output -n fomo-servo-train python -m pytest tests/test_augmentation_suite.py -q
```

### Task 6: Aggregate augmentation statistics in training artifacts

**Files:**
- Modify: `src/fomo_servo/datasets/collate.py`
- Modify: `src/fomo_servo/training/engine.py`
- Modify: `src/fomo_servo/experiments.py`
- Modify: `tests/test_training_engine.py`

- [ ] **Step 1: Preserve sample metadata through collate.**

Extend `FOMOBatch` with a tuple of lightweight metadata records while leaving tensor shapes and device transfer unchanged.

- [ ] **Step 2: Aggregate per-epoch counters.**

Count samples, each operation trigger/rate, clipping/drops, pre/post objects and same/different-class collisions in the train loop. Keep validation excluded from augmentation statistics.

- [ ] **Step 3: Persist history and summaries.**

Add stable CSV columns and JSON fields. Existing history rows remain readable; resume appends compatible rows and preserves resolved preset metadata.

- [ ] **Step 4: Test statistics and metadata.**

Assert counts equal the number of train samples, rates are bounded, object/collision counts are numeric, and saved artifacts contain preset and resolved parameters without absolute paths.

### Task 7: Extend visualization and add integration tests

**Files:**
- Modify: `scripts/visualize_augmentations.py`
- Modify: `tests/test_augmentation_visualize.py`
- Create: `tests/test_augmentation_visualize_suite.py` if the existing test becomes too large

- [ ] **Step 1: Add suite visualization mode.**

Require train split, select at least 16 deterministic real images, render epoch 0/1/2, photometric, underwater and affine panels with boxes, centroids, clipping/drops and sampled parameters, and write only relative paths to JSON.

- [ ] **Step 2: Test synthetic visualization output.**

Use the existing fixture and assert all four required files are created, JSON has the required fields, and no absolute path is present.

### Task 8: Full verification and pause

**Files:**
- No additional source files.

- [ ] **Step 1: Run all required checks.**

```powershell
conda run --no-capture-output -n fomo-servo-train python -m pytest -q
conda run --no-capture-output -n fomo-servo-train python -m compileall src scripts
git -c safe.directory=D:/DL_Project/fomo-visual-servo diff --check
```

- [ ] **Step 2: Run a synthetic smoke check.**

Exercise one train sample through all preset modes and one short CPU DataLoader epoch. Do not run CUDA training or formal training.

- [ ] **Step 3: Verify scope.**

Confirm no model, loss, optimizer, scheduler, training epoch count, checkpoint threshold, inference threshold or evaluator source changed. Confirm outputs are ignored and Git status contains only intended source/config/test/docs changes.

- [ ] **Step 4: Report and pause.**

Report modified files, RNG semantics, preset expansion, operation rules, statistics, test commands/results, visualization paths, initial performance measurement, unresolved issues, and explicitly state no training/commit/push occurred.
