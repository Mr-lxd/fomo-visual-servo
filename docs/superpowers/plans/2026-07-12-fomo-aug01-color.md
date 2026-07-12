# FOMO aug01_color Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add only deterministic train-split RGB color jitter to the existing augmentation framework while preserving every locked baseline variable and validation/test no-op behavior.

**Architecture:** Extend the existing `AugmentationPipeline` with a color-only branch operating on original RGB `uint8 [H,W,3]` images. Pass an explicit per-sample `numpy.random.Generator` derived from the configured seed and PyTorch DataLoader worker seed; keep bbox, letterbox metadata, heatmap labels, and collate tensors unchanged. Extend the visualization CLI to produce a deterministic 16-image contact sheet and JSON audit records.

**Tech Stack:** Python 3.10, NumPy, OpenCV RGB/HSV conversion, PyYAML, PyTorch DataLoader worker metadata, pytest.

---

### Task 1: Lock the experiment comparison and behavioral tests

**Files:**
- Create: `configs/experiments/aug01_color.yaml`
- Modify: `tests/test_augmentation.py`
- Modify: `tests/test_augmentation_visualize.py`

- [ ] Add tests for all-disabled, operation-disabled, probability-zero, and zero-parameter probability-one exact no-op behavior.
- [ ] Add tests for deterministic same-seed output, different-seed output, RGB channel order, clipping/dtype/shape, and unchanged boxes/heatmaps/collision counts.
- [ ] Add validation/test split no-op tests and a two-worker DataLoader repeatability test.
- [ ] Add a resolved-config comparison that removes only `source_path`, experiment name, output directory, and augmentation fields, then asserts aug00 and aug01 are equal elsewhere.
- [ ] Add a CLI contact-sheet smoke test using the synthetic fixture with `--num-images 2`.
- [ ] Run the focused tests and confirm they fail because color jitter, aug01 config, and contact-sheet mode are not implemented.

### Task 2: Implement controlled color jitter and metadata

**Files:**
- Modify: `src/fomo_servo/datasets/augmentation.py`
- Modify: `src/fomo_servo/datasets/yolo.py`
- Modify: `src/fomo_servo/datasets/__init__.py`

- [ ] Add immutable factor and metadata records with neutral defaults: brightness/contrast/saturation `1.0`, hue `0.0`, `applied=False`.
- [ ] Sample only from the passed `numpy.random.Generator`: probability gate, factors `[1-p,1+p]`, and hue `[-p,p]`.
- [ ] Apply brightness and contrast in float RGB, saturation and hue in explicit OpenCV RGB/HSV space, clip to `[0,255]`, and return the original shape/dtype.
- [ ] Short-circuit all no-op cases before consuming RNG and reject any non-color operation that is enabled.
- [ ] Add dataset-local RNG creation from configured seed plus `get_worker_info().seed`; pass `train_split` and seed from the training engine without using global random state.
- [ ] Preserve original/augmented/letterbox boxes, transform, heatmaps, and collate contract; attach metadata only to the sample object.
- [ ] Run the focused tests and confirm green.

### Task 3: Add aug01 configuration and visualization audit output

**Files:**
- Modify: `scripts/visualize_augmentations.py`
- Modify: `tests/test_augmentation_visualize.py`
- Create: `outputs/experiments/aug01_color/visualization/` at runtime only

- [ ] Extend the CLI with config-driven dataset loading, default output directory, fixed image count defaulting to 16, deterministic variant seeds, and optional legacy single-output mode.
- [ ] Render original, one typical color-jitter result, three additional random results, and minimum/neutral/maximum factor examples with bbox, centroid, and factor labels.
- [ ] Write `color_jitter_contact_sheet.jpg` and `color_jitter_samples.json`; keep generated outputs ignored and outside Git staging.
- [ ] Ensure relative image paths and all factor metadata are JSON serializable.
- [ ] Run the visualization smoke test.

### Task 4: Full verification and handoff

**Files:**
- No additional source changes unless a test exposes a scoped defect.

- [ ] Run `conda run --no-capture-output -n fomo-servo-train python -m pytest -q` without formal CUDA training.
- [ ] Run `conda run --no-capture-output -n fomo-servo-train python -m compileall src scripts`.
- [ ] Run `git -c safe.directory=D:/DL_Project/fomo-visual-servo diff --check`.
- [ ] Report modified files, mathematical semantics, RNG source, test results, contact-sheet paths, config drift, and unresolved issues; do not commit or push.
