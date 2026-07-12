# FOMO Augmentation Framework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a disabled-by-default augmentation configuration and no-op train-only pipeline without changing locked baseline behavior.

**Architecture:** `fomo_servo.datasets.augmentation` owns schema-facing no-op pipeline behavior and explicit future-algorithm errors. `YOLOv5FOMODataset` invokes it before letterbox only for the configured train split. Existing geometry and heatmap code remain unchanged. The visualization script consumes the same dataset sample contract.

**Tech Stack:** Python 3.10, NumPy, OpenCV, PyYAML, pytest, standard-library argparse.

---

### Task 1: Add failing schema, pipeline, dataset-equivalence, and CLI tests

**Files:**

- Create: `tests/test_augmentation.py`
- Create: `tests/test_augmentation_visualize.py`
- Modify: `tests/test_config.py`
- Modify: `tests/test_imports.py`

- [ ] **Step 1: Test the complete disabled schema and invalid probability failures**

```python
config = load_config("configs/experiments/aug00_none_locked.yaml")
assert config.augmentation.enabled is False
assert config.augmentation.color_jitter.probability == 0.0
with pytest.raises(ConfigurationError, match="probability"):
    load_config(config_with_probability=1.1)
```

- [ ] **Step 2: Test enabled=false dataset output against the legacy path**

```python
legacy = YOLOv5FOMODataset(..., augmentation=None)[index]
framework = YOLOv5FOMODataset(..., augmentation=config.augmentation, is_train=True)[index]
np.testing.assert_array_equal(legacy.image, framework.image)
assert legacy.original_boxes == framework.original_boxes
assert legacy.transform == framework.transform
np.testing.assert_array_equal(legacy.heatmap.class_index, framework.heatmap.class_index)
np.testing.assert_array_equal(legacy.heatmap.one_hot, framework.heatmap.one_hot)
```

- [ ] **Step 3: Test validation/test no-op and deterministic RNG behavior**

```python
train_pipeline = AugmentationPipeline(config.augmentation, is_train=True)
validation_pipeline = AugmentationPipeline(config.augmentation, is_train=False)
assert train_pipeline.apply(image, boxes, np.random.default_rng(42)) == validation_pipeline.apply(image, boxes, np.random.default_rng(7))
```

- [ ] **Step 4: Test the visualization script writes a disabled-framework panel**

Run: `python scripts/visualize_augmentations.py --dataset-root tests/fixtures/yolo_micro --split train --index 0 --input-size 96 --output <tmp>/augmentation.jpg`

Expected: exit 0 and a readable image with source/letterbox/heatmap panels.

- [ ] **Step 5: Run focused tests and verify missing APIs fail for expected reasons**

Run: `conda run --no-capture-output -n fomo-servo-train python -m pytest tests/test_augmentation.py tests/test_augmentation_visualize.py tests/test_config.py -q`

Expected: FAIL because augmentation config, pipeline, and visualization entry point do not yet exist.

### Task 2: Implement disabled schema and train-only no-op pipeline

**Files:**

- Modify: `src/fomo_servo/config.py`
- Create: `src/fomo_servo/datasets/augmentation.py`
- Modify: `src/fomo_servo/datasets/__init__.py`

- [ ] **Step 1: Add frozen dataclasses for global and five operation settings**

```python
@dataclass(frozen=True)
class AugmentationConfig:
    enabled: bool
    color_jitter: ColorJitterConfig
    horizontal_flip: ProbabilityConfig
    gaussian_blur: ProbabilityConfig
    gaussian_noise: ProbabilityConfig
    affine: ProbabilityConfig
```

- [ ] **Step 2: Parse and validate all YAML fields without changing locked training values**

```python
augmentation = _parse_augmentation(payload.get("augmentation", {}))
```

- [ ] **Step 3: Implement no-op application and explicit future-algorithm error**

```python
result = pipeline.apply(original_rgb, original_boxes, rng)
# disabled: same image, same boxes, no random draw
# enabled future operator: raise AugmentationNotImplementedError
```

- [ ] **Step 4: Run focused schema/pipeline tests**

Run: `conda run --no-capture-output -n fomo-servo-train python -m pytest tests/test_augmentation.py tests/test_config.py -q`

Expected: PASS.

### Task 3: Insert pipeline before letterbox and add visualization interface

**Files:**

- Modify: `src/fomo_servo/datasets/yolo.py`
- Create: `scripts/visualize_augmentations.py`
- Modify: `tests/test_yolo_dataset.py`
- Create: `tests/test_augmentation_visualize.py`

- [ ] **Step 1: Pass original RGB and original pixel-coordinate boxes through no-op pipeline before letterbox**

```python
original_boxes = _normalized_to_absolute(...)
augmented_image, augmented_boxes = pipeline.apply(original_image, original_boxes, rng)
letterbox_image, transform = letterbox_rgb(augmented_image, self.input_size)
```

- [ ] **Step 2: Force `is_train=False` behavior for validation and test splits**

```python
effective_pipeline = pipeline if is_train else AugmentationPipeline.disabled()
```

- [ ] **Step 3: Render original/augmented/letterbox/heatmap panels without implementing augmentation**

The script must accept dataset root, split, index, input size and output path; it must use dataset outputs and never modify model or training configuration.

- [ ] **Step 4: Run focused dataset and visualization tests**

Run: `conda run --no-capture-output -n fomo-servo-train python -m pytest tests/test_yolo_dataset.py tests/test_augmentation.py tests/test_augmentation_visualize.py -q`

Expected: PASS.

### Task 4: Update locked experiment config only with disabled augmentation schema

**Files:**

- Modify: `configs/experiments/aug00_none_locked.yaml`
- Verify: all locked fields from the handoff remain unchanged.

- [ ] **Step 1: Add the full disabled augmentation block**

```yaml
augmentation:
  enabled: false
  color_jitter: {enabled: false, probability: 0.0, brightness: 0.0, contrast: 0.0, saturation: 0.0, hue: 0.0}
  horizontal_flip: {enabled: false, probability: 0.0}
  gaussian_blur: {enabled: false, probability: 0.0}
  gaussian_noise: {enabled: false, probability: 0.0}
  affine: {enabled: false, probability: 0.0}
```

- [ ] **Step 2: Confirm the diff contains no model/loss/training/evaluator/threshold edits**

Run: `git diff -- configs/experiments/aug00_none_locked.yaml`

Expected: only the new augmentation block is added.

### Task 5: Full verification

**Files:**

- Verify: all source, test, script, config and documentation files above.

- [ ] **Step 1: Run the required pytest command**

Run: `conda run --no-capture-output -n fomo-servo-train python -m pytest -q`

- [ ] **Step 2: Run compileall**

Run: `conda run --no-capture-output -n fomo-servo-train python -m compileall src scripts`

- [ ] **Step 3: Run diff whitespace validation**

Run: `git -c safe.directory=D:/DL_Project/fomo-visual-servo diff --check`

- [ ] **Step 4: Confirm no aug01_color implementation or formal CUDA training was executed**

Expected: only disabled framework files are changed and all training variables remain locked.
