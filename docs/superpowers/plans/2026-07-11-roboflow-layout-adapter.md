# Roboflow Layout Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train the FOMO project directly against the confirmed Roboflow YOLO directory layout without copying or altering the original dataset.

**Architecture:** `YOLOv5FOMODataset` owns split path resolution. It first checks the existing project layout and then Roboflow split folders, including the `val`→`valid` alias. A separate YAML preserves all seven source classes and supplies the only user-specific absolute data path.

**Tech Stack:** Python 3.10, existing PyYAML/OpenCV/NumPy/PyTorch stack, pytest fixtures.

---

### Task 1: Specify Roboflow path-resolution behavior in a failing dataset test

**Files:**

- Modify: `tests/test_yolo_dataset.py`

- [ ] **Step 1: Create a temporary Roboflow layout from existing tiny JPEG fixtures**

```python
root / "train" / "images"
root / "train" / "labels"
root / "valid" / "images"
root / "valid" / "labels"
```

- [ ] **Step 2: Require logical `split="val"` to resolve the physical `valid` folder**

```python
dataset = YOLOv5FOMODataset(root, split="val", input_size=96, stride=8, class_mode="preserve")
assert dataset.images_dir == root / "valid" / "images"
assert dataset.labels_dir == root / "valid" / "labels"
```

- [ ] **Step 3: Run the focused test**

Run: `conda run -n fomo-servo-train python -m pytest tests/test_yolo_dataset.py -q`

Expected: FAIL because the current loader only checks `images/val`.

### Task 2: Implement ordered project/Roboflow split resolution

**Files:**

- Modify: `src/fomo_servo/datasets/yolo.py`

- Modify: `tests/test_yolo_dataset.py`

- [ ] **Step 1: Resolve the existing project layout first**

```python
(root / "images" / split, root / "labels" / split)
```

- [ ] **Step 2: Try Roboflow `<split>/images` / `<split>/labels` and `val`→`valid`**

```python
(root / split / "images", root / split / "labels")
(root / "valid" / "images", root / "valid" / "labels")
```

- [ ] **Step 3: Raise one error listing all attempted image directories**

Run: `conda run -n fomo-servo-train python -m pytest tests/test_yolo_dataset.py -q`

Expected: PASS.

### Task 3: Add the confirmed seven-class training YAML and document usage

**Files:**

- Create: `configs/aquarium_pretrain_192.yaml`
- Modify: `README.md`

- [ ] **Step 1: Add the user-confirmed root and preserve seven source classes**

```yaml
dataset:
  root: "${FOMO_DATASET_ROOT}"
  classes: [fish, jellyfish, penguin, puffin, shark, starfish, stingray]
  class_mode: preserve
```

- [ ] **Step 2: Document the training command and output contract**

```powershell
python scripts/train.py --config configs/aquarium_pretrain_192.yaml --device cuda
```

### Task 4: Verification

**Files:**

- Verify: adapted loader, new YAML, tests, and README.

- [ ] **Step 1: Run full pytest without bytecode writes**

Run: `conda run -n fomo-servo-train cmd /c "set PYTHONDONTWRITEBYTECODE=1&&python -m pytest -q"`

Expected: project layout, Roboflow layout, CPU training smoke, CUDA tests, and existing geometry tests all pass; unavailable ONNX tests skip explicitly.

**Note:** This directory is not a Git checkout, so normal plan commit steps do not apply.
