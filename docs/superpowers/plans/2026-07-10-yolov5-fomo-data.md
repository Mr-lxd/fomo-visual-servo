# YOLOv5 FOMO Data Pipeline Implementation Plan

> **For agentic workers:** Execute test-first. Do not add a neural network, training loop, or ONNX export code.

**Goal:** Load YOLOv5-format images and labels, apply reversible letterbox geometry, and emit stride-8 FOMO class heatmaps for single-class merge and preserved multi-class modes.

**Architecture:** Geometry lives in `fomo_servo.geometry` and owns letterbox, bbox, and heatmap-coordinate transformations. `fomo_servo.datasets` owns YOLO parsing, class mapping, image loading, and heatmap target assembly. The visualizer consumes the public dataset API and writes a four-panel OpenCV image; it contains no model inference.

**Tech Stack:** Python 3.10, NumPy, OpenCV, PyYAML, pytest, synthetic JPEG fixtures.

---

### Task 1: Define geometry, parser, dataset, and visualization tests

**Files:**

- Create: `tests/test_letterbox.py`
- Create: `tests/test_yolo_dataset.py`
- Create: `tests/test_heatmap.py`
- Create: `tests/test_visualize_yolo_heatmap.py`
- Create: `tests/fixtures/yolo_micro/data.yaml`
- Create: `tests/fixtures/yolo_micro/labels/train/*.txt`
- Create: `tests/fixtures/yolo_micro/labels/val/*.txt`
- Create: `tests/fixtures/yolo_micro/invalid_labels.txt`

- [ ] Test landscape, portrait, and square letterbox round trips with continuous forward/inverse coordinate error at most one pixel.
- [ ] Test empty, multiple, and edge-centroid heatmaps; require class-index `[G,G]` and one-hot `[1+N,G,G]` targets.
- [ ] Test single-class merge and preserved multi-class mappings.
- [ ] Test malformed YOLO lines raise a descriptive `YoloLabelError`.
- [ ] Test the visualization script writes a four-panel image from the fixture dataset.
- [ ] Run focused tests and verify they fail because the new modules, script, and fixture images do not exist.

### Task 2: Implement reversible letterbox geometry and heatmap target generation

**Files:**

- Create: `src/fomo_servo/geometry/__init__.py`
- Create: `src/fomo_servo/geometry/letterbox.py`
- Create: `src/fomo_servo/datasets/heatmap.py`

- [ ] Implement RGB letterbox to square `[S,S,3]` without crop or non-uniform scaling.
- [ ] Preserve original/resized dimensions, all padding edges, and scale in an immutable transform object.
- [ ] Implement point/bbox forward and inverse operations, plus continuous heatmap-coordinate inversion.
- [ ] Generate background-zero class-index and one-hot stride-8 heatmaps; raise on conflicting-class cell collisions and count same-class collisions.
- [ ] Run geometry and heatmap tests; verify they pass.

### Task 3: Implement YOLOv5 dataset loading and synthetic JPEG fixtures

**Files:**

- Create: `src/fomo_servo/datasets/yolo.py`
- Modify: `src/fomo_servo/datasets/__init__.py`
- Create: `tests/fixtures/yolo_micro/images/train/*.jpg`
- Create: `tests/fixtures/yolo_micro/images/val/*.jpg`

- [ ] Parse `data.yaml` class names as list or index mapping.
- [ ] Parse `class_id x_center y_center width height`, validate finite normalized values and bbox bounds, and treat a missing/empty label file as no targets.
- [ ] Implement class mapping modes `merge_single` and `preserve`; return normalized CHW image `[3,S,S]`, original/letterbox boxes, class-index heatmap, one-hot heatmap, and geometry metadata.
- [ ] Generate deterministic tiny JPEG fixtures with OpenCV because text patches cannot encode valid binary JPEG files.
- [ ] Run dataset tests; verify they pass.

### Task 4: Implement and verify the visualization script

**Files:**

- Create: `scripts/visualize_yolo_heatmap.py`

- [ ] Add CLI parameters for dataset root, split, sample index, input size, stride, mapping mode, output path, and optional display.
- [ ] Render original bbox, letterbox image, scaled heatmap, and original image with decoded grid-centre centroids in a four-panel OpenCV image.
- [ ] Run the visualization test and direct script command against the fixture dataset.

### Task 5: Full validation

**Files:**

- Verify: all new source, tests, fixtures, and script files.

- [ ] Run `conda run -n fomo-servo-train python -m pytest -q`.
- [ ] Run the visualizer on fixtures and inspect the generated panel image dimensions.
- [ ] Confirm no model, optimizer, trainer, or ONNX exporter was added.
