# Edge Impulse Parity Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a reproducible, test-verified parity audit for the locked epoch-58 PyTorch model and the supplied Edge Impulse float32 TFLite model without modifying raw data, training, checkpoint selection, loss, or thresholds.

**Architecture:** A manifest-only dataset view verifies immutable source hashes and removes only declared invalid annotation rows. A separate Edge Impulse compatibility module preserves the public FOMO postprocessor and legacy many-to-one centroid matching semantics, while retaining strict one-to-one evaluation as an explicit alternative. The TFLite script probes the actual model contract, preprocesses in a documented Fit-longest/RGB/zero-pad path, caches raw output tensors, and invokes either evaluator on the same clean view.

**Tech Stack:** Python 3.10, NumPy, PyTorch for the local checkpoint path, OpenCV, pytest, optional `ai-edge-litert` or TensorFlow Lite interpreter; no CUDA-only operation.

---

### Task 1: Record the immutable audit and parity-clean view

**Files:**
- Create: `src/fomo_servo/evaluation/parity_clean.py`
- Modify: `src/fomo_servo/datasets/yolo.py`
- Create: `tests/test_parity_clean.py`
- Modify: `tests/test_yolo_dataset.py`
- Create at runtime: `outputs/parity_audit/edge_impulse_parity_v1/invalid_label_audit.json`
- Create at runtime: `outputs/parity_audit/edge_impulse_parity_v1/invalid_label_audit.csv`
- Create at runtime: `outputs/parity_audit/edge_impulse_parity_v1/parity-clean-v1.json`
- Create: `scripts/audit_yolo_labels.py`

- [ ] **Step 1: Write failing audit and manifest tests**

```python
def test_manifest_removes_only_declared_zero_area_rows(tmp_path: Path) -> None:
    manifest = build_parity_clean_manifest(dataset_root, rules=("width_lte_zero", "height_lte_zero"))
    view = ParityCleanView(dataset_root, manifest)
    assert view.read_label_lines("test/labels/sample.txt") == ("0 0.5 0.5 0.2 0.2",)

def test_manifest_rejects_changed_source_hash(tmp_path: Path) -> None:
    view = ParityCleanView(dataset_root, manifest)
    label_path.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    with pytest.raises(ParityCleanError, match="SHA-256"):
        view.read_label_lines("test/labels/sample.txt")
```

- [ ] **Step 2: Run the new test file and verify it fails because the module is absent**

Run: `python -m pytest tests/test_parity_clean.py -q`

- [ ] **Step 3: Implement strict label auditing and `ParityCleanView`**

```python
class ParityCleanView:
    def read_label_lines(self, relative_path: str) -> tuple[str, ...]:
        """Return manifest-approved YOLO rows or raise on a changed source/undeclared invalid row."""

def audit_yolo_dataset(dataset_root: Path, class_count: int) -> LabelAudit:
    """Audit every image/label pair without mutating the source dataset."""
```

The audit must distinguish physical invalid rows from category counts, retain raw content and file SHA-256, recognize an empty label as a valid no-object sample, and create deterministic full-view/test-view hashes from the virtual cleaned bytes.

`YOLOv5FOMODataset` receives an optional, default-`None` label-reader callback. Its default remains `parse_yolo_label_file`; only the parity CLI supplies `ParityCleanView.parse_label_file`. The callback accepts `(label_path, num_source_classes)` and must raise a diagnostic error for changed source hashes or undeclared invalid rows. This prevents a separate copied dataset and makes use of the cleaning manifest explicit at the dataset boundary.

- [ ] **Step 4: Run the audit tests and hash regression test**

Run: `python -m pytest tests/test_parity_clean.py -q`

- [ ] **Step 5: Generate versioned audit artifacts from the real source dataset**

Run: `python scripts/audit_yolo_labels.py --config configs/experiments/checkpoint_v2_lite_aug03_locked.yaml --output-dir outputs/parity_audit/edge_impulse_parity_v1`

Expected: the command reports two physical zero-area test rows, writes the three requested artifacts, and reports the approved all-view and test-view hashes.

### Task 2: Implement EI FOMO decoding and two matching modes

**Files:**
- Create: `src/fomo_servo/evaluation/edge_impulse.py`
- Modify: `src/fomo_servo/evaluation/__init__.py`
- Create: `tests/test_edge_impulse_compat.py`

- [ ] **Step 1: Write failing postprocess and matching tests**

```python
def test_edge_impulse_decoder_accepts_exact_threshold_and_merges_diagonal_cells() -> None:
    probabilities = torch.zeros((1, 3, 3, 3), dtype=torch.float32)
    probabilities[0, 1, 0, 0] = 0.5
    probabilities[0, 1, 1, 1] = 0.8
    detections = decode_edge_impulse_fomo(probabilities, input_size=24, threshold=0.5)
    assert detections[0].input_bbox == (0.0, 0.0, 16.0, 16.0)

def test_legacy_matching_allows_two_predictions_to_match_one_ground_truth() -> None:
    report = evaluate_centroids(predictions, ground_truths, mode="edge_impulse_legacy")
    assert (report.true_positives, report.false_positives, report.false_negatives) == (2, 0, 0)
```

Tests must also cover no target, one target, different-class adjacent cells, one-to-many and many-to-one cases, incorrect class, false positive/negative, and normalized-distance values immediately below, exactly at, and immediately above `0.2`.

- [ ] **Step 2: Run the new compatibility test file and verify the expected import failure**

Run: `python -m pytest tests/test_edge_impulse_compat.py -q`

- [ ] **Step 3: Implement a separate EI-compatible decoder and metrics report**

```python
def normalized_centroid_distance(prediction: Point, target: Point, width: float, height: float) -> float:
    return hypot((prediction.x - target.x) / width, (prediction.y - target.y) / height)

def evaluate_centroids(..., mode: Literal["edge_impulse_legacy", "strict_one_to_one"]) -> EdgeImpulseEvaluation:
    """Evaluate one image at a time without replacing the repository's CentroidEvaluator."""
```

For local logits, apply softmax exactly once. For known probability tensors such as the EI TFLite output, reject negative/non-normalized data and never apply softmax again. Reproduce the referenced raster-order FOMO cube merge, `score >= threshold`, foreground-channel-only behavior, and legacy repeated nearest-GT assignments.

- [ ] **Step 4: Run the compatibility tests and existing metric/postprocess regression tests**

Run: `python -m pytest tests/test_edge_impulse_compat.py tests/test_centroid_metrics.py tests/test_postprocess.py -q`

### Task 3: Add fixed-threshold local parity evaluation

**Files:**
- Create: `scripts/evaluate_parity_local.py`
- Create: `tests/test_evaluate_parity_local.py`

- [ ] **Step 1: Write a failing CLI contract test**

```python
def test_local_parity_cli_requires_cleaning_manifest() -> None:
    assert main(["--config", "config.yaml", "--checkpoint", "epoch58.pt"]) == 2
```

- [ ] **Step 2: Run the CLI test and verify it fails because the script is missing**

Run: `python -m pytest tests/test_evaluate_parity_local.py -q`

- [ ] **Step 3: Implement a no-sweep local evaluator**

The CLI accepts one explicit checkpoint, one explicit cleaning manifest, one threshold, evaluator mode, and an output directory. It caches epoch-58 FP32 logits once, emits JSON/CSV image matches, and invokes the existing local evaluator or the new EI compatibility evaluator. It rejects thresholds other than the passed value and never alters the locked test manifest.

- [ ] **Step 4: Run script tests and the actual fixed evaluations**

Run: `python -m pytest tests/test_evaluate_parity_local.py -q`

Run exactly once each on the clean test view: local evaluator at `0.50`, EI legacy evaluator at `0.50`, and local evaluator at locked `0.35`. No threshold grid is accepted.

### Task 4: Add actual-TFLite probing, inference, and output caching

**Files:**
- Modify: `pyproject.toml`
- Create: `scripts/evaluate_edge_impulse_tflite.py`
- Create: `tests/test_evaluate_edge_impulse_tflite.py`

- [ ] **Step 1: Write failing TFLite path-selection and preprocessing tests**

```python
def test_zip_with_multiple_float32_candidates_requires_explicit_model_path(tmp_path: Path) -> None:
    with pytest.raises(TFLiteEvaluationError, match="multiple"):
        resolve_tflite_model(zip_path)

def test_fit_longest_zero_padding_places_rgb_image_at_center() -> None:
    image = np.full((10, 20, 3), 255, dtype=np.uint8)
    tensor, metadata = preprocess_edge_impulse_image(image, target_size=192, input_dtype=np.float32)
    assert tensor.shape == (1, 192, 192, 3)
    assert np.all(tensor[0, :48] == 0)
```

- [ ] **Step 2: Run the TFLite tests and verify the expected import failure**

Run: `python -m pytest tests/test_evaluate_edge_impulse_tflite.py -q`

- [ ] **Step 3: Implement optional official runtime discovery and explicit model probing**

The optional `tflite` dependency may use `ai-edge-litert`; the script must fall back to TensorFlow Lite, then `tflite_runtime`, and otherwise fail with one install command rather than silently skipping inference. It must inspect every candidate model's tensor dtype, shape, and quantization before choosing, record all model/ZIP hashes, detect a terminal Softmax through operators plus a probability-sum check, and require the caller to choose among multiple candidates.

- [ ] **Step 4: Implement fixed `0.5` inference and artifact cache**

For each image, save the preprocessed image, raw output tensor, decoded activated cells, merged detections, and image-level match data. A subsequent run with matching model/preprocessing/manifest hashes reuses the raw tensor cache rather than invoking the model again.

- [ ] **Step 5: Run TFLite unit tests and the real ZIP evaluation under an installed interpreter**

Run: `python -m pytest tests/test_evaluate_edge_impulse_tflite.py -q`

Run: `python scripts/evaluate_edge_impulse_tflite.py --model C:\\path\\to\\model.tflite --config configs/experiments/checkpoint_v2_lite_aug03_locked.yaml --cleaning-manifest outputs/parity_audit/edge_impulse_parity_v1/parity-clean-v1.json --split test --threshold 0.5 --output-dir outputs/parity_audit/edge_impulse_parity_v1/ei_float32`

### Task 5: Produce the audit document and final 2×2 matrix

**Files:**
- Create: `docs/experiments/edge_impulse_parity_audit.md`

- [ ] **Step 1: Document confirmed, inferred, unknown, and blocked facts**

Include the user-confirmed Studio settings/results, the exact ZIP/model metadata, the data audit, raw and clean hashes, evaluator source provenance, and the fact that Studio cloud behavior still requires per-sample confirmation.

- [ ] **Step 2: Populate the final matrix only from immutable output artifacts**

The table has local epoch-58 / EI float32 rows and local / EI-legacy columns at cleaned test FP32 threshold `0.5`; the separate local `0.35` result is labelled validation-tuned and is not an EI parity comparison.

- [ ] **Step 3: Verify the complete repository**

Run: `python -m pytest -q`

Run: `python -m compileall -q src scripts`

Run from the repository root: `git diff --check`

Run from the repository root: `git status --short`

Run from the repository root: `git diff --stat`

No commit, push, retraining, loss change, class-weight change, object-weight change, threshold sweep, or checkpoint replacement is allowed.
