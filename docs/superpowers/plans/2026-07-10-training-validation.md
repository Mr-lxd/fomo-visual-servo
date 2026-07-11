# Training and Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train and validate the existing FOMO model from YAML with weighted CE/focal loss, AdamW, configurable scheduler, F1 checkpointing, resume, history CSV, and CPU smoke coverage.

**Architecture:** Configuration parsing owns all run parameters. `datasets.collate` owns conversion from NumPy samples to tensor batches; `losses` and `metrics` remain stateless; `training.engine` owns train/validation epochs, finite-gradient checks, persistence, and early stopping. The CLI only assembles these public components.

**Tech Stack:** Python 3.10, PyTorch 2.5, NumPy, PyYAML, OpenCV, standard-library CSV/path handling, pytest.

---

### Task 1: Add failing configuration, loss, metric, collate, and persistence tests

**Files:**

- Create: `tests/test_losses.py`
- Create: `tests/test_metrics.py`
- Create: `tests/test_training_engine.py`
- Modify: `tests/test_config.py`
- Modify: `tests/test_train_cli.py`

- [ ] **Step 1: Specify CE and focal expectations**

```python
loss = build_classification_loss(LossConfig("focal_cross_entropy", 0.0, (1.0, 2.0)))
assert torch.allclose(loss(logits, targets), F.cross_entropy(logits, targets, weight=weights))
```

- [ ] **Step 2: Specify multi-class foreground micro metrics**

```python
metrics = foreground_micro_metrics(predictions, targets)
assert metrics.precision == pytest.approx(0.5)
assert metrics.recall == pytest.approx(0.5)
assert metrics.f1 == pytest.approx(0.5)
```

- [ ] **Step 3: Specify 2 epoch CPU smoke artifacts and resume behavior**

```python
summary = run_training(config, device_override="cpu")
assert summary.completed_epochs == 2
assert (output_dir / "last.pt").is_file()
assert (output_dir / "best_val_f1.pt").is_file()
assert len(list(csv.DictReader((output_dir / "history.csv").open()))) == 2
```

- [ ] **Step 4: Run the new tests and verify missing APIs fail clearly**

Run: `conda run -n fomo-servo-train python -m pytest tests/test_losses.py tests/test_metrics.py tests/test_training_engine.py -q`

Expected: FAIL because loss, metrics, collate, and training engine APIs do not exist.

### Task 2: Implement YAML schema, collate, loss, and metric boundaries

**Files:**

- Modify: `src/fomo_servo/config.py`
- Modify: `src/fomo_servo/datasets/__init__.py`
- Create: `src/fomo_servo/datasets/collate.py`
- Create: `src/fomo_servo/losses/classification.py`
- Modify: `src/fomo_servo/losses/__init__.py`
- Create: `src/fomo_servo/metrics/classification.py`
- Modify: `src/fomo_servo/metrics/__init__.py`

- [ ] **Step 1: Parse immutable YAML loss/optimizer/scheduler/training fields and validate `1+N` class weights**

```python
LossConfig(name="focal_cross_entropy", gamma=2.0, class_weights=(1.0, 4.0))
OptimizerConfig(name="adamw", learning_rate=1e-3, weight_decay=1e-4)
SchedulerConfig(name="step_lr", step_size=10, gamma=0.5)
```

- [ ] **Step 2: Collate samples into FOMO tensors**

```python
FOMOBatch(
    images=torch.from_numpy(np.stack(sample.image for sample in samples)),
    targets=torch.from_numpy(np.stack(sample.heatmap.class_index for sample in samples)),
)
```

- [ ] **Step 3: Implement CE/focal and foreground micro metrics**

```python
probability_of_target = logits.softmax(dim=1).gather(1, targets.unsqueeze(1)).squeeze(1)
loss = (1.0 - probability_of_target).pow(gamma) * weighted_cross_entropy_per_cell
```

- [ ] **Step 4: Run focused tests**

Run: `conda run -n fomo-servo-train python -m pytest tests/test_losses.py tests/test_metrics.py tests/test_config.py -q`

Expected: PASS.

### Task 3: Implement deterministic engine, checkpointing, early stopping, and resume

**Files:**

- Create: `src/fomo_servo/training/engine.py`
- Modify: `src/fomo_servo/training/__init__.py`
- Modify: `tests/test_training_engine.py`

- [ ] **Step 1: Build seeded datasets/loaders and validate class-name contract**

```python
train_dataset = YOLOv5FOMODataset(..., split=config.dataset.train_split, ...)
assert train_dataset.class_names == config.dataset.class_names
train_loader = DataLoader(train_dataset, shuffle=True, collate_fn=collate_fomo_samples, generator=generator, **runtime.data_loader_kwargs)
```

- [ ] **Step 2: Implement train/validation epoch functions and finite-gradient guard**

```python
scaler.scale(loss).backward()
scaler.unscale_(optimizer)
ensure_finite_gradients(model)
scaler.step(optimizer)
scaler.update()
```

- [ ] **Step 3: Persist and restore full checkpoint state, history CSV, and early stopping state**

```python
torch.save(checkpoint, output_dir / "last.pt")
if improved:
    torch.save(checkpoint, output_dir / "best_val_f1.pt")
```

- [ ] **Step 4: Run CPU smoke and resume test**

Run: `conda run -n fomo-servo-train python -m pytest tests/test_training_engine.py -q`

Expected: PASS and use only temporary fixture output directories.

### Task 4: Replace the preflight CLI and update project configuration/documentation

**Files:**

- Modify: `scripts/train.py`
- Modify: `configs/aquarium_creature_192.yaml`
- Modify: `README.md`
- Modify: `tests/test_train_cli.py`

- [ ] **Step 1: Parse `--config`, `--device`, and `--resume`; pass only overrides to `run_training`**

```python
summary = run_training(config, device_override=args.device, resume_override=args.resume)
print(f"Best validation F1: {summary.best_val_f1:.6f}")
```

- [ ] **Step 2: Document all YAML run parameters and CPU smoke command**

```powershell
python scripts/train.py --config configs/aquarium_creature_192.yaml --device cpu
```

### Task 5: Full verification

**Files:**

- Verify: all production/test/config/document files above.

- [ ] **Step 1: Run all pytest tests without bytecode writes**

Run: `conda run -n fomo-servo-train cmd /c "set PYTHONDONTWRITEBYTECODE=1&&python -m pytest -q"`

Expected: all CPU tests pass; CUDA-only and unavailable ONNX tests report explicit skips.

- [ ] **Step 2: Run a direct 2 epoch CPU CLI smoke against the fixture-created YAML**

Run: the test-created command through `tests/test_train_cli.py` or a temporary config path.

Expected: `last.pt`, `best_val_f1.pt`, and `history.csv` are created under the configured temporary output directory.

**Note:** This directory is not a Git checkout, so normal plan commit steps do not apply.
