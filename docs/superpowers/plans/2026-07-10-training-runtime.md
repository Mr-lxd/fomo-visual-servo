# Training Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make device selection, AMP, DataLoader transfer settings, and the `train.py` configuration preflight usable from YAML and a `--device` CLI override.

**Architecture:** `fomo_servo.config` owns YAML parsing and validation. `fomo_servo.training.runtime` owns model placement, batch placement, AMP context selection, and effective DataLoader settings. `scripts.train` is a transparent preflight CLI that consumes only the public configuration/model/training APIs; it does not implement a loss or epoch loop.

**Tech Stack:** Python 3.10, PyTorch 2.5, PyYAML, argparse, pytest.

---

### Task 1: Specify configuration and training runtime behavior with failing tests

**Files:**

- Create: `tests/test_training_runtime.py`
- Modify: `tests/test_config.py`
- Modify: `tests/test_imports.py`

- [ ] **Step 1: Write failing YAML tests for the four training fields and `TRAIN` alias**

```python
config = load_config(config_path)
assert config.training.device == "auto"
assert config.training.amp is True
assert config.training.num_workers == 4
assert config.training.pin_memory is True

with pytest.raises(ConfigurationError, match="not both"):
    load_config(config_with_training_and_train)
```

- [ ] **Step 2: Write failing runtime tests for device placement, DataLoader arguments, and CPU AMP diagnostics**

```python
runtime = create_training_runtime(TrainingConfig("cpu", True, 4, True))
model = prepare_model(FOMONet(num_classes=1, input_size=96), runtime)
images, targets = move_training_batch(images, targets, runtime)
assert model is not None and images.device.type == targets.device.type == "cpu"
assert runtime.amp_enabled is False
assert runtime.data_loader_kwargs == {"num_workers": 4, "pin_memory": False}
```

- [ ] **Step 3: Run focused tests to verify the intended public APIs fail as missing**

Run: `conda run -n fomo-servo-train python -m pytest tests/test_training_runtime.py tests/test_config.py -q`

Expected: FAIL because AMP/loader config fields and `fomo_servo.training` runtime APIs do not exist.

### Task 2: Implement validated YAML fields and device-aware runtime helpers

**Files:**

- Modify: `src/fomo_servo/config.py`
- Create: `src/fomo_servo/training/__init__.py`
- Create: `src/fomo_servo/training/runtime.py`

- [ ] **Step 1: Add immutable training configuration fields**

```python
@dataclass(frozen=True)
class TrainingConfig:
    device: str
    amp: bool
    num_workers: int
    pin_memory: bool
```

- [ ] **Step 2: Implement runtime construction and model/batch movement**

```python
runtime = create_training_runtime(config.training, device_override)
model = model.to(runtime.device)
images = images.to(runtime.device, non_blocking=True)
targets = targets.to(runtime.device, non_blocking=True)
```

- [ ] **Step 3: Implement the CUDA-only autocast context**

```python
if runtime.amp_enabled:
    return torch.autocast(device_type="cuda", dtype=torch.float16)
return nullcontext()
```

- [ ] **Step 4: Run focused tests and keep CUDA-only assertions conditional**

Run: `conda run -n fomo-servo-train python -m pytest tests/test_training_runtime.py tests/test_config.py -q`

Expected: CPU tests PASS; CUDA-specific test runs if CUDA is available, otherwise skips.

### Task 3: Implement transparent `train.py` configuration preflight

**Files:**

- Modify: `scripts/train.py`
- Modify: `tests/test_scripts.py`
- Create: `tests/test_train_cli.py`

- [ ] **Step 1: Write failing CLI tests for YAML device resolution and `--device cpu` override**

```python
completed = subprocess.run(
    [sys.executable, "scripts/train.py", "--config", str(config_path), "--device", "cpu"],
    check=False,
    capture_output=True,
    text=True,
)
assert completed.returncode == 0
assert "Device: cpu" in completed.stdout
assert "Training loop is not implemented" in completed.stdout
```

- [ ] **Step 2: Implement argparse, configuration loading, model placement, and clear preflight output**

```python
parser.add_argument("--config", type=Path, required=True)
parser.add_argument("--device", default=None)
config = load_config(arguments.config)
runtime = create_training_runtime(config.training, arguments.device)
model = prepare_model(build_fomo_model(config), runtime)
```

- [ ] **Step 3: Run CLI tests**

Run: `conda run -n fomo-servo-train python -m pytest tests/test_train_cli.py tests/test_scripts.py -q`

Expected: PASS without requiring a real dataset or writing checkpoints.

### Task 4: Update shipped configuration and user documentation

**Files:**

- Modify: `configs/aquarium_creature_192.yaml`
- Modify: `README.md`

- [ ] **Step 1: Add explicit `amp`, `num_workers`, and `pin_memory` fields to the shipped YAML**

```yaml
training:
  device: auto
  amp: true
  num_workers: 4
  pin_memory: true
```

- [ ] **Step 2: Document CLI precedence and the fact that current command is a preflight rather than a training loop**

```powershell
python scripts/train.py --config configs/aquarium_creature_192.yaml --device cuda
python scripts/train.py --config configs/aquarium_creature_192.yaml --device cpu
```

### Task 5: Full verification

**Files:**

- Verify: all files listed above.

- [ ] **Step 1: Run all tests with bytecode writes disabled**

Run: `conda run -n fomo-servo-train cmd /c "set PYTHONDONTWRITEBYTECODE=1&&python -m pytest -q"`

Expected: all available CPU/CUDA tests pass; existing ONNX tests still skip until their optional dependencies are installed.

- [ ] **Step 2: Run both CLI override commands against the shipped configuration**

Run: `conda run -n fomo-servo-train python scripts/train.py --config configs/aquarium_creature_192.yaml --device cpu`

Expected: exit 0 and clearly report CPU, AMP disabled, effective DataLoader settings, and no training loop execution.

**Note:** This directory is not a Git checkout, so normal plan commit steps do not apply.
