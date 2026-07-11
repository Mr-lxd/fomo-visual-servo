# PyTorch FOMO Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a CPU/CUDA portable MobileNetV2-lite FOMO model that maps fixed square RGB inputs to stride-8 raw class logits and can be exported to ONNX.

**Architecture:** `fomo_servo.models.fomo` will own shape validation, MobileNetV2-style inverted residual blocks, and the `1×1 → ReLU6 → 1×1` FOMO head. `fomo_servo.config` will explicitly validate the model and device YAML fields. `fomo_servo.runtime` will own device resolution so model code never selects a device. Tests consume only the public API and conditionally exercise CUDA and ONNX Runtime when those packages/devices exist.

**Tech Stack:** Python 3.10, PyTorch 2.5, pytest, ONNX exporter supplied by PyTorch; `onnx` is used only by the exporter and `onnxruntime` only by runtime validation.

---

### Task 1: Define externally observable model and runtime tests

**Files:**

- Create: `tests/test_fomo_model.py`
- Create: `tests/test_runtime.py`
- Modify: `tests/test_config.py`
- Modify: `tests/test_imports.py`

- [ ] **Step 1: Write the failing model contract tests**

```python
@pytest.mark.parametrize(("input_size", "num_classes"), [(96, 1), (192, 1), (224, 7)])
def test_fomo_model_returns_stride_eight_logits(input_size: int, num_classes: int) -> None:
    model = FOMONet(num_classes=num_classes, width_multiplier=0.35).eval()
    logits = model(torch.randn(2, 3, input_size, input_size))
    assert logits.shape == (2, num_classes + 1, input_size // 8, input_size // 8)
    assert logits.dtype == torch.float32
```

- [ ] **Step 2: Add failing device, backward, invalid-input, and parameter-count tests**

```python
def test_fomo_model_cpu_backward_produces_parameter_gradients() -> None:
    model = FOMONet(num_classes=1, width_multiplier=0.35)
    loss = model(torch.randn(1, 3, 96, 96)).square().mean()
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
```

- [ ] **Step 3: Add a failing YAML-field test for model and device configuration**

```python
def test_load_config_reads_model_and_training_runtime_fields(tmp_path: Path) -> None:
    config = load_config(_write_config(tmp_path, width_multiplier=0.35, head_channels=32, device="auto"))
    assert config.model.backbone == "mobilenet_v2_lite"
    assert config.model.width_multiplier == pytest.approx(0.35)
    assert config.model.head_channels == 32
    assert config.training.device == "auto"
```

- [ ] **Step 4: Run focused tests to verify expected missing-public-API failures**

Run: `conda run -n fomo-servo-train python -m pytest tests/test_fomo_model.py tests/test_runtime.py -q`

Expected: FAIL because `FOMONet` and `resolve_device` do not exist.

### Task 2: Implement the model and device resolver

**Files:**

- Create: `src/fomo_servo/models/fomo.py`
- Create: `src/fomo_servo/runtime.py`
- Modify: `src/fomo_servo/models/__init__.py`
- Modify: `src/fomo_servo/config.py`

- [ ] **Step 1: Implement pure validation and rounded-channel helpers**

```python
def _require_positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value

def _make_divisible(value: float, divisor: int = 8) -> int:
    return max(divisor, int(value + divisor / 2) // divisor * divisor)
```

- [ ] **Step 2: Implement Conv-BN-ReLU6 and depthwise inverted residual components**

```python
class InvertedResidual(nn.Module):
    def forward(self, images: Tensor) -> Tensor:
        output = self.block(images)
        return images + output if self.use_residual else output
```

- [ ] **Step 3: Assemble a `/8` MobileNetV2-lite encoder and raw-logit head**

```python
class FOMONet(nn.Module):
    def forward(self, images: Tensor) -> Tensor:
        _validate_images(images)
        features = self.backbone(images)
        return self.head(features)
```

- [ ] **Step 4: Implement device resolution outside the model**

```python
def resolve_device(requested: str | torch.device = "auto") -> torch.device:
    if str(requested).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)
```

- [ ] **Step 5: Extend the validated configuration data structures without changing the existing minimal config contract**

```python
@dataclass(frozen=True)
class ModelConfig:
    input_size: int
    output_stride: int
    backbone: str
    width_multiplier: float
    head_channels: int

@dataclass(frozen=True)
class TrainingConfig:
    device: str
```

`backbone`, `width_multiplier`, `head_channels`, and `training.device` use validated defaults only for legacy minimal YAML; the shipped project YAML supplies all four explicitly.

- [ ] **Step 6: Run focused tests and the existing suite**

Run: `conda run -n fomo-servo-train python -m pytest tests/test_fomo_model.py tests/test_runtime.py -q`

Expected: PASS on CPU, with CUDA-only cases skipped if unavailable.

### Task 3: Add ONNX export and ONNX Runtime conformance tests

**Files:**

- Modify: `tests/test_fomo_model.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Add failing ONNX smoke and runtime-equivalence tests**

```python
def test_onnxruntime_cpu_matches_pytorch_logits(tmp_path: Path) -> None:
    onnxruntime = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    model = FOMONet(num_classes=1, width_multiplier=0.35).eval()
    images = torch.randn(1, 3, 160, 160)
    torch.onnx.export(model, images, tmp_path / "fomo.onnx", opset_version=17)
    session = onnxruntime.InferenceSession(str(tmp_path / "fomo.onnx"), providers=["CPUExecutionProvider"])
    onnx_logits = session.run(None, {session.get_inputs()[0].name: images.numpy()})[0]
    np.testing.assert_allclose(onnx_logits, model(images).detach().numpy(), rtol=1e-4, atol=1e-5)
```

- [ ] **Step 2: Declare the exporter dependency without adding a CUDA-specific package**

```toml
export = [
    "onnx>=1.16,<2.0",
]
deployment = [
    "onnxruntime>=1.17,<2.0",
    "opencv-python-headless>=4.8,<5.0",
]
```

- [ ] **Step 3: Run tests and retain dependency-gated skips when the current environment lacks ONNX packages**

Run: `conda run -n fomo-servo-train python -m pytest tests/test_fomo_model.py -q`

Expected: CPU model tests pass; exporter and ONNX Runtime tests skip until `python -m pip install -e ".[export,deployment]"` has been run by the user.

### Task 4: Full verification and documentation review

**Files:**

- Verify: `src/fomo_servo/models/fomo.py`
- Verify: `src/fomo_servo/runtime.py`
- Verify: `tests/test_fomo_model.py`
- Verify: `tests/test_runtime.py`
- Verify: `pyproject.toml`

- [ ] **Step 1: Run all tests without generating bytecode artifacts**

Run: `conda run -n fomo-servo-train cmd /c "set PYTHONDONTWRITEBYTECODE=1&&python -m pytest -q"`

Expected: all CPU tests pass; CUDA and ONNX Runtime tests report explicit skips only when their prerequisites are unavailable.

- [ ] **Step 2: Run a fresh environment report**

Run: `conda run -n fomo-servo-train python scripts/check_env.py --profile training`

Expected: Python 3.10 and torch are reported; CUDA status reflects the active machine; a missing ONNX Runtime is explicitly reported.

- [ ] **Step 3: Check scope**

Confirm model forward returns logits only, contains no device selection, no loss/training loop, no dataset changes, no custom CUDA operators, and no generated ONNX artifact is committed.

**Note:** This directory is not a Git checkout, so plan steps that normally create commits intentionally do not apply.
