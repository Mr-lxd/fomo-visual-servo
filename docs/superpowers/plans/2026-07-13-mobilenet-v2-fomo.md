# MobileNetV2 FOMO Backbone Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an internally implemented standard MobileNetV2 FOMO backbone cut at `block_6_expand_relu`, preserve the existing lite model, and complete a two-epoch CUDA Stage A smoke without formal training or commit.

**Architecture:** Keep `FOMONet` and its state dict intact. Add a separate standard MobileNetV2 trunk that completes blocks 0–5 and executes only block 6's expansion Conv–BN–ReLU6, then attach the fixed `96→32→1+N` FOMO head. Extend configuration and training artifacts with model identity metadata while retaining legacy defaults.

**Tech Stack:** Python 3.10, PyTorch 2.5.1 CUDA 12.1 build, PyYAML, pytest, optional ONNX/ONNX Runtime.

---

### Task 1: Lock configuration and compatibility contracts in failing tests

**Files:**
- Create: `tests/test_mobilenet_v2_fomo.py`
- Modify: `tests/test_config.py`

- [ ] **Step 1: Add failing model-config tests**

Add tests that load a minimal new-backbone YAML and assert:

```python
assert config.model.backbone == "mobilenet_v2_fomo"
assert config.model.cut_point == "block_6_expand_relu"
assert config.model.pretrained is False
```

Add a legacy-config assertion:

```python
assert legacy.model.backbone == "mobilenet_v2_lite"
assert legacy.model.cut_point == "lite_stride8_output"
assert legacy.model.pretrained is False
```

Add invalid tests for a non-boolean `pretrained` value and a new-backbone cut
point other than `block_6_expand_relu`.

- [ ] **Step 2: Add failing public-model API tests**

In `tests/test_mobilenet_v2_fomo.py`, import the wished-for API:

```python
from fomo_servo.models import (
    MobileNetV2FOMOBackbone,
    MobileNetV2FOMONet,
    build_fomo_model,
    describe_model,
)
```

The initial tests assert the new classes are importable and that the old
`FOMONet` still builds for `mobilenet_v2_lite`.

- [ ] **Step 3: Verify the red state**

Run:

```powershell
conda run --no-capture-output -n fomo-servo-train `
  python -m pytest -q tests/test_config.py tests/test_mobilenet_v2_fomo.py
```

Expected: collection or assertion failure because the new fields and model API
do not exist. Confirm the failure is not caused by a typo.

### Task 2: Extend the model configuration schema without changing old YAML

**Files:**
- Modify: `src/fomo_servo/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Add model identity fields**

Extend `ModelConfig` with:

```python
cut_point: str = "lite_stride8_output"
pretrained: bool = False
```

Parse `model.cut_point` with a backbone-dependent default:

```python
default_cut_point = (
    "block_6_expand_relu"
    if backbone == "mobilenet_v2_fomo"
    else "lite_stride8_output"
)
cut_point = _optional_text(model_mapping, "cut_point", default_cut_point, "model")
pretrained = _optional_boolean(model_mapping, "pretrained", False, "model")
```

For `mobilenet_v2_fomo`, reject any other cut point with a
`ConfigurationError`. Pass both fields into `ModelConfig(...)`.

- [ ] **Step 2: Run config tests to green**

Run:

```powershell
conda run --no-capture-output -n fomo-servo-train `
  python -m pytest -q tests/test_config.py
```

Expected: all config tests pass, and old YAML requires no edits.

### Task 3: Test the exact standard block mapping and cut point

**Files:**
- Modify: `tests/test_mobilenet_v2_fomo.py`

- [ ] **Step 1: Add explicit block-spec assertions**

Assert the exported block specifications resolve to:

```python
expected = (
    (0, 1, 16, 1),
    (1, 6, 24, 2),
    (2, 6, 24, 1),
    (3, 6, 32, 2),
    (4, 6, 32, 1),
    (5, 6, 32, 1),
    (6, 6, 64, 2),
)
```

The tuple fields are `(block_id, expansion, base_output_channels, stride)`.

- [ ] **Step 2: Add cut-point structural assertions**

Construct `MobileNetV2FOMOBackbone(width_multiplier=0.35)` and assert:

```python
assert backbone.cut_point == "block_6_expand_relu"
assert backbone.output_stride == 8
assert backbone.output_channels == 96
assert isinstance(backbone.block_6_expansion[0], torch.nn.Conv2d)
assert backbone.block_6_expansion[0].kernel_size == (1, 1)
assert backbone.block_6_expansion[0].stride == (1, 1)
assert backbone.block_6_expansion[0].in_channels == 16
assert backbone.block_6_expansion[0].out_channels == 96
assert isinstance(backbone.block_6_expansion[1], torch.nn.BatchNorm2d)
assert isinstance(backbone.block_6_expansion[2], torch.nn.ReLU6)
```

Assert the backbone has no block-6 depthwise/projection module and that a
192×192 tensor produces `[1,96,24,24]`.

- [ ] **Step 3: Verify the new tests fail**

Run the focused test file and confirm failure because the standard backbone is
not implemented.

### Task 4: Implement the internal standard MobileNetV2 FOMO trunk

**Files:**
- Create: `src/fomo_servo/models/mobilenet_v2_fomo.py`
- Modify: `src/fomo_servo/models/__init__.py`
- Test: `tests/test_mobilenet_v2_fomo.py`

- [ ] **Step 1: Define immutable standard block specs**

Create a frozen `MobileNetV2BlockSpec` dataclass and the exact block 0–6 tuple
from Task 3. Channel projections use MobileNetV2 `make_divisible(value, 8)`.

- [ ] **Step 2: Implement ordinary ONNX-safe primitives**

Implement private Conv–BN–ReLU6 and standard inverted residual modules using
only `Conv2d`, `BatchNorm2d`, `ReLU6`, and residual addition. Complete only
blocks 0–5.

- [ ] **Step 3: Implement the explicit block-6 expansion cut**

Build:

```python
self.block_6_expansion = nn.Sequential(
    nn.Conv2d(16, 96, kernel_size=1, stride=1, bias=False),
    nn.BatchNorm2d(96),
    nn.ReLU6(inplace=False),
)
```

Derive 16 and 96 programmatically from the prior standard block output and
block-6 expansion factor. Do not instantiate or execute block 6's stride-2
depthwise convolution.

- [ ] **Step 4: Implement `MobileNetV2FOMONet`**

Validate `num_classes`, `input_size`, `width_multiplier`, `head_channels`,
`output_stride`, `cut_point`, and `pretrained`. Reject `pretrained=true` with a
message stating that no download or pretrained source is configured.

Attach exactly:

```python
self.head = nn.Sequential(
    nn.Conv2d(96, head_channels, kernel_size=1, bias=True),
    nn.ReLU6(inplace=False),
    nn.Conv2d(head_channels, num_classes + 1, kernel_size=1, bias=True),
)
```

Do not add an initialization pass: retain the initialization performed by the
stock PyTorch module constructors, including head biases. Forward returns raw
logits and verifies spatial shape `input_size // 8`.

- [ ] **Step 5: Export public APIs and run focused tests**

Export the backbone, network, block specs, and cut-point constant from
`models/__init__.py`. Run the focused tests and reach green for block mapping,
stride, channels, and shape.

### Task 5: Dispatch the factory while preserving the old state dict

**Files:**
- Modify: `src/fomo_servo/models/fomo.py`
- Modify: `tests/test_mobilenet_v2_fomo.py`

- [ ] **Step 1: Add failing factory and legacy checkpoint tests**

Test that `build_fomo_model` returns `FOMONet` for old config and
`MobileNetV2FOMONet` for new config. Save a synthetic old payload:

```python
torch.save({"model_state": old_model.state_dict()}, checkpoint_path)
reloaded = build_fomo_model(old_config)
reloaded.load_state_dict(torch.load(checkpoint_path, weights_only=False)["model_state"])
```

Assert every old state tensor is equal. Add an unknown-backbone diagnostic
test.

- [ ] **Step 2: Add minimal factory dispatch**

Keep the old branch unchanged, add a local import and new branch for
`mobilenet_v2_fomo`, and make the unknown error list both supported names.
Pass all six model fields to the new constructor.

- [ ] **Step 3: Run old and new model tests**

Run:

```powershell
conda run --no-capture-output -n fomo-servo-train `
  python -m pytest -q tests/test_fomo_model.py tests/test_mobilenet_v2_fomo.py
```

Expected: old model tests remain green and factory tests pass.

### Task 6: Complete numerical, device, serialization, and no-download tests

**Files:**
- Modify: `tests/test_mobilenet_v2_fomo.py`

- [ ] **Step 1: Add raw-logit and finite backward tests**

Use random float32 `[0,1]` input. Assert output is finite, output channel sums
are not constrained to one, cross-entropy backward completes, and every
present gradient is finite.

- [ ] **Step 2: Add CPU/CUDA tests**

Always run CPU forward/backward. When CUDA exists, compare CPU/CUDA shapes,
run CUDA backward, and run one CUDA float16 autocast smoke. Use skip markers
when CUDA is absent.

- [ ] **Step 3: Add state-dict round-trip**

Seed, construct, save, reconstruct, load, switch both to eval, and assert
identical logits for a fixed CPU tensor.

- [ ] **Step 4: Prove no download path**

Monkeypatch `torch.hub.load_state_dict_from_url` to raise if called, construct
with `pretrained=false`, and verify success. Separately assert
`pretrained=true` raises `ModelConfigurationError` before forward.

- [ ] **Step 5: Add optional ONNX tests**

Use `pytest.importorskip("onnx")` and fixed `[1,3,192,192]` input. Export opset
17, validate only standard ONNX domains, and, when `onnxruntime` exists,
compare CPU logits with `rtol=1e-4, atol=1e-5`.

### Task 7: Add stable model identity metadata to artifacts

**Files:**
- Create: `src/fomo_servo/models/metadata.py`
- Modify: `src/fomo_servo/models/__init__.py`
- Modify: `src/fomo_servo/training/engine.py`
- Modify: `tests/test_mobilenet_v2_fomo.py`
- Modify: `tests/test_training_engine.py`

- [ ] **Step 1: Add failing metadata tests**

Assert `describe_model(config, model)` returns:

```python
{
    "backbone_name": "mobilenet_v2_fomo",
    "width_multiplier": 0.35,
    "cut_point": "block_6_expand_relu",
    "cut_point_input_channels": 16,
    "cut_point_output_channels": 96,
    "head_channels": 32,
    "pretrained": False,
    "initialization": "pytorch_module_defaults",
    "backbone_parameter_count": 15840,
    "head_parameter_count": 3368,
    "parameter_count": 19208,
}
```

Also assert stable old-model counts and identity. Extend checkpoint tests to
expect a `model_metadata` mapping.

- [ ] **Step 2: Implement the pure metadata helper**

Count trainable parameters separately from `model.backbone`, `model.head`, and
the full model. Use config identity plus `model.backbone.output_channels`.

- [ ] **Step 3: Thread metadata through training artifacts**

Compute the mapping once after model creation. Add it to checkpoint arguments,
every checkpoint payload, `TrainingSummary`, `training_summary.json`, and
`experiment_metadata.json`. Do not make it a required key when restoring old
checkpoints.

- [ ] **Step 4: Run focused metadata and engine tests**

Expected: new metadata tests pass and all existing resume/checkpoint tests stay
green.

### Task 8: Create and lock the formal single-variable config

**Files:**
- Create: `configs/experiments/model01_mobilenet_v2_fomo_aug03.yaml`
- Modify: `tests/test_mobilenet_v2_fomo.py`

- [ ] **Step 1: Copy aug03 values and change only allowed fields**

Set project/experiment name to `model01_mobilenet_v2_fomo_aug03`, output to
`outputs/experiments/model01_mobilenet_v2_fomo_aug03`, and set the model fields
shown in the design. Keep all other YAML values byte-equivalent where possible.

- [ ] **Step 2: Add resolved comparison test**

Load both configs with a fixture dataset root. Convert to dictionaries, remove
only `source_path`, the complete `model` mapping, `experiment.name`, and
`training.output_dir`, then assert equality. Separately assert the only model
differences are backbone, cut point, and pretrained; width, head, input, and
stride remain equal.

- [ ] **Step 3: Run the config comparison test**

Expected: no drift in augmentation, data, loss, optimizer, scheduler, seed,
epochs, thresholds, matching, or evaluator.

### Task 9: Run complete Stage A quality gates

**Files:** none

- [ ] **Step 1: Run complete pytest**

```powershell
conda run --no-capture-output -n fomo-servo-train `
  python -m pytest -q
```

Expected: all non-optional tests pass; ONNX and ONNX Runtime remain explicit
skips if dependencies are absent.

- [ ] **Step 2: Compile and check whitespace**

```powershell
conda run --no-capture-output -n fomo-servo-train `
  python -m compileall src scripts

git -c safe.directory=D:/DL_Project/fomo-visual-servo diff --check
```

Expected: both commands succeed.

### Task 10: Measure both model complexities on identical inputs

**Files:**
- Create temporarily: `scripts/_tmp_compare_mobilenet_models.py`
- Delete after measurement: `scripts/_tmp_compare_mobilenet_models.py`

- [ ] **Step 1: Implement the temporary diagnostic**

Build both models for seven classes and 192 input. Count Conv2d MACs with
forward hooks using actual output shapes. Report FLOPs as `2×MACs`. Serialize a
weights-only payload for comparable checkpoint bytes.

Measure batch-1 CPU FP32 and CUDA FP32 latency with 50 warmup and 200 timed
iterations. Synchronize CUDA before and after timing. Reset and read CUDA peak
allocated memory separately for each model.

- [ ] **Step 2: Run the diagnostic**

```powershell
conda run --no-capture-output -n fomo-servo-train `
  python scripts/_tmp_compare_mobilenet_models.py
```

Expected architecture values include old 29,144 parameters/24 channels and new
19,208 parameters/96 channels. Save only ignored output JSON if needed; do not
stage it.

- [ ] **Step 3: Delete the temporary diagnostic with `apply_patch`**

Confirm Git status contains no temporary script.

### Task 11: Run the controlled two-epoch CUDA smoke

**Files:**
- Create temporarily: `configs/experiments/_tmp_model01_smoke.yaml`
- Delete after smoke: `configs/experiments/_tmp_model01_smoke.yaml`

- [ ] **Step 1: Create a smoke-only config**

Copy the formal model01 config, change only epochs to 2 and output directory to
`outputs/experiments/model01_mobilenet_v2_fomo_aug03_smoke`.

- [ ] **Step 2: Run CUDA training**

In the same PowerShell process, set the temporary dataset environment variable
and run:

```powershell
$env:FOMO_DATASET_ROOT="<DATASET_ROOT>"
conda run --no-capture-output -n fomo-servo-train `
  python -u scripts/train.py `
  --config configs/experiments/_tmp_model01_smoke.yaml `
  --device cuda
```

Check finite train/validation loss, CUDA AMP, two completed epochs,
augmentation stats, model metadata, checkpoint reload, next resume epoch=3,
and stable peak memory.

- [ ] **Step 3: Delete the smoke config and recheck status**

Keep ignored outputs for evidence, delete the temporary YAML, and run Git
status plus `diff --check`.

### Task 12: Stage A pause and report

**Files:** none

- [ ] **Step 1: Report without committing**

Report modified files, commands, tests, unresolved optional dependencies, exact
cut mapping, 96 channels, parameter/MAC/latency/memory comparison, and smoke
results. Do not run 60 epochs, do not commit, and do not push.

- [ ] **Step 2: Wait for explicit Stage B approval**

No further action is authorized until the user approves.

### Task 13: Stage B after explicit approval only

**Files:** none before approval

- [ ] **Step 1: Re-run quality checks and commit**

After approval, verify the branch and scope, rerun pytest/compileall/diff check,
stage only model/config/test/docs changes, and commit:

```powershell
git -c safe.directory=D:/DL_Project/fomo-visual-servo `
  commit -m "feat: add MobileNetV2 FOMO backbone"
```

- [ ] **Step 2: Enforce the formal-training gate**

Require a clean worktree, correct dataset content hash, exact model metadata,
locked non-model config comparison, completed Stage A smoke, and CUDA
availability.

- [ ] **Step 3: Run exactly one formal 60-epoch training**

Use the formal model01 config and CUDA. Do not enable pretrained weights or
change object/class weights.

- [ ] **Step 4: Evaluate six checkpoints with the same FP32 evaluator**

Compare aug03 lite and model01 FOMO `best_centroid_f1.pt`,
`best_grid_f1.pt`, and `last.pt`. Write
`outputs/experiments/model01_mobilenet_v2_fomo_comparison.csv` with fixed and
sweep P/R/F1, thresholds, micro/macro/per-class F1, grid F1, localization,
count metrics, epochs, time, complexity, and latency.

- [ ] **Step 5: Stop**

Report the single-variable result. Do not start another model experiment and
do not push.
