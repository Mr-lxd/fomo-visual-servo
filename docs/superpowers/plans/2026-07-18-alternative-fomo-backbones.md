# Alternative FOMO Backbones Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add strictly validated MobileNetV3-Small and SqueezeNet1.1 stride-8 FOMO candidates without changing existing MobileNetV2 behavior.

**Architecture:** Keep the two existing models untouched. Add one torchvision-backed alternative-model module, one strict local-weight loader, minimal config/factory metadata extensions, two protocol-identical YAML files, and a deterministic ONNX/CPU benchmark report path.

**Tech Stack:** Python 3.10, PyTorch 2.5.1, torchvision 0.20.1, pytest, ONNX 1.22, ONNX Runtime 1.23.

---

### Task 1: Write config and registry contract tests

**Files:**
- Modify: `tests/test_config.py`
- Create: `tests/test_alternative_fomo_backbones.py`
- Modify: `src/fomo_servo/config.py`
- Modify: `src/fomo_servo/models/fomo.py`

- [ ] Write tests that load each new YAML identity, construct each random model by name, assert `[2,8,24,24]`, assert unknown backbone failure, and assert legacy MobileNetV2 identities remain unchanged.
- [ ] Run: `conda run -n fomo-servo-train python -m pytest tests/test_alternative_fomo_backbones.py -q`.
  Expected initial failure: import/model factory does not support both new names.
- [ ] Add only the fields and explicit factory routes required by those tests; do not alter old model constructor calls.
- [ ] Re-run the same test target and then `tests/test_mobilenet_v2_fomo.py -q`.

### Task 2: Implement strict torchvision local-weight provenance

**Files:**
- Create: `src/fomo_servo/models/torchvision_pretrained.py`
- Modify: `src/fomo_servo/models/metadata.py`
- Test: `tests/test_alternative_fomo_backbones.py`

- [ ] Write failing tests using local synthetic torchvision-shaped state dicts for successful strict prefix load, SHA mismatch, source-version mismatch, and missing/unexpected key rejection.
- [ ] Run the individual tests and observe each fails before the loader exists.
- [ ] Implement `load_torchvision_backbone_weights()` to hash a local file before `torch.load(..., weights_only=True)`, validate the full official key set, then strict-load the selected feature prefix. Return an immutable report containing URL, enum, torchvision version, SHA, full-key count, prefix-key count, missing keys and unexpected keys.
- [ ] Re-run the focused tests and assert `describe_model()` serializes the report.

### Task 3: Implement MobileNetV3-Small FOMO with test-first shape and gradient checks

**Files:**
- Create: `src/fomo_servo/models/alternative_fomo.py`
- Modify: `src/fomo_servo/models/__init__.py`
- Test: `tests/test_alternative_fomo_backbones.py`

- [ ] Write failing tests for `features.2` stride-8 `[B,24,24,24]`, D2-equivalent `Conv1x1 -> ReLU -> Conv1x1` head, finite CPU loss/backward, CUDA/AMP when available, parameter count and checkpoint round trip.
- [ ] Run only these tests; failure must be caused by the missing implementation.
- [ ] Implement `MobileNetV3SmallFOMONet` around torchvision `mobilenet_v3_small(weights=None).features[:3]`, add the shared logits head, fixed-input validation and the strict pretrained hook.
- [ ] Re-run the focused tests and inspect exported metadata.

### Task 4: Implement SqueezeNet 1.1 FOMO with test-first cut-point checks

**Files:**
- Modify: `src/fomo_servo/models/alternative_fomo.py`
- Test: `tests/test_alternative_fomo_backbones.py`

- [ ] Write failing tests for `squeezenet1_1(...).features[:7]` Fire4 output `[B,256,24,24]`, no interpolation modules, logits shape, finite CPU loss/backward, CUDA/AMP when available, strict pretrained loading and checkpoint/resume compatibility.
- [ ] Run those tests; they must fail because SqueezeNet is not registered yet.
- [ ] Implement `SqueezeNet1_1FOMONet` using only the real feature prefix through Fire4 and the same FOMO head.
- [ ] Re-run focused new-model tests and old MobileNetV2 tests.

### Task 5: Add comparable protocol configs, ONNX parity and CPU benchmark

**Files:**
- Create: `configs/experiments/stage_e_mobilenet_v3_small_fomo_pretrained.yaml`
- Create: `configs/experiments/stage_e_squeezenet1_1_fomo_pretrained.yaml`
- Create: `scripts/benchmark_backbones.py`
- Create: `docs/experiments/stage_e_config_diff.json`
- Create: `docs/experiments/stage_e_config_diff.md`
- Create: `docs/experiments/stage_e_onnx_parity_report.md`
- Create: `docs/experiments/stage_e_cpu_benchmark_report.md`
- Test: `tests/test_alternative_fomo_backbones.py`

- [ ] Write failing tests for static ONNX export/ORT logits parity for D2 and both new models, plus benchmark JSON schema validation.
- [ ] Run the targeted tests and observe no benchmark/export support for alternatives.
- [ ] Implement a benchmark CLI that reads only existing YAML/checkpoints or random models, exports fixed `[1,3,192,192]`, records named model metadata, parameter/state/ONNX sizes, default/single-thread CPU timings and ORT errors. No dataset split is opened.
- [ ] Download official source weights outside the repository, calculate SHA-256, update the two YAMLs and config diffs, then run the ONNX/benchmark CLI with random and pretrained models.
- [ ] Re-run the new tests, all former ONNX tests, and verify no unsupported operators or parity errors.

### Task 6: Verify, commit implementation, then run validation-only screening

**Files:**
- Modify: `docs/experiments/stage_e_alternative_backbone_validation.md`
- Modify: `docs/experiments/stage_e_onnx_parity_report.md`
- Modify: `docs/experiments/stage_e_cpu_benchmark_report.md`

- [ ] Run `conda run -n fomo-servo-train python -m pytest -q`, `conda run -n fomo-servo-train python -m compileall -q src scripts`, and `git -c safe.directory=D:/DL_Project/fomo-visual-servo diff --check`.
- [ ] Explicitly stage only Stage E source/config/test/docs files and create the local commit `feat: add alternative FOMO backbones`; do not stage the pre-existing untracked handoff file, push, create a PR, or access the dataset test split.
- [ ] Train MobileNetV3-Small then SqueezeNet1.1 once each using validation-only selection and record elapsed time, selected epoch/threshold, global/per-class validation metrics and D2 comparison.
- [ ] Run the same final verification commands, explicitly stage only new results docs, and create local `docs: record alternative backbone validation` commit.
