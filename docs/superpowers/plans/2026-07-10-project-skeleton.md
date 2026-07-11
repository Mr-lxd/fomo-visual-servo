# Project Skeleton Implementation Plan

> **For agentic workers:** Execute test-first and keep this task to configuration, package boundaries, command stubs, and environment checks. Do not add models or dataset loaders.

**Goal:** Create an installable `src`-layout PyTorch FOMO project skeleton with YAML configuration validation, command placeholders, an environment check, and pytest coverage.

**Architecture:** `fomo_servo.config` is the only implemented project module: it reads a small YAML schema and derives the `1 + N` channel contract. Empty feature packages establish the future module boundaries. The environment script imports the package and validates the requested configuration without importing model code.

**Tech Stack:** Python 3.10–3.11, setuptools, PyYAML, pytest.

---

### Task 1: Define packaging and the red tests

**Files:**

- Create: `pyproject.toml`
- Create: `requirements-dev.txt`
- Create: `tests/test_config.py`
- Create: `tests/test_imports.py`
- Create: `tests/test_environment_check.py`

- [ ] Add packaging metadata, a `src` package discovery rule, a `pytest` test path, PyYAML runtime dependency, and pytest development extra.
- [ ] Add tests that expect a valid YAML config to load, a missing required field to raise `ConfigurationError`, all skeleton packages to import, and the environment-check script to exit successfully with the repository configuration.
- [ ] Run `python -m pytest -q`; expect collection/import failures because `fomo_servo` and the environment script do not exist yet.

### Task 2: Implement the smallest configuration contract

**Files:**

- Create: `src/fomo_servo/__init__.py`
- Create: `src/fomo_servo/config.py`
- Create: `configs/aquarium_creature_192.yaml`

- [ ] Implement `ConfigurationError`, immutable config data objects, and `load_config(path)` using `yaml.safe_load`.
- [ ] Validate a non-empty relative dataset root, a non-empty unique class list, positive integer input size/stride, and `input_size % output_stride == 0`.
- [ ] Expose derived `output_channels = 1 + len(class_names)` and `grid_size = input_size // output_stride`.
- [ ] Run the focused configuration tests; expect them to pass.

### Task 3: Establish package and command boundaries

**Files:**

- Create: `src/fomo_servo/{datasets,models,losses,metrics,postprocess,inference,export}/__init__.py`
- Create: `scripts/__init__.py`
- Create: `scripts/check_environment.py`
- Create: `scripts/{train,evaluate,predict_image,predict_video,export_onnx,benchmark}.py`

- [ ] Keep feature packages empty apart from a descriptive module docstring.
- [ ] Make each future command exit explicitly with a clear “not implemented” message rather than silently succeeding.
- [ ] Make `check_environment.py` verify the Python version, import `fomo_servo`, load the supplied YAML configuration, and report whether PyTorch is installed; it must run without importing a model or dataset loader.
- [ ] Run the environment-check test and the script directly; expect a zero exit status for the repository configuration.

### Task 4: Full verification

**Files:**

- Verify: `pyproject.toml`, `requirements-dev.txt`, `src/`, `tests/`, `configs/`, `scripts/`, and `docs/`

- [ ] Run `python -m pytest -q` from the repository root.
- [ ] Run `python scripts/check_environment.py --config configs/aquarium_creature_192.yaml`.
- [ ] Confirm no real model or dataset-loader module was added and report the test count and any remaining limitations.
