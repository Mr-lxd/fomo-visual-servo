# Environment Configuration Implementation Plan

> **For agentic workers:** Execute test-first for the new environment-check script. Do not create conda environments, install dependencies, or alter external YOLO projects.

**Goal:** Describe isolated Python 3.10 training and deployment environments inside the repository and verify their prerequisites without changing the host machine.

**Architecture:** `pyproject.toml` provides separate optional groups for PyTorch training and ONNX Runtime deployment. `environment.yml` creates only a named Python 3.10 training environment when the user later invokes conda. `scripts/check_env.py` reports all requested packages while applying profile-specific pass/fail criteria.

**Tech Stack:** Conda environment YAML, Python 3.10, PyTorch, ONNX Runtime, OpenCV, PyYAML, pytest.

---

### Task 1: Define the failing environment-check contract

**Files:**

- Modify: `tests/test_scripts.py`

- [ ] Add a subprocess test for `scripts/check_env.py --profile all`.
- [ ] Require output lines for Python, PyTorch, CUDA availability, OpenCV, ONNX Runtime, and current device.
- [ ] Make the expected exit status depend on Python 3.10 and the profile-required packages.
- [ ] Run the focused test and verify that it fails because `check_env.py` does not exist.

### Task 2: Add isolated-environment metadata and the checker

**Files:**

- Create: `environment.yml`
- Modify: `pyproject.toml`
- Modify: `requirements-dev.txt`
- Create: `scripts/check_env.py`

- [ ] Set the supported project interpreter to Python 3.10.
- [ ] Move PyTorch into a `training` optional dependency group; add OpenCV to both training and deployment groups; add ONNX Runtime only to the deployment group.
- [ ] Configure `environment.yml` to create `fomo-servo-train` with Python 3.10 and install only the training/development extras after a future explicit conda command.
- [ ] Implement profile-aware, read-only checks. Report missing dependencies and unsupported Python clearly; do not install anything or import model/data modules.
- [ ] Run the focused test and verify it passes.

### Task 3: Document Windows setup and deployment separation

**Files:**

- Modify: `README.md`

- [ ] Document the Windows commands for `conda env create -f environment.yml` and `conda activate fomo-servo-train`.
- [ ] Document CPU-first training, optional CUDA wheel selection inside the project environment, and separate Raspberry Pi deployment installation with `.[deployment]`.
- [ ] Document `scripts/check_env.py --profile training|deployment|all` and state that the command only inspects the active environment.

### Task 4: Verify without changing the host environment

**Files:**

- Verify: `environment.yml`, `pyproject.toml`, `requirements-dev.txt`, `README.md`, `scripts/check_env.py`, and `tests/test_scripts.py`

- [ ] Run `python -m pytest -q`.
- [ ] Run `python -m compileall -q src scripts`.
- [ ] Parse `environment.yml` using the existing YAML parser.
- [ ] Run `python scripts/check_env.py --profile all`; report its status without treating a missing target dependency on the current host as a test failure.
