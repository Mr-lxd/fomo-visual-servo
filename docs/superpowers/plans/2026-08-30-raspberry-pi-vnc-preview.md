# Raspberry Pi VNC Camera Preview Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in OpenCV desktop preview for the existing latest-frame camera CLI while preserving the formal headless ONNX Runtime path and frozen inference semantics.

**Architecture:** Keep preview handling at the CLI boundary. The consumer displays the same annotated BGR frame that is written to MP4, after inference and telemetry are complete; the capture thread retains its one-frame replacement buffer. Use two mutually exclusive Raspberry Pi dependency profiles so the formal headless venv remains unchanged and a separate preview venv supplies Qt/XCB HighGUI.

**Tech Stack:** Python 3.10 tests, Python 3.13 ARM64 deployment, OpenCV HighGUI, ONNX Runtime CPU, pytest, Wayland/labwc + XWayland + WayVNC.

**Completion:** All implementation and automated verification items were completed on 2026-08-30. The user subsequently completed the VNC operator acceptance successfully and authorized the separate final milestone commit/push. The final evidence is recorded in `docs/handoffs/2026-08-28-raspberry-pi4-deployment-handoff.md`.

---

### Task 1: Specify CLI preview behavior with failing tests

**Files:**
- Create: `tests/test_predict_video_display.py`
- Modify: `tests/test_bundle_launcher.py`

- [x] Assert `--display` defaults to false and headless execution never calls HighGUI.
- [x] Assert `q` and Esc stop after the displayed annotated frame has been written to MP4/CSV/JSONL.
- [x] Assert the preview is closed on normal and exceptional exits.
- [x] Assert `GUI: NONE` and a missing desktop-session environment produce actionable `InferenceError` output.
- [x] Run the new tests and confirm they fail because `--display` and preview handling do not exist.

### Task 2: Implement the smallest CLI-local preview adapter

**Files:**
- Modify: `scripts/predict_video.py`

- [x] Add default-off `--display` without changing other parser defaults.
- [x] Add a private preview helper that checks the OpenCV GUI backend and desktop environment before creating a window.
- [x] Wrap OpenCV HighGUI failures in diagnostic `InferenceError` messages.
- [x] Call `imshow`/`waitKey(1)` only on the copied annotated output frame, after prediction, telemetry, and MP4 writing.
- [x] Stop on `q` or Esc and close the named window from `finally` without masking primary failures.
- [x] Run display, camera-limit, video-buffer, launcher, and import tests to green.

### Task 3: Record mutually exclusive Pi runtime profiles

**Files:**
- Create: `requirements-pi4-headless.txt`
- Create: `requirements-pi4-preview.txt`
- Modify: `tests/test_bundle_launcher.py`

- [x] Keep NumPy `2.5.2` and ONNX Runtime `1.29.0` identical in both profiles.
- [x] Pin only `opencv-python-headless==5.0.0.93` in headless and only `opencv-python==5.0.0.93` in preview.
- [x] Assert neither profile includes torch, torchvision, CUDA, models, or training dependencies.
- [x] Run dependency-profile tests to green.

### Task 4: Update and deploy the minimal ignored Pi bundle

**Files:**
- Modify: `outputs/deployment/d2_seed42_epoch40/pi4_bundle/scripts/predict_video.py`
- Create: `outputs/deployment/d2_seed42_epoch40/pi4_bundle/run.py`
- Create: `outputs/deployment/d2_seed42_epoch40/pi4_bundle/requirements-pi4-headless.txt`
- Create: `outputs/deployment/d2_seed42_epoch40/pi4_bundle/requirements-pi4-preview.txt`

- [x] Apply identical display behavior to the ORT-only bundle CLI without adding PyTorch symbols.
- [x] Copy only the launcher, ORT-only video CLI, and dependency profiles to `/home/pi/fomo-ort-d2-epoch40` through a staging directory.
- [x] Verify import closure and formal ONNX/sidecar SHA before publication.

### Task 5: Validate Pi dependencies and noninteractive launch conditions

**Files:**
- Modify: `docs/handoffs/2026-08-28-raspberry-pi4-deployment-handoff.md`

- [x] Leave `/home/pi/venvs/fomo-ort-d2-epoch40` unchanged and prove its headless camera smoke still writes MP4/CSV/JSONL.
- [x] Create `/home/pi/venvs/fomo-ort-d2-epoch40-preview` and install the preview profile from binary wheels.
- [x] Verify `pip check`, versions, `GUI: QT5`, Qt/XCB libraries, no torch, active Wayland/labwc/WayVNC, and inherited desktop session variables.
- [x] From plain SSH, verify `--display` fails before camera use with a clear instruction to launch from a VNC desktop terminal.
- [x] Do not open or operate the GUI window; provide the exact VNC Terminal command for the user.

### Task 6: Final verification and handoff

**Files:**
- Modify: `docs/handoffs/2026-08-28-raspberry-pi4-deployment-handoff.md`

- [x] Run camera/CLI targeted tests and torch-free import tests.
- [x] Run the complete pytest suite and `git diff --check`.
- [x] Recheck the complete diff, formal model SHA, threshold, and absence of test-split access.
- [x] Record USB UVC status, headless/preview dependency separation, exact commands, pending operator GUI confirmation, and camera-on-screen domain-shift warning.
- [x] Suggest a milestone commit message; do not commit or push.
