# Vision Inference v1 — Final Handoff

## Status

- Slice 3 status: **complete**.
- Code merged: **YES**.
- Software verification: **PASS**.
- Raspberry Pi + Qt hardware acceptance A-G: **PASS**.

## Authoritative merged baselines

| Repository | Pull request | Merged `main` squash commit |
| --- | --- | --- |
| fomo-visual-servo | [PR #5](https://github.com/Mr-lxd/fomo-visual-servo/pull/5) | `649cb9a238a2aee75423b5f09aeaffdaff8ac9f7` |
| RoboBeetle | [PR #32](https://github.com/Mr-lxd/RoboBeetle/pull/32) | `c06dd44419c9b56dd97c6424d1b3b555f5697453` |

## Frozen architecture

```text
USB UVC
  |
CameraOwner
  |-- FrameHub [REALTIME_LATEST]
  |     |-- LiveStreamConsumer -> RBVS 47010 -> Qt LIVE
  |     `-- InferenceWorker
  |           -> BGR to RGB
  |           -> OnnxRuntimePredictor
  |           -> existing NumPy postprocess
  |           -> latest InferenceResult
  |
  `-- CaptureManager [SEQUENTIAL_BOUNDED]
        via CameraOwner.frame_callback -> offer_frame
```

`CameraOwner` is the sole `VideoCapture` owner. `InferenceWorker` consumes
`FrameHub` directly as a latest-only consumer: it has no inference FIFO and
does not create a second camera reader. `CaptureManager` is an independent,
sequential, bounded callback branch through
`CameraOwner.frame_callback -> offer_frame`.

## Formal deployment artifact

| Field | Value |
| --- | --- |
| identity | `d2_mobilenet_v2_fomo_seed42_epoch40` |
| ONNX SHA256 | `3dea74511bf2c44844192e75594fd53d4c4ce941f8b53b15767e020832bf9b08` |
| report SHA256 | `9ef9b98d6692d44d71271764ffcadfcbf24a6e668d8b66b1ffd1b91360845aa0` |
| postprocess threshold | `0.40` |
| input | RGB FP32 `[1, 3, 192, 192]`, normalized to `[0, 1]` |
| output | raw logits FP32 `[1, 8, 24, 24]` |
| opset | `17` |

## Runtime surfaces and launch

| Surface | Contract |
| --- | --- |
| `47010` | RBVS LIVE JPEG stream |
| `47011` | Existing `GET /api/v1/vision/status`, snapshot, record start, and record stop control plane |
| absent | No `47012`, no `/api/v1/vision/inference/latest`, and no full-detection HTTP endpoint |

Use paired artifacts with the existing live command; do not add a threshold
override or a temporary machine-specific path:

```bash
python run.py vision_live \
  --source /dev/video0 \
  --width 640 \
  --height 480 \
  --fps 25 \
  --fourcc YUYV \
  --bind 0.0.0.0 \
  --port 47010 \
  --control-port 47011 \
  --capture-output-root datasets_raw/robobeetle \
  --inference-onnx <formal-model.onnx> \
  --inference-report <formal-model.onnx.json>
```

## Verification record

- fomo focused Vision suites: **80 passed**.
- ONNX predictor regression in the existing `fomo-servo-train` environment:
  **18 passed**.
- Combined six-file fomo gate in that environment: **98 passed**.
- RoboBeetle targeted Windows CTest: **4/4 passed**.
- `git diff --check`: **PASS**.
- Raspberry Pi + Qt hardware acceptance A-G: **PASS**.

The accepted Pi preflight was aarch64, Python `3.13.5`, NumPy `2.5.2`, OpenCV
`5.0.0`, and ONNX Runtime `1.29.0`. `CPUExecutionProvider` was available and
selected explicitly. The deployment import closure succeeds without PyYAML or
`torch`; the Vision/ONNX Runtime path does not eagerly load training or config
dependencies.

Hardware observations confirm that one process owns `/dev/video0`; `47010`
stays LIVE; `47011` remains the existing status/capture plane; `47012` is not
listening; and `/api/v1/vision/inference/latest` returns `404`. LIVE inference
and capture/recording/inference coexisted, including sequential recording
frames. An inference initialization failure left camera, LIVE, capture, and
HTTP usable, while Qt showed Inference Error. A clean restart produced fresh
inference diagnostics with no stale generation or capture-state leakage. No
FPS threshold is claimed.

## Lifecycle and UI semantics

The worker states are `disabled`, `starting`, `running`, and `failed`. Failure
does not stop LIVE or Capture. There is no automatic retry: the failed worker
must be stopped and joined before an explicit lifecycle restart; direct
`start()` from `failed` is rejected. That restart creates a new inference
generation and clears result, counters, completion metrics, error, and model
identity until validation succeeds.

`skipped_frames` counts source-frame gaps only between successful inference
results; frames before the first successful result do not contribute.
`latency_ms` is capture timestamp to inference completion. Qt remains compatible
with legacy Slice 2 `GET` status responses, and a capture `POST` action response
that omits inference preserves the UI's prior inference state. Qt diagnostics
show bounded metadata, not full detections.

## Explicit exclusions

This slice includes no overlay, detection list, `TargetTracker`, target
selection, visual servo, PID, actuator behavior, RBRP change, STM32 change,
retraining, threshold tuning, second camera, second `VideoCapture`, inference
FIFO, or inference backlog.

## Next boundary

Slice 3 ends at isolated inference plus diagnostics. Any target selection,
tracking, visual servo, PID, authority, or actuator work requires a new design
gate.
