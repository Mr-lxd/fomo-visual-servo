# Vision Inference v1 Design

## Goal

Integrate the frozen D2 ONNX Runtime predictor into the existing RoboBeetle
Vision Capture v1 service as an optional, read-only, realtime-latest inference
consumer, and expose bounded inference diagnostics through the existing 47011
status endpoint and Qt vision card.

This design applies to the paired `fomo-visual-servo` and `RoboBeetle`
repositories. It does not change the robot-control path, firmware, RBVS
payload, camera ownership, capture recording contract, or standalone
`predict_image`/`predict_video` CLIs.

## Baseline and constraints

- fomo-visual-servo baseline: `22cdf3ef539cb8982b471897fab68a06215cd3fe`.
- RoboBeetle baseline: `9d860348639f154a5b5ed5f8dbbcd8f5fcfcc19c`.
- `CameraOwner` is the only component allowed to open or release `/dev/video0`.
- `FrameHub` is a thread-safe single-slot latest-frame boundary. Inference must
  consume it directly with `wait_for_newer()`; no callback fan-out or second
  queue is allowed.
- The formal predictor is
  `OnnxRuntimePredictor.from_files(onnx_path, report_path)`. Integrated code
  must not construct an independent `onnxruntime.InferenceSession`.
- The integrated model contract is fixed to artifact
  `d2_mobilenet_v2_fomo_seed42_epoch40`, seed `42`, epoch `40`, threshold
  `0.40`, RGB float32 input `[1,3,192,192]` in `[0,1]`, raw-logit output
  `[1,8,24,24]`, opset `17`, and model SHA-256
  `3dea74511bf2c44844192e75594fd53d4c4ce941f8b53b15767e020832bf9b08`.
- The model sidecar remains the source of class mapping, preprocessing,
  postprocessing, and the validated threshold. The integrated CLI has no
  threshold override.
- Inference is perception-only. No target selection, tracking, overlay,
  actuator, RBRP, STM32, heartbeat, or automatic motion behavior is included.

## Architecture

The existing pipeline remains:

```text
USB UVC camera
    -> CameraOwner
    -> FrameHub (one replaceable BGR frame)
         -> LiveStreamConsumer -> RBVS 47010 -> Qt video
         -> InferenceWorker -> BGR-to-RGB -> OnnxRuntimePredictor
                              -> existing NumPy postprocess
                              -> latest InferenceResult
         -> CaptureManager -> snapshot / bounded sequential recording
```

`InferenceWorker` is an independent thread owned by `VisionService`. It never
constructs `VideoCapture`, never releases the camera, and never performs disk,
network, JPEG, video, HighGUI, or Qt work. `CaptureManager.offer_frame()` stays
the existing CameraOwner callback and remains independent from inference.

### InferenceWorker API and state

`src/fomo_servo/vision/inference_worker.py` provides:

- `InferenceState`: `disabled`, `starting`, `running`, and `failed`.
- Frozen `InferenceResult` with `frame_id`,
  `capture_timestamp_ns`, `inference_started_ns`,
  `inference_finished_ns`, and `detections: tuple[Detection, ...]`.
- `InferenceWorker.start()`, `stop()`, `status()`, and `latest_result()`.

On `start()`, the worker transitions to `starting` and launches its own
thread. The thread loads and validates the predictor through
`OnnxRuntimePredictor.from_files()`. Only after successful initialization does
it snapshot the current FrameHub frame ID as the startup floor and transition
to `running`. Frames already cached while the model initialized are therefore
not treated as fresh inference input.

The loop waits for the newest frame newer than its internal floor, converts the
source BGR image to a new RGB array with OpenCV, calls
`predict_rgb_image()` without a threshold argument, and atomically publishes a
new frozen result. The source frame image is not modified. If frames 101, 102,
103, and 104 arrive while frame 100 is being inferred, the next call returns
104; 101--103 are not queued or processed. `skipped_frames` records those
intentional frame-ID gaps between successfully published results.

The worker stores only one latest result, a bounded deque of recent completion
timestamps for FPS, counters, the validated model identity, and the latest
diagnostic error. A stop request is checked before publishing a completed
inference, so shutdown does not intentionally publish a new result. A runtime
or initialization exception transitions the worker to `failed`, preserves a
diagnostic error, and exits without retrying or affecting CameraOwner,
FrameHub, RBVS, CaptureManager, or the HTTP server.

### Frozen contract validation

`OnnxRuntimePredictor.from_files()` remains responsible for sidecar loading,
file hash/size validation, ORT session creation, tensor shape/type checks, and
the existing preprocessing/postprocessing contract. The integration layer
additionally checks the frozen D2 identity listed above before entering
`running`; a mismatch is an inference failure, not a VisionService startup
failure.

## Service and CLI integration

`VisionServiceConfig` adds nullable `inference_onnx` and `inference_report`
paths. Construction rejects exactly one supplied path before CameraOwner is
started. With both omitted, no worker is created and status reports
`disabled`. With both supplied, the worker is started after the LIVE camera and
RBVS path but before the HTTP control server. Model or ORT failure happens in
the worker thread and therefore leaves LIVE and Capture available.

Shutdown follows the frozen order:

1. stop HTTP 47011 and wait for handlers;
2. finalize CaptureManager;
3. stop and join InferenceWorker;
4. stop RBVS;
5. stop CameraOwner and release the camera;
6. clear FrameHub.

`scripts/vision_live.py` adds only `--inference-onnx` and
`--inference-report`. A lone argument is rejected by CLI configuration
validation. No confidence, strategy, tracking, or test-split option is added.

## HTTP status contract

The existing `GET /api/v1/vision/status` response gains this bounded object:

```json
{
  "inference": {
    "state": "disabled|starting|running|failed",
    "artifact_name": null,
    "model_sha256": null,
    "confidence_threshold": null,
    "latest_frame_id": null,
    "capture_timestamp_ns": null,
    "processed_frames": 0,
    "skipped_frames": 0,
    "inference_fps": null,
    "latency_ms": null,
    "detection_count": null,
    "last_error": null
  }
}
```

Artifact identity and threshold are populated only after the model contract is
validated. Result-specific fields are null until a successful inference has
been published; frame ID zero is still a valid non-null result. Full
detections remain internal to `InferenceResult`; no `/inference/latest` route
or new port is added. Existing capture fields and behavior are unchanged.

## Qt diagnostics integration

`VisionControlClient` extends the existing `VisionCaptureStatus` structure and
parses the optional `inference` object through the same 1-second polling path.
Nullable numeric fields use explicit `have...` flags so null is not confused
with zero. A response from a Slice 2 server without `inference` resets the
inference view to disabled and remains valid JSON/status input.

`MainWindow` adds diagnostics to the existing Realtime Video card only:

- `Inference RUNNING`, `Inference STARTING`, `Inference Disabled`, or
  `Inference Error` state text;
- a compact line containing artifact identity, FPS, latency, result frame,
  detection count, and skipped-frame count;
- a short error message when the state is failed.

No `VideoView` changes, overlay, bounding box, class label, tracking marker,
target lock, synchronization visualization, or additional socket/polling loop
is introduced.

## Testing strategy

Tests are written before production changes and run through explicit red-green
cycles.

### fomo-visual-servo

- Worker state and model identity, including disabled configuration,
  starting/running, initialization failure, runtime failure, preserved
  `last_error`, clean stop, and no camera ownership.
- Frame binding and channel order using an explicit BGR fixture, preserving
  source frame ID/timestamp and proving source pixels are unchanged.
- Deterministic slow-predictor latest-only regression: A blocks, B/C/D are
  published, then D is the next processed frame and skipped count is 2 for
  the intermediate frames.
- Service status and failure isolation: inference failure leaves the camera,
  RBVS/Capture composition, and HTTP status available.
- CLI pair validation, disabled/enabled defaults, and absence of a threshold
  override.
- Existing capture/status, vision-service, bundle, and ORT predictor tests
  relevant to changed interfaces.

### RoboBeetle

- `VisionControlClient` parses running, failed, model identity, counters,
  result fields, and explicit nulls.
- Status responses without `inference` remain backward compatible.
- `MainWindow` displays running diagnostics and failed error state without
  adding overlay behavior; existing robot-control transport tests remain
  quiet.

Targeted tests are followed by `git diff --check`. No hardware, camera, VNC,
RBRP, STM32, or RDC action is part of this slice.

## Files in scope

### fomo-visual-servo

- Create `src/fomo_servo/vision/inference_worker.py`.
- Modify `src/fomo_servo/vision/service.py`.
- Modify `src/fomo_servo/capture/control.py`.
- Modify `scripts/vision_live.py`.
- Add targeted worker/service/CLI tests and `docs/vision_inference_v1.md`.

### RoboBeetle

- Modify `RoboBeetleConsole/src/vision/VisionControlClient.h/.cpp`.
- Modify `RoboBeetleConsole/src/ui/MainWindow.h/.cpp`.
- Extend `vision_control_client_tests.cpp` and `main_window_tests.cpp`.
- Update CMake only if the existing test target needs an explicitly listed
  changed source.

No firmware, RBRP, robot-control, standalone `predict_video`, or historical
worktree file is in scope.
