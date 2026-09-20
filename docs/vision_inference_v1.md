# Vision Inference v1 — Frozen Pi Deployment Contract

Vision Inference v1 adds an optional, isolated Pi-side inference worker to the
existing LIVE Vision and capture service. It consumes the camera's latest
FrameHub frame and publishes inference status without changing control,
overlay, or actuator behavior.

## Frozen artifact contract

The deployment artifact is fixed as follows:

| Field | Frozen value |
| --- | --- |
| artifact identity | `d2_mobilenet_v2_fomo_seed42_epoch40` |
| ONNX SHA256 | `3dea74511bf2c44844192e75594fd53d4c4ce941f8b53b15767e020832bf9b08` |

The Pi launch must provide both artifact paths as a pair:

```text
--inference-onnx <path-to-deployed-model.onnx>
--inference-report <path-to-deployed-model.onnx.json>
```

Providing only one path is invalid and is rejected before service startup.
When neither path is provided, inference is disabled and the existing camera,
stream, and capture service remains available.

## Worker lifecycle and scheduling

The inference worker exposes exactly these states:

```text
disabled -> starting -> running
                         |
                         +-> failed
```

`failed` reports the initialization or prediction error through status. The
inference failure is isolated: it does not stop CameraOwner, FrameHub, the
47010 realtime stream, or the 47011 capture/control server. Inference has its
own generation. Starting a new generation clears the prior inference result
and counters, and a stale worker generation cannot publish into the new one.

The runtime policy is `REALTIME_LATEST`: the worker directly consumes the
single replaceable latest frame from `FrameHub`. There is no inference FIFO.
If prediction is busy while newer frames replace the current FrameHub slot,
those superseded frames are not inferred. `skipped_frames` counts those
superseded frame IDs between successfully processed frames; it is a latest-only
freshness metric, not a count of failed predictions.

For each completed prediction:

```text
latency_ms = (inference_finished_ns - capture_timestamp_ns) / 1_000_000.0
```

## Status and HTTP boundary

Inference status is available only through the existing HTTP/JSON control
plane on TCP `47011`:

```text
GET http://<pi-host>:47011/api/v1/vision/status
```

The status payload includes lifecycle state, frozen artifact identity/hash,
latest completed frame metadata, processed/skipped counts, inference FPS,
latency, detection count, and the last error. It does not expose full
detections over HTTP. There is no `47012` service and no
`/api/v1/vision/inference/latest` endpoint. TCP `47010` remains the realtime
JPEG video stream.

Inference does not add control commands, UI overlays, target selection,
visual-servo decisions, or actuator commands. The camera, capture, stream, and
robot-control behavior remain unchanged.

## Pi launch and verification notes

From the deployed bundle directory, provide the paired deployed model and
report paths:

```bash
python run.py vision_live \
  --source /dev/video0 \
  --inference-onnx <path-to-deployed-model.onnx> \
  --inference-report <path-to-deployed-model.onnx.json>
```

Do not add a threshold override or an alternate inference configuration to
this launch contract. Verify the process through `47011` and confirm that the
inference state progresses from `starting` to `running`, the reported artifact
identity and SHA256 match this document, and `processed_frames`/`latency_ms`
update while the camera is live. If model initialization or prediction fails,
confirm `state: "failed"` and `last_error` while the camera/stream/control
services remain independently usable.

## Targeted regression

The Task 10 regression gate is:

```bash
python -m pytest -q tests/test_vision_inference_worker.py tests/test_vision_service.py tests/test_capture_control.py tests/test_vision_capture_service.py tests/test_bundle_launcher.py tests/test_onnx_runtime_predictor.py
```

Fresh result in the Task 10 worktree: pytest interrupted during collection
with `1 error` because `tests/test_onnx_runtime_predictor.py` imports missing
package `torch` (`ModuleNotFoundError: No module named 'torch'`). No warning
summary was emitted; no test cases ran before collection was interrupted.
