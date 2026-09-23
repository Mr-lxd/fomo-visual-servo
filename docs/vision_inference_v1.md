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
| report SHA256 | `9ef9b98d6692d44d71271764ffcadfcbf24a6e668d8b66b1ffd1b91360845aa0` |
| fixed postprocess threshold | `0.40` |
| ONNX input | RGB `float32` `[1, 3, 192, 192]`, normalized to `[0, 1]` |
| ONNX output | raw logits `float32` `[1, 8, 24, 24]` |
| ONNX opset | `17` |

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
47010 realtime stream, or the 47011 capture/control server. A failed worker
does not retry automatically: it must be stopped so its thread is joined before
an explicit new `start()` creates a new inference generation; direct `start()`
from `failed` is rejected. This does not restart CameraOwner. A new inference
generation clears the prior result,
counters, completion-time window, error, and validated model identity until
validation completes; a stale worker generation cannot publish into the new
one.

The runtime policy is `REALTIME_LATEST`: the worker directly consumes the
single replaceable latest frame from `FrameHub`. There is no inference FIFO.
If prediction is busy while newer frames replace the current FrameHub slot,
those superseded frames are not inferred. `skipped_frames` counts source-frame
gaps only between successfully published inference results; frames before the
first successful result do not contribute. It is a latest-only freshness
metric, not a count of failed predictions.

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
detections over HTTP. The `inference` object also reports
`vision_process_rss_bytes` (the RSS of the complete Vision Python process,
including ONNX Runtime/model, camera, HTTP, video, and detection allocations;
this is not model-only memory) and `system_total_memory_bytes` (the Linux
system's total physical RAM). Both values are non-negative byte counts when
available, or `null` when their query fails. Memory = Vision process RSS / total
system physical memory. Full detections are not exposed over HTTP. Detection
metadata is available through the existing TCP `47012` stream;
`/api/v1/vision/inference/latest` does not exist. TCP `47010` remains the
realtime JPEG video stream.

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

## Final verification — Hardware A-G PASS

Final software verification is recorded from the completed Slice 3 gates:

- Focused Vision suites: `80 passed`.
- `tests/test_onnx_runtime_predictor.py` in the existing `fomo-servo-train`
  environment: `18 passed`.
- Complete six-file fomo gate in the same environment: `98 passed`.
- RoboBeetle targeted Windows CTest: `4/4` passed.
- `git diff --check`: PASS.

The Pi runtime intentionally does **not** require `torch` or PyYAML. Hardware
preflight found and fixed the deployment import closure so the Vision/ONNX
Runtime path does not eagerly import training or configuration dependencies.
The accepted Raspberry Pi environment was aarch64 with Python `3.13.5`, NumPy
`2.5.2`, OpenCV `5.0.0`, and ONNX Runtime `1.29.0`; `CPUExecutionProvider` was
available and selected explicitly. PyYAML and `torch` were absent, while the
Vision ONNX Runtime import closure succeeded.

Hardware acceptance A-G passed:

| Gate | Accepted evidence |
| --- | --- |
| A — Formal model identity | The frozen artifact, report hash, threshold, tensor contract, and opset matched this document. |
| B — Sole CameraOwner | Exactly one process owned `/dev/video0`; inference did not open a second `VideoCapture`. |
| C — `REALTIME_LATEST` | Inference consumed `FrameHub` directly with no FIFO or stale-inference backlog. No FPS threshold is asserted. |
| D — LIVE + inference | TCP `47010` remained LIVE while inference diagnostics updated on `47011`. |
| E — LIVE + Capture + inference | Snapshot and sequential recording continued alongside inference; sequential recording frames were verified. |
| F — inference failure isolation | An inference initialization error left camera, LIVE, capture, and HTTP usable; Qt displayed Inference Error. |
| G — clean shutdown/restart | A new inference generation cleared stale result and model-generation state without capture-state leakage. |

The observed runtime surface stayed bounded: `47010` remained the LIVE stream,
`47011` remained the existing status/capture plane, no service listened on
`47012`, and `/api/v1/vision/inference/latest` returned `404`. The reported
model identity, SHA256, and threshold matched the frozen contract. Qt exposes
diagnostics only: it remains compatible with legacy Slice 2 `GET` responses,
and capture `POST` action responses without inference preserve the prior
inference UI state.

## Authoritative merged baseline

Vision Inference v1 was merged with the following authoritative baselines.
These commits are the completed Slice 3 integration baselines:

| Repository | Pull request | Merged `main` commit |
| --- | --- | --- |
| fomo-visual-servo | [PR #5](https://github.com/Mr-lxd/fomo-visual-servo/pull/5) | `649cb9a238a2aee75423b5f09aeaffdaff8ac9f7` |
| RoboBeetle | [PR #32](https://github.com/Mr-lxd/RoboBeetle/pull/32) | `c06dd44419c9b56dd97c6424d1b3b555f5697453` |
