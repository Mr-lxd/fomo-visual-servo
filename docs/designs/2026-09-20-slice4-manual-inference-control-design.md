# Slice 4 Task 01 — Manual Inference Control Design

> **Status:** Frozen implementation record; execution evidence is reported in
> the Task 01 review report. Hardware acceptance remains pending.

## Authority and baseline

This document records the implementation contract frozen in
`Slice4_Codex_Task01_Frozen_Pack.md` (2026-09-20, sections 4 and 5). It is a
traceable implementation record, not a new architecture proposal. The required
fomo baseline is `3f82ee6274c03be1c6ff703971333e0e4cc919a2`.

Task 01 is limited to the fomo Pi Vision service. It adds explicit manual
inference Start, Stop, and Retry coordination on the existing HTTP control
plane at port 47011. It does not change the model artifact, threshold,
preprocessing, postprocessing, camera topology, capture behavior, RBRP, Qt, or
RoboBeetle firmware.

## Startup and status contract

The paired `inference_onnx` and `inference_report` parameters remain an
all-or-nothing startup validation. When both paths are supplied, `VisionService`
constructs one `InferenceWorker` capability but does not call `worker.start()`
during `start_live()`. No predictor, ONNX Runtime session, or model-file
validation runs until an explicit manual Start. When neither path is supplied,
no worker is constructed; camera, stream, capture, and HTTP continue normally.

`GET /api/v1/vision/status` remains the sole authority for live inference
state. Every `inference` object includes the existing worker fields plus:

| Field | Meaning |
| --- | --- |
| `configured` | Both artifact-path parameters were supplied; it does not prove a model loaded. |
| `control_supported` | `true` for the complete Slice 4 `VisionService`; absent/false is retained for standalone legacy server compatibility. |
| `operation` | `null`, `"stopping"`, or `"retrying"`; it is service lifecycle metadata, not a worker state or queue. |

The only completed manual-stop condition is `state == "disabled"` with
`operation == null`. Worker states remain exactly `disabled`, `starting`,
`running`, and `failed`. A configured-but-not-yet-started worker exposes no
validated identity or active metrics.

## Components and ownership

### `InferenceWorker`

The worker retains ownership of predictor initialization, latest-only frame
consumption, generation fencing, and its worker-state lock. It gains two
public lifecycle primitives:

- `request_stop()` idempotently sets the generation stop event under the
  worker lock without joining or clearing state.
- `reset_disabled()` is legal only after `stop()` has released the worker
  thread handle. It clears identity, result, frame metadata, counters,
  completion window, and error, then publishes the disabled snapshot. It never
  touches `FrameHub`, camera, or capture.

If `Thread.start()` fails, `start()` must leave no false `starting` state and
no unusable thread handle: it records a `failed` diagnostic, clears the
unstarted handle, and re-raises the synchronous error.

### `InferenceControl`

`src/fomo_servo/vision/inference_control.py` is the service-level lifecycle
facade. It owns action admission, shutdown state, operation state,
lifecycle-error latching, and one lazy non-daemon coordination thread. Its
public API is fixed as:

```python
InferenceControl(worker, *, is_live, camera_running)
open_actions()
start_inference() -> InferenceActionResult
stop_inference() -> InferenceActionResult
status() -> dict
begin_shutdown()
finish_shutdown()
```

The facade does not own a camera or parse HTTP. It uses an `RLock` and
`Condition`; its lock is acquired before the worker lock and is never held
while joining a worker, stopping an HTTP server, or finalizing capture.

The coordinator is created only for accepted asynchronous Stop or Retry. It
has one operation slot, waits on a condition when idle, and performs
`worker.stop()`/join outside the facade lock. It does not read frames, create a
predictor, use a task queue, or spawn one thread per click.

### `VisionService` and `VisionControlServer`

`VisionService` constructs the worker (when configured) and exactly one
facade. It passes `is_live` and `camera_running` predicates to the facade,
opens actions only after LIVE mode is ready, and starts the existing control
listener afterwards. The service exposes thin `start_inference()` and
`stop_inference()` callbacks. Startup rollback closes actions and cleans the
facade while preserving the original startup exception.

An unsuccessful `start_live()` rollback and `shutdown()` both terminally close
that `VisionService` lifecycle. The caller must construct a new
`VisionService` for a later full restart; a closed facade is never silently
reopened into a partially actionable LIVE service.

`VisionControlServer` remains a routing/validation/mapping boundary. Optional
`inference_start_callback` and `inference_stop_callback` receive the two new
routes; the server never reads or writes worker private fields and never imports
`VisionService`.

## Admission and lifecycle semantics

Admission order is fixed: shutdown; lifecycle failure latch; current
operation; configured capability; current worker state; LIVE/camera checks for
a request that genuinely needs a new generation. Unknown internal state returns
`409 invalid_inference_state` rather than guessing.

- A configured disabled, LIVE service with a running camera accepts Start with
  `202`; `worker.start()` only creates its worker thread.
- Start during `starting` or `running` reports the corresponding `200` duplicate
  outcome without a new generation.
- Stop during `starting`, `running`, or `failed` accepts asynchronously with
  `202` and `operation="stopping"`. It signals the worker without joining in
  the HTTP handler.
- Explicit Start from `failed` accepts Retry with `202` and
  `operation="retrying"`. The coordinator first stops/joins the old generation,
  rechecks shutdown/LIVE/camera, then starts exactly one new generation when
  possible.
- Stopping and retrying have the frozen busy/duplicate responses. There is no
  automatic retry, forced thread termination, or cross-generation idempotency
  key.

After successful Stop or Clear Error cleanup, the worker is cleanly disabled;
configured paths remain intact and camera/capture/session counters are not
reset. A coordinator cleanup failure latches a clear lifecycle error, retains
the operation, and makes later Start/Stop return
`409 inference_control_failed` until process shutdown. The latch is also
logged at error level for operational diagnosis.

## HTTP contract

Only these endpoints are added:

```text
POST /api/v1/vision/inference/start
POST /api/v1/vision/inference/stop
```

Each accepts only an empty body or JSON empty object `{}` (surrounding JSON
whitespace allowed). Bodies are limited to 1024 bytes. Invalid content length,
unsupported transfer encoding, truncated/timeout reads, non-object JSON, and
objects with keys return `400 invalid_request`; declared oversized bodies return
`413 request_too_large` without reading the excess. The read timeout is two
seconds. This validation applies only to the new inference routes; existing
capture POST behavior is unchanged.

The numeric `Content-Length` comparison is bounded before integer conversion,
so an arbitrarily long decimal header is still classified as
`413 request_too_large` rather than becoming an internal server error.

An accepted action returns a small acknowledgement only:

```json
{"ok":true,"action":"inference/start","outcome":"accepted"}
```

`202` means the operation was accepted, never that initialization completed or
the thread exited. Success duplicates use `200`; errors retain a stable string
`error` machine code plus a readable `message`. POST responses do not contain
camera, capture, or inference status objects. Clients must obtain authoritative
state through the existing GET status endpoint.

## Shutdown and exclusion boundary

`begin_shutdown()` closes action admission, records shutdown, signals any
worker, and wakes the coordinator. `VisionService.shutdown()` then stops the
control server, shuts down capture, asks the facade to join/reclaim worker and
coordinator resources, and finally stops the mode/camera owner. Every cleanup
step is attempted; the first cleanup error is ultimately reported. Shutdown
never starts a retry generation and never force-kills native inference.

No second HTTP port, status poller, camera, prediction queue, model policy, UI
surface, deployment bundle tool, or hardware interaction is introduced by this
task.

## Verification boundary

All Task 01 tests use fakes, barriers/events/conditions, `tmp_path`, and a
localhost temporary HTTP port. They do not access `/dev/video0`, a real Pi,
serial hardware, a robot, or a real model. The detailed P01–P30 mapping and
commands are recorded in the paired implementation plan.
