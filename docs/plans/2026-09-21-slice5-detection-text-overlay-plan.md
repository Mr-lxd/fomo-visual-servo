# Slice 5 Plan — Detection Text Overlay

## Repository / branch

Base `fomo-visual-servo` main:

`10e01bb4e2130c0a866f15bd6ee5ba5dd36ac20d`

Working branch:

`chatgpt/slice5-detection-text-overlay`

## Phase A — Pi detection metadata boundary

### A01 — Immutable result geometry

Extend `InferenceResult` with source frame width/height and an optional
`result_sink`.

Acceptance:

- width/height equal the CameraOwner frame processed by inference
- sink receives exactly one completed successful result
- worker disabled/starting/running/failed lifecycle is unchanged
- manual stop/retry behavior is unchanged

### A02 — Latest-only result hub

Add `InferenceResultHub`.

Acceptance:

- starts empty
- `snapshot()` is non-blocking
- waiter wakes only for a newer frame ID
- duplicate/regressing frame IDs fail
- negative timeout fails
- no FIFO/backlog

### A03 — Detection metadata codec

Add bounded NDJSON v1 encoder.

Acceptance:

- exact source frame ID/timestamp/dimensions
- coordinate space is `original_frame_pixels`
- only class ID/name/confidence/original centroid crosses the wire
- empty detections is valid
- no bbox/visual-marker/heatmap/input-coordinate fields
- finite/range validation
- at most 256 detections
- at most 128 UTF-8 bytes per class name
- at most 64 KiB per line

### A04 — Detection TCP server

Add one-client server-to-client stream on default port 47012.

Acceptance:

- cached preconnection result is not replayed
- reconnect does not replay stale metadata
- newest-only behavior under a slow sender
- send timeout ends only the current viewer
- unexpected client bytes close only the metadata session
- invalid port/buffer/timeout is rejected before listening
- asynchronous metadata-server failure is visible but does not stop 47010/47011

### A05 — Service composition

Wire:

`InferenceWorker -> InferenceResultHub -> DetectionTcpServer`

Acceptance:

- 47010 remains RBVS v1 unchanged
- 47011 routes remain unchanged
- 47012 starts/stops with the Vision service
- inference still starts disabled
- metadata server may be connected while inference is disabled and simply stays idle
- status advertises support / actual bound port / version
- startup bind/configuration failure rolls back cleanly
- runtime metadata failure is non-fatal to video/control
- no Robot/RBRP/STM32 dependency

### A06 — Validation

Required:

- focused Vision/inference/detection pytest
- full repository pytest using the existing `fomo-servo-train` Python 3.10 env
- `git diff --check`
- Raspberry Pi smoke:
  - 47010, 47011 and 47012 listen
  - inference initially disabled
  - Start -> metadata records
  - Stop -> inference disabled
  - video/control stay available
  - leave inference disabled

## Phase B — Qt metadata client

Begin only after RoboBeetle Task 03 / PR #34 is merged.

### B01 — Capability parsing

Extend `VisionControlClient` with optional:

- detection stream supported
- detection stream port
- detection stream version

Rules:

- capability is trusted only from fresh authoritative status
- missing fields mean unsupported
- old Pi must not produce 47012 connection spam

### B02 — Detection stream decoder/client

Add bounded NDJSON parser and dedicated TCP client.

Acceptance:

- maximum 64 KiB line
- type/version/schema validation
- maximum 256 detections
- strictly increasing metadata frame IDs
- timestamp/dimension/confidence/coordinate validation
- endpoint generation rejects stale callbacks after host switch
- detection failure does not kill RBVS video
- zero RobotController writes

### B03 — Latest-frame freshness policy

Do **not** cache/replay old video frames.

Keep normal RBVS latest-frame display.

For the newest valid detection result:

```text
age_ns =
    current_video_capture_timestamp_ns
    - detection_capture_timestamp_ns
```

Render only when:

- inference == Running
- HTTP status is fresh
- video is connected
- metadata dimensions match the current video frame
- `0 <= age_ns <= 1_500_000_000`

Suppress otherwise.

The 1500 ms constant is UI-only and must not influence model or control logic.

### B04 — Text-only VideoView rendering

Render:

`<class_name> <confidence>`

near the centroid.

Acceptance:

- no rectangle
- no background bbox
- no dot/circle
- no crosshair
- no centroid glyph
- no fixed-size marker
- source coordinate mapping uses the actual image paint rectangle
- label is clamped into the visible image rectangle
- small text shadow/outline is allowed for readability
- empty/stale/failed/disabled metadata removes text

## Phase C — Hardware acceptance

1. 47010 / 47011 / 47012 listening.
2. Connect Video while inference disabled: no overlay text.
3. Start Inference: 47012 emits increasing source frame IDs.
4. Fresh detection text appears over the latest live video.
5. Empty detections clears old labels.
6. Stop Inference clears/suppresses text while video continues.
7. Disconnect only 47012: video/control remain usable.
8. Reconnect 47012: cached preconnection metadata is not replayed.
9. Host switch clears metadata state.
10. No robot motion or RBRP writes.

## Deferred

TargetTracker, target selection/lock, bbox, visual marker, old-frame replay,
PID, visual servo, threshold/model changes, second camera and capture-time
burned-in overlays.
