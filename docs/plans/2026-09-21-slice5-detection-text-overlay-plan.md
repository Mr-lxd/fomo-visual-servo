# Slice 5 Plan — Frame-Synchronized Detection Text Overlay

Base `fomo-visual-servo` main:

`10e01bb4e2130c0a866f15bd6ee5ba5dd36ac20d`

Working branch:

`chatgpt/slice5-detection-text-overlay`

## Phase A — Pi metadata boundary

### A01 — Inference result geometry

Extend `InferenceResult` with source frame width/height and an optional result sink.

Acceptance:
- width/height equal the processed CameraOwner frame
- one sink publication per successful result
- manual lifecycle semantics unchanged

### A02 — Latest-only result hub

Add `InferenceResultHub`.

Acceptance:
- starts empty
- snapshot is non-blocking
- waits only for a newer frame ID
- duplicate/regressing frame IDs fail
- negative timeout fails
- no queue/backlog

### A03 — Detection metadata codec

Add NDJSON v1 encoder.

Acceptance:
- exact source frame ID/timestamp/dimensions
- `original_frame_pixels` coordinate space
- class ID/name/confidence/original centroid only
- empty detections valid
- no bbox/marker/heatmap/input coordinates
- finite/range validation
- <=256 detections
- <=128 UTF-8 bytes per class name
- <=64 KiB record

### A04 — Detection TCP server

Default port 47012, one viewer, server-to-client only.

Acceptance:
- no cached preconnection replay
- latest-only under slow sender
- send timeout ends viewer without retry queue
- unexpected inbound bytes close viewer
- invalid configuration rejected before listen
- server errors remain visible

### A05 — Service composition

Wire `InferenceWorker -> InferenceResultHub -> DetectionTcpServer`.

Acceptance:
- 47010 RBVS v1 unchanged
- 47011 existing routes unchanged
- 47012 starts/stops with Vision service
- inference still starts disabled
- status advertises supported/actual bound port/version
- detection start failure rolls back LIVE startup
- result received on 47012 has the exact source frame ID from the worker
- no robot/RBRP/STM32 dependency

### A06 — Validation

Required runtime tests:
- `tests/test_detection_streaming.py`
- `tests/test_inference_result_hub.py`
- all `tests/test_vision_*.py`
- `git diff --check`
- `compileall` for Vision runtime

Complete repository pytest also runs in the normal Windows PyTorch training
environment. A minimal runtime environment without `torch` may fail training-test
collection and must report that as an environment limitation rather than installing
training dependencies into the Pi/runtime environment.

## Phase B — Qt exact-frame synchronization

Begin only after RoboBeetle PR #34 is merged.

### B01 — Capability parsing

Parse optional:
- `detection_stream_supported`
- `detection_stream_port`
- `detection_stream_version`

Only a fresh 47011 status may enable the metadata connection. Missing fields mean
unsupported/read-only.

### B02 — Detection client

Add a bounded NDJSON parser and independent TCP client.

Acceptance:
- max 64 KiB line
- type/version/schema validation
- <=256 detections
- finite/range/dimension validation
- endpoint generation protects host switches
- metadata failure does not kill RBVS video or HTTP control
- no RobotController writes

### B03 — Bounded decoded-frame cache

Extend the video side with a fixed-capacity recent decoded-frame cache keyed by
RBVS `frame_id`.

Acceptance:
- fixed capacity; no unbounded image queue
- exact frame ID lookup
- replacement/drop metrics visible in tests
- host switch/video disconnect clears cache
- normal inference-disabled display remains latest-frame behavior

### B04 — Exact synchronized display policy

While inference is Running:
- each metadata record may render only on the cached frame with the exact same
  `frame_id`
- exact match displays that cached frame plus text overlay
- cache miss drops metadata and increments a sync-miss counter
- never move detections to current/latest/nearest frame
- zero detections displays the exact matched frame with no previous text

When inference is not Running, video returns to latest-frame display and overlay
is cleared.

### B05 — Text-only VideoView rendering

Render:

`<class_name> <confidence>`

near the centroid.

Acceptance:
- no rectangle
- no dot/circle
- no crosshair
- no centroid glyph
- no bbox
- source coordinates mapped through the actual image paint rectangle
- text clamped inside the image rectangle
- small text shadow/outline allowed only for readability
- resize/high-DPI/source-aspect tests

## Phase C — Hardware acceptance

1. 47010 / 47011 / 47012 listen.
2. Connect Video with inference Disabled: latest live video, no label.
3. Start Inference: 47012 emits increasing source frame IDs.
4. Qt displays only exact matched cached frames with text labels.
5. Metadata whose frame fell out of cache is dropped, never reattached.
6. Empty detections clears labels on its exact frame.
7. Stop Inference clears overlay and returns to latest live video.
8. Disconnect 47012 only: video/control remain usable.
9. Reconnect does not replay cached preconnection metadata.
10. No robot motion or RBRP writes.

## Deferred

TargetTracker, target selection/lock, bbox, visual marker, PID, visual servo,
threshold/model changes, second camera, and burned-in recording overlays.
