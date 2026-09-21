# Vision Slice 5 — Frame-Associated Detection Text Overlay

## Goal

Expose completed FOMO detections to the Qt operator console while preserving the
latest-frame realtime video path.

The first Slice 5 UI renders **text only** near each fresh detection centroid:

`<class_name> <confidence>`

Example:

`creature 0.87`

No rectangle, crosshair, circle, dot, centroid glyph, fixed-size marker, inferred
bounding box, tracker, target lock, PID, or robot motion is part of this slice.

## Frozen model contract

The deployed D2 model remains unchanged:

- artifact: `d2_mobilenet_v2_fomo_seed42_epoch40`
- ONNX SHA256: `3dea74511bf2c44844192e75594fd53d4c4ce941f8b53b15767e020832bf9b08`
- confidence threshold: `0.40`
- input: RGB FP32 `[1,3,192,192]` in `[0,1]`
- output: raw logits `[1,8,24,24]`
- existing FOMO centroid postprocess

FOMO produces centroids, not model-predicted bounding boxes. Slice 5 must not
reconstruct, approximate, or imply a true bbox.

## Network boundaries

Existing contracts remain authoritative:

- `47010`: RBVS v1 JPEG live video, unchanged
- `47011`: HTTP status/capture/manual inference, existing routes unchanged
- `47012`: new Slice 5 detection metadata stream

RBVS v1 is intentionally not modified, so old Qt builds continue consuming 47010.

## Capability advertisement

The `inference` object returned by `GET /api/v1/vision/status` adds:

- `detection_stream_supported: bool`
- `detection_stream_port: int`
- `detection_stream_version: int`

A new Qt client connects to 47012 only after a fresh authoritative status confirms
support. Missing capability means overlay unavailable, not a connection error.

## 47012 wire contract

Transport:

- TCP
- default port 47012
- one active viewer
- server-to-client only
- UTF-8 NDJSON
- one completed inference result per line
- maximum record size 64 KiB
- maximum 256 detections per record
- maximum UTF-8 class name length 128 bytes
- no handshake
- unexpected client bytes close that metadata viewer

Record:

```json
{
  "type": "detections",
  "version": 1,
  "frame_id": 123,
  "capture_timestamp_ns": 456789000,
  "width": 640,
  "height": 480,
  "coordinate_space": "original_frame_pixels",
  "detections": [
    {
      "class_id": 0,
      "class_name": "creature",
      "confidence": 0.87,
      "original_x": 321.4,
      "original_y": 208.7
    }
  ]
}
```

Only data required for text overlay crosses the wire. Heatmap coordinates,
letterbox/input coordinates, component geometry, bbox fields, target IDs and robot
control fields are not part of the contract.

Empty `detections` is valid and is transmitted so the client can clear old labels
for that processed result.

All numeric fields must be finite and centroid coordinates must lie inside the
source frame.

## Latest-only metadata semantics

The metadata path is realtime/latest-only:

- no inference-result FIFO
- no backlog replay
- slow metadata clients may skip intermediate results
- a new connection fences the result cached before connection
- reconnect never replays stale pre-connection detections

`InferenceWorker` publishes each successful immutable `InferenceResult` into an
`InferenceResultHub`. The result includes source width/height, frame ID, capture
timestamp, model identity and detections.

## Frame association without delaying video

Inference completes after its source camera frame has normally already been
displayed. Delaying or replaying video until the matching inference result arrives
would make the operator console visibly lag by inference latency.

Slice 5 therefore preserves the existing latest-frame RBVS display. Qt does **not**
cache old video frames for overlay playback and does **not** freeze the live view.

Every metadata record still carries the exact source:

- `frame_id`
- `capture_timestamp_ns`
- source width/height

Qt compares the Pi monotonic timestamps of the current displayed live frame and
the newest detection result:

```text
overlay_age_ns =
    current_video_capture_timestamp_ns - detection_capture_timestamp_ns
```

A result is renderable only when all of these are true:

- inference state is `running`
- Vision HTTP status is fresh
- metadata stream record is valid
- current video dimensions equal metadata dimensions
- `overlay_age_ns >= 0`
- `overlay_age_ns <= 1_500_000_000` (1500 ms)

If the metadata is from the future, older than 1500 ms, mismatched in resolution,
or otherwise stale, the overlay is suppressed.

The 1500 ms bound is a **UI freshness constant only**. It is not the model
confidence threshold and must not be used by future target tracking or control.

Future tracking/visual-servo logic must consume source detection timestamps
directly rather than treating the operator overlay as a control signal.

## Text-only Qt rendering contract

The Qt half of Slice 5 will:

- add a bounded newline decoder and dedicated DetectionClient
- validate type/version/frame ID/timestamp/dimensions/confidence/coordinates
- retain only the newest valid metadata record
- render `class_name confidence` near `original_x/original_y`
- map original-frame pixels through the exact VideoView image paint rectangle
- clamp label text into the visible image rectangle
- allow a small text shadow/outline only for readability

Not allowed:

- rectangle/bbox
- circle/dot
- crosshair
- centroid glyph
- fixed-size marker
- background detection box
- target lock indicator
- tracking trail
- servo/PID guidance

## Overlay clear/suppress rules

Overlay text is cleared or suppressed when:

- inference is Disabled, Starting, Stopping, Retrying or Failed
- Vision HTTP status is stale/unavailable
- video is disconnected
- Pi host/endpoint generation changes
- detection metadata stream disconnects and the retained record becomes stale
- metadata is older than 1500 ms
- metadata timestamp is newer than the displayed video timestamp
- metadata dimensions differ from the live video
- `detections` is empty

A 47012 failure must not disconnect 47010 video or 47011 control.

## Lifecycle independence

- Connect Video does not start inference.
- Start Inference remains the only action that starts inference.
- Stop Inference does not stop video.
- Detection stream disconnect does not stop inference.
- Video disconnect does not stop inference.
- Overlay rendering produces zero RobotController/RBRP writes.

## Compatibility

Old Qt + new Pi:

- old Qt ignores additional status fields and port 47012
- RBVS v1 continues unchanged

New Qt + old Pi:

- missing detection capability means no overlay connection
- no repeated 47012 connection attempts
- video/manual inference continue normally

New Qt + new Pi:

- latest live video remains realtime
- fresh frame-associated detection text is displayed when within the 1500 ms UI
  freshness window

## Explicit exclusions

- bbox reconstruction
- visual marker
- old-frame replay for overlay
- unbounded video/detection queues
- TargetTracker
- target selection/lock
- optical flow
- persistent target IDs
- PID
- visual servo
- RBRP changes
- STM32 changes
- actuator authority
- robot motion
- retraining
- threshold tuning
- second camera
- burned-in recording overlays
- modification of RBVS v1
- inference-result backlog
