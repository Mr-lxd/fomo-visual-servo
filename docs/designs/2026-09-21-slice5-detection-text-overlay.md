# Vision Slice 5 — Frame-Synchronized Detection Text Overlay

## Goal

Expose FOMO inference detections to the Qt operator console with exact source-frame
identity. The first Slice 5 UI renders **text only** near each detection centroid:

`<class_name> <confidence>`

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
reconstruct or imply a true bbox.

## Network boundaries

- `47010`: RBVS v1 JPEG live video, unchanged
- `47011`: HTTP status/capture/manual inference, existing routes unchanged
- `47012`: new Slice 5 detection metadata stream

RBVS v1 is not modified, so old Qt builds remain compatible.

## Capability advertisement

The `inference` object returned by `GET /api/v1/vision/status` adds:

- `detection_stream_supported: bool`
- `detection_stream_port: int`
- `detection_stream_version: int`

A new Qt client connects to 47012 only after a fresh authoritative status confirms
support. Missing fields mean overlay unavailable, not an error.

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

Empty detections is valid and is transmitted so the client can clear labels for
that processed frame.

All numbers must be finite and coordinates must be inside the source image.

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

## Exact-frame Qt synchronization

The metadata stream is latest-only, but rendering is **exact-frame only**.

Qt must never transfer a detection result onto a different video frame.

Planned Qt flow:

1. `VisionClient` keeps a bounded cache of recently decoded RBVS frames keyed by
   `frame_id`.
2. `DetectionClient` validates each 47012 result.
3. While inference is Running, a result is renderable only when its exact
   `frame_id` exists in the frame cache.
4. The matching cached frame becomes the inference-synchronized displayed frame.
5. Text is rendered near the result centroid on that exact frame.
6. If the frame has already fallen out of the cache, the metadata is dropped and
   counted as a sync miss.
7. A zero-detection result displays its exact matching frame with no old label.
8. When inference stops/fails/disconnects, display returns to normal latest-frame
   RBVS video.

This intentionally allows inference mode to trail the raw live stream by the model
latency. It does **not** freeze waiting for future results and does not create an
unbounded image queue.

There is no time-window fallback. A nearby timestamp or newer frame is not a valid
substitute for an exact `frame_id` match.

## Bounded frame cache

Qt must use a fixed-capacity recent-frame cache. It may replace old entries but
must never grow without bound.

The exact capacity is selected in Phase B and covered by tests. Host switch, video
disconnect, endpoint generation change, or service reset clears the cache.

## Text-only rendering contract

Allowed:

- `creature 0.87`
- readable foreground text
- a small text shadow/outline for contrast
- offsetting/clamping the text so it stays inside the image rectangle

Not allowed:

- rectangle/bbox
- dot/circle
- crosshair
- centroid glyph
- fixed-size marker
- target lock indicator
- tracking trail
- servo/PID guidance

The source anchor always comes from `original_x/original_y` of the exact matched
frame.

## Lifecycle

- Connect Video does not start inference.
- Start Inference remains the only action that starts the worker.
- Stop Inference does not stop video.
- Detection stream disconnect does not stop inference.
- Video disconnect does not stop inference.
- Inference Disabled/Failed, stale HTTP status, host switch, or detection
  disconnect clears/suppresses overlay state.
- Overlay rendering produces zero RobotController/RBRP writes.

## Compatibility

Old Qt + new Pi:

- old Qt ignores extra status fields and 47012
- RBVS v1 continues unchanged

New Qt + old Pi:

- missing capability means no overlay connection
- no repeated 47012 connection attempts
- video/manual inference continue normally

New Qt + new Pi:

- exact `frame_id` synchronized text overlay

A 47012 failure must not disconnect 47010 video or 47011 control.

## Explicit exclusions

- bbox reconstruction
- visual marker
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
- inference result backlog
