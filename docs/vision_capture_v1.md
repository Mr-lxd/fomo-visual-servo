# Vision Capture v1 — Snapshot and Dataset Recording

Vision Capture v1 extends the already-validated LIVE Vision plane with
source-frame snapshots and bounded sequential dataset recording.

It preserves the Slice 1 ownership rule:

    USB UVC camera
        -> CameraOwner (the only /dev/video0 owner)
            -> FrameHub -> RBVS LIVE
            -> CaptureManager -> snapshot / recording

Snapshot and recording never open the camera themselves and never consume
decoded Qt screenshots.

## Scope

Included:

- source-resolution JPEG snapshot;
- source-resolution MJPG/AVI recording;
- per-video-frame frame_id / capture_timestamp_ns index;
- session metadata and non-overwriting session allocation;
- bounded sequential recording queue;
- runtime free-disk checks;
- independent HTTP/JSON Vision Capture control;
- Qt Snapshot / Start Recording / Stop Recording controls and diagnostics.

Excluded:

- FOMO inference, detection, tracking, target selection, visual servo;
- dataset annotation or automatic labeling;
- ROS2 and systemd;
- STM32, RBRP, gateway, authority, heartbeat, or actuator changes.
## Network separation

The three planes have independent responsibilities:

| Port | Protocol | Purpose |
| ---: | --- | --- |
| 47000 | RBRP v1 / TCP | Robot control through the Pi gateway |
| 47010 | RBVS v1 / TCP | Server-to-client realtime JPEG video |
| 47011 | HTTP/JSON | Low-rate Vision Capture control and status |

RBVS v1 remains server-to-client only. Capture commands are not inserted into
RBVS headers or payloads, and no capture action is encoded as an RBRP command.

The Qt VisionControlClient uses QNetworkProxy::NoProxy, matching the direct
engineering-LAN behavior of the existing RBRP and RBVS clients.

Vision Capture control v1 has no authentication or TLS. It is intended only for
the robot's trusted engineering LAN / direct-link environment and must not be
exposed directly to an untrusted network.

## State model

Camera generation state remains:

    OFF <-> LIVE

Capture is intentionally orthogonal:

    IDLE -> RECORDING -> STOPPING -> IDLE
                         |
                         +-> FAILED

A recording start/stop does not restart CameraOwner, clear FrameHub, change an
RBVS connection generation, acquire robot authority, or send actuator commands.

STOPPING means no new frames are accepted while already-enqueued sequential
frames are being finalized. FAILED means the current recording segment is no
longer accepted as a complete nominal recording.
## Control API

All responses are UTF-8 JSON. Normal actions return HTTP 200 with "ok": true.
A capture precondition failure returns HTTP 409 with "ok": false and an
"error" string.

### Status

    GET /api/v1/vision/status

Representative response:

    {
      "ok": true,
      "camera": {
        "running": true,
        "latest_frame_id": 123,
        "capture_timestamp_ns": 456,
        "observed_width": 640,
        "observed_height": 480,
        "observed_fps": 25.0,
        "measured_capture_fps": 13.75
      },
      "capture": {
        "state": "idle",
        "recording": false,
        "session_id": null,
        "session_dir": null,
        "segment": null,
        "recorded_frames": 0,
        "snapshot_count": 0,
        "queue_frames": 0,
        "queue_bytes": 0,
        "max_queue_bytes": 67108864,
        "free_disk_bytes": null,
        "last_error": null,
        "first_dropped_frame_id": null
      }
    }
### Snapshot

    POST /api/v1/vision/snapshot

The server records the FrameHub frame_id that exists when the request begins
and waits for a strictly newer camera frame. It therefore does not intentionally
save a frame cached before the snapshot request.

The saved JPEG uses the CameraOwner source image and source dimensions. The
snapshot filename includes both a session-local snapshot sequence and the
CameraOwner frame_id.

### Recording

    POST /api/v1/vision/recording/start
    POST /api/v1/vision/recording/stop

Start creates the next recording segment inside the current capture session.
OpenCV requires a positive fixed FPS when opening the MJPG/AVI writer, so the
writer initially uses the camera/driver-reported observed_fps. The recent
CameraOwner measured_capture_fps remains a runtime diagnostic only and is not a
prerequisite for starting recording.

Stop first prevents new enqueues and drains the bounded sequential queue. After
VideoWriter closes, the finalized AVI container frame count is checked against
the indexed frame_count. A mismatch or an unreadable finalized AVI marks the
segment FAILED instead of publishing a false successful dataset. Only a
successfully released writer with a matching frame count is eligible for timing
finalization.

The segment's own first/last capture_timestamp_ns values are then used to
compute final actual_capture_fps. The AVI avih and video strh timing headers are
rewritten in place to that final average cadence. Frame payloads are not
re-encoded, dropped, duplicated, or reordered by this finalization. Ordinary
I/O exceptions during the small header rewrite trigger a best-effort restoration
of the original timing fields; this does not claim power-loss transaction
atomicity. Only after finalization is complete are metadata and the completed
state published.

The Qt client polls status approximately once per second. If a user action is
requested while only a status poll is in flight, one action is queued and sent
immediately after that status response. User actions are not silently discarded.

Changing the capture endpoint increments an endpoint generation, cancels any
queued action for the previous endpoint, may abort an old status GET, and
ignores late status replies from that old generation. Once Snapshot, Start
Recording, or Stop Recording has actually been sent, the endpoint is held
stable until that action receives a definite response or timeout because
aborting the local QNetworkReply cannot undo a POST already executed by the
server. MainWindow close is likewise deferred while such an action is in flight.
Stop Recording uses a longer client timeout than status/snapshot/start so the
server can legally drain and finalize the bounded recording queue.

RBVS connection loss does not stop capture-control polling and does not disable
Stop Recording. A recording remains stoppable through 47011 even if the
realtime video connection on 47010 is unavailable.
## Recorder QoS and backpressure

LIVE and dataset recording intentionally use different QoS.

LIVE is REALTIME_LATEST:

- FrameHub contains one replaceable latest raw frame;
- RBVS may skip intermediate frame IDs;
- low latency is preferred over completeness.

Recording is SEQUENTIAL_BOUNDED:

- CameraOwner calls CaptureManager.offer_frame() after publishing to FrameHub;
- the callback performs no file I/O;
- while recording, it copies the source image and appends it to a memory-bounded
  FIFO queue;
- one recorder worker writes the queued frames to MJPG/AVI in FIFO order;
- every written video frame gets one frame_index.csv row.

The default raw-frame queue budget is 64 MiB.

If the queue cannot accept a source frame:

1. that frame_id becomes first_dropped_frame_id;
2. new frame acceptance stops;
3. state becomes STOPPING;
4. the already-enqueued contiguous prefix is drained and finalized;
5. the segment is marked failed;
6. final state becomes FAILED.

The implementation never converts queue pressure into an unreported frame drop
and never allows an unbounded recording backlog.
## Disk safety and write failures

Before a snapshot or recording starts, available space must be at least the
configured reserve. The default reserve is 512 MiB.

During recording the worker rechecks free disk space every 25 written frames.
If the reserve is crossed, the current not-yet-written frame is recorded as the
first dropped frame, queued data is discarded, the writer/index are closed, and
the segment becomes FAILED with a low_disk_space diagnostic.

The same explicit failure model is used for recorder exceptions, finalized AVI
frame-count mismatches, metadata persistence failures, and file/index write
failures. Start Recording is transactional: if writer creation, frame-index
creation, initial metadata persistence, or worker startup fails, the partially
opened resources are released, newly created segment/index files are removed
when safe, and metadata is returned to the rolled-back state. If any rollback
step itself fails, the cleanup errors are aggregated, the manager enters FAILED,
and the control request reports the incomplete rollback instead of pretending
that an idle rollback succeeded.

If the recorder's final metadata write fails after the segment has already been
finalized in memory, the segment/manager are marked FAILED and the write is
retried once. A later Stop Recording or clean shutdown also retries the
authoritative metadata write even when the recorder thread has already exited.

Snapshot follows the same persistence rule after its JPEG and in-memory record
exist: metadata persistence is retried once immediately. If both attempts fail,
the snapshot record and JPEG are retained for recovery, last_error explicitly
reports snapshot_metadata_persist_failure, and an idle manager enters FAILED.
If recording is active, new recording frames stop being accepted and the
existing recorder failure/finalization path completes the segment as FAILED.
A later Stop Recording or clean shutdown retries the authoritative metadata
write so the retained snapshot can be indexed without deleting or rewriting
the JPEG.

free_disk_bytes, queue bytes, recording state, frame count, snapshot count,
and the most recent error are exposed through the status API and Qt UI.

## Shutdown ordering

The foreground Vision service shuts down in this order:

1. stop the HTTP control server and wait for in-flight handlers;
2. finalize CaptureManager / open recording;
3. stop the RBVS server;
4. stop CameraOwner and release /dev/video0;
5. clear FrameHub at the camera-generation boundary.

CaptureManager does not own or release the camera.
## Dataset layout

Default root:

    datasets_raw/robobeetle/

One runtime lazily creates one capture session when the first snapshot or
recording is requested:

    datasets_raw/robobeetle/
      YYYYMMDD/
        capture-YYYYMMDD-NNN/
          raw.avi
          raw_002.avi
          ...
          frame_index.csv
          metadata.json
          frames/
            snapshot_000001_frame_00000000000000000123.jpg
            ...

Session indices use max(existing NNN)+1 and are never deliberately reused or
overwritten. Slice 2 assumes one VisionService/CaptureManager owns a given
capture root; coordinating simultaneous independent processes that allocate the
same root is outside this slice.

raw.avi and later segments:

- container: AVI;
- codec: MJPG;
- dimensions: CameraOwner observed source dimensions;
- camera reported FPS: retained from CAP_PROP_FPS and used only as the
  writer's initial fixed AVI time base;
- measured capture FPS: recent CameraOwner delivered cadence exposed as
  diagnostics only;
- finalized video frame count must match the indexed frame count before a
  segment can be completed;
- final actual capture FPS: recomputed from first/last Pi monotonic timestamps;
- final container FPS: AVI timing headers rewritten to final actual capture FPS;
- input: source BGR frame;
- no resize, overlay, bounding box, HUD, inference preprocessing, or color
  normalization is applied by the capture subsystem.

frame_index.csv columns:

    segment
    segment_frame_index
    frame_id
    capture_timestamp_ns

The index is flushed periodically and again when the segment closes.
## Metadata

metadata.json is rewritten atomically through a temporary file and includes:

- schema version and session ID;
- session start/update time;
- observed camera source, dimensions, driver-reported FPS and FourCC;
- per-segment writer initial FPS, finalized container FPS, final actual
  capture FPS, capture elapsed time, finalized AVI frame count / verification,
  timing-header rewrite status, and timing error ratio;
- recording state;
- queue budget/current use;
- latest measured free disk space and configured reserve;
- last error and first dropped frame ID;
- finalized recording segment entries;
- the current segment while active;
- snapshot entries with source frame ID and capture timestamp.

A clean service shutdown sets "closed": true.

## Defaults

Current Slice 2 defaults:

| Setting | Default |
| --- | ---: |
| RBVS stream port | 47010 |
| Capture control port | 47011 |
| Camera target | 640x480 YUYV @ 25 FPS |
| LIVE JPEG quality | 80 |
| Snapshot JPEG quality | 95 |
| Recording codec/container | MJPG / AVI |
| Recording queue budget | 64 MiB |
| Minimum free disk reserve | 512 MiB |
| Capture root | datasets_raw/robobeetle |

Additional vision_live CLI arguments:

    --control-port 47011
    --capture-output-root datasets_raw/robobeetle
    --capture-queue-mib 64
    --capture-min-free-mib 512
## Hardware Acceptance status

Software tests do not constitute Slice 2 hardware acceptance.

The real Raspberry Pi, UVC camera, Windows Qt console, storage device, and robot
control link must still verify:

1. **Capture A — startup / ownership**
   - one CameraOwner opens /dev/video0;
   - both 47010 and 47011 listen;
   - capture controls do not create a second camera owner.

2. **Capture B — snapshot**
   - Snapshot saves a fresh source-resolution JPEG;
   - filename/metadata frame_id matches a frame newer than the request floor;
   - LIVE video remains current.

3. **Capture C — nominal recording**
   - record at least 20–30 seconds at the negotiated camera profile;
   - Stop closes a playable MJPG/AVI file;
   - frame_index.csv row count matches the video-frame count;
   - frame IDs are strictly sequential for a nominal non-overloaded run;
   - container FPS is close to timestamp-derived actual capture FPS;
   - container duration is close to timestamp-derived capture duration;
   - metadata is finalized and free-space diagnostics are credible.

4. **Capture D — LIVE + recording coexistence**
   - LIVE remains latest-frame oriented while recording;
   - recording queue remains bounded and does not grow without limit;
   - starting/stopping recording does not restart CameraOwner or RBVS.

5. **Capture E — video loss while recording**
   - disconnect RBVS while recording;
   - capture control remains usable and Stop Recording succeeds;
   - reconnect shows current video, not replayed history.

6. **Capture F — robot-control isolation**
   - RBRP Connect/Acquire/telemetry/DisableServos/Release remain functional
     while snapshotting and recording;
   - capture actions cause no robot-control command or authority transition.

7. **Capture G — shutdown / restart**
   - Ctrl-C or application shutdown finalizes open files;
   - /dev/video0, 47010, and 47011 are released;
   - restart creates/reuses no existing session directory incorrectly.

Do not mark these gates PASS until observed on the actual hardware.
