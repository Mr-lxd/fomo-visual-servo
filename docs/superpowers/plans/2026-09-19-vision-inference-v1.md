# Vision Inference v1 — Authoritative Implementation Plan

This plan supersedes the previous implementation plan.

Execute it task-by-task using TDD.

Recommended execution mode:

```text
Subagent-Driven Development
```

Inline execution is acceptable only if subagents are unavailable.

Do not expand scope.

---

# 0. Repository / Worktree / Baseline

## fomo-visual-servo

Repository:

```text
Mr-lxd/fomo-visual-servo
```

Worktree:

```text
D:\DL_Project\fomo-visual-servo-worktrees\feature-vision-inference-v1
```

Branch:

```text
feature/vision-inference-v1
```

Reviewed design baseline:

```text
9c92788ba177ccb46284d3abff7096d442b607a0
```

Before implementation:

```powershell
git -C "D:\DL_Project\fomo-visual-servo-worktrees\feature-vision-inference-v1" fetch origin
git -C "D:\DL_Project\fomo-visual-servo-worktrees\feature-vision-inference-v1" rev-parse HEAD
git -C "D:\DL_Project\fomo-visual-servo-worktrees\feature-vision-inference-v1" status --short
```

The worktree must be clean except for this implementation-plan document if it has not yet been committed.

---

## RoboBeetle

Repository:

```text
Mr-lxd/RoboBeetle
```

Worktree:

```text
D:\RoboBeetle-worktrees\feature-vision-inference-v1
```

Branch:

```text
feature/vision-inference-v1
```

Expected starting baseline:

```text
9d860348639f154a5b5ed5f8dbbcd8f5fcfcc19c
```

Do not touch the original checkouts.

Forbidden Git operations:

```text
git clean
git reset --hard
force push
destructive rebase
push main
create PR
merge
```

---

# 1. Frozen Architecture

The architecture is:

```text
USB UVC
   ↓
CameraOwner
   ├── FrameHub [REALTIME_LATEST]
   │      ├── LiveStreamConsumer
   │      │       ↓
   │      │     RBVS 47010
   │      │       ↓
   │      │     Qt LIVE
   │      │
   │      └── InferenceWorker
   │              ↓
   │           BGR → RGB
   │              ↓
   │      OnnxRuntimePredictor.from_files()
   │              ↓
   │       existing NumPy postprocess
   │              ↓
   │       latest InferenceResult
   │
   └── CaptureManager [SEQUENTIAL_BOUNDED]
          via CameraOwner.frame_callback -> offer_frame()
```

HARD rules:

```text
CameraOwner remains the sole camera owner.

InferenceWorker consumes FrameHub directly.

InferenceWorker must not use CameraOwner.frame_callback.

InferenceWorker must not create a second frame queue.

InferenceWorker must not use LatestFrameReader.

InferenceWorker must not call cv2.VideoCapture.

CaptureManager remains on CameraOwner.frame_callback.

LIVE = REALTIME_LATEST.

Inference = REALTIME_LATEST.

Recording = SEQUENTIAL_BOUNDED.
```

---

# 2. Frozen Model Contract

Integrated inference must use:

```python
OnnxRuntimePredictor.from_files(onnx_path, report_path)
```

Do not construct a raw `onnxruntime.InferenceSession`.

Frozen identity:

```text
artifact:
d2_mobilenet_v2_fomo_seed42_epoch40

seed:
42

epoch:
40

confidence threshold:
0.40

ONNX SHA-256:
3dea74511bf2c44844192e75594fd53d4c4ce941f8b53b15767e020832bf9b08

input:
RGB float32
[1,3,192,192]
range [0,1]

output:
float32 raw logits
[1,8,24,24]

opset:
17
```

No threshold override may exist in `vision_live`.

No retraining, checkpoint selection, test-set access, TensorRT, TFLite or quantization.

---

# 3. TDD / Commit Rules

Every production change starts with a failing test.

Use:

```text
RED
→ minimal GREEN
→ focused regression
→ incremental commit
```

Keep the incremental task commits.

Do NOT create another duplicate “final implementation commit” at the end.

Final stage is:

```text
fresh verification
→ scope audit
→ clean worktree
→ push
→ exact SHA verification
```

not another implementation commit.

---

# Task 1 — InferenceWorker public contract

Create:

```text
src/fomo_servo/vision/inference_worker.py
tests/test_vision_inference_worker.py
```

Define:

```python
class InferenceState(str, Enum):
    DISABLED = "disabled"
    STARTING = "starting"
    RUNNING = "running"
    FAILED = "failed"
```

Define frozen:

```python
@dataclass(frozen=True)
class InferenceResult:
    frame_id: int
    capture_timestamp_ns: int
    inference_started_ns: int
    inference_finished_ns: int
    detections: tuple[Detection, ...]
```

Worker API:

```text
start()
stop()
status()
latest_result()
```

Initial status before `start()`:

```json
{
  "state": "disabled",
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
```

Run RED first.

Then implement only enough for GREEN.

Commit:

```text
test(vision): define inference worker contract
```

---

# Task 2 — Model initialization and frozen identity validation

Model initialization must happen inside the worker thread.

State:

```text
DISABLED
→ start()
→ STARTING
→ successful validation
→ RUNNING
```

Initialization exception:

```text
STARTING
→ FAILED
```

No automatic retry.

Use an injected predictor factory for tests.

The production default is:

```python
OnnxRuntimePredictor.from_files
```

## Required validator tests

Do NOT only test `epoch`.

Use parameterized mismatch tests covering at least:

```text
artifact_name
checkpoint_seed
checkpoint_epoch
confidence_threshold
onnx_sha256
input_shape
input_dtype
input_color_order
input_value_range
output_shape
output_dtype
output_semantic
opset
```

Each mismatch must prove:

```text
worker reaches FAILED
worker never reaches RUNNING
last_error identifies the mismatched contract field
```

Only threshold comparison may use:

```python
math.isclose(..., rel_tol=0.0, abs_tol=1e-9)
```

Everything else is exact.

After validation:

```python
cached = hub.snapshot()
after_frame_id = None if cached is None else cached.frame_id
```

Only then transition to `RUNNING`.

Cached frames accumulated during model initialization are not processed.

Commit:

```text
feat(vision): validate inference model contract
```

---

# Task 3 — BGR→RGB and source provenance

Write RED tests proving:

```text
FrameHub source is BGR.

Predictor receives RGB.

Original VisionFrame.image remains unchanged.
```

Use non-symmetric channel values, e.g.:

```text
BGR = [11,22,33]
RGB expected = [33,22,11]
```

Result must preserve:

```text
frame_id
capture_timestamp_ns
```

Worker adds:

```text
inference_started_ns
inference_finished_ns
```

Use existing:

```text
Detection
```

Do not create a second detection schema.

Processing order:

```python
frame = hub.wait_for_newer(...)
started = clock_ns()
rgb = cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB)
prediction = predictor.predict_rgb_image(rgb)
finished = clock_ns()
```

Do not pass a threshold argument to `predict_rgb_image()`.

Commit:

```text
feat(vision): publish frame-bound inference results
```

---

# Task 4 — REALTIME_LATEST and bounded metrics

Add a deterministic blocking predictor regression.

Required scenario:

```text
worker starts inference on frame 100

while blocked:
101 arrives
102 arrives
103 arrives

release frame 100 inference

next predictor input must be frame 103
```

It must NOT process:

```text
101
102
```

Expected:

```text
processed_frames = 2
skipped_frames = 2
```

No inference FIFO may exist.

Maintain only:

```text
latest_result
processed_frames
skipped_frames
bounded completion timestamp deque
last_error
validated model identity
```

`skipped_frames` counts gaps between successfully published inference results.

Do not count frames before the first successful result.

---

## latency_ms

Frozen definition:

```python
latency_ms = (
    inference_finished_ns - capture_timestamp_ns
) / 1_000_000.0
```

This is capture → completed inference latency.

Do not expose `processing_ms`.

---

## inference_fps

Use bounded completion timestamps.

For fewer than two timestamps:

```text
null
```

If:

```python
elapsed_seconds <= 0
```

return:

```text
null
```

Do not divide by zero.

Otherwise:

```python
(count - 1) / elapsed_seconds
```

---

# Task 5 — Inference generation reset / restart

Inference generation is independent of camera generation.

Do NOT restart CameraOwner merely because InferenceWorker restarts.

Required test:

```text
start worker
→ process result
→ stop worker
→ start same worker again
```

Immediately after new `start()` while state is STARTING:

```text
latest_result = None
processed_frames = 0
skipped_frames = 0
inference_fps = None
last_error = None
model identity = unset
```

During second model initialization:

```text
publish frame N
```

After validation, N is startup floor and must NOT be processed.

Then publish:

```text
N+1
```

Only N+1 is processed.

Also test FAILED restart explicitly:

```text
generation 1:
factory fails
→ FAILED
→ exactly one factory call

stop()

start()

generation 2:
factory called exactly one additional time
→ can reach RUNNING
```

No automatic retry within the same generation.

Commit:

```text
feat(vision): enforce latest-only inference lifecycle
```

---

# Task 6 — Failure isolation

Test:

### Initialization failure

```text
bad/missing sidecar/model
→ Inference FAILED
→ no automatic retry
```

### Runtime failure

```text
predict_rgb_image raises
→ Inference FAILED
```

A runtime failure must not clear an already published successful latest result from the same generation.

It must preserve:

```text
last_error
```

InferenceWorker must never:

```text
open camera
release camera
restart CameraOwner
restart VisionService
write disk
write network
encode JPEG
record video
open HighGUI
touch Qt
```

Commit:

```text
test(vision): isolate inference failures
```

---

# Task 7 — VisionService integration

Modify:

```text
src/fomo_servo/vision/service.py
src/fomo_servo/capture/control.py
tests/test_capture_control.py
tests/test_vision_capture_service.py
```

`VisionServiceConfig` adds:

```python
inference_onnx: Path | None = None
inference_report: Path | None = None
```

Contract:

```text
both absent
→ disabled

both supplied
→ inference enabled

exactly one supplied
→ ValueError before CameraOwner startup
```

With both paths present, construct exactly one InferenceWorker.

No worker when both absent.

---

## Startup order

```text
1. mode_manager.set_mode(LIVE)
2. inference_worker.start() if configured
3. control_server.start()
```

Asynchronous inference initialization failure must NOT roll back LIVE.

A synchronous failure starting HTTP/control infrastructure should retain existing rollback behavior.

---

## Shutdown order

Must remain:

```text
1. stop HTTP 47011 and wait for handlers
2. CaptureManager.shutdown()
3. InferenceWorker.stop()
4. VisionTcpServer stop
5. CameraOwner stop/release
6. FrameHub clear
```

The existing `VisionModeManager.shutdown()` may perform steps 4–6.

---

# Task 8 — 47011 inference status

Only modify existing:

```text
GET /api/v1/vision/status
```

Add:

```json
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
```

Do NOT add:

```text
47012
/api/v1/vision/inference/latest
full detections over HTTP
```

Model identity is populated only after successful contract validation.

Result-specific values stay null until the first successful result.

Explicitly test:

```text
GET /api/v1/vision/inference/latest
→ 404
```

Also test invalid inference artifact pair:

```text
CameraOwner still running
HTTP 47011 still running
Capture still usable
Inference FAILED
```

Commit:

```text
feat(vision): expose isolated inference status
```

---

# Task 9 — CLI configuration

Modify:

```text
scripts/vision_live.py
tests/test_vision_service.py
tests/test_bundle_launcher.py
```

Add only:

```text
--inference-onnx
--inference-report
```

Do not add:

```text
--confidence-threshold
--strategy
--tracking
```

Tests:

```text
neither provided → disabled
both provided → accepted
one provided → rejected before service/camera startup
```

`main()` should convert incomplete pair ValueError to `parser.error(...)`.

Commit:

```text
feat(vision): configure optional inference artifacts
```

---

# Task 10 — Pi-side targeted regression

Create:

```text
docs/vision_inference_v1.md
```

Document:

```text
formal artifact identity
formal SHA
both required artifact paths
DISABLED/STARTING/RUNNING/FAILED
REALTIME_LATEST
skipped_frames semantics
latency_ms formula
failure isolation
47011-only status
no control/overlay behavior
```

Run:

```powershell
python -m pytest -q `
  tests/test_vision_inference_worker.py `
  tests/test_vision_service.py `
  tests/test_capture_control.py `
  tests/test_vision_capture_service.py `
  tests/test_bundle_launcher.py `
  tests/test_onnx_runtime_predictor.py
```

All must pass.

Commit documentation separately:

```text
docs(vision): describe inference v1 deployment
```

---

# Task 11 — VisionControlClient TDD

Modify only:

```text
RoboBeetleConsole/src/vision/VisionControlClient.h
RoboBeetleConsole/src/vision/VisionControlClient.cpp
RoboBeetleConsole/tests/vision_control_client_tests.cpp
```

Extend existing `VisionCaptureStatus`.

Use explicit `have...` flags for nullable numeric values.

Must distinguish:

```text
JSON null
```

from:

```text
numeric zero
```

In particular:

```text
latest_frame_id = 0
```

is a valid result.

---

## Critical RequestKind rule

Current `applyPayload()` processes both:

```text
GET /api/v1/vision/status
```

and capture action responses:

```text
POST snapshot
POST recording/start
POST recording/stop
```

Capture action responses do NOT contain `inference`.

Therefore:

### GET status response

If `inference` is missing:

```text
treat as legacy Slice 2 server
reset inference fields to disabled defaults
```

### POST action response

If `inference` is missing:

```text
PRESERVE existing inference state
```

Do NOT reset it.

Implement by passing `RequestKind` into payload application or equivalent explicit logic.

Required regression:

```text
GET status says RUNNING
→ inference RUNNING

POST snapshot succeeds without inference object
→ inference remains RUNNING

later legacy GET status without inference
→ inference becomes DISABLED
```

Also test:

```text
null fields
frame ID zero
FAILED
legacy status
```

Do not add:

```text
new request type
new timer
new polling loop
new socket
new endpoint
```

Commit:

```text
feat(qt): parse vision inference diagnostics
```

---

# Task 12 — Qt diagnostics UI

Modify only:

```text
RoboBeetleConsole/src/ui/MainWindow.h
RoboBeetleConsole/src/ui/MainWindow.cpp
RoboBeetleConsole/tests/main_window_tests.cpp
```

Add two labels in existing Realtime Video card:

```text
inferenceState
inferenceDiagnostics
```

Example state text:

```text
Inference RUNNING
Inference STARTING
Inference Disabled
Inference Error
Inference Unavailable
```

Example diagnostics:

```text
d2_mobilenet_v2_fomo_seed42_epoch40
24.1 FPS
13.6 ms
Frame 1824
Det 2
Skip 423
```

Use `--` for unavailable nullable values.

FAILED should show short `last_error`.

Do not modify:

```text
VideoView
```

Do not add:

```text
overlay
bounding box
class label
tracking marker
target lock
target selection
```

Test that inference status updates produce zero robot-control transport writes.

Commit:

```text
feat(qt): show vision inference diagnostics
```

---

# Task 13 — Windows targeted verification

Configure the isolated worktree build if required:

```powershell
$env:PATH = "D:\Qt\Tools\mingw1310_64\bin;D:\Qt\Tools\Ninja;D:\Qt\6.11.2\mingw_64\bin;$env:PATH"

& 'D:\Qt\Tools\CMake_64\bin\cmake.exe' `
  -S "D:\RoboBeetle-worktrees\feature-vision-inference-v1\RoboBeetleConsole" `
  -B "D:\RoboBeetle-worktrees\feature-vision-inference-v1\RoboBeetleConsole\build\mingw-debug" `
  -G Ninja `
  -DBUILD_TESTING=ON `
  -DCMAKE_BUILD_TYPE=Debug `
  -DCMAKE_PREFIX_PATH='D:\Qt\6.11.2\mingw_64'
```

Build:

```powershell
& 'D:\Qt\Tools\CMake_64\bin\cmake.exe' --build `
  "D:\RoboBeetle-worktrees\feature-vision-inference-v1\RoboBeetleConsole\build\mingw-debug"
```

Run:

```powershell
& 'D:\Qt\Tools\CMake_64\bin\ctest.exe' `
  --test-dir "D:\RoboBeetle-worktrees\feature-vision-inference-v1\RoboBeetleConsole\build\mingw-debug" `
  --output-on-failure `
  -R '^(vision_control_client_tests|main_window_tests|vision_client_tests|video_view_tests)$'
```

Do not change CMakeLists unless an actual build dependency proves it necessary.

Current architecture should not require a CMake change.

---

# Task 14 — Scope and architecture audit

## fomo

Run:

```powershell
git -C "D:\DL_Project\fomo-visual-servo-worktrees\feature-vision-inference-v1" diff --check origin/main...HEAD

git -C "D:\DL_Project\fomo-visual-servo-worktrees\feature-vision-inference-v1" diff --stat origin/main...HEAD

git -C "D:\DL_Project\fomo-visual-servo-worktrees\feature-vision-inference-v1" diff --name-only origin/main...HEAD
```

Allowed implementation paths should be limited to:

```text
src/fomo_servo/vision/inference_worker.py
src/fomo_servo/vision/service.py
src/fomo_servo/capture/control.py
scripts/vision_live.py
tests/... relevant Vision tests
docs/vision_inference_v1.md
reviewed design/plan docs
```

No firmware / robot-control / training files.

---

## Forbidden integrated-runtime scans

Use only high-signal forbidden symbols:

```powershell
rg -n 'VideoCapture|LatestFrameReader|cv2\.VideoCapture' `
  "D:\DL_Project\fomo-visual-servo-worktrees\feature-vision-inference-v1\src\fomo_servo\vision\inference_worker.py"

rg -n '47012|/api/v1/vision/inference/latest|confidence-threshold|TargetTracker' `
  "D:\DL_Project\fomo-visual-servo-worktrees\feature-vision-inference-v1\src\fomo_servo\vision\inference_worker.py" `
  "D:\DL_Project\fomo-visual-servo-worktrees\feature-vision-inference-v1\src\fomo_servo\vision\service.py" `
  "D:\DL_Project\fomo-visual-servo-worktrees\feature-vision-inference-v1\src\fomo_servo\capture\control.py" `
  "D:\DL_Project\fomo-visual-servo-worktrees\feature-vision-inference-v1\scripts\vision_live.py"
```

These commands should produce no matches.

Do NOT use broad grep terms such as:

```text
RBRP
STM32
servo
```

as a machine pass/fail gate, because existing comments/docstrings may legitimately mention them.

Use changed-file scope audit instead.

---

## Required positive scan

```powershell
rg -n 'wait_for_newer|predict_rgb_image|cvtColor|BGR2RGB|skipped_frames|capture_timestamp_ns' `
  "D:\DL_Project\fomo-visual-servo-worktrees\feature-vision-inference-v1\src\fomo_servo\vision\inference_worker.py"
```

Must show:

```text
direct FrameHub consumption
BGR→RGB
existing predictor call
capture timestamp binding
latest-only metrics
```

---

## RoboBeetle

Run:

```powershell
git -C "D:\RoboBeetle-worktrees\feature-vision-inference-v1" diff --check origin/main...HEAD

git -C "D:\RoboBeetle-worktrees\feature-vision-inference-v1" diff --stat origin/main...HEAD

git -C "D:\RoboBeetle-worktrees\feature-vision-inference-v1" diff --name-only origin/main...HEAD
```

Allowed paths:

```text
VisionControlClient.h/.cpp
MainWindow.h/.cpp
vision_control_client_tests.cpp
main_window_tests.cpp
```

No:

```text
RoboBeetleFirmware
RoboBeetlePi
RBRP
actuator
heartbeat
protocol command implementation
```

changes.

---

# Task 15 — Fresh final verification

Do not rely on earlier test runs.

## fomo

Run fresh:

```powershell
python -m pytest -q `
  tests/test_vision_inference_worker.py `
  tests/test_vision_service.py `
  tests/test_capture_control.py `
  tests/test_vision_capture_service.py `
  tests/test_bundle_launcher.py `
  tests/test_onnx_runtime_predictor.py
```

Record exact pass count.

---

## RoboBeetle

Fresh build:

```powershell
& 'D:\Qt\Tools\CMake_64\bin\cmake.exe' --build `
  "D:\RoboBeetle-worktrees\feature-vision-inference-v1\RoboBeetleConsole\build\mingw-debug"
```

Fresh targeted tests:

```powershell
& 'D:\Qt\Tools\CMake_64\bin\ctest.exe' `
  --test-dir "D:\RoboBeetle-worktrees\feature-vision-inference-v1\RoboBeetleConsole\build\mingw-debug" `
  --output-on-failure `
  -R '^(vision_control_client_tests|main_window_tests|vision_client_tests|video_view_tests)$'
```

Record exact pass count.

---

# Task 16 — Final Git state and push

There is NO new “final implementation commit”.

All production changes should already be represented by the incremental TDD commits above.

Before push:

```powershell
git -C "D:\DL_Project\fomo-visual-servo-worktrees\feature-vision-inference-v1" status --short
git -C "D:\RoboBeetle-worktrees\feature-vision-inference-v1" status --short
```

Both must be clean.

Then:

```powershell
git -C "D:\DL_Project\fomo-visual-servo-worktrees\feature-vision-inference-v1" push -u origin feature/vision-inference-v1

git -C "D:\RoboBeetle-worktrees\feature-vision-inference-v1" push -u origin feature/vision-inference-v1
```

Verify:

```powershell
git -C "D:\DL_Project\fomo-visual-servo-worktrees\feature-vision-inference-v1" rev-parse HEAD
git -C "D:\DL_Project\fomo-visual-servo-worktrees\feature-vision-inference-v1" ls-remote origin refs/heads/feature-vision-inference-v1

git -C "D:\RoboBeetle-worktrees\feature-vision-inference-v1" rev-parse HEAD
git -C "D:\RoboBeetle-worktrees\feature-vision-inference-v1" ls-remote origin refs/heads/feature-vision-inference-v1
```

Local and remote SHA must match for each repository.

Do NOT create PR.

---

# Final report required from Codex

Return exactly:

```text
1. Baseline verification

2. Execution mode used
   - Subagent-Driven Development or Inline

3. TDD commits
   - fomo commit list
   - RoboBeetle commit list

4. Changed files
   - fomo
   - RoboBeetle

5. Architecture compliance
   - CameraOwner sole owner
   - CaptureManager remains callback branch
   - InferenceWorker direct FrameHub consumer
   - no inference FIFO
   - BGR→RGB
   - source frame provenance
   - capture→completion latency
   - no threshold override
   - failure isolation
   - inference-generation restart independent of camera lifecycle

6. Qt compatibility
   - legacy GET status handling
   - action response preserves inference status
   - null vs zero semantics
   - no new polling/socket/endpoint
   - no overlay

7. Tests run
   - exact commands
   - exact counts/results

8. git diff --check
   - both repos

9. Scope audit
   - changed-file list
   - forbidden scan result

10. Exact HEAD SHA
    - fomo
    - RoboBeetle

11. Exact remote SHA
    - fomo
    - RoboBeetle

12. Worktree status

13. Known limitations

14. Explicit confirmation
    - no PR created
    - no main modification
    - no force push
    - no hardware action
    - no camera/wiring/Qt manual acceptance performed
```

---

# Stop conditions

STOP and report instead of inventing a workaround if:

```text
origin/main unexpectedly changes in a way that conflicts with the frozen design

the formal OrtModelContract differs from the reviewed contract

current service architecture no longer has CameraOwner as sole owner

implementing inference would require changing RBRP/STM32/actuator authority

a second camera path appears necessary

a new inference FIFO appears necessary

full detections over HTTP become necessary

Qt overlay becomes necessary

the requested implementation cannot preserve CaptureManager QoS
```

Do not silently widen scope.

---

# Final frozen exclusions

Still OUT OF SCOPE:

```text
retraining
checkpoint reselection
test split access
threshold tuning
TensorRT
TFLite
quantization
new camera owner
LatestFrameReader integration
inference FIFO
new TCP port
47012
/inference/latest
full detection HTTP protocol
inference MP4/CSV/JSONL
VNC preview
TargetTracker
target selection
Qt overlays
boxes
class labels
tracking
target lock
PID
visual servo
RBRP changes
STM32 changes
actuator authority
robot motion
```
