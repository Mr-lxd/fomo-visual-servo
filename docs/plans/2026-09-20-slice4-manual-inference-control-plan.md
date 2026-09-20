# Slice 4 Task 01 — Manual Inference Control Implementation Plan

> **For implementation:** follow the frozen Task 01 contract step-by-step;
> tests are written and observed failing before their production counterpart.

**Goal:** Add manual Start/Stop/Retry lifecycle coordination for the existing
Pi Vision inference capability and expose it through the existing 47011 HTTP
control plane.

**Architecture:** `InferenceWorker` remains the sole predictor/frame consumer.
`InferenceControl` owns service-level action admission, operation metadata, and
one lazy coordinator thread. `VisionService` composes them and
`VisionControlServer` maps validated HTTP requests to callbacks. GET status is
the only authoritative state surface.

**Tech stack:** Python 3.10 project runtime, standard-library `threading` and
`http.server`, existing pytest suite, fake camera/predictor, localhost HTTP.

**Frozen source:** `Slice4_Codex_Task01_Frozen_Pack.md`, sections 4–7.
**Required baseline:** `3f82ee6274c03be1c6ff703971333e0e4cc919a2`.
**Status:** Frozen execution plan; software evidence is reported separately in
the Task 01 review report; hardware acceptance remains pending.

---

## Scope guard

Allowed production files are:

- `src/fomo_servo/vision/inference_worker.py`
- `src/fomo_servo/vision/inference_control.py` (new)
- `src/fomo_servo/vision/service.py`
- `src/fomo_servo/capture/control.py`
- `scripts/vision_live.py` only if a lifecycle/help-text clarification proves
  necessary; no CLI expansion is planned.

Allowed test files are the two new inference-control suites plus the four
frozen existing Vision/capture suites. No model, capture manager, camera owner,
streaming, deployment, dependency, Qt, or RoboBeetle file is in scope. This
task ends with an uncommitted review diff; it does not commit, push, open a PR,
merge, deploy, or access hardware.

## Task sequence

### Task 0: Preflight and repository isolation

**Files:** no repository content change.

- [ ] Verify the remote, working-tree state, applicable `AGENTS.md`, worktree
  registration, `git fetch origin`, and the exact `origin/main` baseline.
- [ ] Verify the named branch/worktree path are unused before creating
  `codex/slice4-manual-inference-control` at the frozen SHA in the requested
  sibling worktree.
- [ ] Recheck the new worktree HEAD, remote, and clean status before edits.

### Task 1: Contract records and RED worker/service tests

**Files:**

- Create: `docs/designs/2026-09-20-slice4-manual-inference-control-design.md`
- Create: `docs/plans/2026-09-20-slice4-manual-inference-control-plan.md`
- Modify: `tests/test_vision_inference_worker.py`
- Modify: `tests/test_vision_capture_service.py`

- [ ] Record the frozen startup, state, HTTP, concurrency, shutdown, and
  exclusion contracts without declaring implementation or hardware success.
- [ ] Add tests for non-joining `request_stop`, post-join-only
  `reset_disabled`, thread-start rollback, configured service disabled after
  `start_live`, and no predictor invocation before manual Start.
- [ ] Run the focused tests and retain the feature-missing failures as RED
  evidence. A missing interpreter or package is an environment blocker, not
  RED evidence.

### Task 2: Worker lifecycle primitives

**Files:**

- Modify: `src/fomo_servo/vision/inference_worker.py`
- Test: `tests/test_vision_inference_worker.py`

- [ ] Implement `request_stop()` under the worker lock without joining,
  releasing handles, or clearing diagnostics.
- [ ] Make `reset_disabled()` reject any state where `stop()` has not released
  the exact worker thread handle; on success clear only the frozen
  worker/inference diagnostics and notify waiters.
- [ ] Make `start()` recover from `Thread.start()` failure with `failed`, an
  explicit error, and no unstarted thread handle.
- [ ] Run the worker-focused suite after each red/green increment.

### Task 3: Service-level inference facade

**Files:**

- Create: `src/fomo_servo/vision/inference_control.py`
- Create: `tests/test_vision_inference_control.py`

- [ ] Write deterministic fake-worker tests before facade code for every
  admission-table case, one lazy coordinator, duplicate behavior, retry,
  shutdown race, lifecycle fault latch, and non-blocking status behavior.
- [ ] Implement frozen `InferenceActionResult`, `InferenceControlError`, and
  `InferenceControl` public APIs. Use `RLock`/`Condition`, facade-lock then
  worker-lock ordering, one operation slot, and no busy polling.
- [ ] Keep joins outside the facade lock. A failed cleanup retains its operation
  and exposes a high-priority lifecycle error until shutdown.
- [ ] Run `tests/test_vision_inference_control.py` through the complete
  red/green sequence.

### Task 4: VisionService composition

**Files:**

- Modify: `src/fomo_servo/vision/service.py`
- Modify: `tests/test_vision_capture_service.py`

- [ ] Replace auto-start with facade construction and callback wrappers.
- [ ] Open actions only after `VisionMode.LIVE` succeeds and before the control
  listener starts; close them and clean up on startup rollback.
- [ ] Route status through the facade and preserve configured/disabled status
  without validated model identity.
- [ ] Apply the frozen shutdown ordering: close admission, stop HTTP, shut down
  capture, finish facade/worker coordination, then stop mode/camera. Preserve
  the first cleanup error while attempting all later cleanup.
- [ ] Run service and worker focused tests with fake camera and predictor only.

### Task 5: HTTP inference actions

**Files:**

- Modify: `src/fomo_servo/capture/control.py`
- Create: `tests/test_vision_inference_control_http.py`
- Modify: `tests/test_capture_control.py`

- [ ] Write localhost tests before routing code for both ACK routes, full body
  validation, HTTP mapping, standalone-server fallback, status increment,
  capture compatibility, and no additional port/endpoint.
- [ ] Add optional callbacks without importing `VisionService` or accessing
  worker private state.
- [ ] Enforce the new-route-only 1024-byte / 2-second request-body boundary.
  Accepted responses contain only `ok`, `action`, and `outcome`.
- [ ] Re-run the HTTP and legacy capture-control suites.

### Task 6: Regression and review pack

**Files:** only the approved source/test/docs changes above.

- [ ] Run the Task 01 core gate, isolated Vision regression gate, Pi import /
  launcher gate, and optional training-environment gate only if preinstalled
  dependencies make it available.
- [ ] Run `git diff --check`, inspect status, and generate an untracked-aware
  full review diff containing every new file.
- [ ] Report test counts, exit codes, P01–P30 mapping, concurrency design,
  representative HTTP responses, unexecuted hardware acceptance, and any
  BLOCKER / SHOULD FIX / NIT. Stop for Reviewer without committing.

## Acceptance-test mapping

The following are the contract test anchors. A test may cover closely related
assertions only where the frozen matrix explicitly requires one atomic
scenario.

| ID | Planned test anchor |
| --- | --- |
| P01 | `test_configured_service_stays_disabled_until_http_start_and_factory_runs_once` |
| P02 | `test_unconfigured_service_http_actions_keep_camera_and_capture_available` |
| P03 | `test_configured_service_stays_disabled_until_http_start_and_factory_runs_once` |
| P04 | `test_configured_service_stays_disabled_until_http_start_and_factory_runs_once` |
| P05 | `test_http_stop_returns_before_blocked_prediction_and_preserves_capture_state` |
| P06 | `test_stopping_is_async_busy_and_uses_one_non_daemon_coordinator` |
| P07 | `test_http_stop_returns_before_blocked_prediction_and_preserves_capture_state`; `test_worker_discards_result_when_stop_requested_before_publish` |
| P08 | `test_reset_disabled_requires_stop_to_release_handle_then_clears_metrics` |
| P09 | `test_stop_during_factory_initialization_never_publishes_running` |
| P10 | `test_failed_worker_has_no_background_retry` |
| P11 | `test_failed_start_retries_after_old_generation_join_once` |
| P12 | `test_failed_start_retries_after_old_generation_join_once` |
| P13 | `test_failed_clear_error_stops_to_clean_disabled_without_losing_configuration` |
| P14 | `test_ten_controlled_start_stop_cycles_have_no_generation_result_or_counter_leak` |
| P15 | `test_invalid_inference_artifact_fails_only_after_manual_start_and_keeps_live_service` |
| P16 | `test_http_stop_returns_before_blocked_prediction_and_preserves_capture_state` |
| P17 | `test_http_stop_returns_before_blocked_prediction_and_preserves_capture_state` |
| P18 | `test_http_stop_returns_before_blocked_prediction_and_preserves_capture_state` |
| P19 | `test_concurrent_start_stop_retry_never_admits_more_than_one_operation` |
| P20 | `test_begin_shutdown_prevents_retry_from_starting_after_cleanup` |
| P21 | `test_shutdown_reclaims_worker_in_disabled_starting_running_and_failed`; `test_shutdown_reclaims_worker_while_stopping_without_starting_new_generation` |
| P22 | `test_status_remains_available_while_coordinator_waits_for_stop_join`; `test_get_status_runs_while_a_separate_http_action_handler_is_active` |
| P23 | `test_thread_start_failure_rolls_back_false_starting_state`; `test_coordinator_thread_start_failure_rolls_back_without_stop_signal` |
| P24 | `test_cleanup_failure_latches_error_and_blocks_all_new_actions` |
| P25 | `test_retry_camera_loss_does_not_start_and_waits_for_new_manual_action` |
| P26 | `test_inference_routes_accept_only_empty_body_or_empty_json_object`; `test_inference_route_rejects_nonempty_or_nonobject_json`; `test_inference_route_rejects_oversize_and_invalid_framing_before_callback`; `test_inference_route_rejects_truncated_or_timed_out_body` |
| P27 | `test_inference_post_is_only_an_ack_and_get_status_is_authoritative` |
| P28 | `test_inference_body_validation_does_not_change_existing_capture_post_shape`; `test_latest_inference_endpoint_remains_not_found` |
| P29 | `test_reset_disabled_requires_stop_to_release_handle_then_clears_metrics` |
| P30 | `test_status_remains_available_while_coordinator_waits_for_stop_join`; `test_get_status_runs_while_a_separate_http_action_handler_is_active` |

## Required commands

Run commands one line at a time using the existing repository Python runtime
and report `sys.executable` first:

```text
python -m pytest -q tests/test_vision_inference_worker.py tests/test_vision_inference_control.py tests/test_vision_inference_control_http.py tests/test_vision_capture_service.py tests/test_capture_control.py tests/test_vision_service.py
python -m pytest -q tests/test_capture_manager.py tests/test_vision_core.py tests/test_vision_mode.py tests/test_vision_protocol.py tests/test_vision_streaming.py
python -m pytest -q tests/test_imports.py::test_vision_ort_runtime_import_closure_does_not_require_yaml_or_torch tests/test_imports.py::test_ort_numpy_postprocess_modules_never_probe_torch_on_import tests/test_bundle_launcher.py
python -m pytest -q tests/test_imports.py tests/test_onnx_runtime_predictor.py
git diff --check
```

The final command involving the complete imports/predictor suite is conditional
on the preexisting training/export dependencies. No dependency installation,
hardware camera, Pi, serial port, or robot may be used to make a test pass.
