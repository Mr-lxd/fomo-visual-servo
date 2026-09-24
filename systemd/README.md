# RoboBeetle Vision user systemd deployment

This directory records the Raspberry Pi Vision LIVE deployment that was
hardware-verified on 2026-09-24.

The service keeps camera streaming and the Vision HTTP/detection endpoints
available after boot. Inference remains operator-controlled: starting the
service does **not** automatically start the ONNX inference worker.

## Verified runtime layout

The verified Raspberry Pi 4 layout is:

```text
~/fomo-vision-runtime/
    clean source checkout used by the service
    artifacts -> ~/fomo-vision-data/artifacts
    datasets_raw -> ~/fomo-vision-data/datasets_raw

~/fomo-vision-data/
    artifacts/
    datasets_raw/

~/venvs/fomo-ort-d2-epoch40/
    bin/python
```

The source checkout and mutable/model data are intentionally separated. The
runtime checkout can therefore remain clean while model artifacts and capture
data persist independently.

The 2026-09-24 verified source baseline was:

```text
2da2ee97ee490aa276d8fba0a12e9e84314de181
```

For a reproducible redeploy, check out that commit. An intentional upgrade to a
newer `main` should be verified before replacing the active runtime.

## Prepare a clean runtime checkout

Example:

```bash
git clone https://github.com/Mr-lxd/fomo-visual-servo.git ~/fomo-vision-runtime
cd ~/fomo-vision-runtime
git checkout --detach 2da2ee97ee490aa276d8fba0a12e9e84314de181
```

The formal ONNX model and capture data are not stored in Git. Put persistent
data under:

```text
~/fomo-vision-data/artifacts
~/fomo-vision-data/datasets_raw
```

and link them into the runtime checkout:

```bash
cd ~/fomo-vision-runtime
ln -s ~/fomo-vision-data/artifacts artifacts
ln -s ~/fomo-vision-data/datasets_raw datasets_raw
```

The required model files are:

```text
artifacts/d2_mobilenet_v2_fomo_seed42_epoch40.onnx
artifacts/d2_mobilenet_v2_fomo_seed42_epoch40.onnx.json
```

The ONNX SHA-256 remains the frozen D2 artifact SHA documented elsewhere in the
repository.

## Install the user service

The checked-in unit is a **user** systemd unit. It does not require root for
normal installation or service control.

From the repository root:

```bash
./systemd/install_user_service.sh
```

This copies the unit to:

```text
~/.config/systemd/user/robobeetle-vision.service
```

reloads the user manager, verifies the unit, and enables it.

To install and start/restart immediately:

```bash
./systemd/install_user_service.sh --now
```

If an unmanaged `run.py vision_live` process is already running, `--now`
fails closed instead of starting a second process on the same camera/ports.
Stop the old manual process first, then rerun the installer.

## Enable unattended boot startup

A user service starts at boot without an interactive login only when lingering
is enabled for the service user.

Check:

```bash
loginctl show-user "$USER" -p Linger
```

Expected:

```text
Linger=yes
```

If it is disabled, enable it once:

```bash
sudo loginctl enable-linger "$USER"
```

The verified robot already had `Linger=yes`.

## Service contract

The unit runs:

```text
~/venvs/fomo-ort-d2-epoch40/bin/python
~/fomo-vision-runtime/run.py vision_live
```

with the verified camera/runtime settings:

- camera: `/dev/video0`
- source format: `640x480`, `25 FPS`, `YUYV`
- RBVS video: TCP `47010`
- Vision HTTP control/status: TCP `47011`
- detection metadata stream: TCP `47012`
- capture root: `datasets_raw/hardware_gate_g_restart`
- ONNX: `artifacts/d2_mobilenet_v2_fomo_seed42_epoch40.onnx`
- sidecar: `artifacts/d2_mobilenet_v2_fomo_seed42_epoch40.onnx.json`
- restart policy: `Restart=on-failure`, 2-second delay

The service starts with the inference worker disabled. Qt or the HTTP control
API explicitly starts/stops inference.

## Verification

Service state:

```bash
systemctl --user is-enabled robobeetle-vision.service
systemctl --user is-active robobeetle-vision.service
systemctl --user status robobeetle-vision.service --no-pager
```

Expected after normal startup:

```text
enabled
active
```

Check listeners:

```bash
ss -ltnp | grep 4701
```

Expected ports:

```text
47010
47011
47012
```

Check authoritative Vision status:

```bash
curl -s http://127.0.0.1:47011/api/v1/vision/status |
python3 -c '
import json, sys
p = json.load(sys.stdin)
i = p.get("inference", {})
print("camera_running =", p.get("camera", {}).get("running"))
print("state =", i.get("state"))
print("vision_process_rss_bytes =", i.get("vision_process_rss_bytes"))
print("system_total_memory_bytes =", i.get("system_total_memory_bytes"))
print("detection_stream_supported =", i.get("detection_stream_supported"))
print("detection_stream_port =", i.get("detection_stream_port"))
'
```

With the service up and inference not yet requested, the expected state is
`disabled`; the camera remains running, memory fields are present, and the
detection stream capability is advertised on port `47012`.

## Verified 2026-09-24 acceptance

The deployed user service was verified on the real Raspberry Pi 4 with:

- clean runtime checkout at
  `/home/pi/fomo-vision-runtime`
- persistent data root at
  `/home/pi/fomo-vision-data`
- `robobeetle-vision.service` active and enabled
- `Linger=yes`
- user-service restart changed MainPID and recovered all three listeners
- `NRestarts=0` after controlled restarts
- camera running with inference disabled after service startup
- explicit inference Start reached Running
- D2 ONNX loaded from the independent data root
- observed inference about 13.76 FPS / 17.2 ms in the acceptance sample
- process/system memory fields reported correctly
- detection metadata remained supported on port `47012`
- explicit inference Stop returned to Disabled while the Vision service stayed up

No reboot was performed during this acceptance. Boot persistence is established
by the enabled user unit plus `Linger=yes`; a future maintenance reboot can
re-run the verification commands above.

## Rollback

The service is isolated from any older experimental checkout. A previous code
directory can be retained separately for rollback, but persistent
`artifacts` and `datasets_raw` should remain under
`~/fomo-vision-data` rather than being owned by an old checkout.

To stop the managed service:

```bash
systemctl --user stop robobeetle-vision.service
```

Do not run a manual `vision_live` process on the same camera/ports while the
managed service is active.
