# Vision Slice 1 — LIVE realtime video

This slice adds an independent realtime video plane only:

USB UVC camera -> CameraOwner -> FrameHub -> JPEG -> RBVS v1 TCP
-> Windows RoboBeetle Qt VisionClient -> VideoView.

It does not add FOMO, detections, tracking, visual servo, recording, ROS2,
systemd, STM32 changes, RBRP changes, or actuator commands.

## Raspberry Pi runtime

The repository launcher bootstraps the src-layout package without requiring a
manual PYTHONPATH:

    cd <fomo-visual-servo checkout or deployment bundle>
    python run.py vision_live --source /dev/video0 --width 640 --height 480 \
      --fps 25 --fourcc YUYV --bind 0.0.0.0 --port 47010 --jpeg-quality 80

The Pi runtime environment must provide the project dependency PyYAML plus
OpenCV and NumPy. An inference-only venv may already contain OpenCV/NumPy while
omitting PyYAML, so verify imports before hardware startup instead of assuming
the older deployment profile is sufficient.

The service runs in the foreground. Ctrl-C performs OFF shutdown: Vision TCP
connections close, CameraOwner stops, and the camera handle is released.
No Vision systemd unit is introduced in Slice 1.

## Windows Qt client

Build and run RoboBeetleConsole normally. The Realtime Video card has its own:

- Vision Host
- Vision Port (default 47010)
- Connect Video / Disconnect Video control
- connection state
- receive/display/frame/drop diagnostics

The video connection is independent from the top RBRP control connection.
Connecting or disconnecting Vision does not acquire/release authority, change
the RBRP heartbeat, or send STM32 commands.

For direct Ethernet the expected first hardware-acceptance host is
192.168.10.2. Wi-Fi uses the Pi address assigned by the actual network.

## Realtime and reconnect invariants

- FrameHub contains one replaceable latest raw frame.
- The stream sender has no application JPEG queue and synchronously writes at
  most one selected RBVS frame at a time.
- Pi requests a 64 KiB send buffer and uses a 0.5 second frame-write timeout;
  actual kernel buffering remains platform-dependent and is validated on hardware.
- A slow socket is disconnected by the write timeout instead of accumulating
  old encoded frames indefinitely.
- Qt bounds its internal compressed read buffer to 256 KiB; a legal larger
  JPEG is parsed incrementally rather than requiring the socket buffer to hold
  the entire frame.
- The incremental decoder returns only the newest complete frame from a
  coalesced read.
- VideoView stores one latest image; a newer image replaces an unpainted one.
- RBVS v1 is server-to-client only; unexpected client bytes close the Vision
  connection.
- Each TCP connection is one frame-id generation/order session.
- A connection never receives the FrameHub image cached before accept; it waits
  for a newly captured frame.
- CameraOwner generation reset closes Vision connections before frame_id can
  restart at zero.

Expected overload behavior is larger frame-id gaps while displayed video stays
near realtime, not an ever-growing old-frame delay.

## Hardware Acceptance status

Hardware Acceptance A-F passed on 2026-09-19 using the real Raspberry Pi 4,
USB UVC camera, Windows RoboBeetle Console, and robot-control link.

1. CameraOwner negotiated 640x480 YUYV at 25 FPS, captured continuously, and
   released both /dev/video0 and TCP 47010 on Ctrl-C.
2. Direct Ethernet streamed current video with no JPEG decode drops. The Pi
   sustained about 25 selected/sent FPS with roughly 30 KiB JPEG frames.
3. A deliberately slow receiver produced increasing frame-id gaps instead of
   replaying every intermediate frame. A non-reading viewer was released and a
   replacement viewer received a current frame promptly.
4. While Vision streamed, RBRP Connect, Acquire, telemetry/heartbeat,
   DisableServos, and Release all remained functional with CRC=0 and Timeout=0.
5. Stopping/restarting Vision did not change RBRP authority or telemetry, did
   not replay actuator commands, and a new CameraOwner generation reconnected
   without stale video replay.
6. Wi-Fi streaming through the tested 192.168.137.x link remained latest-frame
   oriented. Display FPS was lower than Ethernet (about 14 FPS in the observed
   run), but subjective latency stayed roughly constant and reconnect showed
   the current scene immediately.

These are acceptance observations for the tested hardware/network, not protocol
guarantees or cross-device one-way latency measurements.
