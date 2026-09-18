# RBVS v1 — RoboBeetle Vision Stream

RBVS v1 carries realtime JPEG frames on the dedicated Vision TCP connection.
It is independent from RBRP and robot-control authority.

## Fixed header

Every frame is [32-byte header][JPEG payload]. All multi-byte integers use
network byte order (big-endian).

| Offset | Size | Field |
| ---: | ---: | --- |
| 0 | 4 | ASCII magic RBVS |
| 4 | 1 | version = 1 |
| 5 | 1 | header_size = 32 |
| 6 | 1 | codec = 1 (JPEG) |
| 7 | 1 | flags = 0 |
| 8 | 8 | frame_id, uint64 |
| 16 | 8 | capture_timestamp_ns, uint64 |
| 24 | 2 | width, uint16 |
| 26 | 2 | height, uint16 |
| 28 | 4 | jpeg_payload_size, uint32 |

## Validation profile

- v1 accepts exactly version=1 and header_size=32.
- JPEG is the only codec and flags must be zero.
- width/height are each 1..8192.
- width × height must be at most 8,847,360 pixels (4096×2160).
- JPEG payload is 1..4,194,304 bytes (4 MiB).
- capture_timestamp_ns must be positive.
- frame_id uses the full 0..UINT64_MAX range; zero is valid.
- Receiver validates header limits before allocating the JPEG payload.
- Producer drops and diagnoses an encoded frame larger than 4 MiB; it never
  truncates a JPEG.

The dimension fields describe wire representation capability. The validation
profile intentionally accepts fewer decoded pixels than the uint16 field range.

## Frame identity and connection generation

One TCP connection is one RBVS decoding/order session.

- Within one connection, completed wire frames must have strictly increasing
  frame_id values; gaps are expected under latest-frame-wins backpressure.
- A new TCP connection resets sequence validation and never compares its first
  frame_id with the previous connection.
- CameraOwner generation restart, frame counter reset, or uint64 wrap closes
  every current Vision connection before a new generation can stream.
- A frame_id decrease, duplicate, or reset to zero inside one connection is a
  fatal protocol error.
- On accept, if FrameHub already has a cached frame, the connection records its
  frame_id as an accept floor and waits for a newer captured frame. Cached
  pre-connection video is not replayed.

No stream/generation ID is carried in v1; the TCP connection is that boundary.

## Capture timestamp

The producer samples Python time.monotonic_ns() immediately after the camera
API successfully returns a complete frame. It is an acquisition-return
timestamp in the Raspberry Pi CLOCK_MONOTONIC domain, not exposure-start time.

The epoch is undefined and may change across Pi reboot. CameraOwner restart
does not reset the monotonic clock. Windows may use the value for ordering and
same-Pi frame intervals, but must not subtract it from the Windows clock to
claim one-way latency. The Pi may compare it with its own monotonic clock for
source-side freshness decisions.

RBVS v1 does not use Unix epoch, CLOCK_MONOTONIC_RAW, or camera hardware
timestamps.

## Error policy

Malformed framing is connection-fatal: bad magic, unsupported version/header
size/codec, nonzero flags, invalid timestamp/dimensions/payload size, sequence
regression, or a disconnect/timeout in the middle of a frame all discard parser
state and close the connection. The decoder does not scan JPEG bytes for magic.

A length-correct JPEG that fails image decode is a frame-level failure and may
be dropped while parsing continues. JPEG-declared dimensions must be checked
before full image allocation and must equal the RBVS header dimensions.
Repeated dimension mismatches may be promoted to a connection failure.

v1 treats header_size as an integrity field, not an extension negotiation
mechanism. Any field layout or semantic change requires a new protocol version.

## Golden vectors

Header vector:

    52 42 56 53 01 20 01 00
    01 02 03 04 05 06 07 08
    11 12 13 14 15 16 17 18
    02 80 01 e0 00 01 23 45

Full-frame fixture rbvs_v1_2x2.jpg uses frame_id=0, timestamp=1,
width=2, height=2 and payload_size=686.

Payload SHA-256:
742b658462184778ddf5d92207786bb7f08182f75e7f0f5444f8112192b50cd6

Exact 32-byte header plus fixture payload SHA-256:
635bb1a0371f9c9c4242f2924a4c0ffb0590e3edff16945313a89407520bdfa6

## Transport direction and bounded buffering

RBVS v1 is server-to-client only. The Qt viewer sends no data records. If the
Pi observes readable client data, that connection is closed; an orderly/reset
peer close also ends the connection.

The frame header does not provide credit or ACK semantics. Slice 1 therefore
uses bounded socket behavior plus latest-frame selection:

- Pi application code holds no encoded-frame queue; it synchronously writes one
  selected frame, then asks FrameHub for the newest frame available.
- The configured Pi SO_SNDBUF request defaults to 65,536 bytes and each frame
  write has a 0.5 second deadline. Actual kernel buffer accounting is
  platform-dependent and is verified by hardware latency tests rather than
  treated as an exact one-frame guarantee.
- Qt limits its internal QTcpSocket read buffer to 262,144 bytes. A legal JPEG
  up to the 4 MiB protocol cap is still parsed incrementally by the decoder.
- Once Qt has a partial RBVS frame, every new byte restarts a 2,000 ms
  inactivity watchdog. If the partial frame makes no progress for that interval,
  the Vision connection is aborted and parser state is discarded.
- The decoder keeps at most one partial payload and, when one read contains
  several complete frames, retains only the newest complete frame for JPEG
  decode.
