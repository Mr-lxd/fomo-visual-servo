"""Build an offline session package from recorded Pi video and Console CSVs.

Times are Pi CLOCK_MONOTONIC milliseconds after a lower-envelope MCU clock
fit. Gyro/angles retain CSV units (degrees/s, degrees); gait_phase is radians
in [0, 2*pi). Raw images are saved without overlays or model processing.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


SESSION_FIELDS = [
    "session_id", "date", "location_water", "lighting", "turbidity",
    "targets", "backend", "actions", "split", "notes",
]
FRAME_FIELDS = [
    "frame_id", "capture_ts_ns", "sampling", "image_file", "gyro_x",
    "gyro_y", "gyro_z", "roll", "pitch", "gait_phase", "backend",
    "active_mode", "target_mode", "motion_state", "depth_cal_m",
    "depth_age_ms", "nearest_imu_dt_ms",
]
COUNTERS = ["sampler_drop_total", "gateway_drop_total", "batch_gap_total",
            "fragment_gap_total"]


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def distribution(values: np.ndarray) -> dict | None:
    if not len(values):
        return None
    return {"count": len(values), "min": float(np.min(values)),
            "median": float(np.median(values)), "p95": float(np.percentile(values, 95)),
            "max": float(np.max(values)), "mean": float(np.mean(values)),
            "rms": float(np.sqrt(np.mean(values ** 2)))}


def fit_clock(rows: list[dict], window_ms: float = 1000.0) -> tuple[np.ndarray, dict]:
    """Fit pi_ms=a*mcu_ms+b from one minimum-delay sample per time window.

    Each receive batch may contain older samples. Select minima of y-a*x,
    not minima of y alone, so sample age/drift do not bias the window picks.
    Refine the picks three times, then anchor the intercept at the global
    minimum receive offset. Receive delay statistics are not sensor accuracy.
    """
    x = np.array([float(row["mcu_unwrapped_ms"]) for row in rows])
    y = np.array([float(row["pi_rx_ms"]) for row in rows])
    if len(x) < 3 or np.any(np.diff(x) <= 0):
        raise ValueError("need at least three ordered, distinct MCU sample times")
    if len({row["link_epoch"] for row in rows}) != 1:
        raise ValueError("one package must contain a single link_epoch/clock mapping")
    bins = np.floor((x - x[0]) / window_ms).astype(int)
    a = (y[-1] - y[0]) / (x[-1] - x[0])
    for _ in range(3):
        selected = []
        for window in np.unique(bins):
            indices = np.flatnonzero(bins == window)
            selected.append(indices[np.argmin(y[indices] - a * x[indices])])
        if len(selected) < 2:
            raise ValueError("motion duration must span at least two fit windows")
        xs, ys = x[selected], y[selected]
        a = float(np.dot(xs - xs.mean(), ys - ys.mean()) / np.sum((xs - xs.mean()) ** 2))
    if a <= 0:
        raise ValueError("clock fit has a nonpositive slope")
    b = float(np.min(y - a * x))
    pi = a * x + b
    online = np.array([float(row["sample_pi_ms"]) if row["sample_pi_ms"] else np.nan
                       for row in rows])
    difference = pi - online
    return pi, {
        "method": "iterated_window_minimum_offset_then_lower_intercept",
        "equation": "pi_ms = a * mcu_unwrapped_ms + b", "a": a, "b_ms": b,
        "window_ms": window_ms, "iterations": 3, "envelope_points": len(selected),
        "envelope_residual_ms": distribution(y[selected] - pi[selected]),
        "all_receive_residual_ms": distribution(y - pi),
        "offline_minus_sample_pi_ms": distribution(difference[np.isfinite(difference)]),
        "pi_range_ms": [float(pi[0]), float(pi[-1])],
        "limitation": "lower receive envelope still includes minimum transport/sensor latency",
    }


def find_one(raw: Path, pattern: str, required: bool = True) -> Path | None:
    paths = sorted(raw.glob(pattern))
    if len(paths) != 1 and (required or paths):
        raise ValueError(f"expected one {pattern} in {raw}, found {len(paths)}")
    return paths[0] if paths else None


def input_info(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path.resolve()), "sha256": digest.hexdigest(),
            "bytes": path.stat().st_size}


def motion_statistics(all_rows: list[dict], samples: list[dict], pi: np.ndarray) -> dict:
    counters = {}
    for field in COUNTERS:
        values = [int(row[field]) for row in all_rows if row[field]]
        if np.any(np.diff(values) < 0):
            raise ValueError(f"counter {field} resets inside this session")
        counters[field] = {"start": values[0], "end": values[-1],
                           "delta": values[-1] - values[0]}
    crossings = []
    for i in range(1, len(samples)):
        previous, current = samples[i - 1], samples[i]
        if not all(row["phase_valid"] == "1" and row["active_mode"] == "1"
                   for row in (previous, current)):
            continue
        p0, p1 = float(previous["gait_phase"]), float(current["gait_phase"])
        if p0 - p1 > np.pi:
            crossing = pi[i - 1] + (2 * np.pi - p0) / (p1 + 2 * np.pi - p0) * (pi[i] - pi[i - 1])
            crossings.append((i, crossing))
    periods = [t1 - t0 for (i0, t0), (i1, t1) in zip(crossings, crossings[1:])
               if all(row["active_mode"] == "1" and row["phase_valid"] == "1"
                      for row in samples[i0:i1 + 1])]
    return {"csv_rows": len(all_rows), "samples": len(samples),
            "counter_only_rows": len(all_rows) - len(samples), "counters": counters,
            "forward_phase_period_pi_ms": distribution(np.array(periods)),
            "forward_period_method": "linear interpolation of phase wrap crossings; active_mode=1"}


def interpolate_motion(time_ms: float, rows: list[dict], pi: np.ndarray) -> dict:
    """Interpolate valid bracketing channels; use causal state, no extrapolation."""
    if not len(pi) or time_ms < pi[0] or time_ms > pi[-1]:
        return {}
    right = int(np.searchsorted(pi, time_ms, side="left"))
    if pi[right] == time_ms:
        left = right
    else:
        left = right - 1
    weight = 0.0 if left == right else (time_ms - pi[left]) / (pi[right] - pi[left])
    prior = int(np.searchsorted(pi, time_ms, side="right")) - 1
    result = {field: rows[prior][field]
              for field in ("backend", "active_mode", "target_mode", "motion_state")}
    result["nearest_imu_dt_ms"] = float(min(abs(time_ms - pi[left]), abs(pi[right] - time_ms)))
    for field, valid in [("gyro_x", "gyro_valid"), ("gyro_y", "gyro_valid"),
                         ("gyro_z", "gyro_valid"), ("roll", "angle_valid"),
                         ("pitch", "angle_valid"), ("gait_phase", "phase_valid")]:
        if rows[left][valid] != "1" or rows[right][valid] != "1":
            continue
        v0, v1 = float(rows[left][field]), float(rows[right][field])
        if field == "gait_phase":
            # Work in turns for circular interpolation, then restore CSV radians.
            turns0, turns1 = v0 / (2 * np.pi), v1 / (2 * np.pi)
            delta = (turns1 - turns0 + 0.5) % 1.0 - 0.5
            result[field] = float(((turns0 + weight * delta) % 1.0) * 2 * np.pi)
        else:
            result[field] = v0 + weight * (v1 - v0)
    return result


def extract_frames(raw: Path, index: list[dict], selected: dict[int, dict], output: Path) -> dict:
    """Decode every indexed AVI frame once and verify no missing/extra frames."""
    segments = list(dict.fromkeys(row["segment"] for row in index))
    counts = {}
    (output / "frames").mkdir()
    for segment in segments:
        segment_path = (raw / segment).resolve()
        if not segment_path.is_relative_to(raw.resolve()):
            raise ValueError("segment path must stay inside raw directory")
        rows = [row for row in index if row["segment"] == segment]
        capture = cv2.VideoCapture(str(segment_path))
        if not capture.isOpened():
            raise ValueError(f"cannot open video {segment_path}")
        try:
            for position, row in enumerate(rows):
                if int(row["segment_frame_index"]) != position:
                    raise ValueError(f"nonsequential segment_frame_index in {segment}")
                ok, frame = capture.read()
                if not ok:
                    raise ValueError(f"video {segment} ends before index row {position}")
                frame_id = int(row["frame_id"])
                if frame_id in selected:
                    image = output / selected[frame_id]["image_file"]
                    if not cv2.imwrite(str(image), frame):
                        raise ValueError(f"cannot write {image}")
            if capture.read()[0]:
                raise ValueError(f"video {segment} has more frames than frame_index.csv")
        finally:
            capture.release()
        counts[segment] = len(rows)
    return counts


def build(args: argparse.Namespace) -> dict:
    if args.stride < 1 or args.fit_window_ms <= 0:
        raise ValueError("stride and fit-window-ms must be positive")
    raw, output = args.raw.resolve(), args.output.resolve()
    session_id = args.session_id or raw.name
    _, registrations = read_csv(args.sessions)
    matches = [row for row in registrations if row["session_id"] == session_id]
    if args.dev_sample:
        if matches:
            raise ValueError("unregistered development sample must not be in sessions.csv")
        registration = None
    else:
        if len(matches) != 1 or matches[0]["split"] not in ("dev", "test", "desk"):
            raise ValueError("session_id must have exactly one valid sessions.csv row")
        registration = matches[0]
    inputs = {"sessions_csv": input_info(args.sessions)}
    samples, all_motion, pi, fit, motion_stats = [], [], np.array([]), None, None
    if args.only != "video":
        path = find_one(raw, "motion*.csv")
        motion_fields, all_motion = read_csv(path)
        samples = [row for row in all_motion if row["mcu_unwrapped_ms"]]
        pi, fit = fit_clock(samples, args.fit_window_ms)
        motion_stats = motion_statistics(all_motion, samples, pi)
        inputs["motion_csv"] = input_info(path)
    visual_path = find_one(raw, "visual*.csv", required=False)
    visual, by_id, timed_visual = [], {}, []
    if visual_path:
        _, visual = read_csv(visual_path)
        inputs["visual_csv"] = input_info(visual_path)
        for row in visual:
            if row["depth_cal_m"] and row["depth_age_ms"]:
                if row["frame_id"]:
                    by_id[int(row["frame_id"])] = row
                if row["capture_ts_ns"]:
                    timed_visual.append(row)
        timed_visual.sort(key=lambda row: int(row["capture_ts_ns"]))
    depth_times = np.array([int(row["capture_ts_ns"]) for row in timed_visual], dtype=np.int64)
    frames, index, frame_stats = [], [], None
    if args.only != "motion":
        index_path = raw / "frame_index.csv"
        _, index = read_csv(index_path)
        frame_ids = np.array([int(row["frame_id"]) for row in index], dtype=np.int64)
        timestamps = np.array([int(row["capture_timestamp_ns"]) for row in index], dtype=np.int64)
        if not len(index) or np.any(np.diff(frame_ids) <= 0) or np.any(np.diff(timestamps) <= 0):
            raise ValueError("frame IDs and capture times must be strictly increasing")
        keyframes = set(args.keyframes)
        if not keyframes.issubset(set(frame_ids)):
            raise ValueError("keyframe IDs must exist in frame_index.csv")
        for position, row in enumerate(index):
            frame_id, timestamp = int(row["frame_id"]), int(row["capture_timestamp_ns"])
            if position % args.stride and frame_id not in keyframes:
                continue
            record = dict.fromkeys(FRAME_FIELDS, "")
            record.update(frame_id=frame_id, capture_ts_ns=timestamp,
                          sampling="keyframe" if frame_id in keyframes else "uniform",
                          image_file=f"frames/frame_{frame_id:020d}.jpg")
            record.update(interpolate_motion(timestamp / 1e6, samples, pi))
            depth = by_id.get(frame_id)
            if depth is None and len(depth_times):
                depth = timed_visual[int(np.argmin(np.abs(depth_times - timestamp)))]
            if depth:
                record.update(depth_cal_m=depth["depth_cal_m"], depth_age_ms=depth["depth_age_ms"])
            frames.append(record)
        inputs["frame_index_csv"] = input_info(index_path)
        for segment in dict.fromkeys(row["segment"] for row in index):
            video = (raw / segment).resolve()
            if not video.is_relative_to(raw):
                raise ValueError("video segment path escapes raw directory")
            inputs[f"video:{segment}"] = input_info(video)
        covered = 0
        if len(pi):
            all_times = timestamps / 1e6
            indices = np.searchsorted(pi, all_times)
            nearest = np.minimum(np.abs(all_times - pi[np.clip(indices, 0, len(pi) - 1)]),
                                 np.abs(all_times - pi[np.clip(indices - 1, 0, len(pi) - 1)]))
            covered = int(np.sum((all_times >= pi[0]) & (all_times <= pi[-1]) & (nearest <= 50)))
        frame_stats = {"indexed_frames": len(index), "sampled_frames": len(frames),
                       "stride": args.stride, "keyframes": sorted(keyframes),
                       "frame_id_gap_count": int(np.sum(np.diff(frame_ids) > 1)),
                       "dropped_frame_ids": int(np.sum(np.diff(frame_ids) - 1)),
                       "imu_within_50ms_frames": covered,
                       "imu_within_50ms_fraction": covered / len(index),
                       "sampled_depth_coverage_fraction": sum(bool(row["depth_cal_m"]) for row in frames) / len(frames),
                       "depth_join": "same frame_id, otherwise nearest capture_ts_ns"}
    metadata = raw / "metadata.json"
    if metadata.exists():
        inputs["capture_metadata"] = input_info(metadata)
    output.mkdir(parents=True, exist_ok=False)
    if args.only != "video":
        position = 0
        for row in all_motion:
            row["offline_pi_ms"] = ""
            if row["mcu_unwrapped_ms"]:
                row["offline_pi_ms"] = float(pi[position])
                position += 1
        write_csv(output / "imu.csv", motion_fields + ["offline_pi_ms"], all_motion)
    else:
        write_csv(output / "imu.csv", ["offline_pi_ms"], [])
    if index:
        frame_stats["decoded_segment_frames"] = extract_frames(
            raw, index, {row["frame_id"]: row for row in frames}, output)
    else:
        (output / "frames").mkdir()
    write_csv(output / "frames.csv", FRAME_FIELDS, frames)
    result = {"schema_version": "session-package-v1", "session_id": session_id,
              "registration": registration, "development_sample": args.dev_sample,
              "processing_mode": args.only or "full", "inputs": inputs,
              "clock_fit": fit, "quality": {"motion": motion_stats, "frames": frame_stats},
              "units": {"capture_ts_ns": "Pi CLOCK_MONOTONIC ns", "offline_pi_ms": "Pi CLOCK_MONOTONIC ms",
                        "gyro_xyz": "deg/s", "roll_pitch": "deg", "gait_phase": "rad [0,2*pi)",
                        "depth_cal_m": "m", "depth_age_ms": "ms", "nearest_imu_dt_ms": "absolute ms"}}
    (output / "session.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--sessions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new package directory; never overwrite")
    parser.add_argument("--session-id")
    parser.add_argument("--stride", type=int, default=12)
    parser.add_argument("--keyframes", type=int, nargs="*", default=[])
    parser.add_argument("--fit-window-ms", type=float, default=1000)
    parser.add_argument("--only", choices=("motion", "video"), help="Stage A component validation")
    parser.add_argument("--dev-sample", action="store_true", help="unregistered development sample only")
    args = parser.parse_args()
    print(json.dumps(build(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
