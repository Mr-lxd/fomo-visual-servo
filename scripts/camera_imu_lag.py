"""Estimate camera lag from horizontal image motion versus IMU gyro_z.

Positive lag_ms means the camera signal follows the IMU: camera(t) matches
gyro_z(t-lag_ms). Left-turn gyro_z is positive; scene motion is to the right.
Use stride=1 packages for Stage B, and camera calibration for rate magnitudes.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def analyze(args: argparse.Namespace) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if args.end <= args.start or args.max_lag_ms <= 0 or args.lag_step_ms <= 0:
        raise ValueError("need end>start and positive lag range/step")
    if args.focal_length_px is not None and args.focal_length_px <= 0:
        raise ValueError("focal-length-px must be positive")
    package = args.package.resolve()
    metadata = json.loads((package / "session.json").read_text(encoding="utf-8"))
    rows = read_rows(package / "frames.csv")
    if not rows:
        raise ValueError("package has no sampled frames")
    origin_ms = int(rows[0]["capture_ts_ns"]) / 1e6
    start_ms, end_ms = origin_ms + args.start * 1000, origin_ms + args.end * 1000
    rows = [row for row in rows if start_ms <= int(row["capture_ts_ns"]) / 1e6 <= end_ms]
    if len(rows) < 3:
        raise ValueError("selected interval needs at least three frames")
    imu = [row for row in read_rows(package / "imu.csv")
           if row.get("offline_pi_ms") and row.get("gyro_valid") == "1"]
    if len(imu) < 3:
        raise ValueError("package needs at least three valid gyro samples")
    imu_time = np.array([float(row["offline_pi_ms"]) for row in imu])
    gyro = np.array([float(row["gyro_z"]) for row in imu])
    times, rates, shifts, responses, intervals = [], [], [], [], []
    previous, previous_ms, focal = None, None, None
    for row in rows:
        image_path = (package / row["image_file"]).resolve()
        if not image_path.is_relative_to(package):
            raise ValueError("image_file must stay inside the package")
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"cannot read {image_path}")
        image = image.astype(np.float32)
        current_ms = int(row["capture_ts_ns"]) / 1e6
        if previous is not None:
            if image.shape != previous.shape or current_ms <= previous_ms:
                raise ValueError("images must have fixed shape and increasing capture times")
            window = cv2.createHanningWindow((image.shape[1], image.shape[0]), cv2.CV_32F)
            # OpenCV can apply the window in place for optimal DFT sizes.
            # Keep the next pair's previous frame free of earlier windowing.
            (dx, _), response = cv2.phaseCorrelate(previous.copy(), image.copy(), window)
            dt_s = (current_ms - previous_ms) / 1000
            focal = args.focal_length_px or float(image.shape[1])
            times.append((current_ms + previous_ms) / 2)
            rates.append(np.degrees(np.arctan(dx / focal)) / dt_s)
            shifts.append(dx)
            responses.append(response)
            intervals.append(dt_s)
        previous, previous_ms = image, current_ms
    times, rates = np.array(times), np.array(rates)
    # All lags use the same overlap, to avoid peaks caused by edge truncation.
    mask = ((times - args.max_lag_ms >= imu_time[0]) &
            (times + args.max_lag_ms <= imu_time[-1]) & np.isfinite(rates))
    lags = np.arange(-args.max_lag_ms, args.max_lag_ms + args.lag_step_ms / 2,
                     args.lag_step_ms)
    correlations = np.full(len(lags), np.nan)
    if np.sum(mask) >= 3 and np.std(rates[mask]) > 1e-8:
        for index, lag in enumerate(lags):
            matched_gyro = np.interp(times[mask] - lag, imu_time, gyro)
            if np.std(matched_gyro) > 1e-8:
                correlations[index] = np.corrcoef(rates[mask], matched_gyro)[0, 1]
    measurable = bool(np.any(np.isfinite(correlations)))
    best = int(np.nanargmax(correlations)) if measurable else None
    lag_ms = float(lags[best]) if measurable else None
    max_corr = float(correlations[best]) if measurable else None
    if not measurable:
        status = "insufficient_motion_or_overlap"
    elif best in (0, len(lags) - 1):
        status = "peak_at_search_boundary"
    elif max_corr <= 0:
        status = "no_positive_correlation"
    else:
        status = "estimated"
    args.output.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True)
    axes[0].plot((times - origin_ms) / 1000, rates, label="Image horizontal angular-rate estimate")
    aligned = np.interp(times - (lag_ms or 0), imu_time, gyro, left=np.nan, right=np.nan)
    axes[0].plot((times - origin_ms) / 1000, aligned, label="gyro_z(t - camera lag)")
    title = ("Development sample: code path only" if metadata["development_sample"]
             else f"Camera/IMU alignment: {status}")
    axes[0].set(xlabel="Seconds from first packaged frame", ylabel="deg/s", title=title)
    axes[0].legend()
    axes[1].plot(lags, correlations)
    axes[1].set(xlabel="Camera lag relative to IMU (ms)", ylabel="Correlation")
    if measurable:
        axes[1].axvline(lag_ms, color="tab:red", linestyle="--")
    figure.savefig(args.output / "camera_imu_alignment.png", dpi=160)
    plt.close(figure)
    with (args.output / "camera_motion.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["capture_midpoint_pi_ms", "horizontal_shift_px", "phase_response",
                         "image_angular_rate_deg_s", "aligned_gyro_z_deg_s"])
        writer.writerows(zip(times, shifts, responses, rates, aligned))
    result = {"schema_version": "camera-imu-lag-v1", "session_id": metadata["session_id"],
              "development_sample": metadata["development_sample"],
              "interval_seconds_from_first_packaged_frame": [args.start, args.end],
              "method": "full-image phase correlation, signed Pearson lag scan",
              "status": status, "lag_ms": lag_ms, "peak_correlation": max_corr,
              "sign": "positive: camera(t) follows gyro_z(t-lag); left-turn gyro_z>0, scene dx>0",
              "pair_count": len(rates), "common_overlap_pairs": int(np.sum(mask)),
              "median_frame_interval_ms": float(np.median(intervals) * 1000),
              "focal_length_px": focal, "focal_length_calibrated": args.focal_length_px is not None,
              "max_lag_ms": args.max_lag_ms, "lag_step_ms": args.lag_step_ms,
              "notes": ["Default focal length equals image width: angular-rate proxy, not calibration.",
                        "Lag step is a scan grid, not demonstrated measurement accuracy.",
                        "Use stride=1 and real synchronized hand-turn data for Stage B acceptance."]}
    (args.output / "camera_imu_lag.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--start", type=float, required=True, help="seconds from first packaged frame")
    parser.add_argument("--end", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--focal-length-px", type=float)
    parser.add_argument("--max-lag-ms", type=float, default=500)
    parser.add_argument("--lag-step-ms", type=float, default=5)
    print(json.dumps(analyze(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
