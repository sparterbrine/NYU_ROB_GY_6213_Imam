"""
Lidar calibration script.

For each calibration entry, place the robot so the target surface is at
`known_distance_m` from the robot origin, along the specified angle.
The lidar is physically ~LIDAR_OFFSET_M closer to the surface than the origin.

Outputs: data/lidar_calibration.json
  - per-distance: mean_measured_m, mean_error_m (measured - expected), variance_m2, n_samples
"""

import json
import math
import os
import pickle
import sys

import numpy as np

# ---------------------------------------------------------------------------
# CONFIG — edit these to match your calibration files
# ---------------------------------------------------------------------------

# Physical offset of the lidar from the robot origin, in meters.
# The lidar is this much closer to the target than the robot's coordinate origin.
LIDAR_OFFSET_M = 0.15

# Tolerance around the expected lidar distance used to identify the target cluster.
# Readings within ±DISTANCE_TOLERANCE_M of (known_distance - LIDAR_OFFSET_M) are kept.
DISTANCE_TOLERANCE_M = 0.15

# Each entry: filename (relative to data/), known distance from robot origin (m),
# expected raw lidar angle of the target surface (deg, 0–360),
# and angular half-window (deg) around that angle to accept readings.
CALIBRATION_ENTRIES = [
    {"filename": "calibrate_05m.pkl",        "known_distance_m": 0.50, "target_angle_deg":   0, "angle_window_deg": 20},
    {"filename": "calibrate_1m.pkl",         "known_distance_m": 1.00, "target_angle_deg":   0, "angle_window_deg": 20},
    {"filename": "calibrate_15m.pkl",        "known_distance_m": 1.50, "target_angle_deg": 161, "angle_window_deg": 20},
    {"filename": "calibrate_2m.pkl",         "known_distance_m": 2.00, "target_angle_deg":   0, "angle_window_deg": 20},
    {"filename": "calibrate_1m_45deg.pkl",   "known_distance_m": 1.00, "target_angle_deg": 318, "angle_window_deg": 20},
    {"filename": "calibrate_1m_neg45deg.pkl","known_distance_m": 1.00, "target_angle_deg": 318, "angle_window_deg": 20},
]

# ---------------------------------------------------------------------------

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def angle_diff(a: float, b: float) -> float:
    """Smallest absolute angular difference between two angles in [0, 360)."""
    d = abs(a - b) % 360
    return d if d <= 180 else 360 - d


def process_entry(entry: dict) -> dict:
    filepath = os.path.join(DATA_DIR, entry["filename"])
    known_dist = entry["known_distance_m"]
    target_angle = entry["target_angle_deg"]
    angle_window = entry["angle_window_deg"]

    expected_lidar_dist = known_dist - LIDAR_OFFSET_M

    with open(filepath, "rb") as f:
        data = pickle.load(f)

    readings = []
    for sig in data["robot_sensor_signal"]:
        for angle_raw, dist_mm in zip(sig.angles, sig.distances):
            dist_m = dist_mm / 1000.0
            if angle_diff(angle_raw, target_angle) <= angle_window:
                if abs(dist_m - expected_lidar_dist) <= DISTANCE_TOLERANCE_M:
                    readings.append(dist_m)

    if not readings:
        print(f"  WARNING: no readings matched for {entry['filename']} "
              f"(expected {expected_lidar_dist:.3f}m ±{DISTANCE_TOLERANCE_M}m "
              f"at angle {target_angle}°±{angle_window}°)")
        return {
            "filename": entry["filename"],
            "known_distance_m": known_dist,
            "expected_lidar_distance_m": round(expected_lidar_dist, 4),
            "n_samples": 0,
            "mean_measured_m": None,
            "mean_error_m": None,
            "variance_m2": None,
            "std_dev_m": None,
        }

    arr = np.array(readings)
    mean_measured = float(np.mean(arr))
    mean_error = mean_measured - expected_lidar_dist
    variance = float(np.var(arr))
    std_dev = float(np.std(arr))

    print(f"  {entry['filename']}: {len(arr)} samples, "
          f"expected={expected_lidar_dist:.3f}m, "
          f"mean={mean_measured:.4f}m, "
          f"error={mean_error:+.4f}m, "
          f"std={std_dev:.4f}m")

    return {
        "filename": entry["filename"],
        "known_distance_m": known_dist,
        "expected_lidar_distance_m": round(expected_lidar_dist, 4),
        "n_samples": len(arr),
        "mean_measured_m": round(mean_measured, 6),
        "mean_error_m": round(mean_error, 6),
        "variance_m2": round(variance, 8),
        "std_dev_m": round(std_dev, 6),
    }


def main():
    print(f"Lidar offset: {LIDAR_OFFSET_M}m")
    print(f"Processing {len(CALIBRATION_ENTRIES)} calibration files...\n")

    results = []
    for entry in CALIBRATION_ENTRIES:
        filepath = os.path.join(DATA_DIR, entry["filename"])
        if not os.path.exists(filepath):
            print(f"  SKIP: {entry['filename']} not found")
            continue
        results.append(process_entry(entry))

    # Summary grouped by known distance
    output = {
        "lidar_offset_m": LIDAR_OFFSET_M,
        "calibration_entries": results,
        # Convenience: map known_distance_m -> mean_error and variance
        "by_distance": {
            str(r["known_distance_m"]): {
                "mean_error_m": r["mean_error_m"],
                "variance_m2": r["variance_m2"],
                "std_dev_m": r["std_dev_m"],
                "n_samples": r["n_samples"],
            }
            for r in results if r["n_samples"] > 0
        },
    }

    out_path = os.path.join(DATA_DIR, "lidar_calibration.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nCalibration saved to: {out_path}")

    # Print a quick summary table
    print("\n--- Summary ---")
    print(f"{'File':<35} {'Known':>7} {'Expected':>10} {'Mean meas':>10} {'Error':>8} {'Std':>8} {'N':>5}")
    for r in results:
        if r["n_samples"] > 0:
            print(f"{r['filename']:<35} {r['known_distance_m']:>7.2f} "
                  f"{r['expected_lidar_distance_m']:>10.4f} "
                  f"{r['mean_measured_m']:>10.4f} "
                  f"{r['mean_error_m']:>+8.4f} "
                  f"{r['std_dev_m']:>8.4f} "
                  f"{r['n_samples']:>5}")
        else:
            print(f"{r['filename']:<35} {r['known_distance_m']:>7.2f}  NO MATCHING READINGS")


if __name__ == "__main__":
    main()
