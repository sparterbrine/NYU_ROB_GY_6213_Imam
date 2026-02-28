"""
view_log_frames.py  –  Browse camera frames saved in a robot data log.

Usage:
    python view_log_frames.py <path_to_pkl_file>

Controls:
    Any key / Right arrow  – next frame
    Left arrow             – previous frame
    q / Escape             – quit
"""

import argparse
import pickle
import sys

import cv2
import numpy as np


# ── helpers ──────────────────────────────────────────────────────────────────

def load_log(path: str) -> dict:
    with open(path, 'rb') as fh:
        return pickle.load(fh)


def decode_frame(jpeg_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def annotate(frame: np.ndarray, index: int, total: int, data: dict) -> np.ndarray:
    """Draw a semi-transparent overlay with per-frame metadata."""
    img = frame.copy()
    h, w = img.shape[:2]

    # Collect annotation lines
    lines = [f"Frame {index + 1} / {total}"]

    t = data['time'][index]
    lines.append(f"t = {t:.3f} s")

    cs = data['control_signal'][index]
    lines.append(f"Speed={cs[0]}  Steer={cs[1]}")

    cam = data['camera_sensor_signal'][index]
    try:
        lines.append(f"x={float(cam[0]):.3f}  y={float(cam[1]):.3f}  theta={float(cam[5]):.3f}")
    except (IndexError, TypeError):
        lines.append("camera pose: N/A")

    # Draw a dark rectangle behind the text for readability
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness  = 1
    line_h     = 22
    padding    = 6
    box_h      = len(lines) * line_h + padding * 2
    overlay    = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)

    for i, text in enumerate(lines):
        y = padding + (i + 1) * line_h - 4
        cv2.putText(img, text, (padding, y), font, font_scale,
                    (220, 220, 220), thickness, cv2.LINE_AA)

    return img


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Browse camera frames in a robot data log.")
    parser.add_argument('pkl_file', help="Path to the .pkl log file")
    args = parser.parse_args()

    data = load_log(args.pkl_file)

    frames_raw = data.get('frame', [])
    total = len(frames_raw)
    if total == 0:
        print("No frames found in this log file.")
        sys.exit(1)

    print(f"Loaded {total} frames.  Use arrow keys or any key to step; Q/Esc to quit.")

    index = 0
    cv2.namedWindow('Robot Log Viewer', cv2.WINDOW_NORMAL)

    while True:
        jpeg = frames_raw[index]
        if jpeg is None:
            # Placeholder for missing frames
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "No frame captured", (20, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 80, 80), 2)
        else:
            frame = decode_frame(jpeg)

        display = annotate(frame, index, total, data)
        cv2.imshow('Robot Log Viewer', display)

        key = cv2.waitKey(0) & 0xFF
        if key in (ord('q'), 27):          # Q or Escape → quit
            break
        elif key == 81 and index > 0:      # Left arrow
            index -= 1
        elif index < total - 1:            # Any other key → next
            index += 1

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
