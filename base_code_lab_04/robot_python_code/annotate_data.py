"""
annotate_data.py – Browse and annotate ground-truth x, y, theta on logged frames.

Usage:
    python annotate_data.py <pkl_file> [annotations_file]

    annotations_file defaults to <pkl_file without .pkl>_annotations.json

Controls:
    D / Right arrow   – next frame
    A / Left arrow    – previous frame
    E                 – annotate current frame (opens x, y, theta dialogs)
    X                 – delete annotation for current frame
    S                 – save annotations to file
    Q / Escape        – save and quit
"""

import argparse
import json
import os
import pickle
import sys

import cv2
import numpy as np

try:
    import tkinter as tk
    from tkinter import simpledialog
    HAS_TK = True
except ImportError:
    HAS_TK = False


def load_log(path: str) -> dict:
    with open(path, 'rb') as fh:
        return pickle.load(fh)


def decode_frame(jpeg_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def load_annotations(path: str) -> dict:
    """Load annotations JSON; returns dict keyed by int frame index."""
    if os.path.exists(path):
        with open(path, 'r') as fh:
            raw = json.load(fh)
        return {int(k): v for k, v in raw.items()}
    return {}


def save_annotations(annotations: dict, path: str):
    serializable = {str(k): v for k, v in sorted(annotations.items())}
    with open(path, 'w') as fh:
        json.dump(serializable, fh, indent=2)
    print(f"Saved {len(annotations)} annotation(s) → {path}")


def ask_annotation(current: dict) -> dict:
    """Open three Tkinter dialogs to enter x, y, theta. Returns None if cancelled."""
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    defaults = current or {'x': 0.0, 'y': 0.0, 'theta': 0.0}

    x = simpledialog.askfloat("Ground Truth", "x (m):", initialvalue=defaults['x'], parent=root)
    if x is None:
        root.destroy()
        return None
    y = simpledialog.askfloat("Ground Truth", "y (m):", initialvalue=defaults['y'], parent=root)
    if y is None:
        root.destroy()
        return None
    theta = simpledialog.askfloat("Ground Truth", "theta (°):", initialvalue=defaults['theta'], parent=root)
    root.destroy()
    if theta is None:
        return None
    return {'x': x, 'y': y, 'theta': theta}


def draw_frame(frame: np.ndarray, index: int, total: int, data: dict, annotations: dict) -> np.ndarray:
    img = frame.copy()
    h, w = img.shape[:2]
    ann = annotations.get(index)

    lines = [f"Frame {index + 1} / {total}"]
    try:
        lines.append(f"t = {data['time'][index]:.3f} s")
    except (KeyError, IndexError):
        pass
    try:
        cs = data['control_signal'][index]
        lines.append(f"Speed={cs[0]}  Steer={cs[1]}")
    except (KeyError, IndexError):
        pass
    try:
        cam = data['camera_sensor_signal'][index]
        lines.append(f"cam  x={float(cam[0]):.3f}  y={float(cam[1]):.3f}  θ={float(cam[5]):.3f}")
    except (KeyError, IndexError, TypeError):
        pass
    if ann:
        lines.append(f"GT   x={ann['x']:.3f}  y={ann['y']:.3f}  θ={ann['theta']:.3f}")
    else:
        lines.append("GT   (not annotated)")

    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness  = 1
    line_h     = 22
    padding    = 6
    box_h      = len(lines) * line_h + padding * 2

    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)

    for i, text in enumerate(lines):
        is_gt_line = (i == len(lines) - 1)
        if is_gt_line and ann:
            color = (100, 255, 100)
        elif is_gt_line:
            color = (140, 140, 140)
        else:
            color = (220, 220, 220)
        y_pos = padding + (i + 1) * line_h - 4
        cv2.putText(img, text, (padding, y_pos), font, font_scale, color, thickness, cv2.LINE_AA)

    # Annotated badge in top-right corner
    if ann:
        badge = "ANNOTATED"
        (bw, bh), _ = cv2.getTextSize(badge, font, 0.5, 1)
        bx = w - bw - 10
        by = 20
        cv2.rectangle(img, (bx - 4, by - bh - 4), (bx + bw + 4, by + 4), (0, 160, 0), -1)
        cv2.putText(img, badge, (bx, by), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Key hint bar at bottom
    hint = "E:annotate  X:delete  S:save  A/<:prev  D/>:next  Q:save+quit"
    cv2.putText(img, hint, (padding, h - 8), font, 0.42, (140, 140, 140), 1, cv2.LINE_AA)
    return img


def main():
    parser = argparse.ArgumentParser(description="Annotate ground-truth poses on robot log frames.")
    parser.add_argument('pkl_file', help="Path to the .pkl log file")
    parser.add_argument('annotations_file', nargs='?',
                        help="JSON file to save/load annotations (default: <pkl>_annotations.json)")
    args = parser.parse_args()

    if not HAS_TK:
        print("ERROR: tkinter is required for annotation dialogs but could not be imported.")
        sys.exit(1)

    ann_path = args.annotations_file or os.path.splitext(args.pkl_file)[0] + '_annotations.json'
    data = load_log(args.pkl_file)
    annotations = load_annotations(ann_path)

    frames_raw = data.get('frame', [])
    total = len(frames_raw)
    if total == 0:
        print("No frames found in this log file.")
        sys.exit(1)

    print(f"Loaded {total} frames.  Existing annotations: {len(annotations)}")
    print(f"Annotation file: {ann_path}")

    index = 0
    cv2.namedWindow('Annotate Log', cv2.WINDOW_NORMAL)

    while True:
        jpeg = frames_raw[index]
        if jpeg is None:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "No frame captured", (20, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 80, 80), 2)
        else:
            frame = decode_frame(jpeg)

        display = draw_frame(frame, index, total, data, annotations)
        cv2.imshow('Annotate Log', display)

        key = cv2.waitKey(0) & 0xFF

        if key in (ord('q'), 27):                          # Q / Escape → save and quit
            save_annotations(annotations, ann_path)
            break
        elif key == ord('s'):                              # S → save
            save_annotations(annotations, ann_path)
        elif key == ord('e'):                              # E → annotate current frame
            result = ask_annotation(annotations.get(index))
            if result is not None:
                annotations[index] = result
        elif key == ord('x'):                              # X → delete annotation
            annotations.pop(index, None)
        elif key in (ord('a'), 81) and index > 0:         # A / left arrow → prev
            index -= 1
        elif key in (ord('d'), 83) and index < total - 1: # D / right arrow → next
            index += 1

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
