"""
Generates publication-quality figures from lidar_calibration.json.

Outputs (saved to data/):
  lidar_calibration_plot.pdf  (vector, for paper)
  lidar_calibration_plot.png  (300 dpi raster)
"""

import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
CALIB_JSON = os.path.join(DATA_DIR, "lidar_calibration.json")

# Paper style
plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        10,
    "axes.titlesize":   11,
    "axes.labelsize":   10,
    "legend.fontsize":  9,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "axes.linewidth":   0.8,
    "grid.linewidth":   0.5,
    "lines.linewidth":  1.2,
    "figure.dpi":       150,
})
# ---------------------------------------------------------------------------

with open(CALIB_JSON) as f:
    cal = json.load(f)

entries = cal["calibration_entries"]

# Split into axial (target_angle ~0 or ~180) vs angled (45°) groups
# Exclude the 1.5m entry due to a data collection error
axial   = [e for e in entries if "45" not in e["filename"] and e["known_distance_m"] != 1.5]
angled  = [e for e in entries if "45" in e["filename"]]

# Sort axial by known distance
axial.sort(key=lambda e: e["known_distance_m"])

axial_known    = np.array([e["known_distance_m"]      for e in axial])
axial_expected = np.array([e["expected_lidar_distance_m"] for e in axial])
axial_measured = np.array([e["mean_measured_m"]        for e in axial])
axial_std      = np.array([e["std_dev_m"]              for e in axial])
axial_error_mm = np.array([e["mean_error_m"] * 1000    for e in axial])
axial_std_mm   = np.array([e["std_dev_m"]   * 1000     for e in axial])

angled_labels  = ["+45°", "−45°"]
angled_error   = np.array([e["mean_error_m"] * 1000 for e in angled])
angled_std     = np.array([e["std_dev_m"]    * 1000 for e in angled])

# ── Figure layout: 1×3 subplots ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.8))
fig.subplots_adjust(wspace=0.42, left=0.09, right=0.97, top=0.88, bottom=0.18)

BLUE  = "#2166ac"
RED   = "#d6604d"
GREY  = "#636363"

# ── Panel 1: measured vs. expected (axial) ──────────────────────────────────
ax = axes[0]
ax.errorbar(axial_expected, axial_measured, yerr=axial_std,
            fmt='o', color=BLUE, ecolor=BLUE, elinewidth=1.2,
            capsize=4, capthick=1.2, markersize=5, label="Measured (axial)")
lims = [0.2, 2.0]
ax.plot(lims, lims, '--', color=GREY, linewidth=0.9, label="Ideal (y = x)")
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel("Expected lidar distance (m)")
ax.set_ylabel("Mean measured distance (m)")
ax.set_title("(a) Measured vs. Expected")
ax.legend(loc="upper left", framealpha=0.9)
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
ax.grid(True, linestyle=':', alpha=0.6)

# ── Panel 2: mean error ± 1σ vs. known distance (axial) ────────────────────
ax = axes[1]
ax.axhline(0, color=GREY, linewidth=0.8, linestyle='--')
ax.errorbar(axial_known, axial_error_mm, yerr=axial_std_mm,
            fmt='s-', color=BLUE, ecolor=BLUE, elinewidth=1.2,
            capsize=4, capthick=1.2, markersize=5)
ax.fill_between(axial_known,
                axial_error_mm - axial_std_mm,
                axial_error_mm + axial_std_mm,
                alpha=0.15, color=BLUE)
ax.set_xlabel("Known distance (m)")
ax.set_ylabel("Bias ± 1σ  (mm)")
ax.set_title("(b) Ranging Bias vs. Distance")
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
ax.grid(True, linestyle=':', alpha=0.6)

# ── Panel 3: axial std-dev + angled comparison ──────────────────────────────
ax = axes[2]

# Axial std dev as a line
ax.plot(axial_known, axial_std_mm, 'o-', color=BLUE, markersize=5,
        label="Axial (0°)")

# Angled measurements as separate points at x = 1.0
angled_x = [1.0, 1.0]
ax.errorbar(angled_x, angled_error, yerr=angled_std,
            fmt='^', color=RED, ecolor=RED, elinewidth=1.2,
            capsize=4, capthick=1.2, markersize=6,
            label="±45° (bias shown)")

# Annotate 45° bias values
for xi, ei, label in zip(angled_x, angled_error, angled_labels):
    ax.annotate(f"{label}\n{ei:+.1f} mm",
                xy=(xi, ei), xytext=(xi + 0.18, ei),
                fontsize=7.5, color=RED,
                arrowprops=dict(arrowstyle='->', color=RED, lw=0.8))

ax.axhline(0, color=GREY, linewidth=0.8, linestyle='--')
ax.set_xlabel("Known distance (m)")
ax.set_ylabel("Std. dev / bias  (mm)")
ax.set_title("(c) Noise & Angular Bias")
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
ax.set_xlim(0.2, 2.4)
ax.grid(True, linestyle=':', alpha=0.6)
ax.legend(loc="upper left", framealpha=0.9)

# ── Save ────────────────────────────────────────────────────────────────────
for ext in ("pdf", "png"):
    out = os.path.join(DATA_DIR, f"lidar_calibration_plot.{ext}")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved: {out}")

plt.close(fig)
print("Done.")
