"""
Offline Particle Filter Analysis
Runs the PF on 1_straight, 2_scurve, and 3_ucurve and generates
summary plots of the estimated robot location at each timestep.
"""

import os
import sys
import glob

import matplotlib
matplotlib.use('TkAgg')  # interactive backend; change to 'Agg' to save only
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# Ensure we can import project modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import parameters
import data_handling
from particle_filter import ParticleFilter, Map, State
from robot_python_code import RobotOdomSignal, RobotSensorSignal


# ---------------------------------------------------------------------------
# Data files: pick the most recent file per trajectory automatically.
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

def _latest(pattern: str) -> str:
    matches = sorted(glob.glob(os.path.join(DATA_DIR, pattern)))
    if not matches:
        raise FileNotFoundError(f"No file matching {pattern} in {DATA_DIR}")
    return matches[-1]

DATA_FILES = {
    '1_straight': _latest('1_straight*.pkl'),
    '2_scurve':   _latest('2_scurve*.pkl'),
    '3_ucurve':   _latest('3_ucurve*.pkl'),
}

# Ground truth start/end positions in cm (x_start, y_start, x_end, y_end).
# Divide by 100 to convert to metres for plotting.
_cm = 0.01
GROUND_TRUTH = {
    '1_straight': {'start': (39  * _cm,  244   * _cm), 'end': (48.5 * _cm,  52.5 * _cm)},
    '2_scurve':   {'start': (8  * _cm,    50   * _cm), 'end': (139.5 * _cm, 68.5 * _cm)},
    '3_ucurve':   {'start': (220.5 * _cm,  158.5 * _cm), 'end': (4.5 * _cm,  131.5 * _cm)},
}


# ---------------------------------------------------------------------------
# Core: run the PF on one trial and collect per-timestep estimates
# ---------------------------------------------------------------------------
def run_pf(filename: str, traj_name: str) -> dict:
    """Run offline PF and return a dict of time-series arrays."""
    print(f"\n=== {traj_name}  ({os.path.basename(filename)}) ===")

    map_ = Map(parameters.wall_corner_list, parameters.grid_dimensions)
    pf_data = data_handling.get_file_data_for_pf(filename)

    pf = ParticleFilter(
        num_particles=parameters.num_particles,
        map=map_,
        initial_state=State(0.5, 2.0, 1.57),
        state_stdev=State(0.1, 0.1, 0.1),
        known_start_state=False,
        encoder_counts_0=pf_data[0][2].encoder_counts,
    )

    times, xs, ys, thetas, spreads = [], [], [], [], []

    for t in range(1, len(pf_data)):
        row = pf_data[t]
        delta_t: float = pf_data[t][0] - pf_data[t - 1][0]

        u_t: RobotOdomSignal = RobotOdomSignal(row[1][0], row[1][1])
        z_t: RobotSensorSignal = row[2]
        u_t.encoder_total_count = z_t.encoder_counts

        pf.update(u_t, z_t, delta_t)

        ms = pf.particle_set.mean_state
        times.append(pf_data[t][0])
        xs.append(ms.x)
        ys.append(ms.y)
        thetas.append(ms.theta)

        px = [p.state.x for p in pf.particle_set.particle_list]
        py = [p.state.y for p in pf.particle_set.particle_list]
        spreads.append(float(np.std(px) + np.std(py)))

        if t % 100 == 0:
            print(f"  t={pf_data[t][0]:5.1f}s  "
                  f"x={ms.x:.3f}  y={ms.y:.3f}  θ={np.degrees(ms.theta):.1f}°")

    print(f"  Done — {len(times)} timesteps")
    return {
        'map':     map_,
        'times':   np.array(times),
        'xs':      np.array(xs),
        'ys':      np.array(ys),
        'thetas':  np.array(thetas),
        'spreads': np.array(spreads),
    }


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def _draw_map(ax, map_, grid_dims):
    for wall in map_.wall_list:
        ax.plot([wall.corner1.x, wall.corner2.x],
                [wall.corner1.y, wall.corner2.y], 'k-', linewidth=2)
    x_min, x_max = grid_dims[0]
    y_min, y_max = grid_dims[1]
    ax.plot([x_min, x_max, x_max, x_min, x_min],
            [y_min, y_min, y_max, y_max, y_min], 'k--', alpha=0.3, linewidth=1)
    ax.set_xlim(x_min - 0.15, x_max + 0.15)
    ax.set_ylim(y_min - 0.15, y_max + 0.15)


def plot_trial(res: dict, traj_name: str, output_dir: str):
    """Four-panel figure for one trajectory."""
    times   = res['times']
    xs      = res['xs']
    ys      = res['ys']
    thetas  = res['thetas']
    spreads = res['spreads']
    gt      = GROUND_TRUTH.get(traj_name)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(f'Particle Filter — {traj_name}', fontsize=14, fontweight='bold')

    # ── Panel A: 2-D trajectory coloured by time ──────────────────────────
    ax = axes[0, 0]
    _draw_map(ax, res['map'], parameters.grid_dimensions)

    norm  = plt.Normalize(times.min(), times.max())
    cvals = cm.viridis(norm(times))
    for i in range(len(times) - 1):
        ax.plot(xs[i:i+2], ys[i:i+2], color=cvals[i], linewidth=1.5)

    sc = ax.scatter(xs, ys, c=times, cmap='viridis', s=12, zorder=5)
    fig.colorbar(sc, ax=ax, label='Time (s)', shrink=0.85)
    ax.scatter(xs[0],  ys[0],  color='lime',  s=80, zorder=6, label='PF start', marker='o')
    ax.scatter(xs[-1], ys[-1], color='red',   s=100, zorder=6, label='PF end',  marker='*')
    if gt:
        ax.scatter(*gt['start'], color='cyan',   s=120, zorder=7, label='GT start', marker='D', edgecolors='k', linewidths=0.8)
        ax.scatter(*gt['end'],   color='magenta', s=120, zorder=7, label='GT end',   marker='D', edgecolors='k', linewidths=0.8)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Estimated 2-D Trajectory')
    ax.set_aspect('equal', adjustable='box')
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.3)

    # ── Panel B: x(t) and y(t) ────────────────────────────────────────────
    ax = axes[0, 1]
    ax.plot(times, xs, 'b-', linewidth=1.5, label='x (m)')
    ax.plot(times, ys, 'r-', linewidth=1.5, label='y (m)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Position (m)')
    ax.set_title('x(t) and y(t)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Panel C: heading θ(t) ─────────────────────────────────────────────
    ax = axes[1, 0]
    ax.plot(times, np.degrees(thetas), color='darkgreen', linewidth=1.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Heading θ (deg)')
    ax.set_title('Estimated Heading θ(t)')
    ax.grid(True, alpha=0.3)

    # ── Panel D: particle spread ───────────────────────────────────────────
    ax = axes[1, 1]
    ax.plot(times, spreads, color='purple', linewidth=1.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Spread (m)')
    ax.set_title('Particle Spread  (σ_x + σ_y)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, f'{traj_name}_pf.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Saved → {path}")
    return fig


def plot_combined(all_results: dict, output_dir: str):
    """Overlay all three trajectories on one map + time-series comparison."""
    COLORS = {'1_straight': 'royalblue', '2_scurve': 'darkorange', '3_ucurve': 'crimson'}

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('All Trajectories — Particle Filter Estimates', fontsize=13, fontweight='bold')

    # ── Left: overlay on map ──────────────────────────────────────────────
    ax = axes[0]
    first = next(iter(all_results.values()))
    _draw_map(ax, first['map'], parameters.grid_dimensions)

    gt_start_done = False
    gt_end_done   = False
    for name, res in all_results.items():
        c  = COLORS.get(name, 'gray')
        gt = GROUND_TRUTH.get(name)
        ax.plot(res['xs'], res['ys'], '-', color=c, linewidth=2, label=name)
        ax.scatter(res['xs'][0],  res['ys'][0],  color=c, s=70,  marker='o')
        ax.scatter(res['xs'][-1], res['ys'][-1], color=c, s=100, marker='*')
        if gt:
            ax.scatter(*gt['start'], color='cyan',    s=120, zorder=7, marker='D',
                       edgecolors='k', linewidths=0.8,
                       label='GT start' if not gt_start_done else None)
            ax.scatter(*gt['end'],   color='magenta', s=120, zorder=7, marker='D',
                       edgecolors='k', linewidths=0.8,
                       label='GT end'   if not gt_end_done   else None)
            gt_start_done = True
            gt_end_done   = True

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Estimated Trajectories')
    ax.set_aspect('equal', adjustable='box')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Right: x(t) comparison ────────────────────────────────────────────
    ax = axes[1]
    for name, res in all_results.items():
        c = COLORS.get(name, 'gray')
        ax.plot(res['times'], res['xs'], '-',  color=c, linewidth=1.8, label=f'{name} x')
        ax.plot(res['times'], res['ys'], '--', color=c, linewidth=1.2, alpha=0.7, label=f'{name} y')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Position (m)')
    ax.set_title('x(t) — solid    y(t) — dashed')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'all_trajectories_pf.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Saved → {path}")
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'analysis_plots')
    os.makedirs(output_dir, exist_ok=True)

    all_results = {}
    figs = []
    for traj_name, filename in DATA_FILES.items():
        res = run_pf(filename, traj_name)
        all_results[traj_name] = res
        figs.append(plot_trial(res, traj_name, output_dir))

    figs.append(plot_combined(all_results, output_dir))

    print(f"\nAll plots saved to: {output_dir}")
    plt.show()
