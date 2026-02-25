import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('characterization_output.csv')

def get_angular_residuals(angles_deg, mean_angle_deg):
    """Calculates shortest path distance on a circle."""
    return (angles_deg - mean_angle_deg + 180) % 360 - 180

for run_id in df['Run'].unique():
    run_data = df[df['Run'] == run_id]
    
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Pose Estimation Diagnostics - Run {run_id}', fontsize=16)
    
    # 1. Yaw over time: Shows the "jumps" between clusters
    axs[0, 0].plot(run_data.index, run_data['Yaw'], 'b.', markersize=2)
    axs[0, 0].set_title('Yaw vs Sample Index (Timeline)')
    axs[0, 0].set_ylabel('Yaw (degrees)')
    axs[0, 0].grid(True)
    
    # 2. X-Y Scatter: Shows "ghost" positions
    axs[0, 1].scatter(run_data['X'], run_data['Y'], c='r', s=2, alpha=0.5)
    axs[0, 1].set_title('X-Y Position Scatter (Top-down)')
    axs[0, 1].set_xlabel('X (m)')
    axs[0, 1].set_ylabel('Y (m)')
    axs[0, 1].axis('equal')
    axs[0, 1].grid(True)
    
    # 3. Histogram of Raw Yaw: Shows multiple peaks (non-Gaussian)
    axs[1, 0].hist(run_data['Yaw'], bins=100, color='g', alpha=0.7)
    axs[1, 0].set_title('Raw Yaw Distribution')
    axs[1, 0].set_xlabel('Degrees')
    
    # 4. Residuals: Show the noise spread once "flipped" data is centered
    med_yaw = run_data['Yaw'].median()
    residuals_yaw = get_angular_residuals(run_data['Yaw'], med_yaw)
    axs[1, 1].hist(residuals_yaw, bins=100, color='m', alpha=0.7)
    axs[1, 1].set_title('Yaw Deviation from Median')
    axs[1, 1].set_xlabel('Degrees Error')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'run_{run_id}_analysis.png')
    print(f"Generated analysis plot for Run {run_id}")