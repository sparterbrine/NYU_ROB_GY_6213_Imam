import pandas as pd
import numpy as np
import os

# --- 1. CONFIGURATION ---
INPUT_FILE = "characterization_output.csv"

# --- 2. HELPER FUNCTIONS ---
def get_circular_mean(angles_deg):
    """Safely averages angles, ignoring 360 boundaries."""
    angles_rad = np.radians(angles_deg)
    sin_mean = np.mean(np.sin(angles_rad))
    cos_mean = np.mean(np.cos(angles_rad))
    return np.degrees(np.arctan2(sin_mean, cos_mean))

def get_angular_residuals(angles_deg, mean_angle_deg):
    """Calculates the shortest path (-180 to +180) from the mean."""
    return (angles_deg - mean_angle_deg + 180) % 360 - 180

# --- 3. MAIN ANALYSIS ---
def analyze_bulletproof_covariance():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Could not find '{INPUT_FILE}'.")
        return

    df = pd.read_csv(INPUT_FILE)
    if df.empty:
        return

    print("--- BULLET-PROOF POSE COVARIANCE ANALYSIS ---")
    print("Using shortest-path circular math and rejecting single-frame glitches.\n")
    
    all_cleaned_residuals = []

    for run_id, group in df.groupby('Run'):
        if len(group) < 5: 
            continue
            
        # 1. Find the Median Center (Immune to single-frame glitches)
        med_x = group['X'].median()
        med_y = group['Y'].median()
        
        # Circular median logic for Yaw
        angles_rad = np.radians(group['Yaw'])
        med_yaw = np.degrees(np.arctan2(np.median(np.sin(angles_rad)), np.median(np.cos(angles_rad))))

        # 2. Measure the shortest distance of every point to that median center
        dist_x = np.abs(group['X'] - med_x)
        dist_y = np.abs(group['Y'] - med_y)
        res_yaw_from_median = get_angular_residuals(group['Yaw'], med_yaw)
        
        # 3. Filter out random glitches (Keep points within 0.2m and 20 degrees of the main cluster)
        mask = (dist_x < 0.2) & (dist_y < 0.2) & (np.abs(res_yaw_from_median) < 20)
        cleaned_group = group[mask].copy()

        if len(cleaned_group) < 2:
            print(f"Run {run_id}: Too much noise to find a stable cluster.")
            continue

        # 4. Calculate actual Mean and Variance on the clean, glitch-free cluster
        clean_mean_yaw = get_circular_mean(cleaned_group['Yaw'])
        residuals = pd.DataFrame({
            'X': cleaned_group['X'] - cleaned_group['X'].mean(),
            'Y': cleaned_group['Y'] - cleaned_group['Y'].mean(),
            'Yaw': get_angular_residuals(cleaned_group['Yaw'], clean_mean_yaw) # Shortest path variance!
        })

        run_cov = residuals.cov()
        
        print(f"=== RUN {run_id} ===")
        print(f"Kept {len(cleaned_group)} / {len(group)} stable samples")
        print(f"Estimated Center: X: {cleaned_group['X'].mean():.3f} m, Y: {cleaned_group['Y'].mean():.3f} m, Yaw: {clean_mean_yaw:.2f} deg")
        print("\n  [ RUN COVARIANCE MATRIX ]")
        print(run_cov.to_string(float_format=lambda x: f"{x:.6f}"))
        print("-" * 40 + "\n")

        all_cleaned_residuals.append(residuals)

    # --- CALCULATE TOTAL (POOLED) COVARIANCE ---
    if all_cleaned_residuals:
        print("=== TOTAL POOLED ROBUST COVARIANCE ===")
        print("This is the matrix you should plug into your Kalman Filter.")
        
        combined_residuals = pd.concat(all_cleaned_residuals)
        total_cov_matrix = combined_residuals.cov()
        
        print(f"\nTotal clean samples analyzed: {len(combined_residuals)}")
        print("\n  [ OVERALL COVARIANCE MATRIX ]")
        print(total_cov_matrix.to_string(float_format=lambda x: f"{x:.6f}"))
        print("===============================\n")

if __name__ == "__main__":
    analyze_bulletproof_covariance()