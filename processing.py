import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

def fit_arc(df):
    # Function to calculate residuals (distance from points to circle circumference)
    def calc_residuals(params, x, y):
        h, k, r = params
        return np.sqrt((x - h)**2 + (y - k)**2) - r

    # Initial guess: Center is far to the side (large R), since drift is usually slight
    x_m, y_m = df['x'].mean(), df['y'].mean()
    initial_guess = [0, 10, 10] 
    
    result = least_squares(calc_residuals, initial_guess, args=(df['x'], df['y']))
    h_opt, k_opt, r_opt = result.x
    
    # Generate points for the fitted arc
    # Calculate start and end angles based on data
    angles = np.arctan2(df['y'] - k_opt, df['x'] - h_opt)
    theta_range = np.linspace(np.arctan2(0 - k_opt, 0 - h_opt), max(angles), 100)
    
    arc_x = h_opt + r_opt * np.cos(theta_range)
    arc_y = k_opt + r_opt * np.sin(theta_range)
    
    return arc_x, arc_y, r_opt

def analyze_robot_drift(df):
    # Calculate angle for each datapoint in degrees
    # theta = atan2(y, x) -> Angle of deviation from the intended straight path
    df['angle_deg'] = np.degrees(np.arctan2(df['y'], df['x']))
    
    # Calculate Average Movement Vector
    avg_x, avg_y = df['x'].mean(), df['y'].mean()
    avg_angle = np.degrees(np.arctan2(avg_y, avg_x))
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(df['x'], df['y'], color='blue', label='Final Positions')
    
    # Draw trajectory lines from origin
    for i, row in df.iterrows():
        plt.plot([0, row['x']], [0, row['y']], color='gray', linestyle='--', alpha=0.3)
        
    # Draw average movement vector (The 'Resultant' path)
    plt.arrow(0, 0, avg_x, avg_y, head_width=0.1, head_length=0.15, fc='red', ec='red', 
              label=f'Avg Drift: {avg_angle:.2f}°')
    
    plt.axhline(0, color='black', linewidth=1)
    plt.title('Robot Final Positions & Drift Angles')
    plt.xlabel('X Position (Forward)')
    plt.ylabel('Y Position (Lateral Drift)')
    plt.legend()
    plt.axis('equal')
    plt.grid(True, linestyle=':', alpha=0.7)
    
    plt.savefig('robot_movement_plot.png')
    return df, avg_angle
data = {
    'Time':[8,8,8,8,8,8,8,8,8,8,5,5,5,5,5,5,5,5,5,5,2.5,2.5,2.5,2.5,2.5,2.5,2.5,2.5,2.5,2.5],
    'x':[298.5,283,284,284,288.5,279,288.5,279,287.5,279.5,179,184,181,173.5,186,178,180,178,179,180,91,89,88.5,90.5,86.5,92,89,91,91.5,90.5],
    'y':[13.5,62,46,25.5,29.5,67.5,57,65.5,7.5,51,29.5,30,26,25,26,27,26,24.5,26.5,-3.5,7,4,6.5,6,5,7.5,9,7.5,7,5]
}
processed_df, drift = analyze_robot_drift(pd.DataFrame(data))
diff1, diff2 = fit_arc(pd.DataFrame(data))
# Example usage with sample data:
# processed_df, mean_drift = analyze_robot_drift(your_dataframe)