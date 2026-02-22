# External Libraries
import matplotlib.pyplot as plt
from pathlib import Path
import math
import numpy as np
from matplotlib.patches import Ellipse
# Internal Libraries
import parameters
import robot_python_code
import motion_models

# Open a file and return data in a form ready to plot
def get_file_data(filename):
    data_loader = robot_python_code.DataLoader(filename)
    data_dict = data_loader.load()

    # The dictionary should have keys ['time', 'control_signal', 'robot_sensor_signal', 'camera_sensor_signal']
    time_list = data_dict['time']
    control_signal_list = data_dict['control_signal']
    robot_sensor_signal_list = data_dict['robot_sensor_signal']
    encoder_count_list = []
    velocity_list = []
    steering_angle_list = []
    for row in robot_sensor_signal_list:
        encoder_count_list.append(row.encoder_counts)
    for row in control_signal_list:
        velocity_list.append(row[0])
        steering_angle_list.append(row[1])
    
    return time_list, encoder_count_list, velocity_list, steering_angle_list


# For a given trial, plot the encoder counts, velocities, steering angles
def plot_trial_basics(filename):
    time_list, encoder_count_list, velocity_list, steering_angle_list = get_file_data(filename)
    
    plt.plot(time_list, encoder_count_list)
    plt.title('Encoder Values')
    plt.show()
    plt.plot(time_list, velocity_list)
    plt.title('Speed')
    plt.show()
    plt.plot(time_list, steering_angle_list)
    plt.title('Steering')
    plt.show()


# Plot a trajectory using the motion model, input data ste from a single trial.
def run_my_model_on_trial(filename, show_plot = True, plot_color = 'ko'):
    time_list, encoder_count_list, velocity_list, steering_angle_list = get_file_data(filename)
    
    motion_model = motion_models.MyMotionModel([0,0,0], 0)
    x_list, y_list, theta_list = motion_model.traj_propagation(time_list, encoder_count_list, steering_angle_list)

    plt.plot(x_list, y_list, plot_color)
    plt.title('Motion Model Predicted XY Traj (m)')
    plt.axis((-1, 3, -2, 2))
    if show_plot:
        plt.show()


# Iterate through many trials and plot them as trajectories with motion model
def plot_many_trial_predictions(directory):
    directory_path = Path(directory)
    plot_color_list = ['r.','k.','g.','c.', 'b.', 'r.','k.','g.','c.', 'b.','r.','k.','g.','c.', 'b.', 'r.','k.','g.','c.', 'b.']
    count = 0
    for item in directory_path.iterdir():
        filename = item.name
        plot_color = plot_color_list[count]
        run_my_model_on_trial(directory + filename, False, plot_color)
        count += 1
    plt.axis((0.1, 3.6, -1., 0.5))
    plt.show()

# Calculate the predicted distance from single trial for a motion model
def run_my_model_to_predict_distance(filename):
    time_list, encoder_count_list, velocity_list, steering_angle_list = get_file_data(filename)
    motion_model = motion_models.MyMotionModel([0,0,0], 0)
    x_list, _, _ = motion_model.traj_propagation(time_list, encoder_count_list, steering_angle_list)
    distance = x_list[-30]
    
    return distance

# Calculate the differences between two lists, and square them.
def get_diff_squared(m_list, p_list):
    diff_squared_list = []
    for i in range(len(m_list)):
        diff_squared = math.pow(m_list[i]-p_list[i],2)
        diff_squared_list.append(diff_squared)

    coefficients = np.polyfit(m_list, diff_squared_list, 2)
    p=np.poly1d(coefficients)

    plt.plot(m_list, diff_squared_list,'ko')
    plt.plot(m_list, p(m_list),'ro')
    x_fit = np.linspace(min(m_list), max(m_list), 100)
    plt.plot(x_fit, p(x_fit), 'r-', label='Fit')
    plt.title("Error Squared (m^2)")
    plt.xlabel('Measured distance travelled (m)')
    plt.ylabel('(Actual - Predicted)^2 (m^2)')
    plt.show()

    return diff_squared_list


# Open files, plot them to predict with the motion model, and compare with real values
def process_files_and_plot(files_and_data, directory):
    predicted_distance_list = []
    measured_distance_list = []
    for row in files_and_data:
        filename = row[0]
        print(directory + filename)
        measured_distance = row[1]
        measured_distance_list.append(measured_distance)
        predicted_distance = run_my_model_to_predict_distance(directory + filename)
        predicted_distance_list.append(predicted_distance)

    # Plot predicted and measured distance travelled.
    plt.plot(measured_distance_list, predicted_distance_list, 'ko')
    plt.plot([0, 4.],[0, 4.])
    plt.title('Distance Trials')
    plt.xlabel('Measured Distance (m)')
    plt.ylabel('Predicted Distance (m)')
    plt.legend(['Measured vs Predicted', 'Slope 1 Line'])
    plt.show()

    # Plot the associated variance
    get_diff_squared(measured_distance_list, predicted_distance_list)


# Sample and plot some simulated trials
def sample_model(num_samples):
    traj_duration = 10
    for i in range(num_samples):
        model = motion_models.MyMotionModel([0,0,0], 0)
        t_list, traj_x, traj_y, traj_theta = model.generate_simulated_traj(traj_duration)
        plt.plot(traj_x, traj_y, 'k.')

    plt.title('Sampling the model')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.show()
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import math
import random

# Ensure you have your motion_models imported or defined above
# import motion_models 

def sim_trial(num_simulations=100, traj_duration=2):
    plt.figure(figsize=(10, 6))
    
    final_points = []
    # Use a colormap to differentiate trajectories
    colors = plt.cm.jet(np.linspace(0, 1, num_simulations))

    # --- 1. Run Simulations ---
    for i in range(num_simulations):
        # Initialize model
        model = motion_models.MyMotionModel([0, 0, 0], 0)
        
        # Run simulation
        _, traj_x, traj_y, _ = model.generate_simulated_traj(traj_duration)
        
        if not traj_x: continue
            
        # Plot path (faint)
        plt.plot(traj_x, traj_y, '-', color=colors[i], alpha=0.1, linewidth=1)
        
        # Plot final dot (solid)
        plt.plot(traj_x[-1], traj_y[-1], '.', color=colors[i], alpha=0.6, markersize=3)
        
        # Store final (x, y)
        final_points.append([traj_x[-1], traj_y[-1]])

    final_points = np.array(final_points)

    # --- 2. Calculate 2-Sigma Statistics ---
    if len(final_points) > 1:
        # Calculate Mean and Covariance
        mean = np.mean(final_points, axis=0)
        cov = np.cov(final_points, rowvar=False)
        
        # Get eigenvalues (variance) and eigenvectors (rotation)
        lambda_, v = np.linalg.eig(cov)
        lambda_ = np.sqrt(lambda_) # Convert variance to std dev
        
        # Sort so the largest spread is mapped to width (major axis)
        idx = np.argmax(lambda_)
        angle = np.degrees(np.arctan2(v[1, idx], v[0, idx]))
        
        # SCALE FACTOR: 2 Standard Deviations
        # Width/Height = 2 * (2 * sigma)
        scale_factor = 2
        width = 2 * (scale_factor * lambda_[idx])
        height = 2 * (scale_factor * lambda_[1-idx])

        # Draw the 2-Sigma Ellipse
        ell = Ellipse(xy=mean, width=width, height=height, angle=angle, 
                      edgecolor='black', fc='None', lw=2, linestyle='--', label='2-Sigma Boundary')
        plt.gca().add_patch(ell)
        
        # --- 3. Calculate Percentage Inside ---
        # We use Mahalanobis distance. For 2-sigma, the threshold is 2^2 = 4.
        inv_cov = np.linalg.inv(cov)
        inside_count = 0
        
        for p in final_points:
            diff = p - mean
            # Calculate Mahalanobis distance squared: (x-u)'S^-1(x-u)
            dist_sq = diff.T @ inv_cov @ diff
            
            # Check if inside 2-sigma (dist <= 2, so dist^2 <= 4)
            if dist_sq <= 4.0:
                inside_count += 1
                
        percent_inside = (inside_count / len(final_points)) * 100
        
        # Plot Mean
        plt.plot(mean[0], mean[1], 'kx', markersize=10, markeredgewidth=2, label='Mean Position')
        
        # Update Title
        plt.title(f'Simulations: {num_simulations} | Inside 2-Sigma: {percent_inside:.1f}% (Expected ~86.5%)')

    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.show()
######### MAIN ########

# Some sample data to test with
files_and_data = [
    ['robot_data_0_-1_09_02_26_16_17_04.pkl', 289/100], # filename, measured distance in meters
    ['robot_data_0_-5_09_02_26_16_20_28.pkl', 166/100],
    ['robot_data_0_-20_09_02_26_16_19_12.pkl', 63/100],
    ['robot_data_0_0_09_02_26_16_14_08.pkl', 320/100],
    ['robot_data_0_5_09_02_26_16_21_53.pkl', 98/100],
    ['data_0_0_09_02_26_19_16_33.pkl', 182/100]
    ]

# Plot the motion model predictions for a single trial
if False:
    filename = './data/data_0_0_09_02_26_19_16_33.pkl'
    run_my_model_on_trial(filename)
    time_list, encoder_count_list, velocity_list, steering_angle_list = get_file_data(filename)
    run_my_model_on_trial(filename)

# Plot the motion model predictions for each trial in a folder
if False:
    directory = ('./data/')
    plot_many_trial_predictions(directory)

# A list of files to open, process, and plot - for comparing predicted with actual distances
if False:
    directory = ('./data/')    
    process_files_and_plot(files_and_data, directory)

# Try to sample with the motion model
if True:
    sim_trial(200)
