# External libraries
import math
import numpy as np

# UDP parameters
localIP = "192.168.0.200" # Put your laptop computer's IP here
arduinoIP = "192.168.0.199" # Put your arduino's IP here
localPort = 4010
arduinoPort = 4010
bufferSize = 1024

# Camera parameters
camera_id = 0
marker_length = 0.071
camera_matrix = np.array([[
            1538.7939968143703,
            0.0,
            829.2140966459408
        ],
        [
            0.0,
            1537.2558443313133,
            577.1496540648694
        ],
        [
            0.0,
            0.0,
            1.0
        ]], dtype=np.float32)
dist_coeffs = np.array([
            -0.3965206668901596,
            0.1469937626365962,
            -0.0007861794226302347,
            0.0022527358131867437,
            -0.02308737414959782
        ], dtype=np.float32)


# Robot parameters
num_robot_sensors = 2 # encoder, steering
num_robot_control_signals = 2 # speed, steering

# Logging parameters
max_num_lines_before_write = 1
filename_start = './data/robot_data'
data_name_list = ['time', 'control_signal', 'robot_sensor_signal', 'camera_sensor_signal', 'state_mean', 'state_covariance']

# Experiment trial parameters
trial_time = 10000 # milliseconds
extra_trial_log_time = 2000 # milliseconds

# KF parameters
I3 = np.array([[1, 0, 0],[0, 1, 0], [0, 0, 1]])
covariance_plot_scale = 100