# External libraries
import math
import numpy as np

# UDP parameters
localIP = "192.168.0.199" # Put your laptop computer's IP here 199
arduinoIP = "192.168.0.198" # Put your arduino's IP here 200
localPort = 4010
arduinoPort = 4010
bufferSize = 1024

# Camera parameters
camera_id = 0
marker_length = 0.071
camera_matrix = np.array([[1.41089024e+03, 0.00000000e+00 ,5.34757040e+02],
 [0.00000000e+00 ,1.40977771e+03, 4.63300611e+02],
 [0.00000000e+00 ,0.00000000e+00 ,1.00000000e+00]], dtype=np.float32)
dist_coeffs = np.array([-0.32511173, -0.09273864 ,-0.00295959 , 0.00111094 , 0.2446519 ], dtype=np.float32)


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