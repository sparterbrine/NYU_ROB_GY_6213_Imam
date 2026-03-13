# External libraries
import numpy as np
from typing import Dict

# UDP parameters
localIP = "192.168.0.199" # Put your laptop computer's IP here 199
arduinoIP = "192.168.0.198" # Put your arduino's IP here 200
localPort = 4010
arduinoPort = 4010
bufferSize = 2048

# Camera parameters
camera_id = 1
marker_length = 0.094488 # meters
''' ArUco marker parameters from calibration - the length of the entire thing(not a pixel in it) '''
camera_matrix = np.array([
        [
            682.4174696857553,
            0.0,
            260.6779690883052
        ],
        [
            0.0,
            683.5970425159574,
            252.2280650657449
        ],
        [
            0.0,
            0.0,
            1.0
        ]
    ], dtype=np.float64)
dist_coeffs = np.array([
            -0.3677610637638757,
            0.07099775179738238,
            -0.0007576824618998384,
            0.0028280822538041144,
            0.06022962819399466
        ], dtype=np.float64)


# Robot parameters
num_robot_sensors = 2 # encoder, steering
num_robot_control_signals = 2 # speed, steering

KNOWN_MARKERS: Dict[int, Dict[str, float]] = {
        # 1: {'x': 1.05, 'y': 0.85, 'yaw': 0},   # Close Left
        # 3: {'x': 1.05, 'y': -0.95, 'yaw': 90}, # Close Right
        # 6: {'x': 1.05, 'y': 0.05,  'yaw': 90}, # Close Center
        # 2: {'x': 2.05, 'y': 0.05,  'yaw': 0},  # Mid Center
        # 4: {'x': 3.05, 'y': 0.85, 'yaw': 180}, # Far Left
        # 5: {'x': 3.05, 'y': -0.95,  'yaw': 90},# Far Right

        1: {'x': 0.365, 'y': -1.210, 'yaw': -90},
        2: {'x': 1.485, 'y': -0.263, 'yaw': -90},
        3: {'x': 0.244, 'y': -0.166, 'yaw': -90},
        4: {'x': 1.115, 'y': -1.852, 'yaw': -90},
        5: {'x': 1.064, 'y': -0.603, 'yaw': -90},
        6: {'x': 0.979, 'y': -1.180, 'yaw': -90},
        # 7: {'x': 1.554, 'y': 2.115, 'yaw': -90},
        8: {'x': 0.164, 'y': -2.135, 'yaw': -90},
    }

# Logging parameters
max_num_lines_before_write = 1
filename_start = './data/robot_data'
data_name_list = ['time', 'control_signal', 'robot_sensor_signal', 'camera_sensor_signal', 'state_mean', 'state_covariance', 'frame']

# Experiment trial parameters
trial_time = 10000 # milliseconds
extra_trial_log_time = 2000 # milliseconds

# KF parameters
I3 = np.array([[1, 0, 0],[0, 1, 0], [0, 0, 1]])
covariance_plot_scale = 100

# PF parameters, modify the map and num particles as you see fit.
initial_state_x: float = .5
initial_state_y: float = 2.25
initial_state_theta: float = 0.0
initial_state_stdev_x: float = 0.25
initial_state_stdev_y: float = 0.25
initial_state_stdev_theta: float = 3.0
num_particles = 225
distance_variance = 0.05       # sensor noise variance (metres²) — used in weight calculation
motion_distance_variance = 0.05 # motion noise std dev (metres) — used in propagate_state
theta_variance = 0.05
grid_dimensions = [[0., 2.5], [0., 2.]] # [x_min, x_max], [y_min, y_max]
_d = (0.15*np.sqrt(2)/2)
clustering_radius = 2.
use_clustering: bool = False
lidar_max_range_m: float = 12  # meters; rays at or beyond this distance are ignored (open-space returns)
wall_corner_list = [
    [0.0, 0.0, 2.440, 0.0],
    [2.440, 0.0, 2.440, 1.880],
    [2.440, 1.880, 1.820, 1.880],
    [1.820, 1.880, 1.820, 1.240],
    [1.820, 1.240, 0.600, 1.240],
    [0.600, 1.240, 0.600, 1.880],
    [0.600, 1.880, 0.0, 1.880],
    [0.0, 1.880, 0.0, 0.0]
    # #box 1
    # [1.0, 0.55, 1.0, 0.70],
    # [1.0, 0.70, 1.15, 0.70],
    # [1.15, 0.70, 1.15, 0.55],
    # [1.15, 0.55, 1.0, 0.55],

    # #box2
    # [3.0, -0.35, 3.0, -0.5],
    # [3.0, -0.50, 3.15, -0.50],
    # [3.15, -0.5, 3.15, -0.35],
    # [3.15, -0.35, 3.0, -0.35],

    # #box 3
    # [1.0, -0.5, 1.0 + _d, -0.5 - _d],
    # [1.0 + _d, -0.5 - _d, 1.0, -0.5 - 2*_d],
    # [1.0, -0.5 - 2*_d, 1.0 - _d, -0.5 - _d],
    # [1.0 - _d, -0.5 - _d, 1.0, -0.5],
    ]