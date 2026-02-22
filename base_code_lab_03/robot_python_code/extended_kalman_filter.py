# External libraries
import json
from typing import List, Tuple

import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# Local libraries
import parameters
import data_handling

from motion_models import distance_travelled_s, rotational_velocity_w, state_prediction, State

# Main class
class ExtendedKalmanFilter:
    def __init__(self, x_0: State, Sigma_0, encoder_counts_0: int):
        self.state_mean: State = x_0
        self.state_covariance = np.array(Sigma_0, dtype=float)
        self.predicted_state_mean: State = State(0, 0, 0)
        self.predicted_state_covariance = parameters.I3 * 1.0
        self.last_encoder_counts: int = encoder_counts_0
        self._q_matrix = None

    # Call the prediction and correction steps
    def update(self, u_t, z_t, delta_t):
        """ u_t =Encoder counts, Steering Angle
         z_t = Camera X, Camera Y, Camera Theta"""
        self.prediction_step(u_t, delta_t)
        self.correction_step(z_t)
        # Update encoder memory for the next step's delta calculation
        self.last_encoder_counts = u_t[0] 

    # Set the EKF's predicted state mean and covariance matrix
    def prediction_step(self, u_t, delta_t):
        """ u_t =Encoder counts, Steering Angle"""
        self.predicted_state_mean, s = self.g_function(self.state_mean, u_t, delta_t)
        
        delta_theta = rotational_velocity_w(u_t[1]) * delta_t
        '''In Degrees'''
        G_x = self.get_G_x(self.state_mean, s, delta_theta)
        G_u = self.get_G_u(self.state_mean, delta_t)
        R = self.get_R(s)
        
        self.predicted_state_covariance = G_x @ self.state_covariance @ G_x.T + G_u @ R @ G_u.T

    # Set the EKF's corrected state mean and covariance matrix
    def correction_step(self, z_t):
        """ z_t = Camera X, Camera Y, Camera Theta"""
        H = self.get_H()
        Q = self.get_Q() #z_t covariance matrix from camera sensor model
        
        # Kalman Gain
        S = H @ self.predicted_state_covariance @ H.T + Q
        K = self.predicted_state_covariance @ H.T @ np.linalg.inv(S)
        
        # Innovation (Difference between measurement and prediction)
        z_pred = self.get_h_function(self.predicted_state_mean)
        innovation = z_t - z_pred
        
        # Wrap angle innovation to [-180, 180] to prevent spinning errors
        innovation[2] = (innovation[2] + 180) % 360 - 180
        
        # Update State and Covariance
        self.state_mean = self.predicted_state_mean + K @ innovation
        self.state_covariance = (np.eye(3) - K @ H) @ self.predicted_state_covariance

    # The nonlinear transition equation that provides new states from past states
    def g_function(self, x_tm1: State, u_t: np.ndarray, delta_t: float) -> Tuple[State, float]:
        encoder_counts, steering_angle_command = u_t
        delta_encoder: int = encoder_counts - self.last_encoder_counts
        s: float = distance_travelled_s(delta_encoder)
        
        w_deg: float = rotational_velocity_w(steering_angle_command)
        delta_theta: float = w_deg * delta_t
        
        theta_rad: float = math.radians(x_tm1.theta)
        theta_t: float = theta_rad + math.radians(delta_theta)
        mid_theta: float = (theta_rad + math.radians(delta_theta)) / 2.0
        
        x_t: float = x_tm1.x + s * math.cos(mid_theta)
        y_t: float = x_tm1.y + s * math.sin(mid_theta)
        
        # Keep theta bounds clean
        theta_t = (theta_t + math.pi) % (2 * math.pi) - math.pi
        
        return State(x_t, y_t, theta_t), s
    
    # The nonlinear measurement function
    def get_h_function(self, x_t):
        return x_t
    
    # This function returns a matrix with the partial derivatives dg/dx
    # g outputs x_t, y_t, theta_t, and we take derivatives wrt inputs x_tm1, y_tm1, theta_tm1
    def get_G_x(self, x_tm1: State, s: float, delta_theta: float) -> np.ndarray:
        '''Delta Theta = Degrees'''
        theta_rad = math.radians(x_tm1.theta)
        delta_theta_rad = math.radians(delta_theta)
        G_x = np.eye(3)
        
        # Chain rule includes pi/180 because state theta is stored in degrees
        G_x[0, 2] = -s * math.sin(theta_rad+delta_theta_rad/2) * math.pi/180
        G_x[1, 2] =  s * math.cos(theta_rad+delta_theta_rad/2) * math.pi/180
        
        return G_x

    # This function returns a matrix with the partial derivatives dg/du
    def get_G_u(self, x_tm1: State, delta_t: float) -> np.ndarray:                
        theta_rad = math.radians(x_tm1.theta)
        G_u = np.zeros((3, 3))
        
        # Partial wrt s mapped to col 0
        G_u[0, 0] = math.cos(theta_rad)
        G_u[1, 0] = math.sin(theta_rad)
        G_u[2, 0] = 0.0
        
        # Partial wrt delta_theta mapped to col 1
        G_u[0, 1] = 0.0
        G_u[1, 1] = 0.0
        G_u[2, 1] = 1.0
        
        return G_u

    # This function returns the matrix dh_t/dx_t
    def get_H(self):
        return np.eye(3)
    
    # This function returns the R_t matrix which contains transition function covariance terms.
    def get_R(self, s: float) -> np.ndarray:
        """The covariance matrix for the control input variance"""
        var_s = 0.00027 * abs(s)
        var_theta = 0.00027 * abs(s)
        
        R = np.zeros((3, 3))
        R[0, 0] = var_s + 1e-8      # Small epsilon prevents singular matrix errors
        R[1, 1] = var_theta + 1e-8 
        return R

    # This function returns the Q_t matrix which contains measurement covariance terms.
    def get_Q(self) -> np.ndarray:
        """The covariance matrix for the measurements z_t variance"""
        if self._q_matrix is None:
            with open('../analysis_code/R_matrix.json', 'r') as f:
                json_data = json.load(f)
            
            R_full = np.array(json_data['matrix'])
            # We only need x, y, and rot_z for our state
            self._q_matrix = R_full[np.ix_([0, 1, 5], [0, 1, 5])]
        return self._q_matrix

class KalmanFilterPlot:

    def __init__(self):
        self.dir_length = 0.1
        fig, ax = plt.subplots()
        self.ax = ax
        self.fig = fig

    def update(self, state_mean, state_covaraiance):
        plt.clf()

        # Plot covariance ellipse
        lambda_, v = np.linalg.eig(state_covaraiance)
        lambda_ = np.sqrt(lambda_)
        xy = (state_mean[0], state_mean[1])
        angle=np.rad2deg(np.arctan2(*v[:,0][::-1]))
        ell = Ellipse(xy, alpha=0.5, facecolor='red',width=lambda_[0], height=lambda_[1], angle = angle)
        ax = self.fig.gca()
        ax.add_artist(ell)
        
        # Plot state estimate
        plt.plot(state_mean[0], state_mean[1],'ro')
        plt.plot([state_mean[0], state_mean[0]+ self.dir_length*math.cos(state_mean[2]) ], [state_mean[1], state_mean[1]+ self.dir_length*math.sin(state_mean[2]) ],'r')
        plt.xlabel('X(m)')
        plt.ylabel('Y(m)')
        plt.axis([-0.25, 2, -1, 1])
        plt.grid()
        plt.draw()
        plt.pause(0.1)


# Code to run your EKF offline with a data file.
def offline_efk():

    # Get data to filter
    filename = './data/robot_data_68_0_06_02_26_17_12_19.pkl'
    ekf_data = data_handling.get_file_data_for_kf(filename)

    # Instantiate PF with no initial guess
    x_0 = State(ekf_data[0][3][0]+.5, ekf_data[0][3][1], ekf_data[0][3][5])
    Sigma_0 = parameters.I3
    encoder_counts_0 = ekf_data[0][2].encoder_counts
    extended_kalman_filter = ExtendedKalmanFilter(x_0, Sigma_0, encoder_counts_0)

    # Create plotting tool for ekf
    kalman_filter_plot = KalmanFilterPlot()

    # Loop over sim data
    for t in range(1, len(ekf_data)):
        row = ekf_data[t]
        delta_t = ekf_data[t][0] - ekf_data[t-1][0] # time step size
        u_t = np.array([row[2].encoder_counts, row[2].steering]) # robot_sensor_signal
        z_t = np.array([row[3][0],row[3][1],row[3][5]]) # camera_sensor_signal

        # Run the EKF for a time step
        extended_kalman_filter.update(u_t, z_t, delta_t)
        kalman_filter_plot.update(extended_kalman_filter.state_mean, extended_kalman_filter.state_covariance[0:2,0:2])


####### MAIN #######
if False:
    offline_efk()
