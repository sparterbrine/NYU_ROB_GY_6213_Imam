# External libraries
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# Local libraries
import parameters
import data_handling

# Main class
class ExtendedKalmanFilter:
    def __init__(self, x_0, Sigma_0, encoder_counts_0):
        self.state_mean = np.array(x_0, dtype=float)
        self.state_covariance = np.array(Sigma_0, dtype=float)
        self.predicted_state_mean = np.zeros(3)
        self.predicted_state_covariance = parameters.I3 * 1.0
        self.last_encoder_counts = encoder_counts_0

    # Call the prediction and correction steps
    def update(self, u_t, z_t, delta_t):
        self.prediction_step(u_t, delta_t)
        self.correction_step(z_t)
        # Update encoder memory for the next step's delta calculation
        self.last_encoder_counts = u_t[0] 

    # Set the EKF's predicted state mean and covariance matrix
    def prediction_step(self, u_t, delta_t):
        self.predicted_state_mean, s = self.g_function(self.state_mean, u_t, delta_t)
        
        G_x = self.get_G_x(self.state_mean, s)
        G_u = self.get_G_u(self.state_mean, delta_t)
        R = self.get_R(s)
        
        self.predicted_state_covariance = G_x @ self.state_covariance @ G_x.T + G_u @ R @ G_u.T

    # Set the EKF's corrected state mean and covariance matrix
    def correction_step(self, z_t):
        H = self.get_H()
        Q = self.get_Q()
        
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

    # Function to calculate distance from encoder counts
    def distance_travelled_s(self, encoder_counts):
        return 0.000294 * encoder_counts    
            
    # Function to calculate rotational velocity from steering and dist travelled or speed
    def rotational_velocity_w(self, steering_angle_command):        
        return (2.25 * steering_angle_command) - 0.66

    # The nonlinear transition equation that provides new states from past states
    def g_function(self, x_tm1, u_t, delta_t):
        delta_encoder = u_t[0] - self.last_encoder_counts
        s = self.distance_travelled_s(delta_encoder)
        
        w_deg = self.rotational_velocity_w(u_t[1])
        delta_theta = w_deg * delta_t
        
        theta_rad = math.radians(x_tm1[2])
        
        x_t = x_tm1[0] + s * math.cos(theta_rad)
        y_t = x_tm1[1] + s * math.sin(theta_rad)
        theta_t = x_tm1[2] + delta_theta
        
        # Keep theta bounds clean
        theta_t = (theta_t + 180) % 360 - 180
        
        return np.array([x_t, y_t, theta_t]), s
    
    # The nonlinear measurement function
    def get_h_function(self, x_t):
        return x_t
    
    # This function returns a matrix with the partial derivatives dg/dx
    # g outputs x_t, y_t, theta_t, and we take derivatives wrt inputs x_tm1, y_tm1, theta_tm1
    def get_G_x(self, x_tm1, s):       
        theta_rad = math.radians(x_tm1[2])
        G_x = np.eye(3)
        
        # Chain rule includes pi/180 because state theta is stored in degrees
        G_x[0, 2] = -s * math.sin(theta_rad) * (math.pi / 180.0)
        G_x[1, 2] =  s * math.cos(theta_rad) * (math.pi / 180.0)
        
        return G_x

    # This function returns a matrix with the partial derivatives dg/du
    def get_G_u(self, x_tm1, delta_t):                
        theta_rad = math.radians(x_tm1[2])
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
    def get_R(self, s):
        var_s = 0.00027 * abs(s)
        var_theta = 0.00027 * abs(s)
        
        R = np.zeros((3, 3))
        R[0, 0] = var_s + 1e-8      # Small epsilon prevents singular matrix errors
        R[1, 1] = var_theta + 1e-8 
        return R

    # This function returns the Q_t matrix which contains measurement covariance terms.
    def get_Q(self):
        # Base confidence constants for [X, Y, Theta] from camera. Tune as needed.
        return np.diag([0.01, 0.01, 0.01])

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
    x_0 = [ekf_data[0][3][0]+.5, ekf_data[0][3][1], ekf_data[0][3][5]]
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
