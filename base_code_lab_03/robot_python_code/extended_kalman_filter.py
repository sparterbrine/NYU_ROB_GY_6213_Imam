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
        self.state_mean = x_0
        self.state_covariance = Sigma_0
        self.predicted_state_mean = [0,0,0]
        self.predicted_state_covariance = parameters.I3 * 1.0
        self.last_encoder_counts = encoder_counts_0

    # Call the prediction and correction steps
    def update(self, u_t, z_t, delta_t):
        return

    # Set the EKF's predicted state mean and covariance matrix
    def prediction_step(self, u_t, delta_t):
        return

    # Set the EKF's corrected state mean and covariance matrix
    def correction_step(self, z_t):
        return

    # Function to calculate distance from encoder counts
    def distance_travelled_s(self, encoder_counts):
        return 0    
            
    # Function to calculate rotational velocity from steering and dist travelled or speed
    def rotational_velocity_w(self, steering_angle_command):        
        return 0

    # The nonlinear transition equation that provides new states from past states
    def g_function(self, x_tm1, u_t, delta_t):
        x_t = x_tm1
        s = 0
        return x_t, s
    
    # The nonlinear measurement function
    def get_h_function(self, x_t):
        return x_t
    
    # This function returns a matrix with the partial derivatives dg/dx
    # g outputs x_t, y_t, theta_t, and we take derivatives wrt inputs x_tm1, y_tm1, theta_tm1
    def get_G_x(self, x_tm1, s):       
        return parameters.I3

    # This function returns a matrix with the partial derivatives dg/du
    def get_G_u(self, x_tm1, delta_t):                
        return parameters.I3

    # This function returns a matrix with the partial derivatives dh_t/dx_t
    def get_H(self):
        return parameters.I3
    
    # This function returns the R_t matrix which contains transition function covariance terms.
    def get_R(self, s):
        return parameters.I3

    # This function returns the Q_t matrix which contains measurement covariance terms.
    def get_Q(self):
        return parameters.I3

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
