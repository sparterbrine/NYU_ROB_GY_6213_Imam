# External libraries
import argparse
import json
import os
from typing import List, Tuple

import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# Local libraries
from filters import ThetaFilter
import parameters
import data_handling

from motion_models import distance_travelled_s, rotational_velocity_w, state_prediction, State, variance_distance_travelled_s, variance_rotational_velocity_w

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
    def update(self, u_t, z_t, delta_t) -> State:
        """ u_t =Encoder counts, Steering Angle
         z_t = Camera X, Camera Y, Camera Theta"""
        self.prediction_step(u_t, delta_t)
        self.correction_step(State(z_t[0], z_t[1], z_t[2]))
        # Update encoder memory for the next step's delta calculation
        self.last_encoder_counts = u_t[0]

        return self.predicted_state_mean

    # Prediction step only — used when the camera measurement is stale (no new ArUco detection)
    def predict_only(self, u_t, delta_t) -> State:
        """Run the motion model forward without a camera correction.
        Propagates predicted_state_mean/covariance into state_mean/covariance."""
        self.prediction_step(u_t, delta_t)
        self.state_mean = self.predicted_state_mean
        self.state_covariance = self.predicted_state_covariance
        self.last_encoder_counts = u_t[0]
        return self.predicted_state_mean

    # Set the EKF's predicted state mean and covariance matrix
    def prediction_step(self, u_t, delta_t):
        """ u_t =Encoder counts, Steering Angle"""
        self.predicted_state_mean, s = self.g_function(u_t, delta_t) # s here is the calculated distance travelled based on the change in encoder counts, which is an intermediate variable needed for the Jacobian calculations below
        
        delta_theta = rotational_velocity_w(u_t[1]) * delta_t
        '''In Degrees'''
        G_x = self.get_G_x(self.state_mean, s, delta_theta)
        G_u = self.get_G_u(self.state_mean, s, delta_theta)
        R = self.get_R(s)
        
        self.predicted_state_covariance = G_x @ self.state_covariance @ G_x.T + G_u @ R @ G_u.T

    # Set the EKF's corrected state mean and covariance matrix
    def correction_step(self, z_t: State):
        """ z_t = State = Camera X, Camera Y, Camera Theta"""
        H = self.get_H()
        Q = self.get_Q() #z_t covariance matrix from camera sensor model
        
        # Kalman Gain
        S = H @ self.predicted_state_covariance @ H.T + Q
        K = self.predicted_state_covariance @ H.T @ np.linalg.inv(S)
        
        # Innovation (Difference between measurement and prediction)
        z_pred = self.get_h_function(self.predicted_state_mean)
        innovation: State = z_t - z_pred
        
        # Wrap angle innovation to [-180, 180] to prevent spinning errors
        innovation.theta = (innovation.theta + 180) % 360 - 180
        
        # Update State and Covariance
        state_np_array = K @ np.array([innovation.x, innovation.y, innovation.theta])
        self.state_mean = self.predicted_state_mean + State(state_np_array[0], state_np_array[1], state_np_array[2])
        self.state_covariance = (np.eye(3) - K @ H) @ self.predicted_state_covariance

    # The nonlinear transition equation that provides new states from past states
    def g_function(self, u_t: np.ndarray, delta_t: float) -> Tuple[State, float]:
        encoder_counts, steering_angle_command = u_t
        delta_encoder: int = encoder_counts - self.last_encoder_counts
        s: float = distance_travelled_s(delta_encoder)
        
        w_deg: float = rotational_velocity_w(steering_angle_command)
        delta_theta: float = w_deg * delta_t
        
        theta_rad: float = math.radians(self.state_mean.theta)
        theta_t: float = theta_rad + math.radians(delta_theta)
        mid_theta: float = (theta_rad + theta_t) / 2.0
        
        x_t: float = self.state_mean.x + s * math.cos(mid_theta)
        y_t: float = self.state_mean.y + s * math.sin(mid_theta)
        
        # Keep theta bounds clean
        theta_t = (theta_t + math.pi) % (2 * math.pi) - math.pi
        
        return State(x_t, y_t, math.degrees(theta_t)), s
    
    # The nonlinear measurement function
    def get_h_function(self, x_t):
        """In this case, the measurement function is the identity function because our measurements directly observe the state variables (x, y, theta) without any nonlinear transformation."""
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
    def get_G_u(self, x_tm1: State, s: float, delta_theta: float) -> np.ndarray:
        """u_t is encoder counts and steering angle command, but we take derivatives wrt the intermediate variables s and delta_theta because those are the direct inputs to the state transition function"""
        '''state is [x_g, y_g, theta_g]'''
        '''# Turning
        d_theta = (ds / L) * math.tan(alpha)

        # Use Half-Angle formula for better accuracy (Runge-Kutta 2nd Order)
        theta_mid = state.theta + (d_theta / 2.0)

        x_new = state.x + ds * math.cos(theta_mid)
        y_new = state.y + ds * math.sin(theta_mid)
        theta_new = state.theta + d_theta'''
                        
        d_theta = math.radians(delta_theta)
        theta_t = math.radians(x_tm1.theta)
        theta_mid: float = theta_t + d_theta/2.
        G_u = np.zeros((3, 2))

        # Partial wrt s mapped to col 0
        # x_t, y_t, theta_t wrt  s (speed)
        G_u[0, 0] = math.cos(theta_mid)
        G_u[1, 0] = math.sin(theta_mid)
        G_u[2, 0] = 0.0


        # Partial wrt delta_theta mapped to col 1
        # x_t, y_t, theta_t wrt delta_theta (rotational velocity)
        G_u[0, 1] = -(s/2) * math.sin(theta_mid) * math.pi/180
        G_u[1, 1] =  (s/2) * math.cos(theta_mid) * math.pi/180
        G_u[2, 1] = 1.0



        return G_u

    # This function returns the matrix dh_t/dx_t
    def get_H(self):
        """In this case, the measurement function is the identity function, so the Jacobian is just the identity matrix."""
        return np.eye(3)
    
    # This function returns the R_t matrix which contains transition function covariance terms.
    def get_R(self, s: float) -> np.ndarray:
        """The covariance matrix for the control input variance"""
        var_s = variance_distance_travelled_s(s)
        var_theta = variance_rotational_velocity_w(s)
        
        R = np.zeros((2, 2))
        R[0, 0] = var_s + 1e-8      # Small epsilon prevents singular matrix errors
        R[1, 1] = var_theta + 1e-8 
        return R

    # This function returns the Q_t matrix which contains measurement covariance terms.
    def get_Q(self) -> np.ndarray:
        """The covariance matrix for the measurements z_t variance"""

        #Received from many tests - multiple robot positions, 360, multiple camera heights.
        return np.array([
            [0.002898, 0.000115, 0.005556],
            [0.000115, 0.004000, 0.007781],
            [0.005556, 0.007781, 20.589390]
        ])

class KalmanFilterPlot:

    def __init__(self):
        self.dir_length = 0.1
        fig, ax = plt.subplots()
        self.ax = ax
        self.fig = fig

    def update(self, state_mean: State, state_covaraiance):
        plt.clf()

        # Plot covariance ellipse
        lambda_, v = np.linalg.eig(state_covaraiance)
        lambda_ = np.sqrt(lambda_)
        xy = (state_mean.x, state_mean.y)
        angle=np.rad2deg(np.arctan2(*v[:,0][::-1]))
        ell = Ellipse(xy, alpha=0.5, facecolor='red',width=lambda_[0], height=lambda_[1], angle = angle)
        ax = self.fig.gca()
        ax.add_artist(ell)
        
        # Plot state estimate
        plt.plot(state_mean.x, state_mean.y,'ro')
        plt.plot([state_mean.x, state_mean.x+ self.dir_length*math.cos(state_mean.theta) ], [state_mean.y, state_mean.y+ self.dir_length*math.sin(state_mean.theta) ],'r')
        plt.xlabel('X(m)')
        plt.ylabel('Y(m)')
        plt.axis([-0.25, 3.5, -1.5, 1.5])
        plt.grid()
        plt.draw()
        plt.pause(0.1)


# Code to run your EKF offline with a data file.
def offline_efk(filename: str, title: str = None):
    ekf_data = data_handling.get_file_data_for_kf(filename)

    # Load ground truth annotations if available (produced by annotate_data.py)
    ann_path = os.path.splitext(filename)[0] + '_annotations.json'
    ground_truth = {}
    if os.path.exists(ann_path):
        with open(ann_path, 'r') as fh:
            raw = json.load(fh)
        ground_truth = {int(k): v for k, v in raw.items()}
        print(f"Loaded {len(ground_truth)} ground truth annotation(s) from {ann_path}")
    '''A list, with each entry being a tuple of [timestamp, control_signal, robot_sensor_signal, camera_sensor_signal]\n
    Reminder: camera_sensor_signal is a list of [camera_x, camera_y, camera_z, camera_roll, camera_pitch, camera_theta]'''

    # Instantiate PF with no initial guess
    x_0 = State(ekf_data[0][3][0]+.5, ekf_data[0][3][1], ekf_data[0][3][5])
    Sigma_0 = parameters.I3
    encoder_counts_0 = ekf_data[0][2].encoder_counts
    extended_kalman_filter = ExtendedKalmanFilter(x_0, Sigma_0, encoder_counts_0)

    # Create plotting tool for ekf
    kalman_filter_plot = KalmanFilterPlot()


    # Store results for plotting
    time_list = []
    predicted_x_list: List[float] = []
    predicted_y_list: List[float] = []
    predicted_theta_list: List[float] = []
    pred_sigma_x_list: List[float] = []
    pred_sigma_y_list: List[float] = []
    pred_sigma_theta_list: List[float] = []
    ekf_sigma_x_list: List[float] = []
    ekf_sigma_y_list: List[float] = []
    ekf_sigma_theta_list: List[float] = []
    z_x_list: List[float] = []
    z_y_list: List[float] = []
    z_theta_list: List[float] = []
    filtered_x_list: List[float] = []
    filtered_y_list: List[float] = []
    filtered_theta_list: List[float] = []
    theta_filter = ThetaFilter()
    # Track last raw camera signal to detect stale (no-detection) frames
    prev_z_raw = np.array([ekf_data[0][3][0], ekf_data[0][3][1], ekf_data[0][3][5]], dtype=float)

    # Loop over sim data
    for t in range(1, len(ekf_data)):
        row = ekf_data[t]
        delta_t = ekf_data[t][0] - ekf_data[t-1][0] # time step size
        u_t = np.array([row[2].encoder_counts, row[2].steering]) # robot_sensor_signal
        z_raw = np.array([row[3][0], row[3][1], row[3][5]], dtype=float) # camera_sensor_signal (raw)

        camera_fresh = not np.array_equal(z_raw, prev_z_raw)

        if camera_fresh:
            z_t = z_raw.copy()
            z_t[2] = theta_filter.filter_pose_theta(z_t[2]) # Filter camera theta measurements to remove outliers
            predicted_state: State = extended_kalman_filter.update(u_t, z_t, delta_t)
            prev_z_raw = z_raw.copy()
        else:
            # No new ArUco detection — propagate with motion model only
            z_t = z_raw  # stale; stored for plotting but not fed into correction
            predicted_state: State = extended_kalman_filter.predict_only(u_t, delta_t)
        # kalman_filter_plot.update(extended_kalman_filter.state_mean, extended_kalman_filter.state_covariance[0:2,0:2])

        # Store for plotting
        time_list.append(row[0])
        predicted_x_list.append(predicted_state.x)
        predicted_y_list.append(predicted_state.y)
        predicted_theta_list.append(predicted_state.theta)
        pred_cov = extended_kalman_filter.predicted_state_covariance
        pred_sigma_x_list.append(math.sqrt(max(0.0, pred_cov[0, 0])))
        pred_sigma_y_list.append(math.sqrt(max(0.0, pred_cov[1, 1])))
        pred_sigma_theta_list.append(math.sqrt(max(0.0, pred_cov[2, 2])))
        ekf_cov = extended_kalman_filter.state_covariance  # posterior: shrinks on correction, grows on predict-only
        ekf_sigma_x_list.append(math.sqrt(max(0.0, ekf_cov[0, 0])))
        ekf_sigma_y_list.append(math.sqrt(max(0.0, ekf_cov[1, 1])))
        ekf_sigma_theta_list.append(math.sqrt(max(0.0, ekf_cov[2, 2])))
        z_x_list.append(z_t[0])
        z_y_list.append(z_t[1])
        z_theta_list.append(z_t[2])
        filtered_x_list.append(extended_kalman_filter.state_mean.x)
        filtered_y_list.append(extended_kalman_filter.state_mean.y)
        filtered_theta_list.append(extended_kalman_filter.state_mean.theta)

    # Build ground truth scatter lists from annotations (annotation index → ekf_data time)
    gt_time_list, gt_x_list, gt_y_list, gt_theta_list = [], [], [], []
    for idx, ann in sorted(ground_truth.items()):
        if idx < len(ekf_data):
            gt_time_list.append(ekf_data[idx][0])
            gt_x_list.append(ann['x'])
            gt_y_list.append(ann['y'])
            gt_theta_list.append(ann['theta'])

    # Print average error between EKF estimate and ground truth annotations
    if gt_x_list:
        ekf_x_at_gt, ekf_y_at_gt, ekf_theta_at_gt = [], [], []
        for idx in sorted(ground_truth.keys()):
            if idx < len(ekf_data):
                list_idx = max(0, idx - 1)  # filtered lists start at t=1, so annotation idx → list index idx-1
                if list_idx < len(filtered_x_list):
                    ekf_x_at_gt.append(filtered_x_list[list_idx])
                    ekf_y_at_gt.append(filtered_y_list[list_idx])
                    ekf_theta_at_gt.append(filtered_theta_list[list_idx])

        if ekf_x_at_gt:
            n = len(ekf_x_at_gt)
            x_errors     = [abs(gt - ekf) for gt, ekf in zip(gt_x_list[:n], ekf_x_at_gt)]
            y_errors     = [abs(gt - ekf) for gt, ekf in zip(gt_y_list[:n], ekf_y_at_gt)]
            # Wrap theta error to [-180, 180]
            theta_errors = [abs(((gt - ekf) + 180) % 360 - 180) for gt, ekf in zip(gt_theta_list[:n], ekf_theta_at_gt)]
            x_mean, y_mean, t_mean = sum(x_errors)/n, sum(y_errors)/n, sum(theta_errors)/n
            x_std  = (sum((e - x_mean)**2 for e in x_errors) / n) ** 0.5
            y_std  = (sum((e - y_mean)**2 for e in y_errors) / n) ** 0.5
            t_std  = (sum((e - t_mean)**2 for e in theta_errors) / n) ** 0.5
            print(f"\nAverage EKF error vs ground truth ({n} annotation(s)):")
            print(f"  x:     {x_mean:.4f} ± {x_std:.4f} m")
            print(f"  y:     {y_mean:.4f} ± {y_std:.4f} m")
            print(f"  theta: {t_mean:.4f} ± {t_std:.4f} deg")

    # Plot x, y, theta over time with z_t (measurements)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
    if title:
        fig.suptitle(title)

    # Convert to numpy for fill_between arithmetic
    px  = np.array(predicted_x_list);     py  = np.array(predicted_y_list);     pt  = np.array(predicted_theta_list)
    sx  = np.array(pred_sigma_x_list);    sy  = np.array(pred_sigma_y_list);    st  = np.array(pred_sigma_theta_list)
    fx  = np.array(filtered_x_list);      fy  = np.array(filtered_y_list);      ft  = np.array(filtered_theta_list)
    esx = np.array(ekf_sigma_x_list);     esy = np.array(ekf_sigma_y_list);     est = np.array(ekf_sigma_theta_list)

    # Labels only on ax1 so the shared figure legend has no duplicates
    # ax1.fill_between(time_list, px - 2*sx, px + 2*sx, alpha=0.2, color='blue', label='Predicted ±2σ')
    ax1.fill_between(time_list, fx - 15*esx, fx + 15*esx, alpha=0.2, color='red', label='EKF ±15σ')
    ax1.plot(time_list, predicted_x_list, label='Predicted', color='blue')
    ax1.plot(time_list, z_x_list, label='Measurement', color='green', linestyle='dashed')
    ax1.plot(time_list, filtered_x_list, label='EKF', color='red')
    if gt_x_list:
        ax1.scatter(gt_time_list, gt_x_list, marker='*', s=120, color='black', zorder=5, label='Ground truth')
    ax1.set_ylabel('x (m)')

    # ax2.fill_between(time_list, py - 2*sy, py + 2*sy, alpha=0.2, color='blue')
    ax2.fill_between(time_list, fy - 15*esy, fy + 15*esy, alpha=0.2, color='red')
    ax2.plot(time_list, predicted_y_list, color='blue')
    ax2.plot(time_list, z_y_list, color='green', linestyle='dashed')
    ax2.plot(time_list, filtered_y_list, color='red')
    if gt_y_list:
        ax2.scatter(gt_time_list, gt_y_list, marker='*', s=120, color='black', zorder=5)
    ax2.set_ylabel('y (m)')

    # ax3.fill_between(time_list, pt - 2*st, pt + 2*st, alpha=0.2, color='blue')
    ax3.fill_between(time_list, ft - 15*est, ft + 15*est, alpha=0.2, color='red')
    ax3.plot(time_list, predicted_theta_list, color='blue')
    ax3.plot(time_list, z_theta_list, color='green', linestyle='dashed')
    ax3.plot(time_list, filtered_theta_list, color='red')
    if gt_theta_list:
        ax3.scatter(gt_time_list, gt_theta_list, marker='*', s=120, color='black', zorder=5)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('theta (deg)')

    # Single legend below all subplots
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(handles),
               bbox_to_anchor=(0.5, 0), framealpha=0.9)
    top = 0.93 if title else 1.0
    fig.tight_layout(rect=[0, 0.06, 1, top])
    plt.show()


####### MAIN #######
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run EKF offline on a robot data log.")
    parser.add_argument('pkl_file', help="Path to the .pkl log file")
    parser.add_argument('--title', default=None, help="Title for the plot")
    args = parser.parse_args()
    offline_efk(args.pkl_file, args.title)
