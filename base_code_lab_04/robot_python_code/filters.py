import numpy as np


class ThetaFilter:
    def __init__(self, buffer_size: int = 10, outlier_threshold: float = 5.0):
        self.theta_outlier_threshold: float = outlier_threshold
        '''degrees, if the change in theta is greater than this, consider it an outlier and return the previous theta value.'''
        self._theta_buffer = []  # For robust theta filtering
        self.theta_filter_buffer_size: int = buffer_size

    def filter_pose_theta(self, theta: float) -> float:
        """Robust filter for theta using a rolling buffer and median."""

        self._theta_buffer.append(theta)
        if len(self._theta_buffer) > self.theta_filter_buffer_size:
            self._theta_buffer.pop(0)


            median_theta = float(np.median(self._theta_buffer))
            if abs(theta - median_theta) > self.theta_outlier_threshold:
                # Replace last value in buffer with median
                self._theta_buffer[-1] = median_theta
                return median_theta
        return theta