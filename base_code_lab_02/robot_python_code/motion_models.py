# External Libraries
import math
import random
from typing import List, Tuple

# Motion Model constants


# A function for obtaining variance in distance travelled as a function of distance travelled
def variance_distance_travelled_s(distance: float) -> float:
    var_s: float = 0.00027 * distance
    return var_s

# Function to calculate distance from encoder counts
def distance_travelled_s(encoder_counts: int) -> float: # In meters
    # Add student code here
    s: float = 0.000294 * encoder_counts

    return s

# A function for obtaining variance in distance travelled as a function of distance travelled
def variance_rotational_velocity_w(distance: float) -> float:
    k: float = 0.00027
    
    var_dist: float = k * distance

    return var_dist

def rotational_velocity_w(steering_angle_command: int) -> float:
    slope: float = 2.25
    intercept: float= -0.66
    
    w: float = (slope * steering_angle_command) + intercept
    
    return w

class State:
    def __init__(self, x: float, y: float, theta: float):
        """[x_g, y_g, theta_g]"""
        self.x: float = x
        self.y: float = y
        self.theta: float= theta

# This class is an example structure for implementing your motion model.
class MyMotionModel:

    # Constructor, change as you see fit.
    def __init__(self, initial_state: State, last_encoder_count: int):
        self.state: State = initial_state
        self.last_encoder_count: int = last_encoder_count
        self.wheel_base: float = 0.14

    # This is the key step of your motion model, which implements x_t = f(x_{t-1}, u_t)
    def step_update(self, encoder_counts: int, steering_angle_command: int, delta_t: float) -> State:
        distance_travelled: float = distance_travelled_s(encoder_counts - self.last_encoder_count)
        '''This is delta_s'''
        curavature_radius: float = self.wheel_base / math.tan(math.radians(steering_angle_command))
        delta_theta: float = distance_travelled / curavature_radius
        '''This is delta_theta'''
        delta_x: float= distance_travelled * math.cos(self.state.theta + delta_theta/2)
        delta_y: float= distance_travelled * math.sin(self.state.theta + delta_theta/2)
        
        self.last_encoder_count = encoder_counts
        self.state.x += delta_x
        self.state.y += delta_y
        self.state.theta += delta_theta

        return self.state
    
    # This is a great tool to take in data from a trial and iterate over the data to create 
    # a robot trajectory in the global frame, using your motion model.
    def traj_propagation(self, time_list: List[float],
                         encoder_count_list: List[int],
                         steering_angle_list: List[int]) -> Tuple[List[float], List[float], List[float]]:
        x_list: List[float] = [self.state.x]
        y_list: List[float] = [self.state.y]
        theta_list: List[float] = [self.state.theta]
        self.last_encoder_count = encoder_count_list[0]
        for i in range(1, len(encoder_count_list)):
            delta_t: float = time_list[i] - time_list[i-1]
            r: float = self.wheel_base / math.tan(math.radians(steering_angle_list[i]))
            new_state: State = self.step_update(encoder_count_list[i], steering_angle_list[i], delta_t)
            x_list.append(new_state.x)
            y_list.append(new_state.y)
            theta_list.append(new_state.theta)

        return x_list, y_list, theta_list
    

    def generate_simulated_traj(self, duration: float) -> Tuple[List[float], List[float], List[float], List[float]]:
        delta_t = 0.1
        t_list: List[float] = []
        x_list: List[float] = []
        y_list: List[float] = []
        theta_list: List[float] = []
        t: float = 0
        encoder_counts: int = 0
        while t < duration:

            t += delta_t 
        return t_list, x_list, y_list, theta_list
            