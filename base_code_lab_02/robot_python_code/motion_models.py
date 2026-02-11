# External Libraries
import math
import random

# Motion Model constants


# A function for obtaining variance in distance travelled as a function of distance travelled
def variance_distance_travelled_s(distance):
    # Add student code here
    var_s = 1

    return var_s

# Function to calculate distance from encoder counts
def distance_travelled_s(encoder_counts):
    # Add student code here
    s = 0

    return s

# A function for obtaining variance in distance travelled as a function of distance travelled
def variance_rotational_velocity_w(distance):
    # Add student code here
    var_w = 1

    return var_w

def rotational_velocity_w(steering_angle_command: int) -> float:
    slope: float = 2.25
    intercept: float= -0.66
    
    w: float = (slope * steering_angle_command) + intercept
    w = w*(-1)
    return w

class State:
    def __init__(self, x: float, y: float, theta: float):
        """[x_g, y_g, theta_g]"""
        self.x: float = x
        self.y: float = y
        self.theta: float= theta
        '''In Degrees'''

# This class is an example structure for implementing your motion model.
class MyMotionModel:

    # Constructor, change as you see fit.
    def __init__(self, initial_state, last_encoder_count):
        self.state = initial_state
        self.last_encoder_count = last_encoder_count

    # This is the key step of your motion model, which implements x_t = f(x_{t-1}, u_t)
    def step_update(self, encoder_counts: int, steering_angle_command: int, delta_t: float) -> State:
        distance_travelled: float = distance_travelled_s(encoder_counts - self.last_encoder_count)
        '''This is delta_s'''
        # if abs(steering_angle_command) < 0.001:
        #     delta_x = distance_travelled * math.sin(self.state.theta)
        #     delta_y = distance_travelled * math.cos(self.state.theta)
        #     delta_theta = 0.
        # else:
        #     # curavature_radius: float = self.wheel_base / math.tan(math.radians(steering_angle_command))
        delta_theta: float = 0 if distance_travelled == 0 else delta_t * rotational_velocity_w(steering_angle_command)
        '''This is delta_theta'''
        delta_x: float= distance_travelled * math.cos(math.radians(self.state.theta + delta_theta/2))
        delta_y: float= distance_travelled * math.sin(math.radians(self.state.theta + delta_theta/2))
        
        return self.state
    
    # This is a great tool to take in data from a trial and iterate over the data to create 
    # a robot trajectory in the global frame, using your motion model.
    def traj_propagation(self, time_list, encoder_count_list, steering_angle_list):
        x_list = [self.state[0]]
        y_list = [self.state[1]]
        theta_list = [self.state[2]]
        self.last_encoder_count = encoder_count_list[0]
        for i in range(1, len(encoder_count_list)):
            delta_t: float = time_list[i] - time_list[i-1]
            new_state: State = self.step_update(encoder_count_list[i], steering_angle_list[i], delta_t)
            x_list.append(new_state.x)
            y_list.append(new_state.y)
            theta_list.append(new_state.theta)

        return x_list, y_list, theta_list
    

    # Coming soon
    def generate_simulated_traj(self, duration):
        delta_t = 0.1
        t_list = []
        x_list = []
        y_list = []
        theta_list = []
        t = 0
        encoder_counts = 0
        while t < duration:

            t += delta_t 
        return t_list, x_list, y_list, theta_list
            