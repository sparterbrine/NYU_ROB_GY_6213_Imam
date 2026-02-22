import math
import random
import numpy as np

# --- Motion Model Constants ---
K_ENC: float = 0.000302        # meters per tick
'''Meters per Encoder Tick'''
L: float = 0.140               # <--- NEW: Wheelbase = 140mm (0.14m)
'''Wheelbase in meters'''

# Uncertainty Constants (Keep these for simulation/Part 7)
VAR_S_COEFF: float = 0.002 
VAR_W_CONST: float = 0.0015

# --- Helper Functions (Simulation) ---
def variance_distance_travelled_s(distance) -> float:
    return VAR_S_COEFF * abs(distance)

def distance_travelled_s(encoder_counts) -> float:
    return encoder_counts * K_ENC

def variance_rotational_velocity_w(distance) -> float:
        return VAR_W_CONST


def rotational_velocity_w(steering_angle_command) -> float:
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

    def __init__(self, initial_state: State, last_encoder_count: int):
        # state = [x, y, theta]
        self.state: State = initial_state
        self.last_encoder_count: int = last_encoder_count

    def step_update(self, encoder_counts: int, steering_angle_command: int, delta_t: float) -> State:
        # 1. Calculate Distance (ds)
        delta_enc = encoder_counts - self.last_encoder_count
        ds = distance_travelled_s(delta_enc)
        
        # 2. Calculate Steering Angle (alpha)
        # Assuming the slider (-20 to 20) represents degrees of steering
        alpha = steering_angle_command * (math.pi / 180.0)

        # 3. Extract current state
        x, y, theta = self.state.x, self.state.y, self.state.theta

        # 4. Propagate State using KINEMATIC BICYCLE MODEL
        # Formula: change_in_theta = (distance / wheelbase) * tan(steering_angle)
        
        if abs(alpha) < 0.001: 
            # If moving straight (avoid divide by zero)
            x_new = x + ds * math.cos(theta)
            y_new = y + ds * math.sin(theta)
            theta_new = theta
        else:
            # Turning
            d_theta = (ds / L) * math.tan(alpha)
            
            # Use Half-Angle formula for better accuracy (Runge-Kutta 2nd Order)
            theta_mid = theta + (d_theta / 2.0)
            
            x_new = x + ds * math.cos(theta_mid)
            y_new = y + ds * math.sin(theta_mid)
            theta_new = theta + d_theta

        # Normalize Angle (-pi to pi)
        theta_new = (theta_new + math.pi) % (2 * math.pi) - math.pi

        # 5. Update internal state
        self.state = State(x_new, y_new, theta_new)
        self.last_encoder_count = encoder_counts
        
        return State(x_new, y_new, theta_new)
    
    # This is a great tool to take in data from a trial and iterate over the data to create 
    # a robot trajectory in the global frame, using your motion model.
    def traj_propagation(self, time_list, encoder_count_list, steering_angle_list):
        x_list = [self.state.x]
        y_list = [self.state.y]
        theta_list = [self.state.theta]
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
        t = 0
        
        # Sim params
        sim_speed_enc = 1000  # ticks/step
        sim_steering = -10    # degrees
        
        self.state = State(0., 0., 0.)
        self.last_encoder_count = 0
        
        t_list = [0.]; x_list = [0.]; y_list = [0.]; theta_list = [0.]

        while t < duration:
            t += delta_t 
            
            # Nominal Physics
            step_dist = distance_travelled_s(100) # 100 ticks
            alpha = sim_steering * (math.pi/180.0)
            
            # Add Noise
            var_s = variance_distance_travelled_s(step_dist)
            ds_noisy = step_dist + random.gauss(0, math.sqrt(var_s))
            
            # Calculate d_theta using NOISY distance and L
            d_theta_noisy = (ds_noisy / L) * math.tan(alpha)
            # Add extra rotational noise if desired
            d_theta_noisy += random.gauss(0, math.sqrt(VAR_W_CONST))

            # Propagate
            x, y, theta = self.state.x, self.state.y, self.state.theta
            theta_new = theta + d_theta_noisy
            theta_mid = theta + (d_theta_noisy / 2.0)
            x_new = x + ds_noisy * math.cos(theta_mid)
            y_new = y + ds_noisy * math.sin(theta_mid)
            
            self.state = State(x_new, y_new, theta_new)
            
            t_list.append(t)
            x_list.append(x_new)
            y_list.append(y_new)
            theta_list.append(theta_new)
            
        return t_list, x_list, y_list, theta_list