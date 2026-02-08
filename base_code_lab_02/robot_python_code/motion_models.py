import math
import random
import numpy as np

# --- Motion Model Constants ---
K_ENC = 0.000302        # meters per tick
L = 0.140               # <--- NEW: Wheelbase = 140mm (0.14m)

# Uncertainty Constants (Keep these for simulation/Part 7)
VAR_S_COEFF = 0.002 
VAR_W_CONST = 0.0015

# --- Helper Functions (Simulation) ---
def variance_distance_travelled_s(distance):
    return VAR_S_COEFF * abs(distance)

def distance_travelled_s(encoder_counts):
    return encoder_counts * K_ENC

def variance_rotational_velocity_w(distance):
    return VAR_W_CONST

# --- The Class ---
class MyMotionModel:

    def __init__(self, initial_state, last_encoder_count):
        # state = [x, y, theta]
        self.state = initial_state
        self.last_encoder_count = last_encoder_count

    def step_update(self, encoder_counts, steering_angle_command, delta_t):
        # 1. Calculate Distance (ds)
        delta_enc = encoder_counts - self.last_encoder_count
        ds = distance_travelled_s(delta_enc)
        
        # 2. Calculate Steering Angle (alpha)
        # Assuming the slider (-20 to 20) represents degrees of steering
        alpha = steering_angle_command * (math.pi / 180.0)

        # 3. Extract current state
        x, y, theta = self.state

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
        self.state = [x_new, y_new, theta_new]
        self.last_encoder_count = encoder_counts
        
        return self.state

    # Simulation Support (Updated to use L)
    def generate_simulated_traj(self, duration):
        delta_t = 0.1
        t = 0
        
        # Sim params
        sim_speed_enc = 1000  # ticks/step
        sim_steering = -10    # degrees
        
        self.state = [0, 0, 0]
        self.last_encoder_count = 0
        
        t_list = [0]; x_list = [0]; y_list = [0]; theta_list = [0]

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
            x, y, theta = self.state
            theta_new = theta + d_theta_noisy
            theta_mid = theta + (d_theta_noisy / 2.0)
            x_new = x + ds_noisy * math.cos(theta_mid)
            y_new = y + ds_noisy * math.sin(theta_mid)
            
            self.state = [x_new, y_new, theta_new]
            
            t_list.append(t)
            x_list.append(x_new)
            y_list.append(y_new)
            theta_list.append(theta_new)
            
        return t_list, x_list, y_list, theta_list