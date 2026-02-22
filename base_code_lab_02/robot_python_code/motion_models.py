# External Libraries
import math
import random

# Motion Model constants


# A function for obtaining variance in distance travelled as a function of distance travelled
def variance_distance_travelled_s(distance):
    var_s = 0.00027 * distance
    return var_s
# Function to calculate distance from encoder counts
def distance_travelled_s(encoder_counts): # In meters
    # Add student code here
    s = 0.000294 * encoder_counts

    return s


# A function for obtaining variance in distance travelled as a function of distance travelled
def variance_rotational_velocity_w(distance):
    k = 0.00027
    
    var_dist = k * distance

    return var_dist

def rotational_velocity_w(steering_angle_command):
    slope = 2.25
    intercept = -0.66
    
    w = (slope * steering_angle_command) + intercept
    
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
        # FIX: Ensure state is a State object
        if isinstance(self.state, list):
            self.state = State(self.state[0], self.state[1], self.state[2])

        delta_t = 0.1
        t_list = []
        x_list = []
        y_list = []
        theta_list = []
        t = 0
        
        # Initialize lists
        x_list.append(self.state.x)
        y_list.append(self.state.y)
        theta_list.append(self.state.theta)
        t_list.append(t)

        # --- Simulation Command Parameters ---
        # Constant speed
        cmd_velocity = 0.3 
        # Random steering command per trajectory (-20 to 20 degrees)
        cmd_steering = 4 

        while t < duration:
            t += delta_t 

            # 1. Add Random Noise (Process Noise)
            # We add noise to the *command* to simulate real-world execution error
            
            # Velocity noise: mean=0.3, std_dev=0.05
            noisy_v = random.gauss(cmd_velocity, 0.05) 
            
            # Steering noise: mean=random_angle, std_dev=2.0 degrees
            noisy_steering = random.gauss(cmd_steering, 2.0)

            # 2. Calculate Displacement
            distance_travelled = noisy_v * delta_t
            
            # 3. Calculate Rotation
            w = rotational_velocity_w(noisy_steering) 
            delta_theta = w * delta_t

            # 4. Update State (Kinematics)
            mid_theta_rad = math.radians(self.state.theta + delta_theta / 2.0)
            
            delta_x = distance_travelled * math.cos(mid_theta_rad)
            delta_y = distance_travelled * math.sin(mid_theta_rad)

            self.state.x += delta_x
            self.state.y += delta_y
            self.state.theta += delta_theta

            # 5. Store Data
            x_list.append(self.state.x)
            y_list.append(self.state.y)
            theta_list.append(self.state.theta)
            t_list.append(t)

        return t_list, x_list, y_list, theta_list