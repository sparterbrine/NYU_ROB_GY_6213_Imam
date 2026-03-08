# External libraries
import copy
from typing import List, Tuple
import matplotlib.pyplot as plt
import math
import numpy as np
import random

# Local libraries
from robot_python_code import RobotOdomSignal, RobotSensorSignal
import parameters
import data_handling

XY_range = List[float] # [x_min, x_max, y_min, y_max]
'''[x_min, x_max, y_min, y_max]'''

# Helper function to make sure all angles are between -pi and pi
def angle_wrap(angle: float) -> float:
    if angle > math.pi:
        angle -= 2*math.pi
    elif angle < -math.pi:
        angle += 2*math.pi
    return angle

# Helper class to store and manipulate your states.
class State:

    # Constructor
    def __init__(self, x: float, y: float, theta: float):
        self.x: float = x
        self.y: float = y
        self.theta: float = theta

    # Get the euclidean distance between 2 states
    def distance_to(self, other_state: "State") -> float:
        return math.sqrt(math.pow(self.x - other_state.x, 2) + math.pow(self.y - other_state.y, 2))
        
    # Get the distance squared between two states
    def distance_to_squared(self, other_state: "State") -> float:
        return math.pow(self.x - other_state.x, 2) + math.pow(self.y - other_state.y, 2)

    # return a deep copy of the state.
    def deepcopy(self) -> "State":
        return copy.deepcopy(self)
        
    # Print the state
    def print(self) -> None:
        print("State: ",self.x, self.y, self.theta)

    def to_array(self) -> np.ndarray:
        """Return state as a numpy array [x, y, theta]"""
        return np.array([self.x, self.y, self.theta])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "State":
        """Create State from array-like [x, y, theta]"""
        return cls(arr[0], arr[1], arr[2])

    def __getitem__(self, idx: int) -> float:
        if idx == 0:
            return self.x
        elif idx == 1:
            return self.y
        elif idx == 2:
            return self.theta
        else:
            raise IndexError("State only supports indices 0, 1, 2 for x, y, theta.")

    def __setitem__(self, idx: int, value: float) -> None:
        if idx == 0:
            self.x = value
        elif idx == 1:
            self.y = value
        elif idx == 2:
            self.theta = value
        else:
            raise IndexError("State only supports indices 0, 1, 2 for x, y, theta.")

    def __sub__(self, other: "State") -> "State":
        return State(self.x - other.x, self.y - other.y, self.theta - other.theta)
    
    
    def __add__(self, other: "State") -> "State":
        return State(self.x + other.x, self.y + other.y, self.theta + other.theta)

# Class to store walls as objects (specifically when represented as line segments in a 2D map.)
class Wall:

    # Constructor
    def __init__(self, wall_corners: XY_range):
        self.corner1: State = State(wall_corners[0], wall_corners[1], 0)
        self.corner2: State = State(wall_corners[2], wall_corners[3], 0)
        self.corner1_mm: State = State(wall_corners[0] * 1000, wall_corners[1] * 1000, 0)
        self.corner2_mm: State = State(wall_corners[2] * 1000, wall_corners[3] * 1000, 0)
        
        self.m: float = (wall_corners[3] - wall_corners[1])/(0.0001 + wall_corners[2] -  wall_corners[0])
        self.b: float = wall_corners[3] - self.m * wall_corners[2]
        self.b_mm: float =  wall_corners[3] * 1000 - self.m * wall_corners[2] * 1000
        self.length: float = self.corner1.distance_to(self.corner2)
        self.length_mm_squared: float = self.corner1_mm.distance_to_squared(self.corner2_mm)
        
        if self.m > 1000:
            self.vertical = True
        else:
            self.vertical = False
        if abs(self.m) < 0.1:
            self.horizontal = True
        else:
            self.horizontal = False


# A class to store 2D maps
class Map:
    def __init__(self, wall_corner_list: List[XY_range], grid_dimensions: List[XY_range]):
        self.wall_list = []
        for wall_corners in wall_corner_list:
            self.wall_list.append(Wall(wall_corners))
        min_x: float = grid_dimensions[0][0]
        max_x: float = grid_dimensions[0][1]
        min_y: float = grid_dimensions[1][0]
        max_y: float = grid_dimensions[1][1]
        border: float = 0.5
        self.plot_range: XY_range = [min_x - border, max_x + border, min_y - border, max_y + border]
        
        self.particle_range: XY_range = [min_x , max_x , min_y, max_y]

    # Function to calculate the distance between any state and its closest wall, accounting for directon of the state.
    def closest_distance_to_walls(self, state: State) -> float:
        closest_distance = 999999999999
        for wall in self.wall_list:
            closest_distance = self.get_distance_to_wall(state, wall, closest_distance)
        
        return closest_distance
        
    # Function to get distance to a wall from a state, in the direction of the state's theta angle.
    # Or return the distance currently believed to be the closest if its closer.
    def get_distance_to_wall(self, state: State, wall: Wall, closest_distance: float) -> float:
        # Ray-segment intersection using parametric form.
        # Ray:     P(t) = O + t*D,        t >= 0,  D = unit direction vector
        # Segment: Q(s) = A + s*(B - A),  s in [0, 1]
        #
        # Solving O + t*D = A + s*(B-A):
        #   denom = D × W   (2D cross product, D = ray dir, W = B-A)
        #   t     = (V × W) / denom   (V = A - O)
        #   s     = (D_perp · V) / denom
        # Intersection is valid when t > 0 (forward) and 0 <= s <= 1 (within segment).
        # Because D is a unit vector, t equals the Euclidean distance to the wall.

        ox, oy = state.x, state.y
        dx = math.cos(state.theta)
        dy = math.sin(state.theta)

        ax, ay = wall.corner1.x, wall.corner1.y
        bx, by = wall.corner2.x, wall.corner2.y

        wx = bx - ax   # wall direction
        wy = by - ay

        vx = ax - ox   # vector from ray origin to wall start
        vy = ay - oy

        denom = dx * wy - dy * wx   # D × W
        if abs(denom) < 1e-10:      # ray is parallel to wall
            return closest_distance

        t = (vx * wy - wx * vy) / denom
        s = (dy * vx - dx * vy) / denom

        if t > 1e-10 and 0.0 <= s <= 1.0 and t < closest_distance:
            return t

        return closest_distance


# Class to hold a particle
class Particle:
    
    def __init__(self):
        self.state: State = State(0., 0., 0.)
        self.weight: float = 1.
        
    # Function to create a new random particle state within a range
    def randomize_uniformly(self, xy_range: XY_range):
        # xy_range = [x_min, x_max, y_min, y_max]
        x = random.uniform(xy_range[0], xy_range[1])
        y = random.uniform(xy_range[2], xy_range[3])
        theta = random.uniform(-math.pi, math.pi)
        self.state = State(x, y, theta)
        self.weight = 1.

    # Function to create a new random particle state with a normal distribution
    def randomize_around_initial_state(self, initial_state: State, state_stdev: State):
        x: float     = random.gauss(initial_state.x,     state_stdev.x)
        y: float     = random.gauss(initial_state.y,     state_stdev.y)
        theta: float = angle_wrap(random.gauss(initial_state.theta, state_stdev.theta))
        self.state = State(x, y, theta)
        self.weight = 1.

    # Function to take a particle and "randomly" propagate it forward according to a motion model.
    def propagate_state(self, last_state: State, delta_encoder_counts: int, steering: int, delta_t: float):
        # --- Tune these constants to match your robot ---
        METERS_PER_COUNT = 0.00078  # encoder counts → meters (calibrate!)
        WHEELBASE_M      = 0.17     # front-to-rear axle distance in meters
        # --- Motion noise standard deviations ---
        DIST_NOISE_STD  = 0.01   # m
        THETA_NOISE_STD = 0.02   # rad

        # Convert encoder counts to forward distance with noise
        distance = delta_encoder_counts * METERS_PER_COUNT + random.gauss(0, DIST_NOISE_STD)

        # Steering in degrees → radians, with noise
        steering_rad = math.radians(steering) + random.gauss(0, math.radians(1))

        # Bicycle (Ackermann) kinematic model
        delta_theta = distance * math.tan(steering_rad) / WHEELBASE_M

        mid_theta = last_state.theta + delta_theta / 2
        x     = last_state.x + distance * math.cos(mid_theta)
        y     = last_state.y + distance * math.sin(mid_theta)
        theta = angle_wrap(last_state.theta + delta_theta + random.gauss(0, THETA_NOISE_STD))

        self.state = State(x, y, theta)

    # Function to determine a particles weight based how well the lidar measurement matches up with the map.
    def calculate_weight(self, lidar_signal: RobotSensorSignal, map: Map):
        """Accumulate log-likelihood over all rays to avoid floating-point underflow\n
        Returns a log weight based on the likelihood of the lidar measurement given the particle's state and the map.\n
        Returns a negative value, to be normalised later in correction() across all particles. The more negative, the less likely the particle's state is given the measurement."""
        log_weight = 0.0
        log_weight_list = []
        for i in range(lidar_signal.num_lidar_rays):
            angle_rad: float  = RobotSensorSignal.convert_hardware_angle(lidar_signal.angles[i])
            distance_m: float = RobotSensorSignal.convert_hardware_distance(lidar_signal.distances[i])

            # Build a state pointing in the direction of this lidar ray
            ray_theta: float = angle_wrap(self.state.theta + angle_rad)
            ray_state: State = State(self.state.x, self.state.y, ray_theta)

            expected_distance = map.closest_distance_to_walls(ray_state)
            log_weight = (expected_distance - distance_m) ** 2 / (2 * parameters.distance_variance)
            log_weight_list.append(log_weight)

        self.weight = float(np.mean(log_weight_list))  # stored as log; correction() converts to linear. Hence it's fine as a negative number and doesn't need to be normalised here.
        
    # Return the normal distribution function output.
    def gaussian(self, expected_distance: float, distance: float) -> float:
        return math.exp(-math.pow(expected_distance - distance, 2)/ 2 / parameters.distance_variance)

    # Deep copy the particle
    def deepcopy(self) -> "Particle":
        return copy.deepcopy(self)
        
    # Print the particle
    def print(self) -> None:
        print("Particle: ", self.state.x, self.state.y, self.state.theta, " w: ", self.weight)

# This class holds the collection of particles.
class ParticleSet:
    
    # Constructor, which calls the known start or unknown start initialization.
    def __init__(self, num_particles: int, xy_range: XY_range, initial_state: State, state_stdev: State, known_start_state: bool):
        self.num_particles: int = num_particles
        self.particle_list: List[Particle] = []
        if known_start_state:
            self.generate_initial_state_particles(initial_state, state_stdev)
        else:
            self.generate_uniform_random_particles(xy_range)
        self.mean_state: State = State(0., 0., 0.)
        self._use_clustering: bool = parameters.use_clustering
        self._clustering_radius: float = parameters.clustering_radius
        self.update_mean_state()
        
    # Function to reset particles and random locations in the workspace.
    def generate_uniform_random_particles(self, xy_range: XY_range):
        for i in range(self.num_particles):
            random_particle: Particle = Particle()
            random_particle.randomize_uniformly(xy_range)
            self.particle_list.append(random_particle)

    # Function to reset particles, normally distributed around the initial state. 
    def generate_initial_state_particles(self, initial_state: State, state_stdev: State):
        for i in range(self.num_particles):
            random_particle: Particle = Particle()
            random_particle.randomize_around_initial_state(initial_state, state_stdev)
            self.particle_list.append(random_particle)

    # Function to resample the particles set, i.e. make a new one with more copies of particles with higher weights.
    def resample(self):
        # Systematic resampling (low-variance resampling).
        # Step size is based on total weight so the CDF is swept exactly N times.
        total_weight = sum(p.weight for p in self.particle_list)
        if total_weight == 0:
            return
        new_list: List[Particle] = []
        N = self.num_particles
        step = total_weight / N
        target = random.uniform(0, step)   # single random offset
        cumulative = 0.0
        j = 0
        for i in range(N):
            while cumulative < target and j < N:
                cumulative += self.particle_list[j].weight
                j += 1
            new_list.append(self.particle_list[j - 1].deepcopy())
            target += step
        self.particle_list = new_list

    def clustering(self) -> List[Particle]:
        """Perform subtractive clustering on the particle set to find cluster centers, which can be used as multiple hypotheses for the robot's state"""
        c1, potentials = self._clustering_potentials_calculation()
        cluster_centers: List[Tuple[Particle, int]] = self._clustering_potential_reduction(c1, potentials)
        return [center[0] for center in cluster_centers]

    def _clustering_potentials_calculation(self) -> Tuple[Tuple[Particle, int], List[float]]:
        """Calculate a "potential" for each particle based on how close it is to other particles, and return the particle with the highest potential along with the list of potentials. \n
        This is Step 1 of Subtractive Clustering"""
        particle_potentials: List[float] = []
        for particle in self.particle_list:
            p_pot: float = 0.0
            for other_particle in self.particle_list:
                if particle is other_particle:
                    continue
                distance = particle.state.distance_to(other_particle.state)
                p_pot += math.exp( (-distance)**2 / (0.5*self._clustering_radius)**2)
            particle_potentials.append(p_pot)
        max_particle_index = particle_potentials.index(max(particle_potentials))
        return (self.particle_list[max_particle_index], max_particle_index), particle_potentials
    
    def _clustering_potential_reduction(self, cluster_center: Tuple[Particle, int], particle_potentials: List[float]) -> List[Tuple[Particle, int]]:
        """Reduce the potential of particles based on their distance to a cluster center. \n
        This is Step 2 of Subtractive Clustering"""
        n: int = len(self.particle_list)
        is_covered: List[bool] = [False] * n
        self._update_coverage(cluster_center[0], is_covered)
        k: int = 1
        cluster_centers: List[Tuple[Particle, int]] = [cluster_center]
        while (sum(is_covered) / n) < 0.75:
            for i, particle in enumerate(self.particle_list):
                particle_potentials[i] = particle_potentials[i] - (particle_potentials[cluster_centers[k-1][1]] *
                                                                    math.exp( (-particle.state.distance_to(cluster_centers[k-1][0].state))**2 / 
                                                                             (0.5*self._clustering_radius)**2))
            k += 1
            max_particle_index = particle_potentials.index(max(particle_potentials))
            cluster_centers.append((self.particle_list[max_particle_index], max_particle_index))
            self._update_coverage(cluster_centers[-1][0], is_covered)

        return cluster_centers

    def _update_coverage(self, center_particle: Particle, covered_mask: List[bool]) -> None:
        for i, particle in enumerate(self.particle_list):
            if not covered_mask[i]:
                dist = particle.state.distance_to(center_particle.state)
                if dist <= self._clustering_radius:
                    covered_mask[i] = True

    # Calculate the mean state.
    def update_mean_state(self):
        # x and y: weighted mean (weights need not be normalised).
        # theta: circular mean using unit-vector averaging to handle wrap-around.
        if self._use_clustering:
            cluster_centers: List[Particle] = self.clustering()
            particles_used = []
            for p in self.particle_list:
                if p.state.distance_to(cluster_centers[0].state) < self._clustering_radius:
                    particles_used.append(p)
        else:
            particles_used = self.particle_list

        # breakpoint()
        total_weight = sum(p.weight for p in particles_used)
        if total_weight == 0:
            return

        mean_x = sum(p.state.x * p.weight for p in particles_used) / total_weight
        mean_y = sum(p.state.y * p.weight for p in particles_used) / total_weight

        # Circular mean for theta
        sin_sum = sum(math.sin(p.state.theta) * p.weight for p in particles_used)
        cos_sum = sum(math.cos(p.state.theta) * p.weight for p in particles_used)
        mean_theta = math.atan2(sin_sum, cos_sum)

        self.mean_state.x     = mean_x
        self.mean_state.y     = mean_y
        self.mean_state.theta = mean_theta
        
    # Print the particle set. Useful for debugging.
    def print_particles(self) -> None:
        for particle in self.particle_list:
            particle.print()
        print()

# Class to hold the particle filter and its functions.
class ParticleFilter:
    
    # Constructor
    def __init__(self, num_particles: int, map: Map, initial_state: State, state_stdev: State, known_start_state: bool, encoder_counts_0: int):
        self.map: Map = map
        self.particle_set: ParticleSet = ParticleSet(num_particles, map.particle_range, initial_state, state_stdev, known_start_state)
        self.state_estimate: State = self.particle_set.mean_state
        self.state_estimate_list: List[State] = []
        self.last_time: float = 0.
        self.last_encoder_counts: int = encoder_counts_0

    # Update the states given new measurements
    def update(self, odometery_signal: RobotOdomSignal, measurement_signal: RobotSensorSignal, delta_t: float):
        self.prediction(odometery_signal, delta_t)
        if len(measurement_signal.angles) > 0:
            self.correction(measurement_signal)
        self.particle_set.update_mean_state()
        self.state_estimate_list.append(self.state_estimate.deepcopy())

    # Predict the current state from the last state.
    def prediction(self, odometry_signal: RobotOdomSignal, delta_t: float):
        # odometry_signal.cmd_speed carries the cumulative encoder count;
        # cmd_steering_angle carries the current steering angle (degrees).
        delta_encoder_counts: int = odometry_signal.encoder_total_count - self.last_encoder_counts
        self.last_encoder_counts: int = odometry_signal.encoder_total_count
        steering: float = odometry_signal.cmd_steering_angle

        for particle in self.particle_set.particle_list:
            last_state: State = particle.state.deepcopy()
            particle.propagate_state(last_state, delta_encoder_counts, steering, delta_t)

    # Correct the predicted states.
    def correction(self, measurement_signal: RobotSensorSignal):
        # Compute log-likelihood for every particle.
        for particle in self.particle_set.particle_list:
            particle.calculate_weight(measurement_signal, self.map)

        # Convert log-weights to linear using the log-sum-exp trick (subtract max to prevent underflow).
        log_weights = [p.weight for p in self.particle_set.particle_list]
        max_log = max(log_weights)
        for p, lw in zip(self.particle_set.particle_list, log_weights):
            p.weight = math.exp(lw - max_log)

        self.particle_set.resample()

    # Output to terminal the mean state.
    def print_state_estimate(self):
        print("Mean state: ", self.particle_set.mean_state.x, self.particle_set.mean_state.y, self.particle_set.mean_state.theta)
    

# Class to help with plotting PF data.
class ParticleFilterPlot:

    # Constructor
    def __init__(self, map: Map):
        self.dir_length: float = 0.1
        fig, ax = plt.subplots()
        self.ax = ax
        self.fig = fig
        self.map: Map = map

        self.predicted_states: List[State] = []
        self.means_states: List[State] = []

    # Clear and update the plot with new PF data
    def update(self, state_mean: State, particle_set: ParticleSet, lidar_signal: RobotSensorSignal, hold_show_plot: bool):
        plt.clf()

        # Plot walls
        for wall in self.map.wall_list:
            plt.plot([wall.corner1.x, wall.corner2.x],[wall.corner1.y, wall.corner2.y],'k')

        # Plot lidar
        for i in range(len(lidar_signal.angles)):
            distance = lidar_signal.convert_hardware_distance(lidar_signal.distances[i])
            angle = lidar_signal.convert_hardware_angle(lidar_signal.angles[i]) + state_mean.theta
            x_ray = [state_mean.x, state_mean.x + distance * math.cos(angle)]
            y_ray = [state_mean.y, state_mean.y + distance * math.sin(angle)]
            plt.plot(x_ray, y_ray, 'r')

        # Plot state estimate
        plt.quiver(state_mean.x, state_mean.y,
                   math.cos(state_mean.theta), math.sin(state_mean.theta),
                   color='b', scale=1/self.dir_length)
        x_particles, y_particles = self.to_plot_data(particle_set)
        plt.plot(x_particles, y_particles, 'g.')
        plt.xlabel('X(m)')
        plt.ylabel('Y(m)')
        plt.axis(self.map.plot_range)
        plt.grid()
        if hold_show_plot:
            plt.show()
        else:
            plt.draw()
            plt.pause(0.1)

    # Helper function to make the particles easy to plot.
    def to_plot_data(self, particle_set):
        x_list = []
        y_list = []
        for p in particle_set.particle_list:
            x_list.append(p.state.x)
            y_list.append(p.state.y)
        return x_list, y_list
        

# Function used to test your PF offline with logged data.
def offline_pf(filename: str = './data/robot_data_0_0_25_02_26_21_41_33.pkl'):

    # Make a map of walls
    map: Map = Map(parameters.wall_corner_list, parameters.grid_dimensions)

    # Get data to filter
    pf_data: list = data_handling.get_file_data_for_pf(filename)
    '''List of trios [timestamp, control_signal, robot_sensor_signal]'''

    # Instantiate PF with uniform distribution across the workspace
    particle_filter: ParticleFilter = ParticleFilter(parameters.num_particles, map, initial_state = State(0.5, 2.0, 1.57), state_stdev = State(0.1,0.1,0.1), known_start_state=False, encoder_counts_0=pf_data[0][2].encoder_counts)

    # Create plotting tool for particles
    particle_filter_plot: ParticleFilterPlot = ParticleFilterPlot(map)
    temp_particle = Particle()

    # Loop over pf data
    for t in range(1, len(pf_data)):
        row = pf_data[t]
        delta_t: float = pf_data[t][0] - pf_data[t-1][0] # time step size
        u_t: RobotOdomSignal = RobotOdomSignal(row[1][0], row[1][1]) # robot_sensor_signal
        z_t: RobotSensorSignal = row[2] # lidar_sensor_signal

        u_t.encoder_total_count = z_t.encoder_counts
        if __name__ == '__main__':
            print(f"Control: {u_t.encoder_total_count} counts, {u_t.cmd_steering_angle} degrees | Measurement: {z_t.distances} distances")
        # Run the PF for a time step
        particle_filter.update(u_t, z_t, delta_t)
        if t == 1:
            temp_particle.state = particle_filter.particle_set.mean_state.deepcopy()
        else:
            temp_particle.propagate_state(temp_particle.state, u_t.encoder_total_count, u_t.cmd_steering_angle, delta_t)
        particle_filter_plot.means_states.append(particle_filter.particle_set.mean_state.deepcopy())
        # Propagate the mean state using a temporary Particle and append the result
        particle_filter_plot.predicted_states.append(temp_particle.state.deepcopy())
        particle_filter_plot.update(particle_filter.particle_set.mean_state, particle_filter.particle_set, z_t, False)

    # particle_filter_plot.update(particle_filter.particle_set.mean_state, particle_filter.particle_set, z_t, False)

    # --- Custom summary plot: mean states (green), predicted states (blue), walls (black), grid limits ---
    plt.figure()
    # Plot walls
    for wall in map.wall_list:
        plt.plot([wall.corner1.x, wall.corner2.x], [wall.corner1.y, wall.corner2.y], 'k-', linewidth=2, label='Wall' if wall == map.wall_list[0] else "")

    # Plot grid map limits as a rectangle
    x_min, x_max = parameters.grid_dimensions[0]
    y_min, y_max = parameters.grid_dimensions[1]
    plt.plot([x_min, x_max, x_max, x_min, x_min], [y_min, y_min, y_max, y_max, y_min], 'k--', label='Map limits')

    # Plot mean states (green)
    if particle_filter_plot.means_states:
        mean_x = [s.x for s in particle_filter_plot.means_states]
        mean_y = [s.y for s in particle_filter_plot.means_states]
        plt.plot(mean_x, mean_y, 'go-', label='Mean states')

    # Plot predicted states (blue)
    if particle_filter_plot.predicted_states:
        pred_x = [s.x for s in particle_filter_plot.predicted_states]
        pred_y = [s.y for s in particle_filter_plot.predicted_states]
        plt.plot(pred_x, pred_y, 'bo-', label='Predicted states')

    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('Particle Filter Trajectory and Predictions')
    plt.axis([x_min-0.5, x_max+0.5, y_min-0.5, y_max+0.5])
    plt.legend()
    plt.grid(True)
    plt.show()
    

        


####### MAIN #######
if __name__ == '__main__':
    import sys
    # robot_python_code forces the Agg (non-interactive) backend for NiceGUI.
    # Switch back to an interactive backend when running this script directly.
    plt.switch_backend('TkAgg')
    path = sys.argv[1] if len(sys.argv) > 1 else './data/robot_data_0_0_25_02_26_21_41_33.pkl'
    offline_pf(path)
