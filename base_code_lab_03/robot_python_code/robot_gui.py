# External libraries
import asyncio
import cv2
import math
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib
from nicegui import ui, app, run
import numpy as np
import time
from fastapi import Response
from time import time

# Local libraries
from robot import Robot
import robot_python_code
import parameters

# Global variables
logging = False
stream_video = True


# Frame converter for the video stream, from OpenCV to a JPEG image
def convert(frame: np.ndarray) -> bytes:
    """Converts a frame from OpenCV to a JPEG image.
    This is a free function (not in a class or inner-function),
    to allow run.cpu_bound to pickle it and send it to a separate process.
    """
    _, imencode_image = cv2.imencode('.jpg', frame)
    return imencode_image.tobytes()
    
# Create the connection with a real camera.
def connect_with_camera():
    video_capture = cv2.VideoCapture(1)
    return video_capture
    
def update_video(video_image):
    if stream_video:
        video_image.force_reload()

def get_time_in_ms():
    return int(time()*1000)

# Create the gui page
@ui.page('/')
def main():

    # Robot variables
    robot = Robot()

    # Lidar data
    max_lidar_range = 12
    lidar_angle_res = 2
    num_angles = int(360 / lidar_angle_res)
    lidar_distance_list = []
    lidar_cos_angle_list = []
    lidar_sin_angle_list = []
    for i in range(num_angles):
        lidar_distance_list.append(max_lidar_range)
        lidar_cos_angle_list.append(math.cos(i*lidar_angle_res/180*math.pi))
        lidar_sin_angle_list.append(math.sin(i*lidar_angle_res/180*math.pi))

    # Set dark mode for gui
    dark = ui.dark_mode()
    dark.value = True
    
    # Set up the video stream, not needed for lab 1
    if stream_video:
        video_capture = cv2.VideoCapture(parameters.camera_id)
    
    # Enable frame grabs from the video stream.
    @app.get('/video/frame')
    async def grab_video_frame() -> Response:
        if not video_capture.isOpened():
            return placeholder
        # The `video_capture.read` call is a blocking function.
        # So we run it in a separate thread (default executor) to avoid blocking the event loop.
        _, frame = await run.io_bound(video_capture.read)
        if frame is None:
            return placeholder
        # `convert` is a CPU-intensive function, so we run it in a separate process to avoid blocking the event loop and GIL.
        jpeg = await run.cpu_bound(convert, frame)
        return Response(content=jpeg, media_type='image/jpeg')

    # Convert lidar data to something visible in correct units. This is dummy data for lab 1.
    def update_lidar_data():
        for i in range(robot.robot_sensor_signal.num_lidar_rays):
            distance_in_mm = robot.robot_sensor_signal.distances[i]
            angle = 360-robot.robot_sensor_signal.angles[i]
            if distance_in_mm > 20 and abs(angle) < 360:
                index = max(0,min(int(360/lidar_angle_res-1),int((angle-(lidar_angle_res/2))/lidar_angle_res)))
                lidar_distance_list[index] = distance_in_mm/1000
               
    # Determine what speed and steering commands to send
    def update_commands():

        # Experiment trial controls
        if robot.running_trial:
            delta_time = get_time_in_ms() - robot.trial_start_time
            if delta_time > parameters.trial_time:
                robot.running_trial = False
                speed_switch.value = False
                steering_switch.value = False
                robot.extra_logging = True
                print("End Trial :", delta_time)
        
        if robot.extra_logging:
            delta_time = get_time_in_ms() - robot.trial_start_time
            if delta_time > parameters.trial_time + parameters.extra_trial_log_time:
                logging_switch.value = False
                robot.extra_logging = False
                

        # Regular slider controls
        if speed_switch.value:
            cmd_speed = slider_speed.value
        else:
            cmd_speed = 0
        if steering_switch.value:
            cmd_steering_angle = slider_steering.value
        else:
            cmd_steering_angle = 0
        return cmd_speed, cmd_steering_angle
        
    # Update
    def update_connection_to_robot():
        if udp_switch.value:
            if not robot.connected_to_hardware:
                udp, udp_success = robot_python_code.create_udp_communication(parameters.arduinoIP, parameters.localIP, parameters.arduinoPort, parameters.localPort, parameters.bufferSize)
                if udp_success:
                    robot.setup_udp_connection(udp)
                    robot.connected_to_hardware = True
                    print("Should be set for UDP!")
                else:
                    udp_switch.value = False
                    robot.connected_to_hardware = False
        else:
            if robot.connected_to_hardware:
                robot.eliminate_udp_connection()
                robot.connected_to_hardware = False
        
    # Update the speed slider if steering is not enabled
    def enable_speed():
        d = 0

    # Update the steering slider if steering is not enabled
    def enable_steering():
        d = 0

    # Visualize the lidar scans
    def show_lidar_plot():
        with main_plot:
            fig = main_plot.fig
            fig.patch.set_facecolor('black')
            plt.clf()
            plt.style.use('dark_background')
            plt.tick_params(axis='x', colors='lightgray')
            plt.tick_params(axis='y', colors='lightgray')
                
            for i in range(num_angles):
                distance = lidar_distance_list[i]
                cos_ang = lidar_cos_angle_list[i]
                sin_ang = lidar_sin_angle_list[i]
                x = [distance * cos_ang, max_lidar_range * cos_ang]
                y = [distance * sin_ang, max_lidar_range * sin_ang]
                plt.plot(x, y, 'r')
            plt.grid(True)
            plt.xlim(-2,2)
            plt.ylim(-2,2)

    # Visualize the lidar scans
    def show_localization_plot():
        with main_plot:
            fig = main_plot.fig
            fig.patch.set_facecolor('black')
            plt.clf()
            plt.style.use('dark_background')
            plt.tick_params(axis='x', colors='lightgray')
            plt.tick_params(axis='y', colors='lightgray')
            
            sigma = 3
            covar_matrix = parameters.covariance_plot_scale * robot.extended_kalman_filter.state_covariance[0:2,0:2]#np.array([[sigma, -sigma*0.9],[ -sigma*0.9, sigma]])
            x_est = robot.extended_kalman_filter.state_mean[0]
            y_est = robot.extended_kalman_filter.state_mean[1]
            lambda_, v = np.linalg.eig(covar_matrix)
            lambda_ = np.sqrt(lambda_)
            ell = Ellipse(xy=(x_est, y_est), alpha=0.5, facecolor='red',width=lambda_[0], height=lambda_[1], angle=np.rad2deg(np.arctan2(*v[:,0][::-1])))
            ax = fig.gca()
            ax.add_artist(ell)

            plt.plot(x_est, y_est, 'ro')

            plt.grid(True)
            plot_range = 1
            plt.xlim(-plot_range, plot_range)
            plt.ylim(-plot_range, plot_range)

    # Run an experiment trial from a button push
    def run_trial():
        robot.trial_start_time = get_time_in_ms()
        robot.running_trial = True
        steering_switch.value = True
        speed_switch.value = True
        logging_switch.value = True
        print("Start time:", robot.trial_start_time)


    # Create the gui title bar
    with ui.card().classes('w-full  items-center'):
        ui.label('ROB-GY - 6213: Robot Navigation & Localization').style('font-size: 24px;')
    
    # Create the video camera, lidar, and encoder sensor visualizations. These may be dummys for lab 01.
    with ui.card().classes('w-full'):
        with ui.grid(columns=3).classes('w-full items-center'):
            with ui.card().classes('w-full items-center h-60'):
                if stream_video:
                    video_image = ui.interactive_image('/video/frame').classes('w-full h-full')
                else:
                    ui.image('./a_robot_image.jpg').props('height=2')
                    video_image = None
            with ui.card().classes('w-full items-center h-60'):
                main_plot = ui.pyplot(figsize=(3, 3))
            with ui.card().classes('items-center h-60'):
                ui.label('Encoder:').style('text-align: center;')
                encoder_count_label = ui.label('0')
                logging_switch = ui.switch('Data Logging ')
                udp_switch = ui.switch('Robot Connect')
                run_trial_button = ui.button('Run Trial', on_click=lambda:run_trial())
                
    # Create the robot manual control slider and switch for speed
    with ui.card().classes('w-full'):
        with ui.grid(columns=4).classes('w-full'):
            with ui.card().classes('w-full items-center'):
                ui.label('SPEED:').style('text-align: center;')
            with ui.card().classes('w-full items-center'):
                slider_speed = ui.slider(min=0, max=100, value=0)
            with ui.card().classes('w-full items-center'):
                ui.label().bind_text_from(slider_speed, 'value').style('text-align: center;')
            with ui.card().classes('w-full items-center'):
                speed_switch = ui.switch('Enable', on_change=lambda: enable_speed())

    # Create the robot manual control slider and switch for steering
    with ui.card().classes('w-full'):
        with ui.grid(columns=4).classes('w-full'):
            with ui.card().classes('w-full items-center'):
                ui.label('STEER:').style('text-align: center;')
            with ui.card().classes('w-full items-center'):
                slider_steering = ui.slider(min=-20, max=20, value=0)
            with ui.card().classes('w-full items-center'):
                ui.label().bind_text_from(slider_steering, 'value').style('text-align: center;')
            with ui.card().classes('w-full items-center'):
                steering_switch = ui.switch('Enable', on_change=lambda: enable_steering())
        

    # Update slider values, plots, etc. and run robot control loop
    async def control_loop():
        update_connection_to_robot()
        cmd_speed, cmd_steering_angle = update_commands()
        robot.control_loop(cmd_speed, cmd_steering_angle, logging_switch.value)
        encoder_count_label.set_text(robot.robot_sensor_signal.encoder_counts)
        #update_lidar_data()
        #show_lidar_plot()
        show_localization_plot()
        update_video(video_image)
        
    ui.timer(0.1, control_loop)

# Run the gui
ui.run(native=True)

