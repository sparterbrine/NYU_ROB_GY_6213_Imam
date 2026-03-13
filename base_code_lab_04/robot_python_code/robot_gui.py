# External libraries
import cv2
import math
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from nicegui import ui, app, run
import numpy as np
import time
from fastapi import Response
from time import time

# Local libraries
from robot import Robot
import robot_python_code
import parameters
from trajectories import TRAJECTORIES, TrajectoryRunner
# from aruco_pose_estimator import ArucoPoseEstimator  # Camera operated separately

# Global variables
logging = False
stream_video = False


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
    video_capture = None
    # if stream_video:
    #     video_capture = cv2.VideoCapture(parameters.camera_id + cv2.CAP_DSHOW)
    robot = Robot(video_capture)
    trajectory_runner = TrajectoryRunner()

    # # ArUco pose estimator (camera operated separately)
    # aruco_estimator = ArucoPoseEstimator(
    #     camera_matrix=parameters.camera_matrix,
    #     dist_coeffs=parameters.dist_coeffs,
    #     marker_length=parameters.marker_length,
    #     known_markers=parameters.KNOWN_MARKERS,
    #     robot_marker_id=7,
    # )

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

    # # Set up the video stream
    # if stream_video:
    #     video_capture = cv2.VideoCapture(parameters.camera_id + cv2.CAP_DSHOW)

    # # Enable frame grabs from the video stream.
    # @app.get('/video/frame')
    # async def grab_video_frame() -> Response:
    #     if not video_capture.isOpened():
    #         return Response(status_code=204)
    #     _, frame = await run.io_bound(video_capture.read)
    #     if frame is None:
    #         return Response(status_code=204)
    #     jpeg = await run.cpu_bound(convert, frame)
    #     return Response(content=jpeg, media_type='image/jpeg')

    # Convert lidar data to something visible in correct units.
    def update_lidar_data():
        for i in range(robot.robot_sensor_signal.num_lidar_rays):
            distance_in_mm = robot.robot_sensor_signal.distances[i]
            angle = 360-robot.robot_sensor_signal.angles[i]
            if distance_in_mm > 20 and abs(angle) < 360:
                index = max(0,min(int(360/lidar_angle_res-1),int((angle-(lidar_angle_res/2))/lidar_angle_res)))
                lidar_distance_list[index] = distance_in_mm/1000

    # # ArUco state: last captured pose and a flag to trigger a post-move capture
    # aruco_pose_last = None
    # needs_post_capture = False

    # def capture_aruco_snapshot():
    #     """Open camera, grab one frame, estimate pose, close camera. Updates aruco_pose_last."""
    #     nonlocal aruco_pose_last
    #     cap = cv2.VideoCapture(parameters.camera_id + cv2.CAP_DSHOW)
    #     pose = None
    #     if cap.isOpened():
    #         ret, frame = cap.read()
    #         if ret and frame is not None:
    #             pose, _ = aruco_estimator.estimate_pose(frame)
    #             if pose is not None:
    #                 aruco_pose_last = pose
    #                 robot.camera_sensor_signal = [pose['x'], pose['y'], pose['z'], pose['roll'], pose['pitch'], pose['yaw']]
    #                 aruco_pose_label.set_text(f"{pose['x']:.3f}, {pose['y']:.3f}, {pose['yaw']:.1f}°")
    #             else:
    #                 aruco_pose_label.set_text("No pose (markers not visible)")
    #         else:
    #             aruco_pose_label.set_text("No frame")
    #     cap.release()
    #     return pose

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

        # Trajectory runner overrides manual sliders when active
        if trajectory_runner.is_running:
            cmd_speed, cmd_steering_angle = trajectory_runner.update()
            if not trajectory_runner.is_running:
                logging_switch.value = False
                trajectory_status_label.set_text('Done')
            return cmd_speed, cmd_steering_angle

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

    # Run an experiment trial from a button push
    def run_trial():
        # capture_aruco_snapshot()  # Camera operated separately
        robot.trial_start_time = get_time_in_ms()
        robot.running_trial = True
        steering_switch.value = True
        speed_switch.value = True
        logging_switch.value = True
        print("Start time:", robot.trial_start_time)


    # Create the gui title bar
    with ui.card().classes('w-full  items-center'):
        ui.label('ROB-GY - 6213: Robot Navigation & Localization').style('font-size: 24px;')

    # Create the video camera, lidar, and encoder sensor visualizations.
    with ui.card().classes('w-full'):
        with ui.grid(columns=3).classes('w-full items-center'):
            with ui.card().classes('w-full items-center h-60'):
                # Camera display disabled
                ui.image('./a_robot_image.jpg').props('height=2')
                video_image = None
            with ui.card().classes('w-full items-center h-60'):
                main_plot = ui.pyplot(figsize=(3, 3))
            with ui.card().classes():
                ui.label('Encoder:').style('text-align: center;')
                encoder_count_label = ui.label('0')
                ui.label('z_t: x, y, theta').style('text-align: center;')
                state_label = ui.label('0.0, 0.0, 0.0').style('text-align: center;')
                # ui.label('ArUco pose: x, y, yaw').style('text-align: center;')
                # aruco_pose_label = ui.label('N/A').style('text-align: center;')
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


    # Trajectory runner controls
    def run_trajectory():
        name = trajectory_select.value
        if name:
            # capture_aruco_snapshot()  # Camera operated separately
            robot.data_logger.set_next_session_name(name)
            logging_switch.value = True
            trajectory_runner.start(name)
            trajectory_status_label.set_text(f'Running: {name}')

    def stop_trajectory():
        trajectory_runner.stop()
        # capture_aruco_snapshot()  # Camera operated separately
        logging_switch.value = False
        trajectory_status_label.set_text('Stopped')

    # Create the trajectory control card
    with ui.card().classes('w-full'):
        with ui.grid(columns=4).classes('w-full items-center'):
            with ui.card().classes('w-full items-center'):
                ui.label('TRAJECTORY:').style('text-align: center;')
            with ui.card().classes('w-full items-center'):
                trajectory_select = ui.select(list(TRAJECTORIES.keys()), value=list(TRAJECTORIES.keys())[0])
            with ui.card().classes('w-full items-center'):
                ui.button('Run', on_click=lambda: run_trajectory())
                ui.button('Stop', on_click=lambda: stop_trajectory())
            with ui.card().classes('w-full items-center'):
                trajectory_status_label = ui.label('Idle')

    # Update slider values, plots, etc. and run robot control loop
    async def control_loop():
        update_connection_to_robot()
        cmd_speed, cmd_steering_angle = update_commands()
        robot.control_loop(cmd_speed, cmd_steering_angle, logging_switch.value)
        encoder_count_label.set_text(robot.robot_sensor_signal.encoder_counts)
        # Update x, y, theta label
        # try:
        #     x = float(robot.camera_sensor_signal[0])
        #     y = float(robot.camera_sensor_signal[1])
        #     theta = float(robot.camera_sensor_signal[5])
        #     state_label.set_text(f"{x:.2f}, {y:.2f}, {theta:.2f}")
        # except Exception:
        #     state_label.set_text("N/A, N/A, N/A")

        update_lidar_data()
        show_lidar_plot()

    ui.timer(0.1, control_loop)

# Run the gui
ui.run(native=True)

