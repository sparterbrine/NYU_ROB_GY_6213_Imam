# External libraries
import asyncio
import cv2
import math
import matplotlib
matplotlib.use('Agg') # Fixes the black screen crash
from matplotlib import pyplot as plt
from nicegui import ui, app, run
import numpy as np
import time
from fastapi import Response
from time import time

# Local libraries
import robot_python_code
import parameters
import motion_models 

# Global variables
logging = False
stream_video = False

def convert(frame: np.ndarray) -> bytes:
    _, imencode_image = cv2.imencode('.jpg', frame)
    return imencode_image.tobytes()
    
def get_time_in_ms():
    return int(time()*1000)

@ui.page('/')
def main():
    robot = robot_python_code.Robot()

    # --- Initialize Motion Model ---
    start_enc = robot.robot_sensor_signal.encoder_counts if robot.robot_sensor_signal else 0
    initial_state = [0.0, 0.0, 0.0] 
    
    # Initialize the NEW Bicycle Model
    my_model = motion_models.MyMotionModel(initial_state, start_enc)
    
    traj_x = [0.0]
    traj_y = [0.0]
    last_update_time = get_time_in_ms()
    loop_counter = 0 

    # GUI Setup
    dark = ui.dark_mode()
    dark.value = True
    
    if stream_video:
        video_capture = cv2.VideoCapture(1)
    
    @app.get('/video/frame')
    async def grab_video_frame() -> Response:
        if not stream_video or not video_capture.isOpened():
            return Response(content=b'', media_type='image/jpeg')
        _, frame = await run.io_bound(video_capture.read)
        if frame is None:
            return Response(content=b'', media_type='image/jpeg')
        jpeg = await run.cpu_bound(convert, frame)
        return Response(content=jpeg, media_type='image/jpeg')

    def update_commands():
        if robot.running_trial:
            delta_time = get_time_in_ms() - robot.trial_start_time
            if delta_time > parameters.trial_time:
                robot.running_trial = False
                speed_switch.value = False
                steering_switch.value = False
                logging_switch.value = False
                print("End Trial")

        if speed_switch.value:
            cmd_speed = slider_speed.value
        else:
            cmd_speed = 0
            
        if steering_switch.value:
            cmd_steering_angle = slider_steering.value
        else:
            cmd_steering_angle = 0
            
        return cmd_speed, cmd_steering_angle
        
    def update_connection_to_robot():
        if udp_switch.value:
            if not robot.connected_to_hardware:
                udp, udp_success = robot_python_code.create_udp_communication(
                    parameters.arduinoIP, parameters.localIP, 
                    parameters.arduinoPort, parameters.localPort, 
                    parameters.bufferSize)
                if udp_success:
                    robot.setup_udp_connection(udp)
                    robot.connected_to_hardware = True
                    if robot.robot_sensor_signal:
                        my_model.last_encoder_count = robot.robot_sensor_signal.encoder_counts
                    ui.notify('Robot Connected!')
                else:
                    udp_switch.value = False
                    ui.notify('Connection Failed')
        else:
            if robot.connected_to_hardware:
                robot.eliminate_udp_connection()
                robot.connected_to_hardware = False
        
    def show_motion_plot():
        with main_plot:
            plt.clf()
            fig = plt.gcf()
            fig.patch.set_facecolor('#121212')
            ax = plt.gca()
            ax.set_facecolor('black')
            ax.tick_params(colors='white')
            for spine in ax.spines.values():
                spine.set_color('white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            
            if len(traj_x) > 0:
                plt.plot(traj_x, traj_y, 'c-', linewidth=2)
                plt.plot(traj_x[-1], traj_y[-1], 'ro') 
                
                # Heading Arrow
                th = my_model.state[2]
                plt.arrow(traj_x[-1], traj_y[-1], 
                          0.2*math.cos(th), 0.2*math.sin(th), 
                          color='r', width=0.02)

            plt.grid(True, color='#444444', linestyle='--')
            plt.xlabel("X (m)")
            plt.ylabel("Y (m)")
            plt.axis('equal') 

    def run_trial():
        robot.trial_start_time = get_time_in_ms()
        robot.running_trial = True
        steering_switch.value = True
        speed_switch.value = True
        logging_switch.value = True
        
        traj_x.clear(); traj_y.clear()
        traj_x.append(0); traj_y.append(0)
        my_model.state = [0.0, 0.0, 0.0]
        if robot.robot_sensor_signal:
            my_model.last_encoder_count = robot.robot_sensor_signal.encoder_counts
        print("Trial Started")

    def reset_odom():
        traj_x.clear(); traj_y.clear()
        traj_x.append(0); traj_y.append(0)
        my_model.state = [0.0, 0.0, 0.0]
        if robot.robot_sensor_signal:
            my_model.last_encoder_count = robot.robot_sensor_signal.encoder_counts
        ui.notify('Odometry Reset')

    # --- UI LAYOUT ---
    with ui.card().classes('w-full items-center'):
        ui.label('Robot Navigation (Bicycle Model)').style('font-size: 24px;')
    
    with ui.card().classes('w-full'):
        with ui.grid(columns=3).classes('w-full items-center'):
            with ui.card().classes('w-full items-center h-60'):
                if stream_video:
                    video_image = ui.interactive_image('/video/frame')
                else:
                    ui.label("Video OFF")
                    video_image = None
            
            with ui.card().classes('w-full items-center h-60'):
                main_plot = ui.pyplot(figsize=(3, 3))
            
            with ui.card().classes('items-center h-60'):
                ui.label('Encoder Counts:')
                encoder_count_label = ui.label('0')
                state_label = ui.label('x: 0.0, y: 0.0').style('font-size: 12px; color: cyan')
                
                udp_switch = ui.switch('Robot Connect')
                logging_switch = ui.switch('Data Logging')
                ui.button('Run Trial', on_click=run_trial)
                ui.button('Reset Odom', on_click=reset_odom).props('color=red')
                
    with ui.card().classes('w-full'):
        with ui.grid(columns=4).classes('w-full'):
            ui.label('SPEED:')
            slider_speed = ui.slider(min=0, max=100, value=0)
            ui.label().bind_text_from(slider_speed, 'value')
            speed_switch = ui.switch('Enable', on_change=lambda: enable_speed())

    with ui.card().classes('w-full'):
        with ui.grid(columns=4).classes('w-full'):
            ui.label('STEER:')
            slider_steering = ui.slider(min=-20, max=20, value=0)
            ui.label().bind_text_from(slider_steering, 'value')
            steering_switch = ui.switch('Enable', on_change=lambda: enable_steering())
        
    def enable_speed(): pass
    def enable_steering(): pass

    async def control_loop():
        nonlocal last_update_time, loop_counter
        try:
            update_connection_to_robot()
            cmd_speed, cmd_st = update_commands()
            
            if robot:
                robot.control_loop(cmd_speed, cmd_st, logging_switch.value)
                if robot.robot_sensor_signal:
                    encoder_count_label.set_text(robot.robot_sensor_signal.encoder_counts)
            
            curr_time = get_time_in_ms()
            dt = (curr_time - last_update_time) / 1000.0
            
            if dt > 0 and robot.robot_sensor_signal:
                enc = robot.robot_sensor_signal.encoder_counts
                new_state = my_model.step_update(enc, cmd_st, dt)
                
                traj_x.append(new_state[0])
                traj_y.append(new_state[1])
                state_label.set_text(f"x:{new_state[0]:.2f} y:{new_state[1]:.2f} th:{new_state[2]:.2f}")
                last_update_time = curr_time

            loop_counter += 1
            if loop_counter % 5 == 0:
                show_motion_plot()
                if stream_video and video_image:
                    video_image.force_reload()

        except Exception as e:
            print(f"Loop Error: {e}")
        
    ui.timer(0.1, control_loop)

ui.run(native=False, port=8080)