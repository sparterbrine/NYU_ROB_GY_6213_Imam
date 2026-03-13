"""
online_pf.py

Run the particle filter in real-time with live robot sensor data over UDP,
displayed in a NiceGUI window matching the style of robot_gui.py.

Usage:
    python online_pf.py
"""

import math
import time

from matplotlib import pyplot as plt
from nicegui import ui

import parameters
import robot_python_code
from robot_python_code import RobotOdomSignal, RobotSensorSignal
from particle_filter import ParticleFilter, ParticleSet, Map, State
from trajectories import TRAJECTORIES, TrajectoryRunner

LOOP_PERIOD = 0.1   # seconds (ui.timer interval)


@ui.page('/')
def main():

    # --- Build map and particle filter ---
    map = Map(parameters.wall_corner_list, parameters.grid_dimensions)
    pf = ParticleFilter(
        parameters.num_particles,
        map,
        initial_state=State(
            parameters.initial_state_x,
            parameters.initial_state_y,
            parameters.initial_state_theta,
        ),
        state_stdev=State(
            parameters.initial_state_stdev_x,
            parameters.initial_state_stdev_y,
            parameters.initial_state_stdev_theta,
        ),
        known_start_state=True,
        encoder_counts_0=0,  # overwritten after first UDP packet is received
    )
    mean_states = []        # trajectory history for the best-weight particle

    sensor_signal = RobotSensorSignal([0, 0, 0])
    msg_sender = None
    msg_receiver = None
    last_time = [time.perf_counter()]   # list so closure can mutate it
    tick = [0]

    # --- UDP state ---
    connected = [False]

    # --- Trajectory runner ---
    trajectory_runner = TrajectoryRunner()

    # --- Dark mode ---
    dark = ui.dark_mode()
    dark.value = True

    # --- Title ---
    with ui.card().classes('w-full items-center'):
        ui.label('ROB-GY 6213 — Online Particle Filter').style('font-size: 24px;')

    # --- Main layout: plot on the left, status on the right ---
    with ui.card().classes('w-full'):
        with ui.grid(columns=2).classes('w-full items-start'):

            with ui.card().classes('w-full items-center'):
                pf_plot = ui.pyplot(figsize=(6, 6))

            with ui.card().classes('w-full'):
                ui.label('Connection').style('font-size: 16px; font-weight: bold;')
                udp_switch = ui.switch('Robot Connect')
                status_label = ui.label('Disconnected').style('color: gray;')

                ui.separator()
                ui.label('Trajectory').style('font-size: 16px; font-weight: bold;')
                traj_select = ui.select(
                    options=list(TRAJECTORIES.keys()),
                    value=list(TRAJECTORIES.keys())[0],
                    label='Trajectory',
                ).classes('w-full')
                with ui.row():
                    run_btn  = ui.button('Run',  color='green')
                    stop_btn = ui.button('Stop', color='red')
                traj_label = ui.label('Idle').style('color: gray;')

                ui.separator()
                ui.label('Controls').style('font-size: 16px; font-weight: bold;')
                with ui.row():
                    reset_odom_btn  = ui.button('Reset Odometry',   color='blue')
                    reinit_pf_btn   = ui.button('Reinit Particles',  color='purple')

                ui.separator()
                ui.label('Particle Filter State').style('font-size: 16px; font-weight: bold;')
                mean_label   = ui.label('Best: x=—  y=—  θ=—')
                enc_label    = ui.label('Encoder: —')
                weight_label = ui.label('Best weight: —')
                tick_label   = ui.label('Tick: 0')

    # ------------------------------------------------------------------ #
    def on_run():
        if not connected[0]:
            ui.notify('Connect to robot first', color='warning')
            return
        trajectory_runner.start(traj_select.value)
        traj_label.set_text(f'Running: {traj_select.value}')
        traj_label.style('color: lightgreen;')

    def on_stop():
        trajectory_runner.stop()
        # Send a stop command immediately
        if connected[0] and msg_sender is not None:
            msg_sender.send_control_signal([0, 0])
        traj_label.set_text('Stopped')
        traj_label.style('color: orange;')

    def on_reset_odom():
        """Reset the encoder reference so the next PF delta starts from zero."""
        pf.last_encoder_counts = sensor_signal.encoder_counts
        mean_states.clear()
        ui.notify('Odometry reset', color='info')

    def on_reinit_particles():
        """Replace the particle set with a fresh uniform distribution."""
        pf.particle_set = ParticleSet(
            parameters.num_particles,
            pf.map.particle_range,
            State(parameters.initial_state_x, parameters.initial_state_y, parameters.initial_state_theta),
            State(parameters.initial_state_stdev_x, parameters.initial_state_stdev_y, parameters.initial_state_stdev_theta),
            False,  # uniform — ignore initial_state
        )
        mean_states.clear()
        ui.notify('Particles reinitialized', color='info')

    run_btn.on_click(on_run)
    stop_btn.on_click(on_stop)
    reset_odom_btn.on_click(on_reset_odom)
    reinit_pf_btn.on_click(on_reinit_particles)

    # ------------------------------------------------------------------ #
    def draw_pf():
        """Redraw the particle filter plot inside the NiceGUI pyplot widget."""
        with pf_plot:
            fig = pf_plot.fig
            fig.patch.set_facecolor('#1e1e1e')
            plt.clf()
            plt.style.use('dark_background')
            ax = plt.gca()
            ax.set_facecolor('#1e1e1e')

            x_min, x_max = parameters.grid_dimensions[0]
            y_min, y_max = parameters.grid_dimensions[1]

            # Walls
            for wall in map.wall_list:
                plt.plot([wall.corner1.x, wall.corner2.x],
                         [wall.corner1.y, wall.corner2.y], 'w-', linewidth=2)

            # Best particle
            best = max(pf.particle_set.particle_list, key=lambda p: p.weight)
            bs = best.state

            # All particles — colour-coded by weight (low=dark green, high=yellow)
            weights = [p.weight for p in pf.particle_set.particle_list]
            w_min, w_max = min(weights), max(weights)
            w_range = w_max - w_min if w_max > w_min else 1.0
            norm_w = [(w - w_min) / w_range for w in weights]
            px = [p.state.x for p in pf.particle_set.particle_list]
            py = [p.state.y for p in pf.particle_set.particle_list]
            plt.scatter(px, py, c=norm_w, cmap='YlGn', s=8, alpha=0.7, vmin=0, vmax=1)

            # Lidar rays from best particle (salmon)
            for i in range(len(sensor_signal.angles)):
                dist  = sensor_signal.convert_hardware_distance(sensor_signal.distances[i])
                angle = sensor_signal.convert_hardware_angle(sensor_signal.angles[i]) + bs.theta
                plt.plot([bs.x, bs.x + dist * math.cos(angle)],
                         [bs.y, bs.y + dist * math.sin(angle)],
                         color='salmon', linewidth=0.5, alpha=0.4)

            # Best-particle arrow (orange)
            plt.quiver(bs.x, bs.y,
                       math.cos(bs.theta), math.sin(bs.theta),
                       color='orange', scale=8, label=f'Best (w={best.weight:.4f})')

            # Best-particle trajectory history (orange line)
            if len(mean_states) > 1:
                plt.plot([s.x for s in mean_states],
                         [s.y for s in mean_states], color='orange', linewidth=1, alpha=0.6)

            plt.xlim(x_min - 0.2, x_max + 0.2)
            plt.ylim(y_min - 0.2, y_max + 0.2)
            plt.xlabel('X (m)', color='lightgray')
            plt.ylabel('Y (m)', color='lightgray')
            plt.tick_params(colors='lightgray')
            plt.grid(True, alpha=0.3)
            plt.legend(loc='upper right', fontsize=8)

    # ------------------------------------------------------------------ #
    async def control_loop():
        nonlocal sensor_signal, msg_sender, msg_receiver

        # Handle connect / disconnect
        if udp_switch.value and not connected[0]:
            udp, success = robot_python_code.create_udp_communication(
                parameters.arduinoIP, parameters.localIP,
                parameters.arduinoPort, parameters.localPort,
                parameters.bufferSize,
            )
            if success:
                msg_sender   = robot_python_code.MsgSender(time.perf_counter(), parameters.num_robot_control_signals, udp)
                msg_receiver = robot_python_code.MsgReceiver(time.perf_counter(), parameters.num_robot_sensors, udp)
                
                # Drain one packet to get the real starting encoder count,
                # so the first PF update doesn't compute a giant delta.
                status_label.set_text('Waiting for first packet...')
                first_signal = sensor_signal
                for _ in range(20):
                    first_signal = msg_receiver.receive_robot_sensor_signal(first_signal)
                    if first_signal.encoder_counts != 0:
                        break
                    time.sleep(0.05)
                pf.last_encoder_counts = first_signal.encoder_counts
                sensor_signal = first_signal

                connected[0] = True
                status_label.set_text('Connected')
                status_label.style('color: lightgreen;')
                last_time[0] = time.perf_counter()
                print(f'UDP connected. Initial encoder: {first_signal.encoder_counts}')
            else:
                udp_switch.value = False
                status_label.set_text('Connection failed')
                status_label.style('color: red;')

        elif not udp_switch.value and connected[0]:
            trajectory_runner.stop()
            msg_sender   = None
            msg_receiver = None
            connected[0] = False
            status_label.set_text('Disconnected')
            status_label.style('color: gray;')
            traj_label.set_text('Idle')
            traj_label.style('color: gray;')

        if not connected[0]:
            return

        # Receive sensor data
        sensor_signal = msg_receiver.receive_robot_sensor_signal(sensor_signal)

        # Compute delta_t
        now = time.perf_counter()
        delta_t = now - last_time[0]
        last_time[0] = now

        # --- Trajectory: get command and send it ---
        if trajectory_runner.is_running:
            cmd_speed, cmd_steering = trajectory_runner.update()
            if not trajectory_runner.is_running:
                # Just finished
                traj_label.set_text('Done')
                traj_label.style('color: lightblue;')
            msg_sender.send_control_signal([cmd_speed, cmd_steering])
        else:
            # No active trajectory — send stop so robot doesn't coast
            msg_sender.send_control_signal([0, 0])

        # Build PF inputs
        u_t = RobotOdomSignal(sensor_signal.encoder_counts, sensor_signal.steering)
        z_t = sensor_signal

        # Run PF update
        pf.update(u_t, z_t, delta_t)
        # Track best-particle trajectory
        best = max(pf.particle_set.particle_list, key=lambda p: p.weight)
        mean_states.append(best.state.deepcopy())

        # Update status labels
        mean_label.set_text(f'x={best.state.x:.3f}  y={best.state.y:.3f}  θ={best.state.theta:.3f} rad')
        enc_label.set_text(f'Encoder: {sensor_signal.encoder_counts}')
        weight_label.set_text(f'Best weight: {best.weight:.4f}')
        tick_label.set_text(f'Tick: {tick[0]}')
        tick[0] += 1

        # Redraw plot
        draw_pf()

    ui.timer(LOOP_PERIOD, control_loop)


ui.run(native=True)
