"""
execute_trajectory.py

Headless trajectory executor — no GUI required.

Usage:
    python execute_trajectory.py <trajectory_name>

Example:
    python execute_trajectory.py "Straight Line"
    python execute_trajectory.py scurve

The script will:
  1. Connect to the robot over UDP.
  2. Prompt the user to press Enter to start.
  3. Run the named trajectory with data logging enabled.
  4. Disconnect and flush the log when done.
"""

import argparse
import sys
import time

import parameters
import robot_python_code
from robot import Robot
from trajectories import TRAJECTORIES, TrajectoryRunner

LOOP_PERIOD = 0.05  # seconds


def main():
    parser = argparse.ArgumentParser(description="Execute a named robot trajectory.")
    parser.add_argument("trajectory", help=f"Name of trajectory to run. Choices: {list(TRAJECTORIES.keys())}")
    args = parser.parse_args()

    trajectory_name = args.trajectory
    if trajectory_name not in TRAJECTORIES:
        print(f"Error: unknown trajectory '{trajectory_name}'.")
        print(f"Available trajectories: {list(TRAJECTORIES.keys())}")
        sys.exit(1)

    # --- Set up robot and trajectory runner ---
    robot = Robot(video_capture=None)
    trajectory_runner = TrajectoryRunner()

    # --- Connect to robot over UDP ---
    print(f"Connecting to robot at {parameters.arduinoIP}:{parameters.arduinoPort} ...")
    udp, success = robot_python_code.create_udp_communication(
        parameters.arduinoIP,
        parameters.localIP,
        parameters.arduinoPort,
        parameters.localPort,
        parameters.bufferSize,
    )
    if not success:
        print("Error: failed to create UDP connection. Check network settings in parameters.py.")
        sys.exit(1)

    robot.setup_udp_connection(udp)
    robot.connected_to_hardware = True
    print("UDP connection established.")

    # --- Wait for user to press Enter ---
    input(f"\nPress Enter to start trajectory '{trajectory_name}' ...\n")

    # --- Start trajectory and logging ---
    robot.data_logger.set_next_session_name(trajectory_name)
    trajectory_runner.start(trajectory_name)
    logging_on = True

    print(f"Running '{trajectory_name}' ...")

    # --- Control loop ---
    try:
        while trajectory_runner.is_running:
            loop_start = time.perf_counter()

            cmd_speed, cmd_steering = trajectory_runner.update()
            robot.control_loop(cmd_speed, cmd_steering, logging_on)

            # Sleep for the remainder of the loop period
            elapsed = time.perf_counter() - loop_start
            sleep_time = LOOP_PERIOD - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nInterrupted — sending stop command.")

    # --- Stop robot and flush log ---
    robot.control_loop(0, 0, logging_on)  # one final tick with zero commands
    robot.data_logger.log(False, time.perf_counter(), [0, 0],
                          robot.robot_sensor_signal,
                          robot.particle_filter.particle_set.mean_state,
                          robot.particle_filter.particle_set, None)
    robot.eliminate_udp_connection()

    print("Trajectory complete. Data saved.")


if __name__ == "__main__":
    main()
