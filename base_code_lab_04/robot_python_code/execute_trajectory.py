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

import cv2

import parameters
import robot_python_code
from aruco_pose_estimator import ArucoPoseEstimator
from robot import Robot
from trajectories import TRAJECTORIES, TrajectoryRunner

LOOP_PERIOD = 0.05  # seconds


def capture_aruco_pose(cap: cv2.VideoCapture, estimator: ArucoPoseEstimator) -> dict | None:
    """Grab one frame from an already-open capture and return a pose dict or None."""
    ret, frame = cap.read()
    if not ret or frame is None:
        return None
    pose, _ = estimator.estimate_pose(frame)
    return pose


def main():
    parser = argparse.ArgumentParser(description="Execute a named robot trajectory.")
    parser.add_argument("trajectory", help=f"Name of trajectory to run. Choices: {list(TRAJECTORIES.keys())}")
    args = parser.parse_args()

    trajectory_name = args.trajectory
    if trajectory_name not in TRAJECTORIES:
        print(f"Error: unknown trajectory '{trajectory_name}'.")
        print(f"Available trajectories: {list(TRAJECTORIES.keys())}")
        sys.exit(1)

    # --- Set up robot, trajectory runner, and ArUco estimator ---
    robot = Robot(video_capture=None)
    trajectory_runner = TrajectoryRunner()
    estimator = ArucoPoseEstimator(
        camera_matrix=parameters.camera_matrix,
        dist_coeffs=parameters.dist_coeffs,
        marker_length=parameters.marker_length,
        known_markers=parameters.KNOWN_MARKERS,
        robot_marker_id=7,
    )

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

    # --- Open camera (stays open for the entire trajectory) ---
    cap = cv2.VideoCapture(parameters.camera_id + cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Warning: camera could not be opened — poses will be None.")

    # --- Start trajectory and logging ---
    robot.data_logger.set_next_session_name(trajectory_name)
    trajectory_runner.start(trajectory_name)
    logging_on = True

    print(f"Running '{trajectory_name}' ...")

    # --- Control loop ---
    try:
        while trajectory_runner.is_running:
            loop_start = time.perf_counter()

            pose = capture_aruco_pose(cap, estimator)
            cmd_speed, cmd_steering = trajectory_runner.update()
            robot.control_loop(cmd_speed, cmd_steering, logging_on, aruco_pose=pose)

            # Sleep for the remainder of the loop period
            elapsed = time.perf_counter() - loop_start
            sleep_time = LOOP_PERIOD - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nInterrupted — sending stop command.")

    # --- Final tick and cleanup ---
    pose = capture_aruco_pose(cap, estimator)
    robot.control_loop(0, 0, logging_on, aruco_pose=pose)
    robot.data_logger.log(False, time.perf_counter(), [0, 0],
                          robot.robot_sensor_signal,
                          robot.particle_filter.particle_set.mean_state,
                          robot.particle_filter.particle_set, None)
    cap.release()
    robot.eliminate_udp_connection()

    print("Trajectory complete. Data saved.")


if __name__ == "__main__":
    main()
