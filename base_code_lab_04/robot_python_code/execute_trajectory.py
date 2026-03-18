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
  2. Open the camera.
  3. Prompt the user to press Enter to start.
  4. Run the named trajectory with data logging enabled.
     At every tick: capture frame, estimate ArUco pose, log both.
  5. Disconnect and flush the log when done.

Each logged 'frame' entry is a dict:
    {'pose': <pose dict or None>, 'jpeg': <JPEG bytes or None>}
"""

import argparse
import sys
import time

import cv2
import numpy as np

import parameters
import robot_python_code
from aruco_pose_estimator import ArucoPoseEstimator
from robot import Robot
from trajectories import TRAJECTORIES, TrajectoryRunner

LOOP_PERIOD = 0.05  # seconds


def capture_frame_and_pose(cap: cv2.VideoCapture, estimator: ArucoPoseEstimator) -> dict:
    """Grab one frame, estimate ArUco pose, return {'pose': ..., 'jpeg': ...}.
    Both values may be None if capture or estimation fails.
    """
    ret, frame = cap.read()
    if not ret or frame is None:
        return {'pose': None, 'jpeg': None}

    pose, annotated = estimator.estimate_pose(frame)

    _, buf = cv2.imencode('.jpg', annotated)
    jpeg = buf.tobytes()

    return {'pose': pose, 'jpeg': jpeg}


def main():
    parser = argparse.ArgumentParser(description="Execute a named robot trajectory.")
    parser.add_argument("trajectory", help=f"Trajectory to run. Choices: {list(TRAJECTORIES.keys())}")
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

    # --- Open camera (stays open for the entire trajectory) ---
    cap = cv2.VideoCapture(parameters.camera_id + cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Warning: camera could not be opened — frames and poses will be None.")

    # --- Wait for user to press Enter ---
    input(f"\nPress Enter to start trajectory '{trajectory_name}' ...\n")

    # --- Start trajectory and logging ---
    robot.data_logger.set_next_session_name(trajectory_name)
    trajectory_runner.start(trajectory_name)
    logging_on = True

    print(f"Running '{trajectory_name}' ...")
    print(f"{'Tick':>5}  {'Wall (s)':>8}  {'Loop (ms)':>9}  {'Sleep (ms)':>10}  {'Hz':>6}  Pose")
    print("-" * 70)

    tick = 0
    loop_times = []
    run_start = time.perf_counter()

    # --- Control loop ---
    try:
        while trajectory_runner.is_running:
            loop_start = time.perf_counter()

            frame_data = capture_frame_and_pose(cap, estimator)
            cmd_speed, cmd_steering = trajectory_runner.update()
            robot.control_loop(cmd_speed, cmd_steering, logging_on, aruco_pose=frame_data)

            elapsed = time.perf_counter() - loop_start
            sleep_time = LOOP_PERIOD - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

            # Per-tick profiling
            tick_total = time.perf_counter() - loop_start
            loop_times.append(tick_total)
            wall_time = loop_start - run_start
            hz = 1.0 / tick_total if tick_total > 0 else float('inf')
            pose = frame_data['pose']
            pose_str = f"x={pose['x']:.2f} y={pose['y']:.2f} yaw={pose['yaw']:.1f}" if pose else "no pose"
            print(f"{tick:>5}  {wall_time:>8.3f}  {elapsed*1000:>9.1f}  {max(sleep_time,0)*1000:>10.1f}  {hz:>6.1f}  {pose_str}")
            tick += 1

    except KeyboardInterrupt:
        print("\nInterrupted — sending stop command.")

    # --- Final tick with camera, then cleanup ---
    frame_data = capture_frame_and_pose(cap, estimator)
    robot.control_loop(0, 0, logging_on, aruco_pose=frame_data)
    robot.data_logger.log(False, time.perf_counter(), [0, 0],
                          robot.robot_sensor_signal,
                          robot.particle_filter.particle_set.mean_state,
                          robot.particle_filter.particle_set, None)
    cap.release()
    robot.eliminate_udp_connection()

    # --- Summary ---
    total_duration = time.perf_counter() - run_start
    if loop_times:
        avg_dt = sum(loop_times) / len(loop_times)
        min_dt = min(loop_times)
        max_dt = max(loop_times)
        print()
        print("=" * 70)
        print(f"  Samples collected : {tick}")
        print(f"  Total duration    : {total_duration:.3f} s")
        print(f"  Effective rate    : {tick / total_duration:.1f} Hz  (target {1/LOOP_PERIOD:.0f} Hz)")
        print(f"  Loop dt — avg: {avg_dt*1000:.1f} ms  min: {min_dt*1000:.1f} ms  max: {max_dt*1000:.1f} ms")
        print("=" * 70)

    print("Trajectory complete. Data saved.")


if __name__ == "__main__":
    main()
