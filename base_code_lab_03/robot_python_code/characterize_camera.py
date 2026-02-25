import os
import csv
from typing import Dict

import cv2
import cv2.aruco as aruco
import numpy as np

import parameters

# --- 1. SETUP PARAMETERS ---
camera_matrix = parameters.camera_matrix
dist_coeffs = parameters.dist_coeffs
marker_length: float = parameters.marker_length
'''[m]'''

# Dictionary of known markers: ID -> (x, y, yaw_degrees)
KNOWN_MARKERS: Dict[int, Dict[str, float]] = {
    1: {'x': 1.05, 'y': 0.85, 'yaw': 0},   # Close Left
    3: {'x': 1.05, 'y': -0.95, 'yaw': 90}, # Close Right
    6: {'x': 1.05, 'y': 0.05,  'yaw': 90}, # Close Center
    2: {'x': 2.05, 'y': 0.05,  'yaw': 0},  # Mid Center
    4: {'x': 3.05, 'y': 0.85, 'yaw': 180}, # Far Left
    5: {'x': 3.05, 'y': -0.95,  'yaw': 90},# Far Right
}
ROBOT_MARKER_ID = 7
OUTPUT_FILE = "characterization_output.csv"

# --- 2. HELPER FUNCTIONS ---
def get_marker_corners_world(x, y, yaw_deg, marker_length):
    """Returns the 4 corners of a marker in world coordinates given its center (x, y) and orientation (yaw)."""
    half_l = marker_length / 2.0
    local_corners = np.array([
        [-half_l,  half_l, 0],
        [ half_l,  half_l, 0],
        [ half_l, -half_l, 0],
        [-half_l, -half_l, 0]
    ])
    
    theta: float = np.radians(yaw_deg)
    '''[rad]'''
    c, s = np.cos(theta), np.sin(theta)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    
    world_corners = (Rz @ local_corners.T).T + np.array([x, y, 0])
    return np.float32(world_corners)

def create_transform_matrix(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.flatten()
    return T

def get_euler_angles_from_matrix(R):
    sy = np.sqrt(R[0,0] * R[0,0] + R[1,0] * R[1,0])
    if sy > 1e-6:
        x, y, z = np.arctan2(R[2,1], R[2,2]), np.arctan2(-R[2,0], sy), np.arctan2(R[1,0], R[0,0])
    else:
        x, y, z = np.arctan2(-R[1,2], R[1,1]), np.arctan2(-R[2,0], sy), 0
    return np.degrees([x, y, z])

# --- 3. MAIN LOOP ---
def main():
    cap = cv2.VideoCapture(1)
    
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_100)
    aruco_params = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, aruco_params)

    local_marker_corners = get_marker_corners_world(0, 0, 0, marker_length)

    # --- Saving State Variables ---
    is_saving = False
    run_counter = 0

    # Write CSV header if the file doesn't exist
    if not os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Run", "X", "Y", "Z", "Roll", "Pitch", "Yaw"])

    print("Strict tracking initialized: Requires Robot (ID 7) + >=1 Fixed Marker.")
    print("Press 's' to toggle saving data.")
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        if ids is not None:
            aruco.drawDetectedMarkers(frame, corners, ids)
            
            obj_points_world = []
            img_points = []
            robot_corners = None

            # Sort detections into fixed environment markers vs. the robot marker
            for i, marker_id in enumerate(ids.flatten()):
                if marker_id in KNOWN_MARKERS:
                    info = KNOWN_MARKERS[marker_id]
                    world_corners = get_marker_corners_world(info['x'], info['y'], info['yaw'], marker_length)
                    obj_points_world.extend(world_corners)
                    img_points.extend(corners[i][0])
                elif marker_id == ROBOT_MARKER_ID:
                    robot_corners = corners[i][0]

            # --- STRICT VISIBILITY CHECK ---
            # Require at least 4 environment corners (1 fixed marker) AND the robot marker
            if len(obj_points_world) >= 4 and robot_corners is not None:
                
                obj_points_world = np.array(obj_points_world)
                img_points = np.array(img_points)
                
                # 1. Standard solvePnP for the Camera's World Pose
                success, rvec_cam, tvec_cam = cv2.solvePnP(obj_points_world, img_points, camera_matrix, dist_coeffs)
                
                if success:
                    T_W2C = create_transform_matrix(rvec_cam, tvec_cam)
                    T_C2W = np.linalg.inv(T_W2C) # Camera pose in World

                    # 2. Standard solvePnP for the Robot's Camera Pose
                    _, rvec_rob, tvec_rob = cv2.solvePnP(local_marker_corners, robot_corners, camera_matrix, dist_coeffs)
                    T_R2C = create_transform_matrix(rvec_rob, tvec_rob)

                    # 3. Calculate Robot's World Pose: T_R^W = T_C^W * T_R^C
                    T_R2W = T_C2W @ T_R2C

                    # Extract final coordinates and angles
                    rob_x, rob_y, rob_z = T_R2W[:3, 3]
                    rob_roll, rob_pitch, rob_yaw = get_euler_angles_from_matrix(T_R2W[:3, :3])

                    # Display on screen
                    text_x, text_y = int(robot_corners[0][0]), int(robot_corners[0][1]) - 40
                    
                    cv2.putText(frame, "ROBOT WORLD POS:", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    cv2.putText(frame, f"X:{rob_x:.2f} Y:{rob_y:.2f} Z:{rob_z:.2f}", (text_x, text_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.putText(frame, f"Yaw:{rob_yaw:.1f} deg", (text_x, text_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

                    # --- SAVE DATA ---
                    if is_saving:
                        with open(OUTPUT_FILE, mode='a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([run_counter, rob_x, rob_y, rob_z, rob_roll, rob_pitch, rob_yaw])

        # --- DRAW SAVING STATUS HUD ---
        if is_saving:
            cv2.putText(frame, f"RECORDING - RUN {run_counter}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        else:
            cv2.putText(frame, "NOT RECORDING (Press 's')", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)

        cv2.imshow('Strict World Frame Tracking', frame)

        # --- KEYBOARD CONTROLS ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            is_saving = not is_saving
            if is_saving:
                run_counter += 1
                print(f"[*] Started recording run {run_counter}...")
            else:
                print(f"[*] Stopped recording run {run_counter}.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()