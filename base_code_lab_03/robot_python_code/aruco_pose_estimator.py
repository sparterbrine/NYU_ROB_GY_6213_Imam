from typing import Dict, Tuple, Optional

import cv2
import cv2.aruco as aruco
import numpy as np

import parameters


class ArucoPoseEstimator:
    def __init__(self, camera_matrix: np.ndarray, dist_coeffs: np.ndarray, 
                 marker_length: float, known_markers: Dict[int, Dict[str, float]], 
                 robot_marker_id: int = 7):
        """
        Initializes the Aruco Pose Estimator.
        """
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.marker_length = marker_length
        self.known_markers = known_markers
        self.robot_marker_id = robot_marker_id

        # Set up Aruco detector
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_100)
        self.aruco_params = aruco.DetectorParameters()
        self.detector = aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

        # Pre-calculate the robot marker's local corners
        self.local_marker_corners = self.get_marker_corners_world(0, 0, 0, self.marker_length)

    @staticmethod
    def get_marker_corners_world(x: float, y: float, yaw_deg: float, marker_length: float) -> np.ndarray:
        """Returns the 4 corners of a marker in world coordinates given its center (x, y) and orientation (yaw)."""
        half_l = marker_length / 2.0
        local_corners = np.array([
            [-half_l,  half_l, 0],
            [ half_l,  half_l, 0],
            [ half_l, -half_l, 0],
            [-half_l, -half_l, 0]
        ])
        
        theta: float = np.radians(yaw_deg)
        c, s = np.cos(theta), np.sin(theta)
        Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        
        world_corners = (Rz @ local_corners.T).T + np.array([x, y, 0])
        return np.float32(world_corners)

    @staticmethod
    def create_transform_matrix(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
        """Creates a 4x4 homogenous transformation matrix from rotation and translation vectors."""
        R, _ = cv2.Rodrigues(rvec)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = tvec.flatten()
        return T

    @staticmethod
    def get_euler_angles_from_matrix(R: np.ndarray) -> np.ndarray:
        """Extracts Roll, Pitch, Yaw (in degrees) from a rotation matrix."""
        sy = np.sqrt(R[0,0] * R[0,0] + R[1,0] * R[1,0])
        if sy > 1e-6:
            x, y, z = np.arctan2(R[2,1], R[2,2]), np.arctan2(-R[2,0], sy), np.arctan2(R[1,0], R[0,0])
        else:
            x, y, z = np.arctan2(-R[1,2], R[1,1]), np.arctan2(-R[2,0], sy), 0
        return np.degrees([x, y, z])

    def estimate_pose(self, frame: np.ndarray) -> Tuple[Optional[Dict[str, float]], np.ndarray]:
        """
        Takes an image frame, detects markers, and returns the estimated robot pose and the annotated frame.
        
        Returns:
            pose: A dictionary with keys ('x', 'y', 'z', 'roll', 'pitch', 'yaw') or None if invalid.
            annotated_frame: The image with Aruco markers and text drawn on it.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        
        pose = None

        if ids is not None:
            aruco.drawDetectedMarkers(frame, corners, ids)
            
            obj_points_world = []
            img_points = []
            robot_corners = None

            # Sort detections into fixed environment markers vs. the robot marker
            for i, marker_id in enumerate(ids.flatten()):
                if marker_id in self.known_markers:
                    info = self.known_markers[marker_id]
                    world_corners = self.get_marker_corners_world(info['x'], info['y'], info['yaw'], self.marker_length)
                    obj_points_world.extend(world_corners)
                    img_points.extend(corners[i][0])
                elif marker_id == self.robot_marker_id:
                    robot_corners = corners[i][0]

            # --- STRICT VISIBILITY CHECK ---
            # Require at least 4 environment corners (1 fixed marker) AND the robot marker
            if len(obj_points_world) >= 4 and robot_corners is not None:
                
                obj_points_world = np.array(obj_points_world)
                img_points = np.array(img_points)
                
                # 1. Standard solvePnP for the Camera's World Pose
                success, rvec_cam, tvec_cam = cv2.solvePnP(obj_points_world, img_points, self.camera_matrix, self.dist_coeffs)
                
                if success:
                    T_W2C = self.create_transform_matrix(rvec_cam, tvec_cam)
                    T_C2W = np.linalg.inv(T_W2C) # Camera pose in World

                    # 2. Standard solvePnP for the Robot's Camera Pose
                    _, rvec_rob, tvec_rob = cv2.solvePnP(self.local_marker_corners, robot_corners, self.camera_matrix, self.dist_coeffs)
                    T_R2C = self.create_transform_matrix(rvec_rob, tvec_rob)

                    # 3. Calculate Robot's World Pose: T_R^W = T_C^W * T_R^C
                    T_R2W = T_C2W @ T_R2C

                    # Extract final coordinates and angles
                    rob_x, rob_y, rob_z = T_R2W[:3, 3]
                    rob_roll, rob_pitch, rob_yaw = self.get_euler_angles_from_matrix(T_R2W[:3, :3])

                    # Store the pose to return
                    pose = {
                        'x': rob_x, 'y': rob_y, 'z': rob_z,
                        'roll': rob_roll, 'pitch': rob_pitch, 'yaw': rob_yaw
                    }

                    # Display on screen
                    text_x, text_y = int(robot_corners[0][0]), int(robot_corners[0][1]) - 40
                    
                    cv2.putText(frame, "ROBOT WORLD POS:", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    cv2.putText(frame, f"X:{rob_x:.2f} Y:{rob_y:.2f} Z:{rob_z:.2f}", (text_x, text_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.putText(frame, f"Yaw:{rob_yaw:.1f} deg", (text_x, text_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        return pose, frame


# --- 3. MAIN LOOP (Usage Example) ---
def main():
    # Setup specific constraints
    KNOWN_MARKERS: Dict[int, Dict[str, float]] = {
        1: {'x': 1.05, 'y': 0.85, 'yaw': 0},   # Close Left
        3: {'x': 1.05, 'y': -0.95, 'yaw': 90}, # Close Right
        6: {'x': 1.05, 'y': 0.05,  'yaw': 90}, # Close Center
        2: {'x': 2.05, 'y': 0.05,  'yaw': 0},  # Mid Center
        4: {'x': 3.05, 'y': 0.85, 'yaw': 180}, # Far Left
        5: {'x': 3.05, 'y': -0.95,  'yaw': 90},# Far Right
    }
    
    # Initialize the Estimator Object
    estimator = ArucoPoseEstimator(
        camera_matrix=parameters.camera_matrix,
        dist_coeffs=parameters.dist_coeffs,
        marker_length=parameters.marker_length,
        known_markers=KNOWN_MARKERS,
        robot_marker_id=7
    )

    cap = cv2.VideoCapture(1)

    print("Strict tracking initialized: Requires Robot (ID 7) + >=1 Fixed Marker.")
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # 1-Liner to extract pose and the drawable frame!
        pose, annotated_frame = estimator.estimate_pose(frame)

        if pose is not None:
            # You can access specific elements easily: pose['x'], pose['yaw'], etc.
            pass 

        cv2.imshow('Strict World Frame Tracking', annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()