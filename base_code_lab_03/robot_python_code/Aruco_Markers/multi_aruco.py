import cv2
import cv2.aruco as aruco
import numpy as np

from base_code_lab_03.robot_python_code import parameters

# --- 1. SETUP CAMERA PARAMETERS ---
# Replace these with your 'parameters.camera_matrix' and 'parameters.dist_coeffs'
# Using dummy values here so the script runs out-of-the-box for testing.
camera_matrix = parameters.camera_matrix
dist_coeffs = parameters.dist_coeffs
marker_length: float = parameters.marker_length  # meters

# Define the 3D coordinates of the marker corners in its own coordinate system
obj_points = np.array([
    [-marker_length / 2,  marker_length / 2, 0],
    [ marker_length / 2,  marker_length / 2, 0],
    [ marker_length / 2, -marker_length / 2, 0],
    [-marker_length / 2, -marker_length / 2, 0]
], dtype=np.float32)

def get_euler_angles(rvec):
    """Converts a rotation vector to readable Euler angles (Roll, Pitch, Yaw) in degrees."""
    R, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(R[0,0] * R[0,0] + R[1,0] * R[1,0])
    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(R[2,1], R[2,2])
        y = np.arctan2(-R[2,0], sy)
        z = np.arctan2(R[1,0], R[0,0])
    else:
        x = np.arctan2(-R[1,2], R[1,1])
        y = np.arctan2(-R[2,0], sy)
        z = 0
        
    return np.degrees([x, y, z])

def main():
    cap = cv2.VideoCapture(1)
    
    # Setup ArUco detector
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_100)
    aruco_params = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, aruco_params)

    print("Detecting multiple markers. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        if ids is not None:
            # Draw bounding boxes and IDs for all detected markers
            aruco.drawDetectedMarkers(frame, corners, ids)

            # Iterate through EVERY detected marker in the current frame
            for i in range(len(ids)):
                marker_id = ids[i][0]
                marker_corners = corners[i][0]

                # Standard OpenCV method to estimate pose for a single marker
                _, rvec, tvec = cv2.solvePnP(obj_points, marker_corners, camera_matrix, dist_coeffs)

                # Extract translation (Position)
                tx, ty, tz = tvec.flatten()
                
                # Extract rotation (Orientation) as Roll, Pitch, Yaw
                roll, pitch, yaw = get_euler_angles(rvec)

                # Draw 3D axes on the marker
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, marker_length * 0.75)

                # --- DRAW DATA ON SCREEN ---
                # Define text position based on the top-left corner of the marker
                text_x = int(marker_corners[0][0])
                text_y = int(marker_corners[0][1]) - 40

                # Format the text strings
                pos_str = f"Pos: x{tx:.2f} y{ty:.2f} z{tz:.2f}"
                ori_str = f"Ori: R{roll:.0f} P{pitch:.0f} Y{yaw:.0f}"

                # Display Position
                cv2.putText(frame, pos_str, (text_x, text_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                # Display Orientation underneath
                cv2.putText(frame, ori_str, (text_x, text_y + 15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        cv2.imshow('Multi-Marker ArUco Pose Monitor', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()