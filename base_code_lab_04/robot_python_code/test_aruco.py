import cv2
import cv2.aruco as aruco
import numpy as np
import parameters  # Assumes your parameters file is in the same folder
from robot_python_code import my_estimatePoseSingleMarkers

def main():
    # Use the camera ID defined in your parameters
    cap = cv2.VideoCapture(1)
    
    # Setup ArUco detector using your dictionary settings
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_100)
    aruco_params = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, aruco_params)

    print("Press 'q' to quit. Watch the console for X, Y values.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        if ids is not None:
            # Draw the basic marker borders
            aruco.drawDetectedMarkers(frame, corners, ids)

            for i in range(len(ids)):
                # Get pose using your specific function
                rvecs, tvecs, _ = my_estimatePoseSingleMarkers(
                    corners[i:i+1], 
                    parameters.marker_length, 
                    parameters.camera_matrix, 
                    parameters.dist_coeffs
                )
                
                rvec, tvec = rvecs[0], tvecs[0]
                
                # Extract coordinates
                x = tvec[0][0]
                y = tvec[1][0]
                z = tvec[2][0]

                # --- DEBUGGING OUTPUT ---
                # To make the center of the screen 0,0 and 'Up' positive:
                # We show the RAW value vs the ADJUSTED value
                y_adjusted = -y 
                
                print(f"ID: {ids[i][0]} | RAW X: {x:6.2f} | RAW Y: {y:6.2f} | ADJ Y (Up+): {y_adjusted:6.2f}")

                # Draw the 3D axes on the marker to see the orientation
                cv2.drawFrameAxes(frame, parameters.camera_matrix, parameters.dist_coeffs, rvec, tvec, 0.03)
                
                # Label the screen with the coordinates
                cv2.putText(frame, f"X:{x:.1f} Y:{y:.1f}", 
                            (int(corners[i][0][0][0]), int(corners[i][0][0][1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow('ArUco Pose Monitor', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


