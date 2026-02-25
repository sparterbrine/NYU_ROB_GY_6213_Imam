import cv2
import cv2.aruco as aruco
import numpy as np
import parameters  # Assumes your parameters file is in the same folder
from robot_python_code import my_estimatePoseSingleMarkers
import time  ### ADDED: For tracking the 1-second interval
import csv   ### ADDED: For formatting the output file
import os    ### ADDED: To check if the file already exists

def main():
    # Use the camera ID defined in your parameters
    cap = cv2.VideoCapture(1)
    
    # Setup ArUco detector using your dictionary settings
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_100)
    aruco_params = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, aruco_params)

    ### ADDED: Setup variables for saving state and timing
    is_saving = False
    last_save_time = 0.0
    csv_filename = 'output.csv'
    
    # Check if file exists so we only write the header once
    file_exists = os.path.isfile(csv_filename)
    
    # Open the CSV file in 'append' mode ('a'). 
    # This ensures it picks up where it left off instead of overwriting.
    csv_file = open(csv_filename, mode='a', newline='')
    csv_writer = csv.writer(csv_file)
    
    if not file_exists:
        # Write headers for Time, Marker ID, Position (X,Y,Z) and Orientation (Rx, Ry, Rz)
        csv_writer.writerow(['Timestamp', 'Marker_ID', 'X', 'Y', 'Z', 'Rx', 'Ry', 'Rz'])

    print("Press 'q' to quit.")
    print("Press 's' to toggle saving data to CSV.")
    print("Watch the console for X, Y values and saving status.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        ### ADDED: Check if 1 second has passed
        current_time = time.time()
        time_to_save = is_saving and (current_time - last_save_time >= 1.0)

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
                
                # Extract coordinates and flatten arrays for easy handling
                tx, ty, tz = tvec.flatten()
                rx, ry, rz = rvec.flatten()

                # --- DEBUGGING OUTPUT ---
                y_adjusted = -ty 
                
                # Only print to console occasionally so we don't flood it, 
                # or you can leave it printing every frame if you prefer.
                print(f"ID: {ids[i][0]} | RAW X: {tx:6.2f} | RAW Y: {ty:6.2f} | RAW Z: {tz:6.2f}")

                ### ADDED: Write to CSV if it's time
                if time_to_save:
                    csv_writer.writerow([current_time, ids[i][0], tx, ty, tz, rx, ry, rz])

                # Draw the 3D axes on the marker to see the orientation
                cv2.drawFrameAxes(frame, parameters.camera_matrix, parameters.dist_coeffs, rvec, tvec, 0.03)
                
                # Label the screen with the coordinates
                cv2.putText(frame, f"X:{tx:.1f} Y:{ty:.1f}", 
                            (int(corners[i][0][0][0]), int(corners[i][0][0][1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        ### ADDED: Reset the timer if a save occurred in this frame
        if time_to_save:
            last_save_time = current_time
            # Optional: Visual cue on the frame that saving happened
            cv2.circle(frame, (30, 30), 10, (0, 0, 255), -1) 

        ### ADDED: Display saving status on the screen
        status_text = "SAVING ON" if is_saving else "SAVING OFF"
        status_color = (0, 0, 255) if is_saving else (0, 255, 0)
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

        cv2.imshow('ArUco Pose Monitor', frame)

        ### ADDED: Handle keypresses for 'q' (quit) and 's' (save)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            is_saving = not is_saving
            print(f"\n--- Saving Toggled {'ON' if is_saving else 'OFF'} ---\n")
            # Reset the timer so it saves immediately upon toggling ON
            if is_saving:
                last_save_time = 0.0

    cap.release()
    cv2.destroyAllWindows()
    csv_file.close() ### ADDED: Safely close the file when the script ends

if __name__ == "__main__":
    main()