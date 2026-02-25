import cv2

# Initialize the detector with a specific dictionary (5x5 is standard)
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, parameters)

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break

    # Detect markers
    corners, ids, rejected = detector.detectMarkers(frame)

    # Draw markers if any are found
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

    cv2.imshow('ArUco Detection', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()