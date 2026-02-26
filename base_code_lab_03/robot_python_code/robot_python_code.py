# External libraries

from typing import List, Tuple

import serial
import time
import pickle
import cv2
import cv2.aruco as aruco
import numpy as np
import matplotlib.pyplot as plt
import socket
from time import strftime
import os

# Local libraries
import parameters
from aruco_pose_estimator import ArucoPoseEstimator

from filters import ThetaFilter


# Function to try to connect to the robot via udp over wifi
def create_udp_communication(arduinoIP, localIP, arduinoPort, localPort, bufferSize):
    try:
        udp = UDPCommunication(arduinoIP, localIP, arduinoPort, localPort, bufferSize)
        print("Success in creating udp communication")
        return udp, True
    except:
        print("Failed to create udp communication!")
        return _, False
        
        
# Class to hold the UPD over wifi connection setup
class UDPCommunication:
    def __init__(self, arduinoIP, localIP, arduinoPort, localPort, bufferSize):
        self.arduinoIP = arduinoIP
        self.arduinoPort = arduinoPort
        self.localIP = localIP
        self.localPort = localPort
        self.bufferSize = bufferSize
        self.UDPServerSocket = socket.socket(family = socket.AF_INET, type = socket.SOCK_DGRAM)
        self.UDPServerSocket.bind((localIP, localPort))
        
    # Receive a message from the robot
    def receive_msg(self):
        bytesAddressPair = self.UDPServerSocket.recvfrom(self.bufferSize)
        message = bytesAddressPair[0]
        address = bytesAddressPair[1]
        clientMsg = "{}".format(message.decode())
        clientIP = "{}".format(address)
        
        return clientMsg
       
    # Send a message to the robot
    def send_msg(self, msg):
        bytesToSend = str.encode(msg)
        self.UDPServerSocket.sendto(bytesToSend, (self.arduinoIP, self.arduinoPort))


# Class to hold the data logger that records data when needed
class DataLogger:

    # Constructor
    def __init__(self, filename_start, data_name_list):
        # Get the absolute path to the data directory relative to this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(script_dir, 'data')
        self.data_dir = data_dir
        self.filename_start = os.path.join(self.data_dir, os.path.basename(filename_start))
        self.filename = self.filename_start
        self.line_count = 0
        self.dictionary = {}
        self.data_name_list = data_name_list
        for name in data_name_list:
            self.dictionary[name] = []
        self.currently_logging = False
        

    # Open the log file
    def reset_logfile(self, control_signal):
        # Ensure the data directory exists
        os.makedirs(self.data_dir, exist_ok=True)
        self.filename = self.filename_start + "_"+str(control_signal[0])+"_"+str(control_signal[1]) + strftime("_%d_%m_%y_%H_%M_%S.pkl")
        self.dictionary = {}
        for name in self.data_name_list:
            self.dictionary[name] = []

        
    # Log one time step of data
    def log(self, logging_switch_on, time, control_signal, robot_sensor_signal, camera_sensor_signal, state_mean, state_covariance):
        if not logging_switch_on:
            if self.currently_logging:
                self.currently_logging = False
        else:
            if not self.currently_logging:
                self.currently_logging = True
                self.reset_logfile(control_signal)

        if self.currently_logging:
            self.dictionary['time'].append(time)
            self.dictionary['control_signal'].append(control_signal)
            self.dictionary['robot_sensor_signal'].append(robot_sensor_signal)
            self.dictionary['camera_sensor_signal'].append(camera_sensor_signal)
            self.dictionary['state_mean'].append(state_mean)
            self.dictionary['state_covariance'].append(state_covariance)

            self.line_count += 1
            if self.line_count > parameters.max_num_lines_before_write:
                self.line_count = 0
                with open(self.filename, 'wb') as file_handle:
                    pickle.dump(self.dictionary, file_handle)

# Utility for loading saved data
class DataLoader:

    # Constructor
    def __init__(self, filename):
        self.filename = filename
        
    # Load a dictionary from file.
    def load(self):
        with open(self.filename, 'rb') as file_handle:
            loaded_dict = pickle.load(file_handle)
        return loaded_dict

# Class to hold a message sender
class MsgSender:

    # Time step size between message to robot sends, in seconds
    delta_send_time = 0.1

    # Constructor
    def __init__(self, last_send_time, msg_size, udp_communication):
        self.last_send_time = last_send_time
        self.msg_size = msg_size
        self.udp_communication = udp_communication
        
    # Pack and send a control signal to the robot.
    def send_control_signal(self, control_signal):
        packed_send_msg = self.pack_msg(control_signal)
        self.send(packed_send_msg)
    
    # If its time, send the control signal to the robot.
    def send(self, msg):
        new_send_time = time.perf_counter()
        if new_send_time - self.last_send_time > self.delta_send_time:
            message = ""
            for data in msg:
                message = message + str(data)
            self.udp_communication.send_msg(message)
            self.last_send_time = new_send_time
      
    # Pack a message so it is in the correct format for the robot to receive it.
    def pack_msg(self, msg):
        packed_msg = ""
        for data in msg:
            if packed_msg == "":
                packed_msg = packed_msg + str(data)
            else:
                packed_msg = packed_msg + ", "+ str(data)
        packed_msg = packed_msg + "\n"
        return packed_msg
        
# The robot's message receiver
class MsgReceiver:

    # Determines how often to look for incoming data from the robot.
    delta_receive_time = 0.05

    # Constructor
    def __init__(self, last_receive_time, msg_size, udp_communication):
        self.last_receive_time = last_receive_time
        self.msg_size = msg_size
        self.udp_communication = udp_communication
      
    # Check if its time to look for a new message from the robot.
    def receive(self):
        new_receive_time = time.perf_counter()
        if new_receive_time - self.last_receive_time > self.delta_receive_time:
            received_msg = self.udp_communication.receive_msg()
            self.last_receive_time = new_receive_time
            return True, received_msg
            
        return False, ""
    
    # Given a new message, put it in a digestable format
    def unpack_msg(self, packed_msg):
        unpacked_msg = []
        msg_list = packed_msg.split(',')
        if len(msg_list) >= self.msg_size:
            for data in msg_list:
                unpacked_msg.append(float(data))
            return True, unpacked_msg

        return False, unpacked_msg
        
    # Check for new message and unpack it if there is one.
    def receive_robot_sensor_signal(self, last_robot_sensor_signal):
        robot_sensor_signal = last_robot_sensor_signal
        receive_ret, packed_receive_msg = self.receive()
        if receive_ret:
            unpack_ret, unpacked_receive_msg = self.unpack_msg(packed_receive_msg)
            if unpack_ret:
                robot_sensor_signal = RobotSensorSignal(unpacked_receive_msg)
            
        return robot_sensor_signal

# Class to hold a camera sensor data. Not needed for lab 1.
class CameraSensor:

    # Constructor
    def __init__(self, camera_id, video_capture=None):
        self.camera_id = camera_id
        if video_capture:
            self.cap = video_capture
        else:
            self.cap = cv2.VideoCapture(camera_id)
        self.pose_estimator = ArucoPoseEstimator(
            camera_matrix=parameters.camera_matrix,
            dist_coeffs=parameters.dist_coeffs,
            marker_length=parameters.marker_length,
            known_markers=parameters.KNOWN_MARKERS,
            robot_marker_id=7
        )
        self.theta_filter = ThetaFilter()
        
    # Get a new pose estimate from a camera image
    def get_signal(self, last_camera_signal: List[float]) -> List[float]:
        camera_signal = last_camera_signal
        ret, pose_estimate = self.get_pose_estimate()
        if ret:
            camera_signal = pose_estimate
            camera_signal[5] = self.theta_filter.filter_pose_theta(camera_signal[5])
        
        return camera_signal
        
    # If there is a new image, calculate a pose estimate from the fiducial tag on the robot.
    def get_pose_estimate(self) -> Tuple[bool, List[float]]:
        """Returns a tuple of (bool, List[float]). The bool indicates if a valid pose estimate was obtained. The list contains the pose estimate in the format [x, y, z, roll, pitch, yaw]."""
        # Try to read a frame a few times
        for _ in range(5):
            ret, frame = self.cap.read()
            if ret:
                break
            time.sleep(0.01) # Wait a bit for the camera to be ready

        if not ret:
            print("Failed to grab frame from camera")
            return False, []
        
        # --- NEW LOGIC ---
        # Pass the raw frame to your new estimator class
        pose_dict, annotated_frame = self.pose_estimator.estimate_pose(frame)

        # If the strict tracking criteria are met, it returns a valid pose
        if pose_dict is not None:
            # Extract the values into the 6-element list format your system expects
            pose_estimate: List[float] = [
                pose_dict['x'], 
                pose_dict['y'], 
                pose_dict['z'], 
                pose_dict['roll'], 
                pose_dict['pitch'], 
                pose_dict['yaw']
            ]
            
            # Optional: Show the live annotated camera feed for debugging
            # cv2.imshow('Aruco Tracking View', annotated_frame)
            # cv2.waitKey(1)
            
            return True, pose_estimate
            
        # If the robot marker or the fixed reference marker isn't visible
        return False, []
    
    # Close the camera stream
    def close(self):
        self.cap.release()
        cv2.destroyAllWindows()


# A storage vessel for an instance of a robot signal
class RobotSensorSignal:

    # Constructor
    def __init__(self, unpacked_msg):
        self.encoder_counts: int = int(unpacked_msg[0])
        self.steering: int = int(unpacked_msg[1])
        self.num_lidar_rays = int(unpacked_msg[2])
        self.angles = []
        self.distances = []
        for i in range(self.num_lidar_rays):
            index = 3 + i*2
            self.angles.append(unpacked_msg[index])
            self.distances.append(unpacked_msg[index+1])
    
    # Print the robot sensor signal contents.
    def print(self):
        print("Robot Sensor Signal")
        print(" encoder: ", self.encoder_counts)
        print(" steering:" , self.steering)
        print(" num_lidar_rays: ", self.num_lidar_rays)
        print(" angles: ",self.angles)
        print(" distances: ", self.distances)
    
    # Convert the sensor signal to a list of ints and floats.
    def to_list(self):
        sensor_data_list = []
        sensor_data_list.append(self.encoder_counts)
        sensor_data_list.append(self.steering)
        sensor_data_list.append(self.num_lidar_rays)
        for i in range(self.num_lidar_rays):
            sensor_data_list.append(self.angles[i])
            sensor_data_list.append(self.distances[i])
        
        return sensor_data_list


# Source - https://stackoverflow.com/a/76802895
# Posted by M lab, modified by community. See post 'Timeline' for change history
# Retrieved 2026-02-22, License - CC BY-SA 4.0

# def my_estimatePoseSingleMarkers(corners, marker_size, mtx, distortion):
#     '''
#     This will estimate the rvec and tvec for each of the marker corners detected by:
#        corners, ids, rejectedImgPoints = detector.detectMarkers(image)
#     corners - is an array of detected corners for each detected marker in the image
#     marker_size - is the size of the detected markers
#     mtx - is the camera matrix
#     distortion - is the camera distortion matrix
#     RETURN list of rvecs, tvecs, and trash (so that it corresponds to the old estimatePoseSingleMarkers())
#     '''
#     marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
#                               [marker_size / 2, marker_size / 2, 0],
#                               [marker_size / 2, -marker_size / 2, 0],
#                               [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)
#     trash = []
#     rvecs = []
#     tvecs = []
#     for c in corners:
#         nada, R, t = cv2.solvePnP(marker_points, c, mtx, distortion, False, cv2.SOLVEPNP_IPPE_SQUARE)
#         rvecs.append(R)
#         tvecs.append(t)
#         trash.append(nada)
#     return rvecs, tvecs, trash
