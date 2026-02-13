// Wifi setup
#include <SPI.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#define SendDeltaTimeInMs 100      // Number ms between messages sent to laptop
#define ReceiveDeltaTimeInMs 10    // Number ms between checking for control signals sent from laptop
#define NoSignalDeltaTimeInMs 2000 // Number ms between message receives from laptop before stopping robot
char ssid[] = "Tenda_9C90E0";//"Tenda_9C95E0";//"Tenda_9C90E0";      // REPLACE with your team's router ssid
char pass[] = "78972629";  //"82168926";//"78972629";        // REPLACE with your team's router password"78972629"
char remoteIP[] = "192.168.0.199"; // "192.168.0.199"; // REPLACE with your laptop's IP address on your team's router
unsigned int localPort = 4010;     // local port to listen on - no need to change
unsigned int remotePort = 4010;    // local port to listen on - no need to change
int status = WL_IDLE_STATUS;
int last_time_rx = 0;
int last_time_tx = 0;
WiFiUDP Udp;
char packetBuffer[256];            //buffer to hold incoming packet

// Lidar setup
#include "RPLidar.h"
#define RPLidarMotorPin 3          // Lidar motor control pin
#define NumLidarRaysPerMsg 50      // How many lidar ray measurements to send at a time
RPLidar lidar;
String current_lidar_scan_data;
int current_num_lidar_rays;

// Motor Control setup
#define RightSpeedPin 9            // Right PWM pin connect MODEL-X ENA
#define RightMotorDirPin1 12       // Right Motor direction pin 1 to MODEL-X IN1 
#define RightMotorDirPin2 11       // Right Motor direction pin 2 to MODEL-X IN2
#define LeftSpeedPin 6             // Left PWM pin connect MODEL-X ENB
#define LeftMotorDirPin1 7         // Left Motor direction pin 1 to MODEL-X IN3 
#define LeftMotorDirPin2 8         // Left Motor direction pin 1 to MODEL-X IN4 

// Servo control setup
#include <Servo.h>
#define ServoPin 10                // Servo control pin
Servo myServo;

// Encoder setup
#define EncoderOutputA 4          // Encoder output pin A
#define EncoderOutputB 5          // Encoder output pin B
#define steering_angle_center 75  // REPLACE with team center angle for servor steering
int a_state;
int encoder_a_last_state; 
int encoder_count;

// Structure for storing control signals received from laptop
struct ControlSignal {
  int speed = 0;
  int steering_angle = 0;
};
ControlSignal last_control_signal;


// Structure for storing sensor signals sent to laptop
struct SensorSignal {
  int encoder_count = 0;
  int steering_angle = 0;
  int num_lidar_rays = 0;
  String lidar_scan_data = "";
};
SensorSignal last_sensor_signal;


// Main setup to run on Arduino power up or reset
void setup() 
{
  // Set up terminal serial messages
  Serial.begin(115200);
  Serial.println("Running robot base code!");

  // Need WiFi module
  if (WiFi.status() == WL_NO_MODULE) {
    Serial.println("Communication with WiFi module failed!");
    while (true);
  }

  // Attempt to connect to WiFi network
  while (status != WL_CONNECTED) {
    Serial.print("Attempting to connect to SSID: ");
    Serial.println(ssid);
    status = WiFi.begin(ssid, pass);

    // wait 10 seconds for connection:
    delay(10000);
  }
  Serial.println("Connected to WiFi");
  printWifiStatus();
  Serial.println("\nStarted connection to server...");
  Udp.begin(localPort);

  // Bind the RPLIDAR driver to the arduino hardware serial
  //Serial2.begin(460800);
  //lidar.begin(Serial2);
  //delay(1000);
  //if (lidar.begin(Serial2)) {
  //  Serial.println("Started Lidar!");
  //} else {
  //  Serial.println("Failed Lidar!");
  //}
  //pinMode(RPLidarMotorPin, OUTPUT);
  //reset_lidar_message();

  // Set up speed control
	pinMode(RightMotorDirPin1, OUTPUT); 
	pinMode(RightMotorDirPin2, OUTPUT); 
	pinMode(LeftSpeedPin, OUTPUT);  
	pinMode(LeftMotorDirPin1, OUTPUT);
  pinMode(LeftMotorDirPin2, OUTPUT); 
  pinMode(RightSpeedPin, OUTPUT); 
  stop();

  // Set up steering control
  myServo.attach(ServoPin);
  myServo.write(steering_angle_center);

  // Set up encoder
  pinMode (EncoderOutputA, INPUT);
  pinMode (EncoderOutputB, INPUT);

  // Set up timers for message sends and receives over bluetooth
  last_time_rx = millis();
  last_time_tx = millis();
}
 
// Main loop to be run sequentially 
void loop() 
{
  // Receive control signal messages
  ControlSignal control_signal = receive_control_signals(last_control_signal);
  last_control_signal = control_signal;
  
  // Send control signals to robot hardware actuators
  control_robot(control_signal);

  // Send sensor data in messages to laptop
  SensorSignal sensor_signal = get_sensor_signal(control_signal.steering_angle);
  send_sensor_signal(sensor_signal);
}

// After sending lidar data to the laptop, reset the count and message.
void reset_lidar_message()
{
  current_num_lidar_rays = 0;
  current_lidar_scan_data = "";
}

// Stop all robot motors
void stop()
{
  digitalWrite(RightMotorDirPin1, LOW);
  digitalWrite(RightMotorDirPin2,LOW);
  digitalWrite(LeftMotorDirPin1,LOW);
  digitalWrite(LeftMotorDirPin2,LOW);
}

// Drive robot forward a desired speed
void forward(int speed)
{
  digitalWrite(RightMotorDirPin1, HIGH);
  digitalWrite(RightMotorDirPin2,LOW);
  digitalWrite(LeftMotorDirPin1,HIGH);
  digitalWrite(LeftMotorDirPin2,LOW);
  analogWrite(LeftSpeedPin, speed * 0.75);
  analogWrite(RightSpeedPin, speed);
}

// Receive control signal messages from laptop, but only have delta time has passed, e.g. 10ms
ControlSignal receive_control_signals(ControlSignal last_control_signal) {
  ControlSignal control_signal = last_control_signal;

  int new_time_rx = millis();
  if (new_time_rx - last_time_rx > ReceiveDeltaTimeInMs) {
    int packetSize = Udp.parsePacket();
    if (packetSize) {
      // read the packet into packetBufffer
      int len = Udp.read(packetBuffer, 255);
      if (len > 0) {
        packetBuffer[len] = 0;
      }
      control_signal = unpack_control_signal(packetBuffer);
      Serial.print("Received cmd: ");
      Serial.print(control_signal.speed);
      Serial.print(", ");
      Serial.println(control_signal.steering_angle);
      last_time_rx = new_time_rx;
    }
  }

  // Stop robot movement if no control message recieved recently
  if (new_time_rx - last_time_rx > NoSignalDeltaTimeInMs) {
    control_signal.speed = 0;
    control_signal.steering_angle = 0;
  }
  return control_signal;
}

// Update the encoder and lidar data
SensorSignal get_sensor_signal(float steering_angle) {

  // Update the encoder count
  encoder_update();
  last_sensor_signal.steering_angle = steering_angle;
  last_sensor_signal.encoder_count = encoder_count;

  // Update the lidar scan
  //lidar_update();

  return last_sensor_signal;
}

// Get new lidar measurements
void lidar_update() {
  if (IS_OK(lidar.waitPoint())) {
    float distance = lidar.getCurrentPoint().distance;
    if (distance > 100 && current_num_lidar_rays < NumLidarRaysPerMsg) {
      int angle = int(lidar.getCurrentPoint().angle);
      current_num_lidar_rays += 1;
      current_lidar_scan_data +=  "," + String(angle) + "," + String(int(distance));
    }
  } else {
    analogWrite(RPLidarMotorPin, 255); //stop the rplidar motor
    
    // try to detect RPLIDAR... 
    rplidar_response_device_info_t info;
    if (IS_OK(lidar.getDeviceInfo(info, 100))) {
       // Detected...
       lidar.startScan();
       
       // Start motor rotating at max allowed speed
       analogWrite(RPLidarMotorPin, 255);
       delay(1000);
    }
  }
}

// Get new encoder measurements
void encoder_update() { 
   a_state = digitalRead(EncoderOutputA); // Reads the "current" state of the outputA
   // If the previous and the current state of the outputA are different, that means a Pulse has occured
   if (a_state != encoder_a_last_state){     
     // If the outputB state is different to the outputA state, that means the encoder is rotating clockwise
     if (digitalRead(EncoderOutputB) != a_state) { 
       encoder_count ++;
     } else {
       encoder_count --;
     }
   } 
   encoder_a_last_state = a_state; // Updates the previous state of the outputA with the current state
 }

// Send a message with sensor signals to the laptop, but only after a preset time, e.g. 100 ms
void send_sensor_signal(SensorSignal sensor_signal)
{
  int new_time_tx = millis();
  if (new_time_tx - last_time_tx > SendDeltaTimeInMs) {
    String msg = String(sensor_signal.encoder_count) + ",";
    msg = msg + String(sensor_signal.steering_angle) + ",";
    msg = msg + String(current_num_lidar_rays);
    msg = msg + current_lidar_scan_data;
    reset_lidar_message();
    //Serial.print("Sending msg: ");
    //Serial.println(msg);

    Udp.beginPacket(remoteIP, remotePort);
    int   array_length  = msg.length()+1;
    char  msg_as_char_array[array_length];
    msg.toCharArray(msg_as_char_array, array_length);
    Udp.write(msg_as_char_array);
    Udp.endPacket();

    // Restart next msg
    last_sensor_signal.lidar_scan_data = "";
    last_sensor_signal.num_lidar_rays = 0;
    last_time_tx = new_time_tx;
  }
}

// Control the robot with desired control signal from laptop
void control_robot(ControlSignal control_signal){
  // Set robot speed
  forward(2 * control_signal.speed);

  // Set robot steering
  int desired_angle = steering_angle_center + control_signal.steering_angle;
  myServo.write(desired_angle);
}

// Unpack a control signal message sent from the laptop
ControlSignal unpack_control_signal(char* packed_control_signal_as_char) {
  ControlSignal control_signal;
  char* token;

  // Unpack the desired speeed
  token = strtok(packed_control_signal_as_char, ",");
  control_signal.speed = atof(token);

  // Unpack the desired steering angle
  token = strtok(NULL, ",");
  control_signal.steering_angle = atof(token);

  return control_signal;
}

// Print the wifi status to the serial terminal
void printWifiStatus() {
  // print the SSID of the network you're attached to:
  Serial.print("SSID: ");
  Serial.println(WiFi.SSID());

  // print your board's IP address:
  IPAddress ip = WiFi.localIP();
  Serial.print("IP Address: ");
  Serial.println(ip);

  // print the received signal strength:
  long rssi = WiFi.RSSI();
  Serial.print("signal strength (RSSI):");
  Serial.print(rssi);
  Serial.println(" dBm");
}

