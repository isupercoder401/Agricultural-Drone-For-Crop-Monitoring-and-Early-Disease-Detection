 I developed an Autonomous Agricultural Drone for smart crop monitoring and early disease detection. The drone integrates Pixhawk flight controller, Raspberry Pi 4, GPS, rangefinder, and high-res cameras to autonomously navigate fields, capture imagery, and analyze data in real time.

Using EfficientNetB7 and YOLOv5, the system detects plant diseases (38 classes), differentiates crops from weeds, and estimates yields. Models were trained on curated datasets and optimized with TensorFlow Lite for deployment on the Raspberry Pi, enabling low-latency inference in the field.

The system uses ROS and MAVLink for communication, with mission planning through QGroundControl and simulations using MAVProxy. Safety features like Return-to-Launch, auto-landing, and sensor validation ensure robust real-world operation.

This project demonstrates the potential of combining AI, robotics, and edge computing to reduce labor, minimize chemical use, and support sustainable farming. Code, models, and configs are publicly available on GitHub.
