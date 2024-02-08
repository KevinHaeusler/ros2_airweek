# ros2_airweek

# Object Detection ROS 2 Package

This ROS 2 package provides functionality for object detection using various methods.

## Installation

### Prerequisites

- ROS 2 Humble
- Python 3.x

### Clone the Repository

```bash
git clone <repository_url>
cd object_detection
```



### Package Dependencies

    rclpy
    sensor_msgs
    cv_bridge
    numpy
    colorsys
    ultralytis

```bash
sudo apt install ros-humble-rclpy ros-humble-sensor-msgs ros-humble-cv-bridge python3-numpy python3-colorsys 
```

```bash
pip3 install openpifpaf ultralytics numpy opencv-python
```

### Run the Node

```bash
ros2 run object_detection image_converter_openpifpaf
```

```bash
ros2 run object_detection image_converter_yolo
```

It publishes a topic called converted_image