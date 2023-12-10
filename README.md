# Carla-ROS-Darknet Integration for Object Detection

This Project was developed for CMPE 249: Intelligent Autonomous Systems 

## Project Overview

This project integrates the Carla simulator, ROS (Robot Operating System), and Darknet for real-time object detection in a simulated environment. It leverages the YOLO (You Only Look Once) algorithm for efficient detection of objects.

## Installation
### ROS Installation

Follow the instructions at ROS Noetic Installation to install [ROS Noetic](http://wiki.ros.org/noetic/Installation/Ubuntu) on Ubuntu.

## Carla Simulator and ROS Bridge

Install the Carla simulator and set up the Carla-ROS bridge using the guide provided here: [Carla-ROS Bridge.](https://carla.readthedocs.io/projects/ros-bridge/en/latest/)

## Darknet ROS

Clone and build the darknet_ros package for ROS integration with YOLO object detection. Refer to the [Darknet ROS GitHub](https://github.com/leggedrobotics/darknet_ros) for detailed instructions.

### Training
Refer to the \Darknet_training folder for training guidelines
[Rumeysa Keskin](https://github.com/Rumeysakeskin/YOLO-Darknet-Video-and-Image-Detection-Traffic-Signs)


## Configuration

Place your custom YOLO model weights (yolov3_traffic.weights) in the weights/ directory and the network configuration file (yolov3_traffic.cfg) in the cfg/ directory of the darknet_ros package. Update the ros.yaml and yolo_traffic.yaml files in the config/ directory accordingly.

## Running the Project

#### Launch Carla Simulator: 
Start the Carla simulator using Utils/Carla_launch.sh , set the path and simulation options based on your needs<br>
```
Bash Carla_launch.sh
```

#### Chnage Map
edit Utils/changeMap.py and run using<br> 
```
python3 changeMap.py
```

#### Run Carla-ROS Bridge:<br>
 source the files before launch
```
source carla-ros-bridge/catkin_ws/devel/setup.bash 
```
 Launch the bridge to communicate between Carla and ROS
```
roslaunch carla_ros_bridge carla_ros_bridge_with_example_ego_vehicle.launch<br>
```
#### Execute Darknet ROS: 
Use the launch file to run darknet_ros.<br>
```
source carla-ros-bridge/catkin_ws/devel/setup.bash 
```
```
roslaunch darknet_ros darknet_ros.launch
```
#### Note: ***Launch the custom ROS node only once darknet_ros starts publishing the bounding box co-ordinates***, you can doublecheck this by running 
```
rostopic echo \darknet_ros\bouding_boxes
```

#### Launch the custom ROS node
In another terminal open up the virtual environment for Keras
```
source carla-ros-bridge/Keras/keras_env/bin/activate
```
##### source in virtual environment also
```
source /opt/ros/noetic/setup.bash
source carla-ros-bridge/catkin_ws/devel/setup.bash 
```
#####  to detect the subclasses using Keras and publish the topic in ROS to view in Rviz
```
python3 detect_subclass
```

### Launch Rviz
click `Add` to create a visualization<br>
choose by topic and select `\traffic_sign_recognition\image_with_boxes`


## Custom Node Explanation

detect_subclass.py subscribes to image from the ego_vehicle rgb camera sensor and bounding box topics from darknet_ros, overlays the detected bounding boxes on images, extracts the image within the bouding box and feeds it to the Keras Model, the output from the model is then published as a topic to ROS under the name `\traffic_sign_recognition\image_with_boxes`. This allows visualization of detections in RViz.

## screenshots and results


![rostopic_lsit](https://github.com/KJhari/carla-ros-bridge/assets/44090664/20594837-8d99-42c8-a529-f406425e895f)

![Carla_map04_detected](https://github.com/KJhari/carla-ros-bridge/assets/44090664/f69b9b8c-2b1b-4fe4-a6f3-269a413afa82)

![3](https://github.com/KJhari/carla-ros-bridge/assets/44090664/8aaa3ec7-0406-425e-b214-8ffa8aca1eea)

[Town7+detection+3.webm](https://github.com/KJhari/carla-ros-bridge/assets/44090664/2d445b7a-cce0-4d10-b3e3-2e658a616b53)

## Troubleshooting

For common issues, refer to each component's respective documentation and troubleshooting guides.

## Contributions and License

Contributions are welcome. Please follow the contribution guidelines specified in the repository.
Acknowledgments

Thanks to the teams behind Carla, ROS, and Darknet for their open-source software, and to [Rumeysa Keskin](https://github.com/Rumeysakeskin/YOLO-Darknet-Video-and-Image-Detection-Traffic-Signs) for the YOLO training on traffic signs.

