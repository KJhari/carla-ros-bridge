# Carla-ROS-Darknet Integration for Object Detection

This Project was developed for CMPE 249: Intelligent Autonomous Systems 

## Project Overview

This project integrates the Carla simulator, ROS (Robot Operating System), and Darknet for real-time object detection in a simulated environment. It leverages the YOLO (You Only Look Once) algorithm for efficient detection of objects.

## Installation
### ROS Installation

Follow the instructions at ROS Noetic Installation to install ROS Noetic on Ubuntu.

## Carla Simulator and ROS Bridge

Install the Carla simulator and set up the Carla-ROS bridge using the guide provided here: Carla-ROS Bridge.

## Darknet ROS

Clone and build the darknet_ros package for ROS integration with YOLO object detection. Refer to the Darknet ROS GitHub for detailed instructions.

## Configuration

Place your custom YOLO model weights (yolov3_traffic.weights) in the weights/ directory and the network configuration file (yolov3_traffic.cfg) in the cfg/ directory of the darknet_ros package. Update the ros.yaml and yolo_traffic.yaml files in the config/ directory accordingly.

## Running the Project

    Launch Carla Simulator: Start the Carla simulator.
    Run Carla-ROS Bridge: Launch the bridge to communicate between Carla and ROS.
    Execute Darknet ROS: Use the launch file to run darknet_ros.
    Bounding Box Overlay: Run the bounding_box_node.py script to overlay bounding boxes on detected objects.

## Custom Node Explanation

bounding_box_node.py subscribes to image and bounding box topics, overlays the detected bounding boxes on images, and publishes the result. This allows visualization of detections in RViz.

## Troubleshooting

For common issues, refer to each component's respective documentation and troubleshooting guides.

## Contributions and License

Contributions are welcome. Please follow the contribution guidelines specified in the repository. The project is released under [specify the license].
Acknowledgments

Thanks to the teams behind Carla, ROS, and Darknet for their open-source software, and to Rumeysa Keskin for the YOLO training on traffic signs.
