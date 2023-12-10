#!/bin/bash

#set the path to your simulator folder
path="/opt/carla-simulator"

#set the quality as Low/Epic based
quality="Low"
#set the simulation fps
framesPerSecond=15

# Navigate to the CARLA simulator directory
cd "$path"
# Set Vulkan ICD (Installable Client Driver) for NVIDIA
export VK_ICD_FILENAMES="/usr/share/vulkan/icd.d/nvidia_icd.json"
# Launch CARLA with the specified FPS and quality level
./CarlaUE4.sh -fps=$framesPerSecond -quality-level=$quality
