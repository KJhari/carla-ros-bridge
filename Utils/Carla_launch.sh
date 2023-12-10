#!/bin/bash

#set the path to your simulator folder
path=/opt/carla-simulator/

#set the quality as Low/Epic based
quality=Low
#set the simulation fps
framesPerSecond=15


cd $(path)
export VK_ICD_FILENAMES="/usr/share/vulkan/icd.d/nvidia_icd.json"  && ./CarlaUE4.sh -fps=$(framesPerSecond) -quality-level=$(quality)