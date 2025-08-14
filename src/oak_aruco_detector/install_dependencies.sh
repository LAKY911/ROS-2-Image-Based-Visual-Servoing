#!/bin/bash

# Install Python dependencies for OAK ArUco Detector
echo "Installing Python dependencies for OAK ArUco Detector..."

# Install DepthAI and related packages
pip3 install -r requirements.txt

echo "Dependencies installed successfully!"

# Build the workspace
echo "Building ROS2 workspace..."
cd ../../../
colcon build --packages-select oak_aruco_detector_interfaces oak_aruco_detector

echo "Build complete! Don't forget to source the install/setup.bash file."
