#!/usr/bin/env python3
"""
Launch file for Visual Servo Plotter
====================================

Launches the visual servo plotter node for monitoring IBVS performance.
This tool provides:
- Real-time corner tracking error visualization
- Camera velocity command monitoring
- Data logging to CSV files
- Interactive keyboard controls

Usage:
    ros2 launch oak_aruco_detector visual_servo_plotter.launch.py
"""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    """Generate the launch description for visual servo plotter."""
    
    visual_servo_plotter_node = Node(
        package='oak_aruco_detector',
        executable='visual_servo_plotter',
        name='visual_servo_plotter',
        output='screen',
        parameters=[
            # Add any parameters here if needed
        ],
        remappings=[
            # Add any topic remappings here if needed
        ]
    )
    
    return LaunchDescription([
        visual_servo_plotter_node,
    ])
