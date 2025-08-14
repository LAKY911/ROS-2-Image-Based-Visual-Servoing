#!/usr/bin/env python3
"""
Launch file for OAK ArUco Annotator
===================================

Launches the OAK ArUco annotator node with configurable parameters.

Usage:
    # With image saving and live display
    ros2 launch oak_aruco_detector oak_aruco_annotator.launch.py save_images:=true show_live_image:=true
    
    # Live display only
    ros2 launch oak_aruco_detector oak_aruco_annotator.launch.py save_images:=false show_live_image:=true
    
    # Headless mode (only ROS topics)
    ros2 launch oak_aruco_detector oak_aruco_annotator.launch.py save_images:=false show_live_image:=false
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Generate the launch description for OAK ArUco annotator."""
    
    # Declare launch arguments
    save_images_arg = DeclareLaunchArgument(
        'save_images',
        default_value='false',
        description='Save annotated images to folder'
    )
    
    show_live_image_arg = DeclareLaunchArgument(
        'show_live_image', 
        default_value='true',
        description='Show live image in OpenCV window'
    )
    
    save_directory_arg = DeclareLaunchArgument(
        'save_directory',
        default_value='~/aruco_sequence',
        description='Directory to save images'
    )
    
    save_interval_arg = DeclareLaunchArgument(
        'save_interval',
        default_value='0.1', 
        description='Interval between saved images (seconds)'
    )
    
    image_format_arg = DeclareLaunchArgument(
        'image_format',
        default_value='jpg',
        description='Image format: jpg, png, or webp'
    )
    
    target_marker_id_arg = DeclareLaunchArgument(
        'target_marker_id',
        default_value='4',
        description='Target marker ID (-1 for all markers)'
    )
    
    marker_size_arg = DeclareLaunchArgument(
        'marker_size',
        default_value='0.05',
        description='Size of ArUco markers in meters'
    )
    
    aruco_dictionary_id_arg = DeclareLaunchArgument(
        'aruco_dictionary_id',
        default_value='DICT_4X4_50',
        description='ArUco dictionary ID'
    )
    
    # Create the node
    oak_aruco_annotator_node = Node(
        package='oak_aruco_detector',
        executable='oak_aruco_annotator',
        name='oak_aruco_annotator',
        output='screen',
        parameters=[
            {
                'save_images': LaunchConfiguration('save_images'),
                'show_live_image': LaunchConfiguration('show_live_image'),
                'save_directory': LaunchConfiguration('save_directory'),
                'save_interval': LaunchConfiguration('save_interval'),
                'image_format': LaunchConfiguration('image_format'),
                'target_marker_id': LaunchConfiguration('target_marker_id'),
                'marker_size': LaunchConfiguration('marker_size'),
                'aruco_dictionary_id': LaunchConfiguration('aruco_dictionary_id'),
                # Decoupled visualization defaults: subscribe to relayed topic and no throttling
                'input_topic': '/oak_undistorted_image/visual',
                'publish_rate': 0.0,
            }
        ],
        remappings=[
            # Add any topic remappings here if needed
        ]
    )
    
    return LaunchDescription([
        save_images_arg,
        show_live_image_arg,
        save_directory_arg,
        save_interval_arg,
        image_format_arg,
        target_marker_id_arg,
        marker_size_arg,
        aruco_dictionary_id_arg,
        oak_aruco_annotator_node,
    ])
