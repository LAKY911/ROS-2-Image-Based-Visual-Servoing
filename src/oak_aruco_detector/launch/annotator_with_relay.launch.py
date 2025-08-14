#!/usr/bin/env python3
"""
Launch the image visual relay and the annotator, decoupling visualization.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    input_topic_arg = DeclareLaunchArgument(
        'input_topic', default_value='/oak_undistorted_image',
        description='Camera input image topic (RELIABLE)'
    )
    output_topic_arg = DeclareLaunchArgument(
        'output_topic', default_value='/oak_undistorted_image/visual',
        description='Visual relay output topic (BEST_EFFORT)'
    )
    publish_rate_arg = DeclareLaunchArgument(
        'publish_rate', default_value='0.0',
        description='Optional annotated publish rate (Hz); 0 disables throttling'
    )

    relay_node = Node(
        package='oak_aruco_detector',
        executable='image_visual_relay',
        name='image_visual_relay',
        output='screen',
        parameters=[
            {
                'input_topic': LaunchConfiguration('input_topic'),
                'output_topic': LaunchConfiguration('output_topic'),
                'output_depth': 1,
            }
        ],
    )

    annotator_node = Node(
        package='oak_aruco_detector',
        executable='oak_aruco_annotator',
        name='oak_aruco_annotator',
        output='screen',
        parameters=[
            {
                'input_topic': LaunchConfiguration('output_topic'),
                'publish_rate': LaunchConfiguration('publish_rate'),
            }
        ],
    )

    return LaunchDescription([
        input_topic_arg,
        output_topic_arg,
        publish_rate_arg,
        relay_node,
        annotator_node,
    ])
