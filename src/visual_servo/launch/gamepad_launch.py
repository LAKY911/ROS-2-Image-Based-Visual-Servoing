from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='joy',
            executable='joy_node',
            name='joy_node',
            parameters=[
                {'deadzone': 0.3}
            ],
        ),
        Node(
            package='visual_servo',
            executable='joy_servo_bridge',
            name='bridge'
        )
    ])
