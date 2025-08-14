from typing import List, Optional, Union

from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


class GazeboMixin:
    @staticmethod
    def include_gazebo(**kwargs) -> IncludeLaunchDescription:
        return IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution(
                    [
                        FindPackageShare("ros_gz_sim"),
                        "launch",
                        "gz_sim.launch.py",
                    ]
                ),
            ),
            launch_arguments={"gz_args": "-r empty.sdf"}.items(),
            **kwargs,
        )

    @staticmethod
    def node_create(
        robot_name: Optional[Union[LaunchConfiguration, str]] = LaunchConfiguration(
            "robot_name", default="lbr"
        ),
        tf: List[float] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        **kwargs,
    ) -> Node:
        label = ["-x", "-y", "-z", "-R", "-P", "-Y"]
        tf = [str(x) for x in tf]
        return Node(
            package="ros_gz_sim",
            executable="create",
            arguments=[
                "-topic",
                "robot_description",
                "-name",
                robot_name,
                "-allow_renaming",
            ]
            + [item for pair in zip(label, tf) for item in pair],
            output="screen",
            namespace=robot_name,
            **kwargs,
        )

    @staticmethod
    def node_clock_bridge(**kwargs) -> Node:
        return Node(
            package="ros_gz_bridge",
            executable="parameter_bridge",
            arguments=["/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock"],
            output="screen",
            **kwargs,
        )

    @staticmethod
    def node_camera_bridge(**kwargs) -> Node:
        return Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            arguments=[
                '/oak_undistorted_image@sensor_msgs/msg/Image@ignition.msgs.Image',
                '/oak_camera_info@sensor_msgs/msg/CameraInfo@ignition.msgs.CameraInfo',
                ],
            output='screen',
            **kwargs,
        )

    @staticmethod
    def node_spawn_entity(
        sdf_filename: str = "/home/lukas/Diploma-Thesis-2025/lbr-stack/src/aruco_gazebo/aruco_box/model.sdf",
        entity_name: str = "aruco_box_1",
        x: float = 0.5,
        y: float = 0.0,
        z: float = 0.25,
        **kwargs,
    ) -> Node:
        return Node(
            package="ros_gz_sim",
            executable="create",
            arguments=[
                "-world", "empty",
                "-file", sdf_filename,
                "-name", entity_name,
                "-x", str(x),
                "-y", str(y),
                "-z", str(z),
            ],
            output="screen",
            **kwargs,
        )