from dataclasses import dataclass
from typing import Optional

import numpy as np
import rclpy
from control_msgs.msg import JointJog
from geometry_msgs.msg import TwistStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, qos_profile_sensor_data
from sensor_msgs.msg import JointState


class ForwardKeyboardNode(Node):
    @dataclass
    class VelocityScales:
        @dataclass
        class Translation:
            x: float = 0.0
            y: float = 0.0
            z: float = 0.0

        @dataclass
        class Rotation:
            x: float = 0.0
            y: float = 0.0
            z: float = 0.0

        joints: list
        translation: Translation
        rotation: Rotation

    @dataclass
    class KeyboardLayout:
        @dataclass
        class Translation:
            @dataclass
            class X:
                increase: str = "w"
                decrease: str = "s"

            @dataclass
            class Y:
                increase: str = "a"
                decrease: str = "d"

            @dataclass
            class Z:
                increase: str = "q"
                decrease: str = "e"

            x: X = X()
            y: Y = Y()
            z: Z = Z()

        @dataclass
        class Rotation:
            @dataclass
            class X:
                increase: str = "r"
                decrease: str = "f"

            @dataclass
            class Y:
                increase: str = "t"
                decrease: str = "g"

            @dataclass
            class Z:
                increase: str = "y"
                decrease: str = "h"

            x: X = X()
            y: Y = Y()
            z: Z = Z()

        joints: list
        translation: Translation
        rotation: Rotation
        escape: str = "Key.esc"
        pause: str = "p"
        reverse_joints: str = "r"

    def __init__(self):
        super().__init__("forward_keyboard")
        
        # Create QoS profile that matches servo node expectations
        servo_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        self._joint_cmd_pub = self.create_publisher(
            JointJog, "/lbr/servo_node/delta_joint_cmds", servo_qos
        )
        self._twist_cmd_pub = self.create_publisher(
            TwistStamped, "/lbr/servo_node/delta_twist_cmds", servo_qos
        )
        self._joint_state_sub = self.create_subscription(
            JointState,
            "/lbr/joint_states",
            self._on_joint_state,
            qos_profile_sensor_data,
        )
        self._cmd_timer = self.create_timer(0.1, self._on_cmd_timer)

        self._joint_state = None
        self._dof = None
        self._twist_cmd = TwistStamped()
        self._joint_cmd = JointJog()

        while not self._joint_state:
            self.get_logger().info(
                f"Waiting for joint state on {self._joint_state_sub.topic_name}..."
            )
            rclpy.spin_once(self, timeout_sec=1.0)
        self._velocity_scales = self.VelocityScales(
            joints=[0.0] * (self._dof or 7),
            translation=self.VelocityScales.Translation(),
            rotation=self.VelocityScales.Rotation()
        )
        self._keyboard_layout = self.KeyboardLayout(
            joints=[str(i) for i in range(self._dof or 7)],
            translation=self.KeyboardLayout.Translation(),
            rotation=self.KeyboardLayout.Rotation()
        )
        self._declare_parameters()
        self._get_parameters()

    @property
    def joint_state(self) -> Optional[JointState]:
        return self._joint_state

    @property
    def dof(self) -> Optional[int]:
        return self._dof

    @property
    def twist_cmd(self) -> np.ndarray:
        return np.array(
            [
                self._twist_cmd.twist.linear.x,
                self._twist_cmd.twist.linear.y,
                self._twist_cmd.twist.linear.z,
                self._twist_cmd.twist.angular.x,
                self._twist_cmd.twist.angular.y,
                self._twist_cmd.twist.angular.z,
            ]
        )

    @property
    def keyboard_layout(self) -> KeyboardLayout:
        return self._keyboard_layout

    def _declare_parameters(self):
        # velocity scales - use default values since _dof might not be set yet
        self.declare_parameter("velocity_scales.joints", [0.1] * (self._dof or 7))
        self.declare_parameter("velocity_scales.translation.x", 0.1)
        self.declare_parameter("velocity_scales.translation.y", 0.1)
        self.declare_parameter("velocity_scales.translation.z", 0.1)
        self.declare_parameter("velocity_scales.rotation.x", 0.1)
        self.declare_parameter("velocity_scales.rotation.y", 0.1)
        self.declare_parameter("velocity_scales.rotation.z", 0.1)

        # keyboard layout
        self.declare_parameter("keyboard_layout.joints", [str(i) for i in range(self._dof or 7)])
        self.declare_parameter("keyboard_layout.translation.x.increase", "w")
        self.declare_parameter("keyboard_layout.translation.x.decrease", "s")
        self.declare_parameter("keyboard_layout.translation.y.increase", "a")
        self.declare_parameter("keyboard_layout.translation.y.decrease", "d")
        self.declare_parameter("keyboard_layout.translation.z.increase", "Key.up")
        self.declare_parameter("keyboard_layout.translation.z.decrease", "Key.down")
        self.declare_parameter("keyboard_layout.rotation.x.increase", "u")
        self.declare_parameter("keyboard_layout.rotation.x.decrease", "j")
        self.declare_parameter("keyboard_layout.rotation.y.increase", "h")
        self.declare_parameter("keyboard_layout.rotation.y.decrease", "k")
        self.declare_parameter("keyboard_layout.rotation.z.increase", "Key.left")
        self.declare_parameter("keyboard_layout.rotation.z.decrease", "Key.right")
        self.declare_parameter("keyboard_layout.escape", "Key.esc")
        self.declare_parameter("keyboard_layout.pause", "p")
        self.declare_parameter("keyboard_layout.reverse_joints", "r")

    def _get_parameters(self):
        # velocity scales
        joints_param = (
            self.get_parameter("velocity_scales.joints")
            .get_parameter_value()
            .double_array_value
        )
        self._velocity_scales.joints = list(joints_param)
        if len(self._velocity_scales.joints) != self._dof:
            raise ValueError(
                f"Number of joint velocity scales ({len(self._velocity_scales.joints)}) "
                f"does not match the number of joints ({self._dof})."
            )
        self._velocity_scales.translation.x = (
            self.get_parameter("velocity_scales.translation.x")
            .get_parameter_value()
            .double_value
        )
        self._velocity_scales.translation.y = (
            self.get_parameter("velocity_scales.translation.y")
            .get_parameter_value()
            .double_value
        )
        self._velocity_scales.translation.z = (
            self.get_parameter("velocity_scales.translation.z")
            .get_parameter_value()
            .double_value
        )
        self._velocity_scales.rotation.x = (
            self.get_parameter("velocity_scales.rotation.x")
            .get_parameter_value()
            .double_value
        )
        self._velocity_scales.rotation.y = (
            self.get_parameter("velocity_scales.rotation.y")
            .get_parameter_value()
            .double_value
        )
        self._velocity_scales.rotation.z = (
            self.get_parameter("velocity_scales.rotation.z")
            .get_parameter_value()
            .double_value
        )

        # keyboard layout
        keyboard_joints_param = (
            self.get_parameter("keyboard_layout.joints")
            .get_parameter_value()
            .string_array_value
        )
        self._keyboard_layout.joints = list(keyboard_joints_param)
        self._keyboard_layout.translation.x.increase = (
            self.get_parameter("keyboard_layout.translation.x.increase")
            .get_parameter_value()
            .string_value
        )
        self._keyboard_layout.translation.x.decrease = (
            self.get_parameter("keyboard_layout.translation.x.decrease")
            .get_parameter_value()
            .string_value
        )
        self._keyboard_layout.translation.y.increase = (
            self.get_parameter("keyboard_layout.translation.y.increase")
            .get_parameter_value()
            .string_value
        )
        self._keyboard_layout.translation.y.decrease = (
            self.get_parameter("keyboard_layout.translation.y.decrease")
            .get_parameter_value()
            .string_value
        )
        self._keyboard_layout.translation.z.increase = (
            self.get_parameter("keyboard_layout.translation.z.increase")
            .get_parameter_value()
            .string_value
        )
        self._keyboard_layout.translation.z.decrease = (
            self.get_parameter("keyboard_layout.translation.z.decrease")
            .get_parameter_value()
            .string_value
        )
        self._keyboard_layout.rotation.x.increase = (
            self.get_parameter("keyboard_layout.rotation.x.increase")
            .get_parameter_value()
            .string_value
        )
        self._keyboard_layout.rotation.x.decrease = (
            self.get_parameter("keyboard_layout.rotation.x.decrease")
            .get_parameter_value()
            .string_value
        )
        self._keyboard_layout.rotation.y.increase = (
            self.get_parameter("keyboard_layout.rotation.y.increase")
            .get_parameter_value()
            .string_value
        )
        self._keyboard_layout.rotation.y.decrease = (
            self.get_parameter("keyboard_layout.rotation.y.decrease")
            .get_parameter_value()
            .string_value
        )
        self._keyboard_layout.rotation.z.increase = (
            self.get_parameter("keyboard_layout.rotation.z.increase")
            .get_parameter_value()
            .string_value
        )
        self._keyboard_layout.rotation.z.decrease = (
            self.get_parameter("keyboard_layout.rotation.z.decrease")
            .get_parameter_value()
            .string_value
        )
        self._keyboard_layout.escape = (
            self.get_parameter("keyboard_layout.escape")
            .get_parameter_value()
            .string_value
        )
        self._keyboard_layout.pause = (
            self.get_parameter("keyboard_layout.pause")
            .get_parameter_value()
            .string_value
        )
        self._keyboard_layout.reverse_joints = (
            self.get_parameter("keyboard_layout.reverse_joints")
            .get_parameter_value()
            .string_value
        )

        self.get_logger().info("Parameters:")
        # veloctiy scales
        self.get_logger().info("  Velocity Scales:")
        self.get_logger().info("    Translation:")
        self.get_logger().info(f"      x: {self._velocity_scales.translation.x}")
        self.get_logger().info(f"      y: {self._velocity_scales.translation.y}")
        self.get_logger().info(f"      z: {self._velocity_scales.translation.z}")
        self.get_logger().info("    Rotation:")
        self.get_logger().info(f"      x: {self._velocity_scales.rotation.x}")
        self.get_logger().info(f"      y: {self._velocity_scales.rotation.y}")
        self.get_logger().info(f"      z: {self._velocity_scales.rotation.z}")
        self.get_logger().info(f"    Joints: {self._velocity_scales.joints}")

        # keyboard layout
        self.get_logger().info("  Keyboard Layout:")
        self.get_logger().info("    Translation:")
        self.get_logger().info(
            f"      x: {self._keyboard_layout.translation.x.increase} / {self._keyboard_layout.translation.x.decrease}"
        )
        self.get_logger().info(
            f"      y: {self._keyboard_layout.translation.y.increase} / {self._keyboard_layout.translation.y.decrease}"
        )
        self.get_logger().info(
            f"      z: {self._keyboard_layout.translation.z.increase} / {self._keyboard_layout.translation.z.decrease}"
        )
        self.get_logger().info("    Rotation:")
        self.get_logger().info(
            f"      x: {self._keyboard_layout.rotation.x.increase} / {self._keyboard_layout.rotation.x.decrease}"
        )
        self.get_logger().info(
            f"      y: {self._keyboard_layout.rotation.y.increase} / {self._keyboard_layout.rotation.y.decrease}"
        )
        self.get_logger().info(
            f"      z: {self._keyboard_layout.rotation.z.increase} / {self._keyboard_layout.rotation.z.decrease}"
        )
        self.get_logger().info(f"    Joints: {self._keyboard_layout.joints}")
        self.get_logger().info(f"    Escape: {self._keyboard_layout.escape}")
        self.get_logger().info(f"    Pause: {self._keyboard_layout.pause}")
        self.get_logger().info(
            f"    Reverse Joints: {self._keyboard_layout.reverse_joints}"
        )

    def _on_cmd_timer(self) -> None:
        self._twist_cmd.header.stamp = self.get_clock().now().to_msg()
        self._twist_cmd.header.frame_id = "lbr_link_ee"  # End effector frame
        self._twist_cmd_pub.publish(self._twist_cmd)
        self._joint_cmd.header.stamp = self.get_clock().now().to_msg()
        self._joint_cmd_pub.publish(self._joint_cmd)

    @twist_cmd.setter
    def twist_cmd(self, twist: np.ndarray) -> None:
        if len(twist) != 6:
            raise ValueError("Twist command must be a 6-element array.")
        self._twist_cmd.twist.linear.x = twist[0] * self._velocity_scales.translation.x
        self._twist_cmd.twist.linear.y = twist[1] * self._velocity_scales.translation.y
        self._twist_cmd.twist.linear.z = twist[2] * self._velocity_scales.translation.z
        self._twist_cmd.twist.angular.x = twist[3] * self._velocity_scales.rotation.x
        self._twist_cmd.twist.angular.y = twist[4] * self._velocity_scales.rotation.y
        self._twist_cmd.twist.angular.z = twist[5] * self._velocity_scales.rotation.z

    @property
    def joint_veloctiy_cmd(self) -> np.ndarray:
        return np.array(self._joint_cmd.velocities)

    @joint_veloctiy_cmd.setter
    def joint_veloctiy_cmd(self, velocities: np.ndarray) -> None:
        if not self._joint_state:
            return
        if len(velocities) != self._dof:
            raise ValueError(f"Joint command must be a {self._dof}-element array.")

        self._joint_cmd.joint_names = self._joint_state.name
        # account for potentially unsorted joint state broadcaster
        index_map = np.argsort(list(self._joint_cmd.joint_names))
        self._joint_cmd.velocities = velocities[index_map].tolist()
        self._joint_cmd.velocities = [
            v * s
            for v, s in zip(self._joint_cmd.velocities, self._velocity_scales.joints)
        ]

    def _on_joint_state(self, msg: JointState):
        self._joint_state = msg
        self._dof = len(self._joint_state.name)
