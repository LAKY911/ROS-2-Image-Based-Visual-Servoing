#!/usr/bin/env python3
"""
Image Visual Relay

Subscribes RELIABLE to a camera image topic and republishes it as BEST_EFFORT
with a small queue. This decouples visualization/annotators from the camera
pipeline to avoid backpressure.

Parameters:
  - input_topic (string, default '/oak_undistorted_image')
  - output_topic (string, default '/oak_undistorted_image/visual')
  - output_depth (int, default 1)

Author: GitHub Copilot
"""

from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image


class ImageVisualRelay(Node):
    def __init__(self) -> None:
        super().__init__('image_visual_relay')

        # Parameters
        self.declare_parameter('input_topic', '/oak_undistorted_image')
        self.declare_parameter('output_topic', '/oak_undistorted_image/visual')
        self.declare_parameter('output_depth', 1)

        input_topic: str = self.get_parameter('input_topic').get_parameter_value().string_value
        output_topic: str = self.get_parameter('output_topic').get_parameter_value().string_value
        output_depth: int = self.get_parameter('output_depth').get_parameter_value().integer_value
        output_depth = max(1, int(output_depth))

        # QoS: RELIABLE in (match camera), BEST_EFFORT out (decoupled)
        in_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
        )
        out_qos = QoSProfile(
            depth=output_depth,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
        )

        # Publisher and subscriber
        self.pub = self.create_publisher(Image, output_topic, out_qos)
        self.sub = self.create_subscription(Image, input_topic, self._cb, in_qos)

        self._count_in = 0
        self._count_out = 0

        self.get_logger().info(
            f"Relaying images: '{input_topic}' (RELIABLE) -> '{output_topic}' (BEST_EFFORT, depth={output_depth})"
        )

        # Optional periodic stats
        self.create_timer(5.0, self._stats)

    def _cb(self, msg: Image) -> None:
        try:
            self._count_in += 1
            # Publish the same message; no modification needed
            self.pub.publish(msg)
            self._count_out += 1
        except Exception as e:
            self.get_logger().warn(f'Failed to relay image: {e}')

    def _stats(self) -> None:
        self.get_logger().debug(f"relay stats: in={self._count_in} out={self._count_out}")


def main(args: Optional[list] = None) -> None:
    rclpy.init(args=args)
    node = ImageVisualRelay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
