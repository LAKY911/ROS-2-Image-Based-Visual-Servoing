#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from oak_aruco_detector_interfaces.msg import Marker, MarkerArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import signal
import sys


class GazeboArucoDetector(Node):
    def __init__(self):
        super().__init__('gazebo_aruco_detector')

        # Parameters
        self.declare_parameter('image_topic', '/oak_undistorted_image')
        self.declare_parameter('queue_size', 10)
        self.declare_parameter('aruco_dictionary', 'DICT_4X4_50')

        self.image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        self.queue_size = self.get_parameter('queue_size').get_parameter_value().integer_value
        dict_name = self.get_parameter('aruco_dictionary').get_parameter_value().string_value

        # Publishers
        self.publisher_ = self.create_publisher(MarkerArray, '/aruco_detections', 10)

        # CV bridge
        self.bridge = CvBridge()

        # ArUco setup
        try:
            dict_id = getattr(cv2.aruco, dict_name)
        except AttributeError:
            self.get_logger().warn(f'Unknown ArUco dictionary {dict_name}, falling back to DICT_4X_4_50')
            dict_id = cv2.aruco.DICT_4X4_50
        self.dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
        self.parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.parameters)
        self.get_logger().info('Using OpenCV ArUco ArucoDetector API')

        # Subscriber
        self.subscription = self.create_subscription(
            Image, self.image_topic, self.image_callback, self.queue_size
        )
        self.get_logger().info(f'Subscribed to image topic: {self.image_topic}')

        # Cached header
        self._header_template = Header()

    def image_callback(self, msg: Image):
        try:
            # Normalize to BGR for consistent processing
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect markers
            corners, ids, _ = self.detector.detectMarkers(gray)

            # Prepare header from incoming image
            self._header_template.stamp = msg.header.stamp
            self._header_template.frame_id = msg.header.frame_id or 'camera_frame'

            if ids is not None and len(ids) > 0:
                markers = MarkerArray()
                markers.header = self._header_template

                for i in range(len(ids)):
                    corner = corners[i][0]  # (4,2) TL, TR, BR, BL
                    cx = float(np.mean(corner[:, 0]))
                    cy = float(np.mean(corner[:, 1]))
                    marker_id = int(ids[i][0])

                    corner_x = [float(corner[j, 0]) for j in range(4)]
                    corner_y = [float(corner[j, 1]) for j in range(4)]

                    m = Marker()
                    m.header = self._header_template
                    m.id = marker_id
                    m.x = cx
                    m.y = cy
                    m.corner_x = corner_x
                    m.corner_y = corner_y
                    markers.markers.append(m)

                self.publisher_.publish(markers)
                self.get_logger().debug(f'Published {len(markers.markers)} ArUco markers')
        except Exception as e:
            self.get_logger().warn(f'Image processing error: {e}')


def signal_handler(signum, frame):
    print('\nReceived interrupt signal. Shutting down gracefully...')
    rclpy.shutdown()
    sys.exit(0)


def main(args=None):
    signal.signal(signal.SIGINT, signal_handler)
    rclpy.init(args=args)
    node = None
    try:
        node = GazeboArucoDetector()
        print('Press Ctrl+C to shutdown gracefully')
        rclpy.spin(node)
    except KeyboardInterrupt:
        print('\nKeyboard interrupt received')
    except Exception as e:
        print(f'Failed to start node: {e}')
    finally:
        if node is not None:
            print('Cleaning up node...')
            node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass
        print('Shutdown complete')


if __name__ == '__main__':
    main()
