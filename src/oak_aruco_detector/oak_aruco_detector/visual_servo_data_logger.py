#!/usr/bin/env python3
"""
Visual Servo Data Logger for OAK ArUco Detector (ROS2)
=====================================================
Logs data from visual servoing topics to CSV files for offline analysis.
This is a lightweight data collection tool without plotting overhead.

Subscribed Topics:
- /aruco_detections (oak_aruco_detector_interfaces/MarkerArray): ArUco marker detections with corner coordinates
- /ibvs/desired_features (visual_servo_interfaces/IBVSDesiredFeatures): Desired IBVS features
- /lbr/servo_node/delta_twist_cmds (geometry_msgs/TwistStamped): Camera velocity commands from servo node

Output Files:
- aruco_detections.csv: Raw ArUco detection data with corner coordinates
- ibvs_desired_features.csv: Desired IBVS features when available  
- camera_velocities.csv: Camera velocity commands
- corner_errors.csv: Computed corner tracking errors
- summary.csv: Summary statistics and metadata
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from geometry_msgs.msg import TwistStamped
import csv
import os
import math
from datetime import datetime

# Import ArUco and IBVS interfaces if available
try:
    from oak_aruco_detector_interfaces.msg import MarkerArray
    ARUCO_AVAILABLE = True
except ImportError:
    ARUCO_AVAILABLE = False
    MarkerArray = None

try:
    from visual_servo_interfaces.msg import IBVSDesiredFeatures
    IBVS_AVAILABLE = True
except ImportError:
    IBVS_AVAILABLE = False
    IBVSDesiredFeatures = None


class VisualServoDataLogger(Node):
    def __init__(self):
        super().__init__('visual_servo_data_logger')
        
        # Parameters (simplified like plotter)
        self.declare_parameter('sync_csv_writes', True)  # Flush after each write for real-time monitoring
        
        self.sync_writes = self.get_parameter('sync_csv_writes').get_parameter_value().bool_value
        
        # Start time for relative timestamps
        self.start_time = self.get_clock().now()
        
        # Storage for desired features (for error computation)
        self.desired_features_msg = None
        
        # Message counters for statistics
        self.msg_counts = {
            'aruco_detections': 0,
            'ibvs_desired_features': 0,
            'camera_velocities': 0
        }
        
        # Setup CSV files
        self.setup_csv_files()
        
        # Setup subscribers
        self.setup_subscribers()
        
        # Log available topics
        self.log_topic_status()
        
    def log_topic_status(self):
        """Log which topics are being monitored."""
        topics_logged = []
        
        if ARUCO_AVAILABLE:
            topics_logged.append("/aruco_detections (oak_aruco_detector_interfaces/MarkerArray)")
        else:
            self.get_logger().warn("ArUco interfaces not available - /aruco_detections logging disabled")
            
        if IBVS_AVAILABLE:
            topics_logged.append("/ibvs/desired_features (visual_servo_interfaces/IBVSDesiredFeatures)")
        else:
            self.get_logger().warn("IBVS interfaces not available - /ibvs/desired_features logging disabled")
            
        topics_logged.append("/lbr/servo_node/delta_twist_cmds (geometry_msgs/TwistStamped)")
        
        self.get_logger().info("VISUAL SERVO DATA LOGGER - TOPICS BEING SAVED:")
        for topic in topics_logged:
            self.get_logger().info(f"  ✓ {topic}")
        self.get_logger().info("=" * 60)
        
    def setup_csv_files(self):
        """Setup CSV files for data logging."""
        self.csv_files = {}
        self.csv_writers = {}
        
        # Create data directory with timestamp (same style as plotter)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.data_directory = f"visual_servo_data_{timestamp}"
        os.makedirs(self.data_directory, exist_ok=True)

        # Corner errors CSV (same format as plotter)
        error_csv_path = os.path.join(self.data_directory, 'corner_errors.csv')
        self.csv_files['corner_error'] = open(error_csv_path, 'w', newline='')
        self.csv_writers['corner_error'] = csv.writer(self.csv_files['corner_error'])
        self.csv_writers['corner_error'].writerow([
            'time', 'norm_of_norms', 'corner_0_error', 'corner_1_error',
            'corner_2_error', 'corner_3_error'
        ])

        # Velocity CSV (same format as plotter)
        velocity_csv_path = os.path.join(self.data_directory, 'velocities.csv')
        self.csv_files['velocity'] = open(velocity_csv_path, 'w', newline='')
        self.csv_writers['velocity'] = csv.writer(self.csv_files['velocity'])
        self.csv_writers['velocity'].writerow([
            'time', 'linear_x', 'linear_y', 'linear_z',
            'angular_x', 'angular_y', 'angular_z'
        ])
        
        self.get_logger().info(f'Data will be saved to: {self.data_directory}')
        
    def setup_subscribers(self):
        """Setup ROS2 subscribers."""
        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST
        )
        
        # ArUco detections subscriber
        if ARUCO_AVAILABLE and MarkerArray is not None:
            self.create_subscription(
                MarkerArray, '/aruco_detections', self.aruco_callback, qos
            )
            self.get_logger().info("Subscribed to /aruco_detections")
            
        # IBVS desired features subscriber
        if IBVS_AVAILABLE and IBVSDesiredFeatures is not None:
            self.create_subscription(
                IBVSDesiredFeatures, '/ibvs/desired_features', self.ibvs_features_callback, qos
            )
            self.get_logger().info("Subscribed to /ibvs/desired_features")
            
        # Camera velocity subscriber
        self.create_subscription(
            TwistStamped, '/lbr/servo_node/delta_twist_cmds', self.velocity_callback, qos
        )
        self.get_logger().info("Subscribed to /lbr/servo_node/delta_twist_cmds")
        
    def aruco_callback(self, msg):
        """Callback for ArUco marker detections."""
        if not msg.markers:
            return

        try:
            # Get the first marker (same as plotter)
            marker = msg.markers[0]
            current_time = (self.get_clock().now() - self.start_time).nanoseconds * 1e-9

            # Extract current corner positions from corner_x and corner_y arrays
            if len(marker.corner_x) < 4 or len(marker.corner_y) < 4:
                self.get_logger().warn(f'Marker has {len(marker.corner_x)} corner_x and {len(marker.corner_y)} corner_y coordinates, expected 4 each')
                return

            current_corners = [
                (marker.corner_x[i], marker.corner_y[i]) for i in range(4)
            ]

            # Get desired corner positions (same logic as plotter)
            if self.desired_features_msg is not None and len(self.desired_features_msg.corners) >= 4:
                desired_corners = [
                    (self.desired_features_msg.corners[i].x, self.desired_features_msg.corners[i].y)
                    for i in range(4)
                ]
            else:
                # Use default desired corners (centered in image)
                image_center_u, image_center_v = 320, 240  # Assuming 640x480 image
                marker_half_size = 50  # pixels
                desired_corners = [
                    (image_center_u - marker_half_size, image_center_v - marker_half_size),  # Top-left
                    (image_center_u + marker_half_size, image_center_v - marker_half_size),  # Top-right
                    (image_center_u + marker_half_size, image_center_v + marker_half_size),  # Bottom-right
                    (image_center_u - marker_half_size, image_center_v + marker_half_size),  # Bottom-left
                ]

            # Calculate individual corner errors and their norms (same as plotter)
            corner_norms = []
            for i, (current_corner, desired_corner) in enumerate(zip(current_corners, desired_corners)):
                error_u = current_corner[0] - desired_corner[0]
                error_v = current_corner[1] - desired_corner[1]
                corner_norm = math.sqrt(error_u**2 + error_v**2)
                corner_norms.append(corner_norm)

            # Calculate norm of corner norms (overall error metric)
            norm_of_norms = math.sqrt(sum(norm**2 for norm in corner_norms))

            # Log to CSV (same format as plotter)
            if 'corner_error' in self.csv_writers:
                row = [current_time, norm_of_norms] + corner_norms
                self.csv_writers['corner_error'].writerow(row)
                if self.sync_writes:
                    self.csv_files['corner_error'].flush()

            self.msg_counts['aruco_detections'] += 1
            
            if self.msg_counts['aruco_detections'] % 100 == 0:
                self.get_logger().info(f"Logged {self.msg_counts['aruco_detections']} ArUco detection messages")

        except Exception as e:
            self.get_logger().warn(f'Error in aruco_callback: {e}')
            
    def ibvs_features_callback(self, msg):
        """Callback for IBVS desired features."""
        self.desired_features_msg = msg
        self.get_logger().info(f'Received desired features: {len(msg.corners)} corners')
        
        # Log the desired corner positions for debugging
        for i, corner in enumerate(msg.corners):
            self.get_logger().info(f'  Corner {i}: x={corner.x:.2f}, y={corner.y:.2f}')
            
        self.msg_counts['ibvs_desired_features'] += 1
            
    def velocity_callback(self, msg: TwistStamped):
        """Callback for camera velocity commands."""
        try:
            current_time = (self.get_clock().now() - self.start_time).nanoseconds * 1e-9
            
            # Log to CSV (same format as plotter)
            if 'velocity' in self.csv_writers:
                row = [
                    current_time,
                    msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z,
                    msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z
                ]
                self.csv_writers['velocity'].writerow(row)
                if self.sync_writes:
                    self.csv_files['velocity'].flush()

            self.msg_counts['camera_velocities'] += 1
            
            if self.msg_counts['camera_velocities'] % 100 == 0:
                self.get_logger().info(f"Logged {self.msg_counts['camera_velocities']} camera velocity messages")

        except Exception as e:
            self.get_logger().warn(f'Error in velocity_callback: {e}')
            
    def create_summary_file(self):
        """Create a summary file with session metadata and statistics."""
        summary_path = os.path.join(self.data_directory, 'summary.csv')
        
        with open(summary_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Parameter', 'Value'])
            writer.writerow(['Session Start Time', datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
            writer.writerow(['Output Directory', self.data_directory])
            writer.writerow(['ArUco Available', ARUCO_AVAILABLE])
            writer.writerow(['IBVS Available', IBVS_AVAILABLE])
            writer.writerow(['Sync CSV Writes', self.sync_writes])
            
            writer.writerow([])  # Empty row
            writer.writerow(['Message Type', 'Count'])
            for msg_type, count in self.msg_counts.items():
                writer.writerow([msg_type, count])
                
        self.get_logger().info(f"Summary file created: {summary_path}")
        
    def destroy_node(self):
        """Clean up resources when node is destroyed."""
        self.get_logger().info("Shutting down visual servo data logger...")
        
        # Create summary file
        self.create_summary_file()
        
        # Close all CSV files safely
        try:
            for file_handle in self.csv_files.values():
                if not file_handle.closed:
                    file_handle.close()
        except Exception as e:
            self.get_logger().warn(f"Error closing files: {e}")
                
        # Log final statistics
        self.get_logger().info("=" * 60)
        self.get_logger().info("DATA LOGGING SESSION COMPLETED")
        self.get_logger().info(f"All data saved to: {self.data_directory}")
        self.get_logger().info("Message counts:")
        for msg_type, count in self.msg_counts.items():
            self.get_logger().info(f"  {msg_type}: {count}")
        self.get_logger().info("=" * 60)
        
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    logger = None
    try:
        logger = VisualServoDataLogger()
        rclpy.spin(logger)
    except KeyboardInterrupt:
        print("\nShutting down data logger...")
    except Exception as e:
        print(f"Error in data logger: {e}")
    finally:
        if logger is not None:
            logger.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
