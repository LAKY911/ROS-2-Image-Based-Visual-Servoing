

#!/usr/bin/env python3
"""
Visual Servo Plotter for OAK ArUco Detector (ROS2)
=================================================
Plots real-time IBVS corner errors and camera velocity commands.
Compatible with oak_aruco_detector topics and message types.
Optimized for computational efficiency and responsive plotting.
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from geometry_msgs.msg import TwistStamped
from collections import deque
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import csv
import threading
import os
import math
from datetime import datetime

# Import ArUco and IBVS interfaces if available
try:
    from oak_aruco_detector_interfaces.msg import MarkerArray
except ImportError:
    MarkerArray = None
try:
    from visual_servo_interfaces.msg import IBVSDesiredFeatures
except ImportError:
    IBVSDesiredFeatures = None


class VisualServoPlotter(Node):
    def __init__(self):
        super().__init__('visual_servo_plotter')
        
        # Parameters
        self.declare_parameter('error_history_len', 10000)  # Increased for longer data retention
        self.declare_parameter('velocity_history_len', 10000)  # Increased for longer data retention
        self.declare_parameter('csv_log', True)
        self.declare_parameter('csv_log_dir', '~/visual_servo_logs')
        self.declare_parameter('update_rate_hz', 20.0)

        self.error_history_len = self.get_parameter('error_history_len').get_parameter_value().integer_value
        self.velocity_history_len = self.get_parameter('velocity_history_len').get_parameter_value().integer_value
        self.csv_log = self.get_parameter('csv_log').get_parameter_value().bool_value
        self.csv_log_dir = self.get_parameter('csv_log_dir').get_parameter_value().string_value
        self.update_rate = self.get_parameter('update_rate_hz').get_parameter_value().double_value

        # Thread safety
        self.data_lock = threading.Lock()
        self.data_dirty = False
        self.paused = False

        # Data buffers for errors (4 corners + norm of norms) - no maxlen to prevent data loss
        self.error_times = deque()
        self.corner_errors = [deque() for _ in range(4)]
        self.norm_of_norms = deque()

        # Data buffers for velocities - no maxlen to prevent data loss
        self.velocity_times = deque()
        self.velocity_data = {
            'linear_x': deque(),
            'linear_y': deque(),
            'linear_z': deque(),
            'angular_x': deque(),
            'angular_y': deque(),
            'angular_z': deque(),
        }

        # For desired features (if available)
        self.desired_features_msg = None

        # Start time for relative timestamps
        self.start_time = self.get_clock().now()

        # Setup data storage
        self.setup_data_storage()

        # Subscribers
        qos = QoSProfile(
            depth=10, 
            reliability=ReliabilityPolicy.BEST_EFFORT, 
            durability=DurabilityPolicy.VOLATILE, 
            history=HistoryPolicy.KEEP_LAST
        )
        
        if MarkerArray is not None:
            self.create_subscription(MarkerArray, '/aruco_detections', self.aruco_callback, qos)
            self.get_logger().info("Subscribed to /aruco_detections")
        else:
            self.get_logger().warn("MarkerArray not available - aruco detection plotting disabled")
            
        if IBVSDesiredFeatures is not None:
            self.create_subscription(IBVSDesiredFeatures, '/ibvs/desired_features', self.ibvs_features_callback, qos)
            self.get_logger().info("Subscribed to /ibvs/desired_features")
        else:
            self.get_logger().warn("IBVSDesiredFeatures not available - desired features plotting disabled")
            
        self.create_subscription(TwistStamped, '/lbr/servo_node/delta_twist_cmds', self.velocity_callback, qos)
        self.get_logger().info("Subscribed to /lbr/servo_node/delta_twist_cmds")

        # Setup plotting
        self.setup_plots()

    def aruco_callback(self, msg):
        """Callback for ArUco marker detections."""
        if self.paused or not msg.markers:
            return

        try:
            # Get the first marker
            marker = msg.markers[0]
            current_time = (self.get_clock().now() - self.start_time).nanoseconds * 1e-9

            # Extract current corner positions from corner_x and corner_y arrays
            if len(marker.corner_x) < 4 or len(marker.corner_y) < 4:
                self.get_logger().warn(f'Marker has {len(marker.corner_x)} corner_x and {len(marker.corner_y)} corner_y coordinates, expected 4 each')
                return

            current_corners = [
                (marker.corner_x[i], marker.corner_y[i]) for i in range(4)
            ]

            # Get desired corner positions
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

            # Calculate individual corner errors and their norms
            corner_norms = []
            with self.data_lock:
                self.error_times.append(current_time)

                for i, (current_corner, desired_corner) in enumerate(zip(current_corners, desired_corners)):
                    error_u = current_corner[0] - desired_corner[0]
                    error_v = current_corner[1] - desired_corner[1]
                    corner_norm = math.sqrt(error_u**2 + error_v**2)
                    corner_norms.append(corner_norm)
                    self.corner_errors[i].append(corner_norm)

                # Calculate norm of corner norms (overall error metric)
                norm_of_norms = math.sqrt(sum(norm**2 for norm in corner_norms))
                self.norm_of_norms.append(norm_of_norms)
                self.data_dirty = True

            # Log to CSV
            if hasattr(self, 'csv_writers') and 'corner_error' in self.csv_writers:
                row = [current_time, norm_of_norms] + corner_norms
                self.csv_writers['corner_error'].writerow(row)

        except Exception as e:
            self.get_logger().warn(f'Error in aruco_callback: {e}')

    def ibvs_features_callback(self, msg):
        """Callback for IBVS desired features."""
        self.desired_features_msg = msg
        self.get_logger().info(f'Received desired features: {len(msg.corners)} corners')
        
        # Log the desired corner positions for debugging
        for i, corner in enumerate(msg.corners):
            self.get_logger().info(f'  Corner {i}: x={corner.x:.2f}, y={corner.y:.2f}')

    def velocity_callback(self, msg: TwistStamped):
        """Callback for camera velocity commands."""
        if self.paused:
            return

        try:
            current_time = (self.get_clock().now() - self.start_time).nanoseconds * 1e-9
            
            with self.data_lock:
                self.velocity_times.append(current_time)
                self.velocity_data['linear_x'].append(msg.twist.linear.x)
                self.velocity_data['linear_y'].append(msg.twist.linear.y)
                self.velocity_data['linear_z'].append(msg.twist.linear.z)
                self.velocity_data['angular_x'].append(msg.twist.angular.x)
                self.velocity_data['angular_y'].append(msg.twist.angular.y)
                self.velocity_data['angular_z'].append(msg.twist.angular.z)
                self.data_dirty = True

            # Log to CSV
            if hasattr(self, 'csv_writers') and 'velocity' in self.csv_writers:
                row = [
                    current_time,
                    msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z,
                    msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z
                ]
                self.csv_writers['velocity'].writerow(row)

        except Exception as e:
            self.get_logger().warn(f'Error in velocity_callback: {e}')

    def setup_plots(self):
        """Setup matplotlib plots for 2-plot vertical layout."""
        plt.ion()  # Enable interactive mode

        # Create figure with 2 subplots arranged vertically
        self.fig, self.axes = plt.subplots(2, 1, figsize=(12, 10))
        # self.fig.suptitle('IBVS Visual Servoing Performance', fontsize=16)

        # Setup corner error plot (top)
        self.error_ax = self.axes[0]
        self.error_ax.set_title('Corner Tracking Errors', fontsize=16, fontweight='bold')
        self.error_ax.set_xlabel('Time (s)', fontsize=14)
        self.error_ax.set_ylabel('Error (pixels)', fontsize=14)
        self.error_ax.tick_params(axis='both', which='major', labelsize=12)
        self.error_ax.grid(True, alpha=0.3)
        self.error_ax.set_xlim(0, 30)  # Initial wider range, will auto-scale
        self.error_ax.set_ylim(0, 100)

        # Setup velocity plot (bottom)
        self.velocity_ax = self.axes[1]
        self.velocity_ax.set_title('Camera Velocity Commands', fontsize=16, fontweight='bold')
        self.velocity_ax.set_xlabel('Time (s)', fontsize=14)
        self.velocity_ax.set_ylabel('Velocity', fontsize=14)
        self.velocity_ax.tick_params(axis='both', which='major', labelsize=12)
        self.velocity_ax.grid(True, alpha=0.3)
        self.velocity_ax.set_xlim(0, 30)  # Initial wider range, will auto-scale
        self.velocity_ax.set_ylim(-1.1, 1.1)  # Fixed Y range for velocities

        # Create persistent line objects
        colors = ['red', 'green', 'blue', 'orange']
        self.error_lines = [
            self.error_ax.plot([], [], color=colors[i], label=f'Corner {i}', linewidth=1.5)[0] for i in range(4)
        ]
        self.error_norm_line = self.error_ax.plot([], [], label='Norm of Norms', color='black', linewidth=2)[0]
        self.error_ax.legend(loc='upper right', fontsize=12)

        self.vel_lines = {
            'linear_x': self.velocity_ax.plot([], [], 'r-', label='Vx (m/s)', linewidth=1.5)[0],
            'linear_y': self.velocity_ax.plot([], [], 'g-', label='Vy (m/s)', linewidth=1.5)[0],
            'linear_z': self.velocity_ax.plot([], [], 'b-', label='Vz (m/s)', linewidth=1.5)[0],
            'angular_x': self.velocity_ax.plot([], [], 'r--', label='ωx (rad/s)', linewidth=1.5)[0],
            'angular_y': self.velocity_ax.plot([], [], 'g--', label='ωy (rad/s)', linewidth=1.5)[0],
            'angular_z': self.velocity_ax.plot([], [], 'b--', label='ωz (rad/s)', linewidth=1.5)[0],
        }
        self.velocity_ax.legend(loc='upper right', fontsize=12)

        # Add key press handler for controls
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

        # Use a GUI timer to refresh plots
        self.gui_timer = self.fig.canvas.new_timer(interval=int(1000 / self.update_rate))
        self.gui_timer.add_callback(self.redraw_plots)
        self.gui_timer.start()

        plt.tight_layout(pad=0.5, h_pad=1.0, w_pad=0.5)
        plt.show()

    def redraw_plots(self):
        """Redraw plots on the GUI thread using the latest buffered data."""
        with self.data_lock:
            if not self.data_dirty:
                return

            # Copy data for thread safety
            time_error = list(self.error_times)
            corner_errors_data = [list(self.corner_errors[i]) for i in range(4)]
            norm_data = list(self.norm_of_norms)

            time_vel = list(self.velocity_times)
            vel_data = {k: list(v) for k, v in self.velocity_data.items()}

            self.data_dirty = False

        # Update error plot lines
        if len(time_error) > 0:
            for i in range(4):
                ce = corner_errors_data[i]
                if len(ce) > 0:
                    n = min(len(time_error), len(ce))
                    self.error_lines[i].set_data(time_error[-n:], ce[-n:])
                else:
                    self.error_lines[i].set_data([], [])

            if len(norm_data) > 0:
                n = min(len(time_error), len(norm_data))
                self.error_norm_line.set_data(time_error[-n:], norm_data[-n:])
            else:
                self.error_norm_line.set_data([], [])

            # Keep time axis starting from 0 with some padding at the end
            time_max = max(time_error)
            padding = max(2.0, time_max * 0.05)  # At least 2 seconds padding or 5% of max time
            self.error_ax.set_xlim(0, time_max + padding)
            
            # Continuously auto-scale y-axis for errors to accommodate larger data
            all_error_values = []
            for ce in corner_errors_data:
                all_error_values.extend(ce)
            all_error_values.extend(norm_data)
            
            if all_error_values:
                y_min = 0.0  # Fixed minimum at 0 for errors
                y_max = max(all_error_values)
                y_padding = max(5.0, y_max * 0.1)  # At least 5 pixels padding or 10% of max value
                y_negative_padding = max(2.0, y_max * 0.05)  # Small negative padding for better view
                self.error_ax.set_ylim(y_min - y_negative_padding, y_max + y_padding)
        else:
            self.error_ax.set_xlim(0, 30)
            self.error_ax.set_ylim(0, 100)
            for line in self.error_lines:
                line.set_data([], [])
            self.error_norm_line.set_data([], [])

        # Update velocity plot lines
        if len(time_vel) > 0:
            for key, line in self.vel_lines.items():
                series = vel_data.get(key, [])
                if len(series) > 0:
                    n = min(len(time_vel), len(series))
                    line.set_data(time_vel[-n:], series[-n:])
                else:
                    line.set_data([], [])
            
            # Keep time axis starting from 0 with some padding at the end
            time_max = max(time_vel)
            padding = max(2.0, time_max * 0.05)  # At least 2 seconds padding or 5% of max time
            self.velocity_ax.set_xlim(0, time_max + padding)
            
            # Fixed Y-axis for velocities from -1.1 to 1.1
            self.velocity_ax.set_ylim(-1.1, 1.1)
        else:
            self.velocity_ax.set_xlim(0, 30)
            self.velocity_ax.set_ylim(-1.1, 1.1)  # Fixed Y range for velocities
            for line in self.vel_lines.values():
                line.set_data([], [])

        try:
            self.fig.canvas.draw_idle()
        except Exception:
            pass

    def setup_data_storage(self):
        """Setup data storage and CSV writing."""
        if not self.csv_log:
            return

        # Create data directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.data_directory = f"visual_servo_data_{timestamp}"
        os.makedirs(self.data_directory, exist_ok=True)

        # Setup CSV files for data logging
        self.csv_files = {}
        self.csv_writers = {}

        # Corner error CSV
        error_csv_path = os.path.join(self.data_directory, 'corner_errors.csv')
        self.csv_files['corner_error'] = open(error_csv_path, 'w', newline='')
        self.csv_writers['corner_error'] = csv.writer(self.csv_files['corner_error'])
        self.csv_writers['corner_error'].writerow([
            'time', 'norm_of_norms', 'corner_0_error', 'corner_1_error',
            'corner_2_error', 'corner_3_error'
        ])

        # Velocity CSV
        velocity_csv_path = os.path.join(self.data_directory, 'velocities.csv')
        self.csv_files['velocity'] = open(velocity_csv_path, 'w', newline='')
        self.csv_writers['velocity'] = csv.writer(self.csv_files['velocity'])
        self.csv_writers['velocity'].writerow([
            'time', 'linear_x', 'linear_y', 'linear_z',
            'angular_x', 'angular_y', 'angular_z'
        ])

        self.get_logger().info(f'Data will be saved to: {self.data_directory}')

    def on_key_press(self, event):
        """Handle keyboard shortcuts."""
        if event.key == 's':
            # Save screenshot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.data_directory, f'screenshot_{timestamp}.png')
            self.fig.savefig(filename, dpi=300, bbox_inches='tight')
            self.get_logger().info(f'Screenshot saved: {filename}')

        elif event.key == 'r':
            # Reset/clear data
            self.reset_data()
            self.get_logger().info('Data reset')

        elif event.key == 'c':
            # Clear plots (keep data shown cleared)
            self.clear_plots()
            self.get_logger().info('Plots cleared')

        elif event.key == ' ':
            # Toggle pause
            self.paused = not self.paused
            status = "paused" if self.paused else "resumed"
            self.get_logger().info(f'Data collection {status}')

        elif event.key == 'h':
            # Show help
            self.show_help()

        elif event.key == 'q':
            # Quit
            self.quit_application()

    def reset_data(self):
        """Reset all data arrays."""
        with self.data_lock:
            self.velocity_times.clear()
            self.error_times.clear()

            for key in self.velocity_data:
                self.velocity_data[key].clear()

            for i in range(4):
                self.corner_errors[i].clear()
            self.norm_of_norms.clear()

            self.data_dirty = True

        self.start_time = self.get_clock().now()

    def clear_plots(self):
        """Clear plot displays but keep data."""
        with self.data_lock:
            for line in self.error_lines:
                line.set_data([], [])
            self.error_norm_line.set_data([], [])
            for line in self.vel_lines.values():
                line.set_data([], [])
            self.data_dirty = True

        try:
            self.fig.canvas.draw_idle()
        except Exception:
            pass

    def show_help(self):
        """Show help information."""
        help_text = """
        Keyboard Controls:
        s - Save screenshot
        r - Reset data
        c - Clear plots
        space - Pause/resume data collection
        h - Show this help
        q - Quit application
        """
        self.get_logger().info(help_text)

    def quit_application(self):
        """Clean shutdown of the application."""
        self.get_logger().info('Shutting down visual servo plotter...')

        # Stop GUI timer
        try:
            if hasattr(self, 'gui_timer'):
                self.gui_timer.stop()
        except Exception:
            pass

        # Close CSV files safely
        try:
            if hasattr(self, 'csv_files'):
                for file_handle in self.csv_files.values():
                    if not file_handle.closed:
                        file_handle.close()
        except Exception as e:
            self.get_logger().warn(f"Error closing files: {e}")

        # Close matplotlib
        try:
            plt.close('all')
        except Exception as e:
            self.get_logger().warn(f"Error closing matplotlib: {e}")

    def destroy_node(self):
        """Clean up resources when node is destroyed."""
        try:
            if hasattr(self, 'gui_timer'):
                self.gui_timer.stop()
        except Exception:
            pass

        try:
            if hasattr(self, 'csv_files'):
                for file_handle in self.csv_files.values():
                    if not file_handle.closed:
                        file_handle.close()
        except Exception:
            pass

        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    plotter = None
    try:
        plotter = VisualServoPlotter()

        # Start ROS2 spinning in a separate thread
        ros_thread = threading.Thread(target=lambda: rclpy.spin(plotter), daemon=True)
        ros_thread.start()

        # Keep matplotlib window open
        plt.show(block=True)

    except KeyboardInterrupt:
        print("Shutting down plotter...")
    except Exception as e:
        print(f"Error in main: {e}")
    finally:
        if plotter is not None:
            try:
                plotter.get_logger().info("Saving final data and closing files...")
                if hasattr(plotter, 'csv_files'):
                    for file_handle in plotter.csv_files.values():
                        if not file_handle.closed:
                            file_handle.close()
                if hasattr(plotter, 'data_directory'):
                    plotter.get_logger().info(f"All data saved to: {plotter.data_directory}")
            except Exception as e:
                print(f"Error during cleanup: {e}")

            plotter.destroy_node()

        # Safe shutdown
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception as e:
            print(f"Error during ROS2 shutdown: {e}")


if __name__ == '__main__':
    main()
