#!/usr/bin/env python3
"""
Simplified Visual Servoing Performance Plotter for IBVS
======================================================

A simplified real-time plotting tool for monitoring IBVS (Image-Based Visual Servoing) performance.
Features:
- 2 vertical plots: Corner tracking errors (top) and camera velocity commands (bottom)
- Real-time data visualization and CSV logging
- Keyboard controls for interaction
- Integrates with oak_aruco_publisher for corner-based error calculation

Author: Assistant
Date: 2025
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy

# Message types
from geometry_msgs.msg import TwistStamped
from collections import deque
import math
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for better GUI support
import matplotlib.pyplot as plt
import numpy as np
import csv
import os
from datetime import datetime
import threading

# Try to import ArUco interfaces
ARUCO_INTERFACES_AVAILABLE = False
try:
    from oak_aruco_detector_interfaces.msg import MarkerArray
    ARUCO_INTERFACES_AVAILABLE = True
except ImportError:
    print("Warning: oak_aruco_detector_interfaces not available")

# Try to import IBVS interfaces  
IBVS_FEATURES_AVAILABLE = False
try:
    from visual_servo_interfaces.msg import IBVSDesiredFeatures
    IBVS_FEATURES_AVAILABLE = True
except ImportError:
    print("Warning: ibvs_interfaces not available - using default desired positions")


class VisualServoPlotter(Node):
    def __init__(self):
        super().__init__('visual_servo_plotter')
        
        # Data storage
        self.max_points = 1000  # Maximum number of data points to store
        
        # Control flags
        self.paused = False
        
        # Time arrays
        self.velocity_time = deque(maxlen=self.max_points)
        self.error_time = deque(maxlen=self.max_points)
        
        # Velocity data (IBVS only)
        self.velocity_data = {
            'linear_x': deque(maxlen=self.max_points),
            'linear_y': deque(maxlen=self.max_points),
            'linear_z': deque(maxlen=self.max_points),
            'angular_x': deque(maxlen=self.max_points),
            'angular_y': deque(maxlen=self.max_points),
            'angular_z': deque(maxlen=self.max_points),
        }
        
        # Error data (corner-based)
        self.error_data = {
            'individual_corner_errors': [
                deque(maxlen=self.max_points) for _ in range(4)  # Individual corner errors (corner 0-3)
            ],
            'norm_of_norms': deque(maxlen=self.max_points),  # Overall error metric
        }
        
        # ArUco marker data storage
        self.desired_features_msg = None   # Storage for desired features
        
        # Start time
        self.start_time = self.get_clock().now()
        
        # Setup subscribers
        self.setup_subscribers()
        
        # Setup matplotlib plots
        self.setup_plots()
        
        # Data storage setup
        self.setup_data_storage()
        
        # Setup periodic data saving (every 10 seconds)
        self.data_save_timer = self.create_timer(10.0, self.save_data_to_files)
        
        self.get_logger().info('Simplified Visual Servo Plotter initialized')
        self.get_logger().info(f'Data will be saved to: {self.data_directory}')
        self.get_logger().info('Keyboard shortcuts: s=screenshot, r=reset, c=clear, space=pause, h=help, q=quit')

    def setup_subscribers(self):
        """Setup ROS2 subscribers for IBVS monitoring."""
        
        # Subscribe to ArUco detections from oak_aruco_publisher
        if ARUCO_INTERFACES_AVAILABLE:
            self.aruco_sub = self.create_subscription(
                MarkerArray,
                '/aruco_detections',
                self.aruco_callback,
                10
            )
            self.get_logger().info("Subscribed to /aruco_detections")
        else:
            self.get_logger().error("ArUco interfaces not available!")
        
        # Subscribe to IBVS desired features
        if IBVS_FEATURES_AVAILABLE:
            try:
                ibvs_qos = QoSProfile(
                    depth=1,
                    durability=DurabilityPolicy.TRANSIENT_LOCAL,
                    reliability=ReliabilityPolicy.RELIABLE
                )
                
                self.ibvs_features_sub = self.create_subscription(
                    IBVSDesiredFeatures,
                    '/ibvs/desired_features',
                    self.ibvs_features_callback,
                    ibvs_qos
                )
                self.get_logger().info("Subscribed to /ibvs/desired_features with TRANSIENT_LOCAL")
                
            except Exception as e:
                self.get_logger().warn(f"Failed to create TRANSIENT_LOCAL subscription: {e}")
                # Fallback to standard QoS
                self.ibvs_features_sub = self.create_subscription(
                    IBVSDesiredFeatures,
                    '/ibvs/desired_features',
                    self.ibvs_features_callback,
                    10
                )
                self.get_logger().info("Subscribed to /ibvs/desired_features with standard QoS")
        else:
            self.get_logger().warn("IBVS features not available - using default desired positions")
        
        # Subscribe to IBVS velocity commands
        self.velocity_sub = self.create_subscription(
            TwistStamped,
            '/lbr/servo_node/delta_twist_cmds',
            self.velocity_callback,
            10
        )
        self.get_logger().info("Subscribed to /lbr/servo_node/delta_twist_cmds")

    def velocity_callback(self, msg: TwistStamped):
        """Callback for IBVS velocity commands."""
        if self.paused:
            return
            
        current_time = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
        self.velocity_time.append(current_time)
        
        # Store velocity data
        self.velocity_data['linear_x'].append(msg.twist.linear.x)
        self.velocity_data['linear_y'].append(msg.twist.linear.y)
        self.velocity_data['linear_z'].append(msg.twist.linear.z)
        self.velocity_data['angular_x'].append(msg.twist.angular.x)
        self.velocity_data['angular_y'].append(msg.twist.angular.y)
        self.velocity_data['angular_z'].append(msg.twist.angular.z)
        
        self.update_velocity_plot()

    def aruco_callback(self, msg):
        """Callback for ArUco detections from oak_aruco_publisher."""
        if self.paused:
            return
            
        current_time = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
        self.error_time.append(current_time)
        
        # Calculate corner-based errors
        self.calculate_corner_errors(msg, current_time)

    def calculate_corner_errors(self, msg, current_time: float):
        """Calculate corner-based errors for IBVS using oak_aruco_publisher data format."""
        if len(msg.markers) == 0:
            return
            
        # Use the first detected marker
        marker = msg.markers[0]
        
        # Extract current corner positions from the marker
        # oak_aruco_publisher provides corner_x and corner_y arrays with 4 values each
        # Order: [top_left, top_right, bottom_right, bottom_left]
        current_corners = [
            (marker.corner_x[0], marker.corner_y[0]),  # Top-left
            (marker.corner_x[1], marker.corner_y[1]),  # Top-right  
            (marker.corner_x[2], marker.corner_y[2]),  # Bottom-right
            (marker.corner_x[3], marker.corner_y[3]),  # Bottom-left
        ]
        
        # Get desired corner positions
        if self.desired_features_msg is not None and len(self.desired_features_msg.desired_features) >= 4:
            desired_corners = [
                (self.desired_features_msg.desired_features[0].u, self.desired_features_msg.desired_features[0].v),
                (self.desired_features_msg.desired_features[1].u, self.desired_features_msg.desired_features[1].v),
                (self.desired_features_msg.desired_features[2].u, self.desired_features_msg.desired_features[2].v),
                (self.desired_features_msg.desired_features[3].u, self.desired_features_msg.desired_features[3].v),
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
        for i, (current_corner, desired_corner) in enumerate(zip(current_corners, desired_corners)):
            error_u = current_corner[0] - desired_corner[0]
            error_v = current_corner[1] - desired_corner[1]
            corner_norm = math.sqrt(error_u**2 + error_v**2)
            corner_norms.append(corner_norm)
            
            # Store individual corner error
            self.error_data['individual_corner_errors'][i].append(corner_norm)
        
        # Calculate norm of corner norms (overall error metric)
        norm_of_norms = math.sqrt(sum(norm**2 for norm in corner_norms))
        
        # Store the overall error metric
        self.error_data['norm_of_norms'].append(norm_of_norms)
        
        # Update the error plot
        self.update_error_plot()

    def ibvs_features_callback(self, msg):
        """Callback for IBVS desired features."""
        self.desired_features_msg = msg
        self.get_logger().info(f'Received desired features: {len(msg.desired_features)} features')
        
        # Log the desired corner positions for debugging
        for i, feature in enumerate(msg.desired_features):
            self.get_logger().info(f'  Corner {i}: u={feature.u:.2f}, v={feature.v:.2f}')

    def setup_plots(self):
        """Setup matplotlib plots for simplified 2-plot vertical layout."""
        plt.ion()  # Enable interactive mode
        
        # Create figure with 2 subplots arranged vertically
        self.fig, self.axes = plt.subplots(2, 1, figsize=(12, 10))
        self.fig.suptitle('IBVS Visual Servoing Performance', fontsize=16)
        
        # Setup corner error plot (top)
        self.error_ax = self.axes[0]
        self.error_ax.set_title('Corner Tracking Errors')
        self.error_ax.set_xlabel('Time (s)')
        self.error_ax.set_ylabel('Error (pixels)')
        self.error_ax.grid(True, alpha=0.3)
        self.error_ax.legend(['Corner 0', 'Corner 1', 'Corner 2', 'Corner 3', 'Norm of Norms'], 
                           loc='upper right')
        
        # Setup velocity plot (bottom)
        self.velocity_ax = self.axes[1]
        self.velocity_ax.set_title('Camera Velocity Commands')
        self.velocity_ax.set_xlabel('Time (s)')
        self.velocity_ax.set_ylabel('Velocity')
        self.velocity_ax.grid(True, alpha=0.3)
        self.velocity_ax.legend(['Vx (m/s)', 'Vy (m/s)', 'Vz (m/s)', 
                               'ωx (rad/s)', 'ωy (rad/s)', 'ωz (rad/s)'],
                              loc='upper right')
        
        # Add key press handler for controls
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        plt.tight_layout()
        plt.show()

    def update_error_plot(self):
        """Update the corner error plot."""
        if not hasattr(self, 'error_ax'):
            return
            
        self.error_ax.clear()
        self.error_ax.set_title('Corner Tracking Errors')
        self.error_ax.set_xlabel('Time (s)')
        self.error_ax.set_ylabel('Error (pixels)')
        self.error_ax.grid(True, alpha=0.3)
        
        if len(self.error_time) > 0:
            time_data = list(self.error_time)
            
            # Plot individual corner errors
            colors = ['red', 'green', 'blue', 'orange']
            for i in range(4):
                if len(self.error_data['individual_corner_errors'][i]) > 0:
                    error_data = list(self.error_data['individual_corner_errors'][i])
                    if len(error_data) == len(time_data):
                        self.error_ax.plot(time_data, error_data, 
                                         color=colors[i], label=f'Corner {i}', linewidth=1.5)
            
            # Plot norm of norms (overall error)
            if len(self.error_data['norm_of_norms']) > 0:
                norm_data = list(self.error_data['norm_of_norms'])
                if len(norm_data) == len(time_data):
                    self.error_ax.plot(time_data, norm_data, 
                                     color='black', label='Norm of Norms', linewidth=2)
        
        self.error_ax.legend()
        self.fig.canvas.draw()

    def update_velocity_plot(self):
        """Update the velocity plot.""" 
        if not hasattr(self, 'velocity_ax'):
            return
            
        self.velocity_ax.clear()
        self.velocity_ax.set_title('Camera Velocity Commands')
        self.velocity_ax.set_xlabel('Time (s)')
        self.velocity_ax.set_ylabel('Velocity')
        self.velocity_ax.grid(True, alpha=0.3)
        
        if len(self.velocity_time) > 0:
            time_data = list(self.velocity_time)
            
            # Plot linear velocities
            if len(self.velocity_data['linear_x']) > 0:
                self.velocity_ax.plot(time_data, list(self.velocity_data['linear_x']), 
                                    'r-', label='Vx (m/s)', linewidth=1.5)
            if len(self.velocity_data['linear_y']) > 0:
                self.velocity_ax.plot(time_data, list(self.velocity_data['linear_y']), 
                                    'g-', label='Vy (m/s)', linewidth=1.5)
            if len(self.velocity_data['linear_z']) > 0:
                self.velocity_ax.plot(time_data, list(self.velocity_data['linear_z']), 
                                    'b-', label='Vz (m/s)', linewidth=1.5)
            
            # Plot angular velocities
            if len(self.velocity_data['angular_x']) > 0:
                self.velocity_ax.plot(time_data, list(self.velocity_data['angular_x']), 
                                    'r--', label='ωx (rad/s)', linewidth=1.5)
            if len(self.velocity_data['angular_y']) > 0:
                self.velocity_ax.plot(time_data, list(self.velocity_data['angular_y']), 
                                    'g--', label='ωy (rad/s)', linewidth=1.5)
            if len(self.velocity_data['angular_z']) > 0:
                self.velocity_ax.plot(time_data, list(self.velocity_data['angular_z']), 
                                    'b--', label='ωz (rad/s)', linewidth=1.5)
        
        self.velocity_ax.legend()
        self.fig.canvas.draw()

    def setup_data_storage(self):
        """Setup data storage and CSV writing."""
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

    def save_data_to_files(self):
        """Periodically save data to CSV files."""
        try:
            # Save error data
            if len(self.error_time) > 0 and 'corner_error' in self.csv_writers:
                time_data = list(self.error_time)
                norm_data = list(self.error_data['norm_of_norms'])
                
                for i, time_val in enumerate(time_data):
                    if i < len(norm_data):
                        row = [time_val, norm_data[i]]
                        # Add individual corner errors
                        for j in range(4):
                            if i < len(self.error_data['individual_corner_errors'][j]):
                                row.append(self.error_data['individual_corner_errors'][j][i])
                            else:
                                row.append(0.0)
                        self.csv_writers['corner_error'].writerow(row)
                        
            # Save velocity data
            if len(self.velocity_time) > 0 and 'velocity' in self.csv_writers:
                time_data = list(self.velocity_time)
                
                for i, time_val in enumerate(time_data):
                    row = [time_val]
                    # Add velocity components
                    for key in ['linear_x', 'linear_y', 'linear_z', 'angular_x', 'angular_y', 'angular_z']:
                        if i < len(self.velocity_data[key]):
                            row.append(self.velocity_data[key][i])
                        else:
                            row.append(0.0)
                    self.csv_writers['velocity'].writerow(row)
                    
            # Flush files
            for file_handle in self.csv_files.values():
                file_handle.flush()
                
        except Exception as e:
            self.get_logger().warn(f"Error saving data: {e}")

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
            # Clear plots
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
        self.velocity_time.clear()
        self.error_time.clear()
        
        for key in self.velocity_data:
            self.velocity_data[key].clear()
            
        for i in range(4):
            self.error_data['individual_corner_errors'][i].clear()
        self.error_data['norm_of_norms'].clear()
        
        self.start_time = self.get_clock().now()

    def clear_plots(self):
        """Clear plot displays but keep data."""
        if hasattr(self, 'error_ax'):
            self.error_ax.clear()
            self.error_ax.set_title('Corner Tracking Errors')
            self.error_ax.set_xlabel('Time (s)')
            self.error_ax.set_ylabel('Error (pixels)')
            self.error_ax.grid(True, alpha=0.3)
            
        if hasattr(self, 'velocity_ax'):
            self.velocity_ax.clear()
            self.velocity_ax.set_title('Camera Velocity Commands')
            self.velocity_ax.set_xlabel('Time (s)')
            self.velocity_ax.set_ylabel('Velocity')
            self.velocity_ax.grid(True, alpha=0.3)
            
        self.fig.canvas.draw()

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
        
        # Close CSV files
        for file_handle in self.csv_files.values():
            file_handle.close()
            
        # Close matplotlib
        plt.close('all')
        
        # Shutdown node
        rclpy.shutdown()


def main():
    rclpy.init()
    
    try:
        plotter = VisualServoPlotter()
        
        # Start ROS2 spinning in a separate thread
        ros_thread = threading.Thread(target=lambda: rclpy.spin(plotter), daemon=True)
        ros_thread.start()
        
        # Keep matplotlib window open
        plt.show(block=True)
        
    except KeyboardInterrupt:
        print("Shutting down plotter...")
    finally:
        if 'plotter' in locals():
            plotter.get_logger().info("Saving final data and closing files...")
            plotter.save_data_to_files()
            for file_handle in plotter.csv_files.values():
                file_handle.close()
            plotter.get_logger().info(f"All data saved to: {plotter.data_directory}")
        rclpy.shutdown()


if __name__ == '__main__':
    main()
