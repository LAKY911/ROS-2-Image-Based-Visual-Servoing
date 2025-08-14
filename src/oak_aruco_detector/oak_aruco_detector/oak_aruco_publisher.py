#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from oak_aruco_detector_interfaces.msg import Marker, MarkerArray
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from ament_index_python.packages import get_package_share_directory
import signal
import sys
import threading
import yaml
import os

import depthai as dai
import cv2
import numpy as np

def getMesh(calibData, ispSize, camSocket):
    """Generate mesh for undistortion based on camera calibration data"""
    M1 = np.array(calibData.getCameraIntrinsics(camSocket, ispSize[0], ispSize[1]))
    d1 = np.array(calibData.getDistortionCoefficients(camSocket))
    R1 = np.identity(3)
    mapX, mapY = cv2.initUndistortRectifyMap(M1, d1, R1, M1, ispSize, cv2.CV_32FC1)

    meshCellSize = 16
    mesh0 = []
    # Creates subsampled mesh which will be loaded on to device to undistort the image
    for y in range(mapX.shape[0] + 1): # iterating over height of the image
        if y % meshCellSize == 0:
            rowLeft = []
            for x in range(mapX.shape[1]): # iterating over width of the image
                if x % meshCellSize == 0:
                    if y == mapX.shape[0] and x == mapX.shape[1]:
                        rowLeft.append(mapX[y - 1, x - 1])
                        rowLeft.append(mapY[y - 1, x - 1])
                    elif y == mapX.shape[0]:
                        rowLeft.append(mapX[y - 1, x])
                        rowLeft.append(mapY[y - 1, x])
                    elif x == mapX.shape[1]:
                        rowLeft.append(mapX[y, x - 1])
                        rowLeft.append(mapY[y, x - 1])
                    else:
                        rowLeft.append(mapX[y, x])
                        rowLeft.append(mapY[y, x])
            if (mapX.shape[1] % meshCellSize) % 2 != 0:
                rowLeft.append(0)
                rowLeft.append(0)

            mesh0.append(rowLeft)

    mesh0 = np.array(mesh0)
    meshWidth = mesh0.shape[1] // 2
    meshHeight = mesh0.shape[0]
    mesh0.resize(meshWidth * meshHeight, 2)

    mesh = list(map(tuple, mesh0))

    return mesh, meshWidth, meshHeight

class OakArucoPublisher(Node):
    def __init__(self):
        super().__init__('oak_aruco_publisher')
        self.publisher_ = self.create_publisher(MarkerArray, '/aruco_detections', 10)
        self.image_publisher_ = self.create_publisher(Image, '/oak_undistorted_image', 10)
        self.camera_info_publisher_ = self.create_publisher(CameraInfo, '/oak_camera_info', 10)
        
        # Declare parameters
        self.declare_parameter('publish_undistorted_image', True)
        self.declare_parameter('camera_fps', 30)  # Default FPS for camera
        self.declare_parameter('image_publish_hz', 15)  # Separate image publishing rate
        self.declare_parameter('image_scale', 1.0)  # Image scaling for bandwidth reduction
        
        # Get default calibration file path using package discovery
        try:
            package_share_dir = get_package_share_directory('oak_aruco_detector')
            default_calibration_file = os.path.join(package_share_dir, 'config', 'camera_calibration.yaml')
        except Exception as e:
            self.get_logger().warn(f"Could not find package directory: {e}")
            default_calibration_file = ''
        
        self.declare_parameter('camera_calibration_file', default_calibration_file)  # Path to camera calibration YAML file
        
        # Cache parameter values to avoid repeated lookups
        self.publish_image = self.get_parameter('publish_undistorted_image').get_parameter_value().bool_value
        self.camera_fps = self.get_parameter('camera_fps').get_parameter_value().integer_value
        self.image_publish_hz = self.get_parameter('image_publish_hz').get_parameter_value().integer_value
        self.image_scale = self.get_parameter('image_scale').get_parameter_value().double_value
        self.calibration_file = self.get_parameter('camera_calibration_file').get_parameter_value().string_value
        
        # Log parameter settings
        self.get_logger().info(f"Undistorted image publishing: {'enabled' if self.publish_image else 'disabled'}")
        self.get_logger().info(f"Camera FPS set to: {self.camera_fps}")
        self.get_logger().info(f"Image publishing: {self.image_publish_hz} Hz, scale: {self.image_scale}")
        self.get_logger().info(f"Calibration file path: {self.calibration_file}")

        # Initialize CV Bridge for image conversion
        self.cv_bridge = CvBridge()
        
        # Publishing control for image rate limiting
        self.image_publish_counter = 0
        self.image_publish_interval = max(1, 60 // self.image_publish_hz)  # Calculate interval for 60 Hz timer
        
        # Cached header for efficiency
        self._header_template = Header()
        self._header_template.frame_id = "oak1w_camera_frame"
        
        # Initialize device and pipeline
        self.device = None
        self.q_undistorted = None
        self.camera_matrix = None
        self.distortion_coeffs = None
        self.camera_info_msg = None
        
        # Calibration data from YAML file
        self.calibration_data = None
        
        # Frame processing control
        self._running = True
        self._frame_thread = None
        
        # Camera configuration (matching camera_undistort.py)
        self.camRes = dai.ColorCameraProperties.SensorResolution.THE_1080_P
        self.camSocket = dai.CameraBoardSocket.CAM_A
        
        # Initialize ArUco detector
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.parameters)
        self.get_logger().info("Using OpenCV ArUco ArucoDetector API")
        
        # Load calibration data from YAML file if provided
        if self.calibration_file:
            if os.path.exists(self.calibration_file):
                self.load_calibration_from_yaml(self.calibration_file)
            else:
                self.get_logger().error(f"Calibration file not found: {self.calibration_file}")
                self.get_logger().info("Will fallback to device calibration data")
        else:
            self.get_logger().warn("No calibration file specified, using device calibration data")
                
        self.init_device()
        
        if self.device is not None:
            # Use frame-driven processing instead of timer-based
            # process_detections will be called automatically when frames arrive
            self.get_logger().info("OAK ArUco Publisher with frame-driven undistorted image processing started successfully.")
        else:
            self.get_logger().error("Failed to initialize OAK device")

    def init_device(self):
        try:
            # Connect to device first to get calibration data
            self.device = dai.Device()
            
            # Get calibration data
            calibData = self.device.readCalibration()
            
            # Create pipeline
            pipeline = dai.Pipeline()

            # Define source - RGB Camera 
            cam_rgb = pipeline.create(dai.node.ColorCamera)
            cam_rgb.setBoardSocket(self.camSocket)
            cam_rgb.setResolution(self.camRes)
            cam_rgb.setFps(self.camera_fps)  # Set camera FPS to configured value

            # Create ImageManip node for undistortion using mesh
            manip = pipeline.create(dai.node.ImageManip)
            mesh, meshWidth, meshHeight = getMesh(calibData, cam_rgb.getIspSize(), self.camSocket)
            manip.setWarpMesh(mesh, meshWidth, meshHeight)
            manip.setMaxOutputFrameSize(cam_rgb.getIspWidth() * cam_rgb.getIspHeight() * 3)
            
            # Link camera to image manip
            cam_rgb.isp.link(manip.inputImage)

            # Output for undistorted image
            xout_undistorted = pipeline.create(dai.node.XLinkOut)
            xout_undistorted.setStreamName("Undistorted")
            manip.out.link(xout_undistorted.input)
 
            # Start pipeline
            self.device.startPipeline(pipeline)
            
            # Get camera intrinsics for the ISP size (after scaling)
            isp_size = cam_rgb.getIspSize()
            intrinsics = calibData.getCameraIntrinsics(self.camSocket, isp_size[0], isp_size[1])
            self.camera_matrix = np.array(intrinsics).reshape(3, 3)
            
            # Get distortion coefficients
            distortion = calibData.getDistortionCoefficients(self.camSocket)
            self.distortion_coeffs = np.array(distortion)
            
            # Create camera info message using external calibration if available
            self.camera_info_msg = self._create_camera_info_msg(isp_size)
            
            self.get_logger().info("Using proper mesh-based undistortion")
            
            # Print device info
            self.get_logger().info(f'Connected cameras: {self.device.getConnectedCameraFeatures()}')
            self.get_logger().info(f'USB speed: {self.device.getUsbSpeed().name}')
            self.get_logger().info(f'Device name: {self.device.getDeviceName()}, Product name: {self.device.getProductName()}')
            self.get_logger().info(f'ISP size: {isp_size}')
            self.get_logger().info(f'Camera matrix of the camera:\n{self.camera_matrix}')
            self.get_logger().info(f'Distortion coeffs of the camera:\n{self.distortion_coeffs}')
            
            # Output queue for undistorted images
            self.q_undistorted = self.device.getOutputQueue(name="Undistorted", maxSize=4, blocking=False)
            
            # Start frame processing thread
            self._frame_thread = threading.Thread(target=self._frame_processing_loop, daemon=True)
            self._frame_thread.start()
            self.get_logger().info("Started frame processing thread")
            
        except Exception as e:
            self.get_logger().error(f"Failed to initialize device: {e}")
            self.device = None

    def load_calibration_from_yaml(self, yaml_file):
        """Load camera calibration data from YAML file"""
        try:
            with open(yaml_file, 'r') as file:
                self.calibration_data = yaml.safe_load(file)
            
            # Validate that required fields are present
            required_fields = ['camera_matrix', 'distortion_coefficients']
            missing_fields = [field for field in required_fields if field not in self.calibration_data]
            
            if missing_fields:
                self.get_logger().error(f"Missing required fields in calibration file: {missing_fields}")
                self.calibration_data = None
                return
            
            self.get_logger().info(f"Successfully loaded calibration data from: {yaml_file}")
            self.get_logger().info(f"Image dimensions: {self.calibration_data.get('image_width', 'N/A')}x{self.calibration_data.get('image_height', 'N/A')}")
            
        except Exception as e:
            self.get_logger().error(f"Failed to load calibration file {yaml_file}: {e}")
            self.calibration_data = None

    def _create_camera_info_msg(self, image_size):
        """Create a CameraInfo message from calibration data"""
        camera_info = CameraInfo()
        camera_info.header.frame_id = "oak_camera_frame"
        camera_info.width = image_size[0]
        camera_info.height = image_size[1]
        
        # Use calibration data from YAML file if available, otherwise fallback to device calibration
        if self.calibration_data is not None:
            # Parse camera matrix from nested list format
            camera_matrix_nested = self.calibration_data.get('camera_matrix', [])
            if camera_matrix_nested:
                # Flatten the nested list: [[a,b,c],[d,e,f],[g,h,i]] -> [a,b,c,d,e,f,g,h,i]
                camera_info.k = [item for row in camera_matrix_nested for item in row]
            
            # Parse distortion coefficients from nested list format
            distortion_nested = self.calibration_data.get('distortion_coefficients', [])
            if distortion_nested:
                # Flatten the nested list: [[a,b,c,d,e]] -> [a,b,c,d,e]
                camera_info.d = [item for row in distortion_nested for item in row]
            
            # Set rectification matrix (identity for monocular camera)
            camera_info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
            
            # Create projection matrix from camera matrix (K matrix with added column of zeros)
            if camera_matrix_nested:
                # P = [fx, 0, cx, 0; 0, fy, cy, 0; 0, 0, 1, 0]
                camera_info.p = [
                    camera_matrix_nested[0][0], camera_matrix_nested[0][1], camera_matrix_nested[0][2], 0.0,
                    camera_matrix_nested[1][0], camera_matrix_nested[1][1], camera_matrix_nested[1][2], 0.0,
                    camera_matrix_nested[2][0], camera_matrix_nested[2][1], camera_matrix_nested[2][2], 0.0
                ]
            
            camera_info.distortion_model = "plumb_bob"
            
            self.get_logger().info("Using external calibration data from YAML file")
            self.get_logger().info(f"Camera matrix (K): {camera_info.k}")
            self.get_logger().info(f"Distortion coefficients (D): {camera_info.d}")
        else:
            # Fallback to device calibration data
            if self.camera_matrix is not None and self.distortion_coeffs is not None:
                camera_info.k = self.camera_matrix.flatten().tolist()
                camera_info.d = self.distortion_coeffs.tolist()
                camera_info.r = np.eye(3).flatten().tolist()
                
                # Create projection matrix from camera matrix
                camera_info.p = np.zeros((3, 4))
                camera_info.p[:3, :3] = self.camera_matrix
                camera_info.p = camera_info.p.flatten().tolist()
                
                camera_info.distortion_model = "plumb_bob"
                self.get_logger().info("Using device calibration data")
            else:
                self.get_logger().warn("No calibration data available")
        
        return camera_info

    def _frame_processing_loop(self):
        """Frame-driven processing loop that runs in a separate thread"""
        while self._running and rclpy.ok():
            try:
                if self.device is None or self.q_undistorted is None:
                    break
                
                # Wait for new frame (non-blocking with small delay)
                in_frame = self.q_undistorted.tryGet()
                if in_frame is not None:
                    # Process the frame immediately when it arrives
                    self.process_detections(in_frame)
                else:
                    # Small sleep to avoid busy waiting
                    threading.Event().wait(0.001)  # 1ms sleep
                    
            except Exception as e:
                if self._running:  # Only log if we're still supposed to be running
                    self.get_logger().warn(f"Frame processing error: {e}")
                break

    def process_detections(self, in_frame=None):
        # If no frame provided, try to get one (backward compatibility for timer-based calls)
        if in_frame is None:
            if self.device is None or self.q_undistorted is None:
                return
            in_frame = self.q_undistorted.tryGet()
            if in_frame is None:
                return

        try:
            # Get undistorted image frame using getCvFrame() method
            frame = in_frame.getCvFrame()  # type: ignore
            
            # Convert to grayscale for ArUco detection (already undistorted)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect ArUco markers using the new API on undistorted image
            try:
                corners, ids, _ = self.detector.detectMarkers(gray)
            except Exception as e:
                self.get_logger().warn(f"ArUco detection error: {e}")
                return

            # Update timestamp once per cycle for efficiency
            self._header_template.stamp = self.get_clock().now().to_msg()

            # Always publish markers at full rate (60 Hz) for responsiveness
            if ids is not None and len(ids) > 0:
                # Create ROS message
                markers = MarkerArray()
                markers.header = self._header_template  # Use cached header

                for i in range(len(ids)):
                    # Get corner coordinates for the marker
                    corner = corners[i][0]  # Shape: (4, 2) - 4 corners, each with (x, y)
                    
                    # Calculate center of marker
                    cx = float(np.mean(corner[:, 0]))
                    cy = float(np.mean(corner[:, 1]))
                    marker_id = int(ids[i][0])
                    
                    # Extract all 4 corner coordinates
                    # OpenCV ArUco returns corners in order: top-left, top-right, bottom-right, bottom-left
                    corner_x = [float(corner[j, 0]) for j in range(4)]
                    corner_y = [float(corner[j, 1]) for j in range(4)]
                    
                    marker = Marker()
                    marker.header = self._header_template  # Use cached header
                    marker.id = marker_id
                    marker.x = cx  # Center x coordinate
                    marker.y = cy  # Center y coordinate
                    marker.corner_x = corner_x  # All 4 corner x coordinates
                    marker.corner_y = corner_y  # All 4 corner y coordinates
                    markers.markers.append(marker)

                self.publisher_.publish(markers)
                self.get_logger().debug(f"Published {len(markers.markers)} ArUco markers with corner coordinates")
                
                # Log detailed corner information for the first marker (for debugging)
                if len(markers.markers) > 0:
                    first_marker = markers.markers[0]
                    self.get_logger().debug(f"Marker {first_marker.id}: Center=({first_marker.x:.1f}, {first_marker.y:.1f}), "
                                          f"Corners: TL=({first_marker.corner_x[0]:.1f}, {first_marker.corner_y[0]:.1f}), "
                                          f"TR=({first_marker.corner_x[1]:.1f}, {first_marker.corner_y[1]:.1f}), "
                                          f"BR=({first_marker.corner_x[2]:.1f}, {first_marker.corner_y[2]:.1f}), "
                                          f"BL=({first_marker.corner_x[3]:.1f}, {first_marker.corner_y[3]:.1f})")

            # Publish camera info with every frame for synchronization
            if self.camera_info_msg is not None:
                self.camera_info_msg.header = self._header_template
                self.camera_info_publisher_.publish(self.camera_info_msg)

            # Publish images at reduced rate to save bandwidth and CPU
            if self.publish_image:
                self.image_publish_counter += 1
                if self.image_publish_counter >= self.image_publish_interval:
                    self.image_publish_counter = 0
                    try:
                        # Optionally resize image to reduce bandwidth
                        if self.image_scale != 1.0:
                            # Calculate new dimensions
                            new_width = int(frame.shape[1] * self.image_scale)
                            new_height = int(frame.shape[0] * self.image_scale)
                            scaled_frame = cv2.resize(frame, (new_width, new_height))
                            image_msg = self.cv_bridge.cv2_to_imgmsg(scaled_frame, encoding='bgr8')
                        else:
                            image_msg = self.cv_bridge.cv2_to_imgmsg(frame, encoding='bgr8')
                        
                        image_msg.header = self._header_template  # Use cached header
                        self.image_publisher_.publish(image_msg)
                        self.get_logger().debug(f"Published image at {self.image_publish_hz} Hz (scale: {self.image_scale})")
                    except Exception as e:
                        self.get_logger().warn(f"Failed to publish undistorted image: {e}")

        except Exception as e:
            self.get_logger().warn(f"Error processing detections: {e}")

    def destroy_node(self):
        """Clean up device resources"""
        self.get_logger().info("Shutting down OAK ArUco Publisher...")
        
        # Stop frame processing thread
        self._running = False
        if self._frame_thread is not None and self._frame_thread.is_alive():
            self.get_logger().info("Waiting for frame processing thread to stop...")
            self._frame_thread.join(timeout=2.0)
        
        if self.device is not None:
            try:
                self.get_logger().info("Closing OAK device...")
                self.device.close()
                self.get_logger().info("OAK device closed successfully")
            except Exception as e:
                self.get_logger().warn(f"Error closing device: {e}")
        
        super().destroy_node()
        self.get_logger().info("Node shutdown complete")

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    print("\nReceived interrupt signal. Shutting down gracefully...")
    rclpy.shutdown()
    sys.exit(0)

def main(args=None):
    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    rclpy.init(args=args)
    
    node = None
    try:
        node = OakArucoPublisher()
        print("Press Ctrl+C to shutdown gracefully")
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received")
    except Exception as e:
        print(f"Failed to start node: {e}")
    finally:
        if node is not None:
            print("Cleaning up node...")
            node.destroy_node()
        try:
            rclpy.shutdown()
        except:
            pass  # ROS might already be shut down
        print("Shutdown complete")

if __name__ == '__main__':
    main()
