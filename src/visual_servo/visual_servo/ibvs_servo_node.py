"""
Image-Based Visual Servoing (IBVS) Controller for ROS2 - Refactored Version

This module implements an IBVS controller with improved structure:
- Separated concerns into multiple classes
- Better error handling and validation
- Cleaner parameter management
- Improved readability and maintainability
"""

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from geometry_msgs.msg import TwistStamped, TransformStamped, Point
from std_srvs.srv import Trigger
from scipy.spatial.transform import Rotation as R
import numpy as np
import tf2_ros
import signal
import sys
import threading
import time
from typing import Optional, List, Tuple, Any
from numpy.typing import NDArray
from rclpy.time import Time as RclpyTime
from oak_aruco_detector_interfaces.msg import MarkerArray # type: ignore
from visual_servo_interfaces.msg import IBVSDesiredFeatures # type: ignore
from visual_servo_interfaces.srv import SetDesiredFeatures # type: ignore
from sensor_msgs.msg import CameraInfo
from dataclasses import dataclass
from enum import Enum
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor

# Optional OpenCV import for PnP depth estimation
try:
    import cv2  # type: ignore
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False


class InteractionMatrixMethod(Enum):
    """Enum for interaction matrix computation methods."""
    CURRENT = "current"
    DESIRED = "desired"
    MEAN = "mean"


@dataclass
class IBVSConfig:
    """Configuration parameters for IBVS controller."""
    # Control parameters
    lambda_gain: float = 0.5
    max_linear_velocity: float = 0.2
    max_angular_velocity: float = 0.5
    convergence_threshold: float = 5.0
    interaction_matrix_method: InteractionMatrixMethod = InteractionMatrixMethod.MEAN
    
    # Timeouts and monitoring
    aruco_timeout: float = 1.5
    
    # Frame IDs
    camera_frame: str = "oak1w_camera_frame"
    gazebo_camera_frame: str = "gazebo_camera_frame"
    base_frame: str = "lbr_link_0"
    ee_frame: str = "lbr_link_ee"
    
    # Default depths
    default_depth: float = 0.6
    
    # ArUco marker parameters    
    marker_size_meters: float = 0.05
    
    # Depth estimation limits
    min_depth: float = 0.2
    max_depth: float = 3.0
    
    def validate(self) -> List[str]:
        """Validate configuration parameters and return list of errors."""
        errors = []
        
        if self.lambda_gain <= 0:
            errors.append(f"lambda_gain must be positive, got {self.lambda_gain}")
        
        if self.max_linear_velocity <= 0:
            errors.append(f"max_linear_velocity must be positive, got {self.max_linear_velocity}")
            
        if self.max_angular_velocity <= 0:
            errors.append(f"max_angular_velocity must be positive, got {self.max_angular_velocity}")
            
        if self.convergence_threshold <= 0:
            errors.append(f"convergence_threshold must be positive, got {self.convergence_threshold}")
            
        if self.aruco_timeout <= 0:
            errors.append(f"aruco_timeout must be positive, got {self.aruco_timeout}")
            
        if self.marker_size_meters <= 0:
            errors.append(f"marker_size_meters must be positive, got {self.marker_size_meters}")
            
        return errors


@dataclass
class FeatureData:
    """Container for image feature data."""
    features: NDArray[np.float64]
    depth: float
    timestamp: RclpyTime
    
    @property
    def center(self) -> Point:
        """Compute center from features."""
        center = Point()
        u_coords = self.features[::2]  # [u1, u2, u3, u4]
        v_coords = self.features[1::2]  # [v1, v2, v3, v4]
        center.x = float(np.mean(u_coords))
        center.y = float(np.mean(v_coords))
        center.z = 0.0
        return center
    
    @property
    def corners(self) -> List[Point]:
        """Convert features to corner points."""
        corners = []
        for i in range(4):
            corner = Point()
            corner.x = float(self.features[2*i])
            corner.y = float(self.features[2*i + 1])
            corner.z = 0.0
            corners.append(corner)
        return corners


class DepthEstimator:
    """Handles depth estimation from ArUco marker corners."""
    
    def __init__(self, config: IBVSConfig, logger=None):
        self.config = config
        self.camera_matrix: Optional[NDArray[np.float64]] = None
        self.dist_coeffs: Optional[NDArray[np.float64]] = None
        self.logger = logger
    
    def set_camera_matrix(self, camera_matrix: NDArray[np.float64]) -> None:
        """Set the camera intrinsic matrix."""
        self.camera_matrix = camera_matrix.copy()
    
    def set_distortion(self, dist_coeffs: Optional[NDArray[np.float64]]) -> None:
        """Set the camera distortion coefficients."""
        if dist_coeffs is None:
            self.dist_coeffs = None
        else:
            self.dist_coeffs = dist_coeffs.astype(np.float64).reshape(-1, 1)
    
    def _estimate_with_pnp(self, corners: List[Point]) -> Optional[float]:
        """Estimate depth using OpenCV solvePnP if available."""
        if not _HAS_CV2 or self.camera_matrix is None:
            return None
        try:
            # Define 3D object points for a square marker centered at origin (Z=0)
            s = self.config.marker_size_meters / 2.0
            # Order must match the 2D corners order; assumed [tl, tr, br, bl]
            obj_pts = np.array([
                [-s,  s, 0.0],
                [ s,  s, 0.0],
                [ s, -s, 0.0],
                [-s, -s, 0.0],
            ], dtype=np.float64)
            img_pts = np.array([[c.x, c.y] for c in corners], dtype=np.float64)
            camera_matrix = self.camera_matrix
            dist_coeffs = self.dist_coeffs if self.dist_coeffs is not None else np.zeros((5, 1), dtype=np.float64)

            # Prefer IPPE for planar squares if available
            flags = getattr(cv2, 'SOLVEPNP_IPPE_SQUARE', 0)
            if flags != 0:
                ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, camera_matrix, dist_coeffs, flags=flags)
            else:
                ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok:
                return None
            z = float(tvec[2, 0])
            return float(np.clip(z, self.config.min_depth, self.config.max_depth))
        except Exception as e:
            if self.logger:
                self.logger.debug(f"solvePnP failed, falling back to heuristic: {e}")
            return None

    def estimate_from_corners(self, corners: List[Point]) -> float:
        """Estimate depth from corner positions using PnP if possible, else heuristic."""
        if len(corners) != 4 or self.camera_matrix is None:
            return self.config.default_depth
        
        # Try solvePnP first
        z = self._estimate_with_pnp(corners)
        if z is not None and np.isfinite(z) and z > 0:
            return z
        
        # Fallback heuristic: pinhole model from average side length in pixels
        u_coords = [corner.x for corner in corners]
        v_coords = [corner.y for corner in corners]
        
        side_lengths = []
        for i in range(4):
            j = (i + 1) % 4
            dx = u_coords[i] - u_coords[j]
            dy = v_coords[i] - v_coords[j]
            side_lengths.append(np.sqrt(dx*dx + dy*dy))
        
        avg_side_length = float(np.mean(side_lengths))
        
        if avg_side_length <= 0:
            return self.config.default_depth
        
        fx, fy = self.camera_matrix[0, 0], self.camera_matrix[1, 1]
        f_avg = (fx + fy) / 2.0
        
        estimated_depth = (f_avg * self.config.marker_size_meters) / avg_side_length
        
        return float(np.clip(estimated_depth, self.config.min_depth, self.config.max_depth))


class InteractionMatrixComputer:
    """Computes interaction matrices for IBVS control."""
    
    def __init__(self, config: IBVSConfig, logger=None):
        self.config = config
        self.camera_matrix: Optional[NDArray[np.float64]] = None
        self.logger = logger
        
        # Precomputed pseudoinverses for optimization
        self._desired_L_pinv: Optional[NDArray[np.float64]] = None
        self._desired_features_cache: Optional[NDArray[np.float64]] = None
        self._desired_depth_cache: Optional[float] = None
    
    def set_camera_matrix(self, camera_matrix: NDArray[np.float64]) -> None:
        """Set the camera intrinsic matrix."""
        self.camera_matrix = camera_matrix.copy()
        # Invalidate cache when camera matrix changes
        self._desired_L_pinv = None
    
    def set_desired_features(self, features: NDArray[np.float64], depth: float) -> None:
        """Precompute pseudoinverse for desired features."""
        if self.camera_matrix is None:
            return
            
        # Check if we need to recompute
        if (self._desired_features_cache is not None and 
            self._desired_depth_cache is not None and
            np.allclose(features, self._desired_features_cache) and
            abs(depth - self._desired_depth_cache) < 0.001):
            return  # Already cached
        
        if self.logger:
            self.logger.debug("Precomputing desired interaction matrix pseudoinverse...")
        
        L_desired = self.compute_matrix(features, depth)
        self._desired_L_pinv = np.linalg.pinv(L_desired)
        
        # Cache the features and depth
        self._desired_features_cache = features.copy()
        self._desired_depth_cache = depth
        
        if self.logger:
            self.logger.debug("Precomputation complete")
    
    def compute_matrix(self, features: NDArray[np.float64], depth: float) -> NDArray[np.float64]:
        """Compute interaction matrix for given features and depth."""
        if self.camera_matrix is None:
            raise ValueError("Camera matrix not set")
        
        fx, fy = self.camera_matrix[0, 0], self.camera_matrix[1, 1]
        cx, cy = self.camera_matrix[0, 2], self.camera_matrix[1, 2]
        
        # Guard against near-zero depth
        depth = float(np.clip(depth, 1e-6, np.inf))
        
        L_s = np.zeros((8, 6))
        
        for i in range(4):
            u, v = features[2*i], features[2*i + 1]
            
            # Convert to normalized coordinates
            x = (u - cx) / fx
            y = (v - cy) / fy

            if self.logger:
                self.logger.debug(
                    f"Normalized coordinates: ({x:.6f}, {y:.6f}), "
                    f"depth: {depth:.6f}, u: {u:.3f}, v: {v:.3f}, "
                    # f"fx: {fx:.3f}, fy: {fy:.3f}, cx: {cx:.3f}, cy: {cy:.3f}"
                )
            # Interaction matrix rows for this corner
            row_x, row_y = 2*i, 2*i + 1
            
            # X coordinate row
            L_s[row_x] = [-1/depth, 0, x/depth, x*y, -(1 + x*x), y]
            
            # Y coordinate row
            L_s[row_y] = [0, -1/depth, y/depth, 1 + y*y, -x*y, -x]
        
        return L_s
    
    def compute_pseudoinverse(self, current_data: FeatureData, 
                            desired_data: Optional[FeatureData] = None) -> NDArray[np.float64]:
        """Compute interaction matrix pseudoinverse based on method."""
        if self.config.interaction_matrix_method == InteractionMatrixMethod.CURRENT:
            L_s = self.compute_matrix(current_data.features, current_data.depth)
            L_pinv = np.linalg.pinv(L_s)
        
        elif self.config.interaction_matrix_method == InteractionMatrixMethod.DESIRED:
            if desired_data is None:
                raise ValueError("Desired data required for DESIRED method")
                
            # Use precomputed pseudoinverse if available
            if (self._desired_L_pinv is not None and 
                self._desired_features_cache is not None and
                self._desired_depth_cache is not None and
                np.allclose(desired_data.features, self._desired_features_cache) and
                abs(desired_data.depth - self._desired_depth_cache) < 0.001):
                L_pinv = self._desired_L_pinv
                if self.logger:
                    self.logger.debug("Using precomputed desired pseudoinverse")
            else:
                L_s = self.compute_matrix(desired_data.features, desired_data.depth)
                L_pinv = np.linalg.pinv(L_s)
        
        elif self.config.interaction_matrix_method == InteractionMatrixMethod.MEAN:
            if desired_data is None:
                raise ValueError("Desired data required for MEAN method")
            L_current = self.compute_matrix(current_data.features, current_data.depth)
            L_desired = self.compute_matrix(desired_data.features, desired_data.depth)
            L_mean = 0.5 * (L_current + L_desired)
            L_pinv = np.linalg.pinv(L_mean)
        
        else:
            raise ValueError(f"Unknown interaction matrix method: {self.config.interaction_matrix_method}")
        
        return L_pinv


class VelocityNormalizer:
    """Normalizes velocity commands for robot control."""
    
    def __init__(self, config: IBVSConfig):
        self.config = config
    
    def normalize(self, velocity: NDArray[np.float64]) -> NDArray[np.float64]:
        """Normalize velocity to [-1, 1] range for MoveIt Servo."""
        v_normalized = np.zeros(6)
        
        # Normalize linear velocities
        for i in range(3):
            v_normalized[i] = np.clip(velocity[i] / self.config.max_linear_velocity, -1.0, 1.0)
        
        # Normalize angular velocities
        for i in range(3, 6):
            v_normalized[i] = np.clip(velocity[i] / self.config.max_angular_velocity, -1.0, 1.0)
        
        return v_normalized


class TransformManager:
    """
    Handles coordinate frame transformations with static transform optimization.
    
    Since the transformation from gazebo_camera_frame to ee_frame is static 
    (just a coordinate system conversion), this class precomputes the transform
    matrix once during initialization instead of doing expensive TF2 lookups
    on every control cycle. This eliminates the 1000ms+ TF2 bottleneck.
    """
    
    def __init__(self, tf_buffer: tf2_ros.Buffer, config: IBVSConfig, logger=None):
        self.tf_buffer = tf_buffer
        self.config = config
        self.logger = logger
        
        # Precomputed static transform matrix from gazebo_camera_frame to ee_frame
        self._static_transform_matrix: Optional[NDArray[np.float64]] = None
        self._transform_initialized = False
        
        # Initialize the static transform lazily
        self._transform_initialized = False
    
    def _initialize_static_transform(self) -> None:
        """Initialize the static transformation from gazebo_camera_frame to ee_frame."""
        if self._transform_initialized:
            return
            
        if self.logger:
            self.logger.debug("Attempting to initialize static transform from gazebo_camera_frame to ee_frame...")
        
        try:
            transform = self.tf_buffer.lookup_transform(
                self.config.ee_frame,
                self.config.gazebo_camera_frame,
                Time(),
                timeout=Duration(seconds=0, nanoseconds=100_000_000)  # 0.1s
            )
            
            # Extract rotation and convert to matrix
            q = [transform.transform.rotation.x,
                 transform.transform.rotation.y,
                 transform.transform.rotation.z,
                 transform.transform.rotation.w]
            rot = R.from_quat(q)
            self._static_transform_matrix = rot.as_matrix()
            self._transform_initialized = True
            
            if self.logger:
                self.logger.debug("Static transform initialized successfully!")
                self.logger.debug(f"Transform matrix:\n{self._static_transform_matrix}")
            
        except Exception as e:
            # Don't log error every time - just on first few attempts
            if hasattr(self, '_init_attempts'):
                self._init_attempts += 1
            else:
                self._init_attempts = 1
                
            if self._init_attempts <= 3 and self.logger:
                self.logger.warn(f"Static transform initialization attempt {self._init_attempts} failed: {e}")
                
            if self._init_attempts >= 10:
                if self.logger:
                    self.logger.error("Could not initialize static transform after 10 attempts, using identity matrix")
                self._static_transform_matrix = np.eye(3)
                self._transform_initialized = True
    
    def reinitialize_static_transform(self) -> bool:
        """Force reinitialize the static transform. Returns True if successful."""
        self._transform_initialized = False
        self._init_attempts = 0
        self._initialize_static_transform()
        return self._transform_initialized and self._static_transform_matrix is not None
    
    def transform_velocity_to_ee_frame(self, v_camera: NDArray[np.float64]) -> NDArray[np.float64]:
        """Transform velocity from camera frame to end-effector frame using precomputed static transform."""
        # Lazy initialization: try to initialize static transform on first use
        if not self._transform_initialized:
            self._initialize_static_transform()
        
        if not self._transform_initialized or self._static_transform_matrix is None:
            if self.logger:
                self.logger.warn("Static transform not available, using identity")
            return v_camera
        
        # Apply precomputed static transformation
        v_linear = self._static_transform_matrix @ v_camera[:3]
        v_angular = self._static_transform_matrix @ v_camera[3:]
        result = np.concatenate([v_linear, v_angular])
        
        return result


class IBVSController(Node):
    """Main IBVS controller with improved structure."""
    
    def __init__(self, config: Optional[IBVSConfig] = None):
        super().__init__('ibvs_controller')
        
        # Initialize configuration
        self.config = config or IBVSConfig()
        self._setup_parameters()
        
        # Validate configuration
        config_errors = self.config.validate()
        if config_errors:
            for error in config_errors:
                self.get_logger().error(f"Configuration error: {error}")
            raise ValueError(f"Invalid configuration: {config_errors}")
        
        # Shutdown guard
        self._shutting_down: bool = False
        
        # Callback groups for concurrency control
        self.cbgroup_sensors = ReentrantCallbackGroup()
        self.cbgroup_control = ReentrantCallbackGroup()
        self.cbgroup_services = MutuallyExclusiveCallbackGroup()
        
        # State for camera intrinsics and readiness
        self.intrinsics_ready: bool = False
        self._camera_matrix: Optional[NDArray[np.float64]] = None
        self._dist_coeffs: Optional[NDArray[np.float64]] = None
        self._image_width: Optional[int] = None
        self._image_height: Optional[int] = None
        
        # Initialize components (pass logger)
        self.depth_estimator = DepthEstimator(self.config, logger=self.get_logger())
        self.interaction_computer = InteractionMatrixComputer(self.config, logger=self.get_logger())
        self.velocity_normalizer = VelocityNormalizer(self.config)
        
        # TF setup
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.tf_static_broadcaster = tf2_ros.StaticTransformBroadcaster(self)
        
        self.transform_manager = TransformManager(self.tf_buffer, self.config, logger=self.get_logger())
        
        # Cache for static gazebo camera rotation
        self._gazebo_static_sent: bool = False
        self._gazebo_quat: Optional[Tuple[float, float, float, float]] = None
        
        # State variables
        self.current_features: Optional[FeatureData] = None
        self.desired_features: Optional[FeatureData] = None
        self.last_aruco_time: Optional[RclpyTime] = None
        self.is_robot_stopped = False
        
        # Lock to guard shared state between callbacks and timer
        self._state_lock = threading.Lock()
        
        # Setup ROS interfaces
        self._setup_subscribers()
        self._setup_publishers()
        self._setup_services()
        self._setup_timers()
        
        # Send static gazebo camera transform once on startup
        self._broadcast_gazebo_camera_frame(self.get_clock().now().to_msg())
        
        self._log_initialization()
    
    def _setup_parameters(self) -> None:
        """Setup ROS parameters and update config."""
        # Declare parameters
        self.declare_parameter('lambda_gain', self.config.lambda_gain)
        self.declare_parameter('max_linear_velocity', self.config.max_linear_velocity)
        self.declare_parameter('max_angular_velocity', self.config.max_angular_velocity)
        self.declare_parameter('convergence_threshold', self.config.convergence_threshold)
        self.declare_parameter('interaction_matrix_method', self.config.interaction_matrix_method.value)
        self.declare_parameter('control_rate_hz', 100.0)
        # Periodic republish of desired features (Hz). 0 disables.
        self.declare_parameter('desired_features_republish_rate_hz', 1.0)
        
        # Get parameters and update config
        self.config.lambda_gain = self.get_parameter('lambda_gain').get_parameter_value().double_value
        self.config.max_linear_velocity = self.get_parameter('max_linear_velocity').get_parameter_value().double_value
        self.config.max_angular_velocity = self.get_parameter('max_angular_velocity').get_parameter_value().double_value
        self.config.convergence_threshold = self.get_parameter('convergence_threshold').get_parameter_value().double_value
        self.control_rate_hz = float(self.get_parameter('control_rate_hz').get_parameter_value().double_value)
        # Store republish rate
        self.desired_features_republish_rate_hz = float(self.get_parameter('desired_features_republish_rate_hz').get_parameter_value().double_value)
        
        method_str = self.get_parameter('interaction_matrix_method').get_parameter_value().string_value
        try:
            self.config.interaction_matrix_method = InteractionMatrixMethod(method_str)
        except ValueError:
            self.get_logger().warn(f'Invalid interaction_matrix_method: {method_str}. Using MEAN.')
            self.config.interaction_matrix_method = InteractionMatrixMethod.MEAN
        
        # Add parameter callback
        self.add_on_set_parameters_callback(self._parameters_callback)
    
    def _setup_subscribers(self) -> None:
        """Setup ROS subscribers."""
        self.markers_sub = self.create_subscription(
            MarkerArray, '/aruco_detections', self._markers_callback, 10,
            callback_group=self.cbgroup_sensors)
        
        self.camera_info_sub = self.create_subscription(
            CameraInfo, '/oak_camera_info', self._camera_info_callback, 10,
            callback_group=self.cbgroup_sensors)
    
    def _setup_publishers(self) -> None:
        """Setup ROS publishers."""
        self.twist_pub = self.create_publisher(
            TwistStamped, '/lbr/servo_node/delta_twist_cmds', 10)
        
        # IBVS features publisher with persistent QoS
        ibvs_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        self.ibvs_features_pub = self.create_publisher(
            IBVSDesiredFeatures, '/ibvs/desired_features', ibvs_qos)
    
    def _setup_services(self) -> None:
        """Setup ROS services."""
        self.set_desired_service = self.create_service(
            Trigger, 'set_current_as_desired', self._set_current_as_desired_callback,
            callback_group=self.cbgroup_services)
        
        self.set_manual_service = self.create_service(
            SetDesiredFeatures, 'set_manual_desired_features', self._set_manual_desired_callback,
            callback_group=self.cbgroup_services)
    
    def _setup_timers(self) -> None:
        """Setup ROS timers."""
        self.monitor_timer = self.create_timer(0.5, self._monitor_aruco_detection,
                                              callback_group=self.cbgroup_control)
        # Control loop timer
        if self.control_rate_hz <= 0:
            self.get_logger().warn('control_rate_hz must be > 0. Falling back to 100 Hz')
            self.control_rate_hz = 100.0
        control_period = 1.0 / self.control_rate_hz
        self.control_timer = self.create_timer(control_period, self._execute_ibvs_control,
                                               callback_group=self.cbgroup_control)
        
        # Periodic republish of desired features so late subscribers can latch
        try:
            rate = getattr(self, 'desired_features_republish_rate_hz', 0.0)
            if rate and rate > 0.0:
                period = 1.0 / float(rate)
                self.desired_features_timer = self.create_timer(
                    period, self._republish_desired_features, callback_group=self.cbgroup_control
                )
            else:
                self.desired_features_timer = None
        except Exception:
            # Do not fail controller if timer cannot be created
            self.desired_features_timer = None
    
    def _republish_desired_features(self) -> None:
        """Timer callback to republish last desired features for latching/visualization."""
        try:
            if self.desired_features is None:
                return
            self._publish_desired_features_msg()
        except Exception:
            pass
    
    def _log_initialization(self) -> None:
        """Log initialization information."""
        self.get_logger().info('IBVS Controller initialized with configuration:')
        self.get_logger().info(f'  Lambda gain: {self.config.lambda_gain}')
        self.get_logger().info(f'  Max velocities: {self.config.max_linear_velocity} m/s, {self.config.max_angular_velocity} rad/s')
        self.get_logger().info(f'  Convergence threshold: {self.config.convergence_threshold} pixels')
        self.get_logger().info(f'  Interaction matrix method: {self.config.interaction_matrix_method.value}')
    
    def _parameters_callback(self, params):
        """Handle runtime parameter updates."""
        from rcl_interfaces.msg import SetParametersResult
        result = SetParametersResult(successful=True)
        
        for param in params:
            try:
                if param.name == 'lambda_gain':
                    if param.value > 0:
                        old_value = self.config.lambda_gain
                        self.config.lambda_gain = param.value
                        self.get_logger().info(f'Updated lambda_gain: {old_value} -> {param.value}')
                    else:
                        result.successful = False
                        result.reason = f"lambda_gain must be positive, got {param.value}"
                        break
                        
                elif param.name == 'max_linear_velocity':
                    if param.value > 0:
                        old_value = self.config.max_linear_velocity
                        self.config.max_linear_velocity = param.value
                        self.get_logger().info(f'Updated max_linear_velocity: {old_value} -> {param.value}')
                        # Update velocity normalizer
                        self.velocity_normalizer = VelocityNormalizer(self.config)
                    else:
                        result.successful = False
                        result.reason = f"max_linear_velocity must be positive, got {param.value}"
                        break
                        
                elif param.name == 'max_angular_velocity':
                    if param.value > 0:
                        old_value = self.config.max_angular_velocity
                        self.config.max_angular_velocity = param.value
                        self.get_logger().info(f'Updated max_angular_velocity: {old_value} -> {param.value}')
                        # Update velocity normalizer
                        self.velocity_normalizer = VelocityNormalizer(self.config)
                    else:
                        result.successful = False
                        result.reason = f"max_angular_velocity must be positive, got {param.value}"
                        break
                        
                elif param.name == 'convergence_threshold':
                    if param.value > 0:
                        old_value = self.config.convergence_threshold
                        self.config.convergence_threshold = param.value
                        self.get_logger().info(f'Updated convergence_threshold: {old_value} -> {param.value}')
                    else:
                        result.successful = False
                        result.reason = f"convergence_threshold must be positive, got {param.value}"
                        break
                        
                elif param.name == 'interaction_matrix_method':
                    try:
                        old_method = self.config.interaction_matrix_method
                        self.config.interaction_matrix_method = InteractionMatrixMethod(param.value)
                        self.get_logger().info(f'Updated interaction_matrix_method: {old_method.value} -> {param.value}')
                        # Update interaction computer and restore camera matrix if we have it
                        self.interaction_computer = InteractionMatrixComputer(self.config, logger=self.get_logger())
                        if self._camera_matrix is not None:
                            self.interaction_computer.set_camera_matrix(self._camera_matrix)
                    except ValueError:
                        result.successful = False
                        result.reason = f"Invalid interaction_matrix_method: {param.value}. Must be 'current', 'desired', or 'mean'"
                        break
                        
                elif param.name == 'control_rate_hz':
                    if param.value > 0:
                        old_rate = self.control_rate_hz
                        self.control_rate_hz = param.value
                        # Update control timer
                        control_period = 1.0 / self.control_rate_hz
                        self.control_timer.cancel()
                        self.control_timer = self.create_timer(control_period, self._execute_ibvs_control)
                        self.get_logger().info(f'Updated control_rate_hz: {old_rate} -> {param.value} Hz')
                    else:
                        result.successful = False
                        result.reason = f"control_rate_hz must be positive, got {param.value}"
                        break
                        
                elif param.name == 'desired_features_republish_rate_hz':
                    if param.value >= 0:
                        old_rate = self.desired_features_republish_rate_hz
                        self.desired_features_republish_rate_hz = param.value
                        self.get_logger().info(f'Updated desired_features_republish_rate_hz: {old_rate} -> {param.value} Hz')
                        
                        # Update republish timer
                        if hasattr(self, 'desired_features_timer') and self.desired_features_timer is not None:
                            self.desired_features_timer.cancel()
                        if param.value > 0:
                            period = 1.0 / param.value
                            self.desired_features_timer = self.create_timer(
                                period, self._republish_desired_features, callback_group=self.cbgroup_control
                            )
                        else:
                            self.desired_features_timer = None
                    else:
                        result.successful = False
                        result.reason = f"desired_features_republish_rate_hz must be non-negative, got {param.value}"
                        break
                        
                else:
                    result.successful = False
                    result.reason = f"Unknown parameter: {param.name}"
                    break
                    
            except (ValueError, TypeError) as e:
                result.successful = False
                result.reason = f"Parameter validation failed for {param.name}: {e}"
                break
        
        return result
    
    def _camera_info_callback(self, msg: CameraInfo) -> None:
        """Process camera info updates."""
        K = msg.k
        camera_matrix = np.array([
            [K[0], K[1], K[2]],
            [K[3], K[4], K[5]],
            [K[6], K[7], K[8]]
        ], dtype=np.float64)
        
        # Store data and propagate to helpers
        self._camera_matrix = camera_matrix
        self._dist_coeffs = np.array(msg.d, dtype=np.float64) if len(msg.d) > 0 else None
        self._image_width = msg.width
        self._image_height = msg.height
        
        self.depth_estimator.set_camera_matrix(camera_matrix)
        self.depth_estimator.set_distortion(self._dist_coeffs)
        self.interaction_computer.set_camera_matrix(camera_matrix)
        
        # Mark ready once we have camera intrinsics
        if not self.intrinsics_ready:
            self.intrinsics_ready = True
            self.get_logger().info('Camera intrinsics received. Controller is ready.')
            self.get_logger().info(f'Camera matrix:\n{camera_matrix}')
            self.get_logger().info(f'Distortion coefficients:\n{self._dist_coeffs}')
            self.get_logger().info(f'Image width: {self._image_width}')
            self.get_logger().info(f'Image height: {self._image_height}')

    def _find_marker_id(self, msg: MarkerArray, target_id: int) -> Optional[Any]:
        """Find a marker with a specific ID in the MarkerArray."""
        for m in msg.markers:
            mid = getattr(m, 'id', getattr(m, 'marker_id', None))
            if mid == target_id:
                return m
        return None
    
    def _markers_callback(self, msg: MarkerArray) -> None:
        """Process ArUco marker detections (track marker ID 4)."""
        if not msg.markers:
            self.get_logger().warn('No ArUco markers detected')
            return
        
        # Track specific marker ID 4
        marker = self._find_marker_id(msg, 4)
        if marker is None:
            self.get_logger().warn('ArUco marker ID 4 not detected in this frame')
            return
        
        # Update detection time
        self.last_aruco_time = self.get_clock().now()
        self.is_robot_stopped = False
        
        # Extract corners from the marker message
        corners = []
        for i in range(4):
            corner = Point()
            corner.x = float(marker.corner_x[i])
            corner.y = float(marker.corner_y[i])
            corner.z = 0.0
            corners.append(corner)
        
        # Create feature data
        features = self._corners_to_features(corners)
        depth = self.depth_estimator.estimate_from_corners(corners)
        
        with self._state_lock:
            self.current_features = FeatureData(features, depth, self.last_aruco_time)
        
        # Log current features and depth like the original
        self.get_logger().info(f'Current features: {features}')
        self.get_logger().info(f'Estimated depth: {depth:.3f} m')
        
        # Compute moved to control timer
        # self._execute_ibvs_control()
    
    def _execute_ibvs_control(self) -> None:
        """Execute IBVS control loop."""
        # Suppress control during shutdown
        if getattr(self, "_shutting_down", False):
            self._stop_robot()
            return
        
        # Stop if previously flagged as stopped
        if self.is_robot_stopped:
            self._stop_robot()
            return
        
        # Check ArUco staleness before doing anything
        now_time = self.get_clock().now()
        if self.last_aruco_time is None:
            # No detections yet or cleared due to timeout
            self._stop_robot()
            return
        time_since_detection = (now_time - self.last_aruco_time).nanoseconds / 1e9
        if time_since_detection > self.config.aruco_timeout:
            if not self.is_robot_stopped:
                self.get_logger().warn(f'AruCo timeout in control loop ({time_since_detection:.2f}s) - stopping robot')
                self.is_robot_stopped = True
            self._stop_robot()
            return
        
        # Snapshot state under lock
        with self._state_lock:
            current_features = self.current_features
            desired_features = self.desired_features
        
        # Guard start until camera intrinsics / interaction matrix can be formed
        if not self.intrinsics_ready or self.interaction_computer.camera_matrix is None:
            self.get_logger().debug('Camera intrinsics not ready. Waiting before control...')
            self._stop_robot()
            return
        
        if not current_features or not desired_features:
            if not desired_features:
                self.get_logger().warn('No desired features set. Use service "set_current_as_desired" first.')
            else:
                self.get_logger().warn('Current features not available')
            # Always command zero when features are unavailable
            self._stop_robot()
            return
        
        # Compute feature error in pixels (for logging/convergence)
        feature_error = current_features.features - desired_features.features
        error_norm = np.linalg.norm(feature_error)

        # Compute feature error in normalized image coordinates for control
        # x = (u - cx) / fx, y = (v - cy) / fy -> e_norm = [(u-cu)-(u*-cu)]/fx, [(v-cy)-(v*-cy)]/fy
        # which simplifies to (u - u*)/fx and (v - v*)/fy
        if self._camera_matrix is not None:
            fx = float(self._camera_matrix[0, 0])
            fy = float(self._camera_matrix[1, 1])
            # Build a per-component scale: even indices by fx, odd by fy
            scales = np.empty(8, dtype=np.float64)
            scales[0::2] = fx
            scales[1::2] = fy
            feature_error_norm = feature_error / scales
        else:
            # Fallback: if no intrinsics, use pixel error (will be gated earlier anyway)
            feature_error_norm = feature_error

        self.get_logger().info(f'Current features: {current_features.features}')
        self.get_logger().info(f'Desired features: {desired_features.features}')
        self.get_logger().info(f'Feature error: {feature_error}')
        self.get_logger().info(f'Feature error norm: {error_norm:.3f} pixels')
        
        # Check convergence
        if error_norm < self.config.convergence_threshold:
            self.get_logger().info('Convergence reached - stopping robot')
            self._stop_robot()
            return
        
        # --- TF2 Streaming for RViz ---
        now = self.get_clock().now().to_msg()
        # Static TF (sent once) for gazebo camera frame
        self._broadcast_gazebo_camera_frame(now)
        # Dynamic TF for feature error visualization
        self._broadcast_feature_error(feature_error, now)
        
        # Compute control law
        try:
            method_name = self.config.interaction_matrix_method.value.upper()
            self.get_logger().info(f'Using {method_name} interaction matrix for IBVS control law')
            
            L_pinv = self.interaction_computer.compute_pseudoinverse(
                current_features, desired_features)
            
            # Use normalized error for consistency with interaction matrix built from normalized coords
            v_camera = -self.config.lambda_gain * (L_pinv @ feature_error_norm)
            
            self.get_logger().info(
                f'Camera velocity command: linear=({v_camera[0]:.3f}, {v_camera[1]:.3f}, {v_camera[2]:.3f}), '
                f'angular=({v_camera[3]:.3f}, {v_camera[4]:.3f}, {v_camera[5]:.3f})'
            )
            
            # Transform to end-effector frame
            v_ee = self.transform_manager.transform_velocity_to_ee_frame(v_camera)
            
            # Normalize and publish
            v_normalized = self.velocity_normalizer.normalize(v_ee)
            
            self.get_logger().debug(
                f'Normalized EE velocity: linear=({v_normalized[0]:.3f}, {v_normalized[1]:.3f}, {v_normalized[2]:.3f}), '
                f'angular=({v_normalized[3]:.3f}, {v_normalized[4]:.3f}, {v_normalized[5]:.3f})'
            )
            
            self._publish_twist_command(v_normalized)
        
        except Exception as e:
            self.get_logger().error(f'IBVS control failed: {e}')
    
    def _publish_twist_command(self, velocity: NDArray[np.float64]) -> None:
        """Publish normalized twist command."""
        # Force zero command if shutting down
        if getattr(self, "_shutting_down", False):
            velocity = np.zeros(6)
        
        twist_msg = TwistStamped()
        twist_msg.header.stamp = self.get_clock().now().to_msg()
        twist_msg.header.frame_id = self.config.ee_frame
        
        twist_msg.twist.linear.x = float(velocity[0])
        twist_msg.twist.linear.y = float(velocity[1])
        twist_msg.twist.linear.z = float(velocity[2])
        twist_msg.twist.angular.x = float(velocity[3])
        twist_msg.twist.angular.y = float(velocity[4])
        twist_msg.twist.angular.z = float(velocity[5])
        
        self.twist_pub.publish(twist_msg)
    
    def _publish_desired_features_msg(self) -> None:
        """Publish IBVSDesiredFeatures with current desired corners and center."""
        try:
            if getattr(self, 'ibvs_features_pub', None) is None:
                return
            if self.desired_features is None or self.desired_features.features is None:
                return

            msg = IBVSDesiredFeatures()

            # Extract 4 corners from desired feature vector [u1,v1,u2,v2,u3,v3,u4,v4]
            feats = list(self.desired_features.features)
            if len(feats) < 8:
                return
            u1, v1, u2, v2, u3, v3, u4, v4 = [float(feats[i]) for i in range(8)]
            msg.corners = [
                Point(x=u1, y=v1, z=0.0),
                Point(x=u2, y=v2, z=0.0),
                Point(x=u3, y=v3, z=0.0),
                Point(x=u4, y=v4, z=0.0),
            ]
            # Center (average of corners)
            cx = (u1 + u2 + u3 + u4) / 4.0
            cy = (v1 + v2 + v3 + v4) / 4.0
            if hasattr(msg, 'aruco_desired_center'):
                msg.aruco_desired_center = Point(x=float(cx), y=float(cy), z=0.0)

            self.ibvs_features_pub.publish(msg)
            if hasattr(self, 'get_logger'):
                self.get_logger().info('Published IBVS desired features (corners + center)')
        except Exception as e:
            try:
                self.get_logger().warn(f'Failed to publish IBVS desired features: {e}')
            except Exception:
                pass

    def shutdown_handler(self) -> None:
        """Handle graceful shutdown."""
        # Set shutdown flag first to prevent any further non-zero commands
        self._shutting_down = True
        self.get_logger().info('Shutting down - stopping robot')
        
        # Cancel timers to stop callbacks scheduling new control cycles
        try:
            if hasattr(self, 'control_timer') and self.control_timer is not None:
                self.control_timer.cancel()
            if hasattr(self, 'monitor_timer') and self.monitor_timer is not None:
                self.monitor_timer.cancel()
            if hasattr(self, 'desired_features_timer') and self.desired_features_timer is not None:
                self.desired_features_timer.cancel()
        except Exception as _:
            pass
        
        # Publish a few zero commands to ensure robot stops
        for _ in range(5):
            self._stop_robot()
            time.sleep(0.05)
    
    def _monitor_aruco_detection(self) -> None:
        """Monitor ArUco detection timeout."""
        if not self.last_aruco_time:
            return
        
        current_time = self.get_clock().now()
        time_since_detection = (current_time - self.last_aruco_time).nanoseconds / 1e9
        
        if time_since_detection > self.config.aruco_timeout and not self.is_robot_stopped:
            self.get_logger().warn(f'AruCo timeout ({time_since_detection:.2f}s) - stopping robot')
            self._stop_robot()
            self.is_robot_stopped = True
            # Clear stale features to avoid using outdated data
            with self._state_lock:
                self.current_features = None
    
    def _stop_robot(self) -> None:
        """Send zero velocity command to stop robot."""
        self._publish_twist_command(np.zeros(6))
    
    def _corners_to_features(self, corners: List[Point]) -> NDArray[np.float64]:
        """Convert corner points to feature vector."""
        features = np.zeros(8)
        for i, corner in enumerate(corners):
            features[2*i] = corner.x
            features[2*i + 1] = corner.y
        return features
    
    def _set_current_as_desired_callback(self, request: Trigger.Request, 
                                       response: Trigger.Response) -> Trigger.Response:
        """Set current features as desired target."""
        if not self.current_features:
            response.success = False
            response.message = "No current features available"
            return response
        
        with self._state_lock:
            self.desired_features = FeatureData(
                self.current_features.features.copy(),
                self.current_features.depth,
                self.get_clock().now()
            )
            desired_copy = FeatureData(self.desired_features.features.copy(), self.desired_features.depth, self.desired_features.timestamp)
        
        # Precompute interaction matrix for desired features
        self.interaction_computer.set_desired_features(
            desired_copy.features, desired_copy.depth)
        
        
        # Create command string for manual reproduction
        features_list = [float(f) for f in desired_copy.features]
        command_str = (
            f"ros2 service call /set_manual_desired_features visual_servo_interfaces/srv/SetDesiredFeatures "
            f'"desired_features: {features_list}"'
        )
        
        response.success = True
        response.message = (
            f"Desired features set to current position. "
            f"To set these features manually later, use:\n{command_str}"
        )
        self.get_logger().info("Desired features updated to current position")
        self.get_logger().info(f"Manual command: {command_str}")
        
        # After updating desired features and precomputing interaction matrix, publish for consumers
        try:
            self._publish_desired_features_msg()
        except Exception:
            pass
        return response
    
    def _set_manual_desired_callback(self, request: SetDesiredFeatures.Request,
                                   response: SetDesiredFeatures.Response) -> SetDesiredFeatures.Response:
        """Set desired features manually."""
        if len(request.desired_features) != 8:
            response.success = False
            response.message = f"Expected 8 features, got {len(request.desired_features)}"
            return response
        
        # Validate coordinates against camera info if available
        if self._image_width is not None and self._image_height is not None:
            w, h = self._image_width, self._image_height
            for i, coord in enumerate(request.desired_features):
                if i % 2 == 0:  # u coordinate
                    if not (0 <= coord <= w):
                        response.success = False
                        response.message = f"Invalid u coordinate: {coord}"
                        return response
                else:  # v coordinate
                    if not (0 <= coord <= h):
                        response.success = False
                        response.message = f"Invalid v coordinate: {coord}"
                        return response
        else:
            # Fallback conservative check
            for coord in request.desired_features:
                if not (0 <= coord <= 4096):
                    response.success = False
                    response.message = f"Invalid coordinate: {coord}"
                    return response
        
        features = np.array(request.desired_features, dtype=np.float64)
        corners = [Point(x=features[2*i], y=features[2*i+1], z=0.0) for i in range(4)]
        depth = self.depth_estimator.estimate_from_corners(corners)
        
        with self._state_lock:
            self.desired_features = FeatureData(features, depth, self.get_clock().now())
            desired_copy = FeatureData(self.desired_features.features.copy(), self.desired_features.depth, self.desired_features.timestamp)
        
        # Precompute interaction matrix for desired features
        self.interaction_computer.set_desired_features(
            desired_copy.features, desired_copy.depth)
        
        
        response.success = True
        response.message = "Manual desired features set successfully"
        self.get_logger().info("Manual desired features set")
        
        # After updating desired features and precomputing interaction matrix, publish for consumers
        try:
            self._publish_desired_features_msg()
        except Exception:
            pass
        return response
    
   
    def _broadcast_gazebo_camera_frame(self, timestamp) -> None:
        """
        Broadcast the gazebo camera frame with 90° Y-axis rotation from camera frame.
        """
        gazebo_tf = TransformStamped()
        gazebo_tf.header.stamp = timestamp
        gazebo_tf.header.frame_id = self.config.camera_frame # TODO: Is this static transform?
        gazebo_tf.child_frame_id = self.config.gazebo_camera_frame

        # 90° rotation around Y and 90° around Z (extrinsic XYZ)
        rotation = R.from_euler('XYZ', [-np.pi/2, np.pi/2, 0])
        gazebo_quat = rotation.as_quat()  # [x, y, z, w]
        gazebo_tf.transform.translation.x = 0.0
        gazebo_tf.transform.translation.y = 0.0
        gazebo_tf.transform.translation.z = 0.0
        gazebo_tf.transform.rotation.x = gazebo_quat[0]
        gazebo_tf.transform.rotation.y = gazebo_quat[1]
        gazebo_tf.transform.rotation.z = gazebo_quat[2]
        gazebo_tf.transform.rotation.w = gazebo_quat[3]
        self.tf_static_broadcaster.sendTransform(gazebo_tf)

    def _broadcast_feature_error(self, feature_error: NDArray[np.float64], timestamp) -> None:
        """
        Broadcast feature error for visualization as a dynamic TF.
        """
        if self.desired_features is None:
            return

        # Centroid of corner errors in pixels
        error_centroid_u = float(np.mean(feature_error[0::2]))
        error_centroid_v = float(np.mean(feature_error[1::2]))

        # Scale pixels to meters for visualization only
        scale = 0.001

        error_tf = TransformStamped()
        error_tf.header.stamp = timestamp
        error_tf.header.frame_id = self.config.gazebo_camera_frame
        error_tf.child_frame_id = "ibvs_feature_error"

        error_tf.transform.translation.x = error_centroid_u * scale
        error_tf.transform.translation.y = error_centroid_v * scale
        error_tf.transform.translation.z = 0.0

        error_tf.transform.rotation.x = 0.0
        error_tf.transform.rotation.y = 0.0
        error_tf.transform.rotation.z = 0.0
        error_tf.transform.rotation.w = 1.0

        self.tf_broadcaster.sendTransform(error_tf)


def main():
    """Main entry point."""
    rclpy.init()
    
    # Create controller with custom config if needed
    config = IBVSConfig()
    controller = IBVSController(config)
    
    # Setup signal handler
    def signal_handler(sig, frame):
        controller.shutdown_handler()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(controller)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        controller.shutdown_handler()
        executor.shutdown()
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
