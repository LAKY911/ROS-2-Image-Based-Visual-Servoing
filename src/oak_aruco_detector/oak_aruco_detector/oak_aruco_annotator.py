#!/usr/bin/env python3
"""
OAK ArUco Annotator Node

This node subscribes to rectified images from the OAK camera and annotates them
with ArUco marker visualization features. It does not perform detection itself;
it consumes detections from /aruco_detections.

Subscriptions:
  - /oak_undistorted_image (sensor_msgs/Image)
  - /aruco_detections (oak_aruco_detector_interfaces/MarkerArray)
  - /ibvs/desired_features (visual_servo_interfaces/IBVSDesiredFeatures)

Publications:
  - /oak_aruco/annotated_image (sensor_msgs/Image)

Services:
  - /oak_aruco/reset (std_srvs/Trigger): clears trajectories and start pose (will be re-set on next detection)
  - /oak_aruco/save_now (std_srvs/Trigger): saves one frame now (PNG/JPG or SVG overlay per parameter)

Hotkeys (when show_live_image=true):
  - r: reset (same as /oak_aruco/reset)
    - s: save now (saves raster PNG/JPG, SVG overlay, and original/raw frame)
  - p: toggle periodic saving (save_images)
  - q: close window

Parameters:
  - target_marker_id (int, default 4): only draw this marker's info
  - save_images (bool, default false): periodically save annotated outputs
  - save_interval (double, default 0.2): seconds between saves
  - image_format (string, default 'jpg'): 'jpg' | 'png' | 'svg' (svg saves overlay/mask only)
  - save_directory (string, default '~/aruco_sequence')
  - show_live_image (bool, default true)
  - trajectory_maxlen (int, default 5000)
  - draw_thickness (int, default 2)

Author: GitHub Copilot
"""

from typing import Deque, List, Optional, Tuple
from collections import deque
from pathlib import Path
import os
import time

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from rclpy.executors import SingleThreadedExecutor, ExternalShutdownException
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger

# Optional ArUco/IBVS interfaces
try:
    from oak_aruco_detector_interfaces.msg import MarkerArray, Marker  # type: ignore
    ARUCO_INTERFACES_AVAILABLE = True
except Exception:
    MarkerArray = None  # type: ignore
    Marker = None  # type: ignore
    ARUCO_INTERFACES_AVAILABLE = False

try:
    from visual_servo_interfaces.msg import IBVSDesiredFeatures  # type: ignore
    IBVS_INTERFACES_AVAILABLE = True
except Exception:
    IBVSDesiredFeatures = None  # type: ignore
    IBVS_INTERFACES_AVAILABLE = False


Corner = Tuple[float, float]
Corners4 = List[Corner]  # 4 ordered corners: TL, TR, BR, BL


class OakArucoAnnotator(Node):
    def __init__(self) -> None:
        super().__init__('oak_aruco_annotator')
        # Parameters
        # Visualization subscribes to the relayed BEST_EFFORT topic by default
        self.declare_parameter('input_topic', '/oak_undistorted_image/visual')
        self.declare_parameter('target_marker_id', 4)
        self.declare_parameter('save_images', False)
        self.declare_parameter('save_interval', 0.2)
        self.declare_parameter('image_format', 'png')  # 'jpg' | 'png' | 'svg'
        self.declare_parameter('save_directory', '~/aruco_sequence')
        self.declare_parameter('show_live_image', True)
        self.declare_parameter('trajectory_maxlen', 5000)
        self.declare_parameter('draw_thickness', 2)
        # Optional throttle for annotated publish rate (Hz); 0 disables throttling
        self.declare_parameter('publish_rate', 0.0)

        self.input_topic: str = self.get_parameter('input_topic').get_parameter_value().string_value
        self.target_marker_id: int = self.get_parameter('target_marker_id').get_parameter_value().integer_value
        self.save_images: bool = self.get_parameter('save_images').get_parameter_value().bool_value
        self.save_interval: float = self.get_parameter('save_interval').get_parameter_value().double_value
        self.image_format: str = self.get_parameter('image_format').get_parameter_value().string_value.lower()
        self.save_directory: str = self.get_parameter('save_directory').get_parameter_value().string_value
        self.show_live_image: bool = self.get_parameter('show_live_image').get_parameter_value().bool_value
        self.trajectory_maxlen: int = self.get_parameter('trajectory_maxlen').get_parameter_value().integer_value
        self.draw_thickness: int = self.get_parameter('draw_thickness').get_parameter_value().integer_value
        self.publish_rate: float = float(self.get_parameter('publish_rate').get_parameter_value().double_value)

        # IO / state
        self.bridge = CvBridge()
        self.last_save_time: float = 0.0

        # Detected data for target marker
        self.last_corners: Optional[Corners4] = None
        self.last_center: Optional[Corner] = None

        # Desired features (from IBVS)
        self.desired_corners: Optional[Corners4] = None
        self.desired_center: Optional[Corner] = None

        # Start pose (set on first detection after reset, or manually via service)
        self.start_corners: Optional[Corners4] = None
        self.start_center: Optional[Corner] = None

        # Corner trajectories (4 deques)
        self.trajectories: List[Deque[Corner]] = [deque(maxlen=self.trajectory_maxlen) for _ in range(4)]

        # Cached annotated image and update flag
        self.cached_annotated_image: Optional[np.ndarray] = None
        self.trajectories_updated: bool = False

        # GUI updates are driven from the main thread in main(), not by a ROS timer

        # Publisher
        self.annotated_pub = self.create_publisher(Image, '/oak_aruco/annotated_image', 10)

        # QoS Profiles
        # Visualization input is BEST_EFFORT to avoid back-pressuring camera
        sensor_qos = QoSProfile(
            depth=2,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
        )
        detection_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
        )
        # IBVS desired features: want latched last value
        ibvs_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
        )

        # Subscriptions
        self.image_sub = self.create_subscription(Image, self.input_topic, self.image_callback, sensor_qos)
        if ARUCO_INTERFACES_AVAILABLE:
            self.detections_sub = self.create_subscription(MarkerArray, '/aruco_detections', self.detections_callback, detection_qos)
        if IBVS_INTERFACES_AVAILABLE:
            self.ibvs_sub = self.create_subscription(IBVSDesiredFeatures, '/ibvs/desired_features', self.ibvs_features_callback, ibvs_qos)

        # Services
        self.reset_srv = self.create_service(Trigger, '/oak_aruco/reset', self.handle_reset)
        self.save_now_srv = self.create_service(Trigger, '/oak_aruco/save_now', self.handle_save_now)

        # Output directory
        self.out_dir = Path(os.path.expanduser(self.save_directory))
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # GUI window state
        self.window_name = 'OAK ArUco Annotator'
        self.cv_window_ready = False
        self._got_first_image = False

        self.get_logger().info(
            f"Annotating marker ID {self.target_marker_id}; input='{self.input_topic}', saving={self.save_images} format={self.image_format} dir={self.out_dir}"
        )

    # ------------------------ Callbacks ------------------------
    def gui_update_callback(self) -> None:
        """Drive OpenCV window updates and key handling on a steady timer."""
        if not self.show_live_image:
            return

        try:
            # Create the window if needed
            if not self.cv_window_ready:
                cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(self.window_name, 1280, 720)
                self.cv_window_ready = True

            # Show the latest cached image if available; otherwise draw a 'waiting' placeholder
            if self.cached_annotated_image is not None:
                cv2.imshow(self.window_name, self.cached_annotated_image)
            else:
                placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(placeholder, f"Waiting for {self.input_topic} ...", (16, 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow(self.window_name, placeholder)

            # Process one round of GUI events and handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('r'):
                self._reset_all()
                self.get_logger().info('Reset by key')
            elif key == ord('s'):
                img = self.cached_annotated_image if self.cached_annotated_image is not None else np.zeros((720, 1280, 3), dtype=np.uint8)
                self._save_both(img)
                self.get_logger().info('Saved PNG/JPG + SVG + RAW by key')
            elif key == ord('p'):
                self.save_images = not self.save_images
                self.get_logger().info(f'Toggled save_images -> {self.save_images}')
            elif key == ord('q'):
                cv2.destroyWindow(self.window_name)
                self.cv_window_ready = False
                self.show_live_image = False
                self.get_logger().info('Window closed by key')
            # Ignore other keys

        except Exception as e:
            # Disable GUI on failure (e.g., headless environment)
            self.get_logger().warn(f'GUI disabled due to error: {e}')
            self.show_live_image = False
            if self.cv_window_ready:
                try:
                    cv2.destroyWindow(self.window_name)
                except Exception:
                    pass
                self.cv_window_ready = False

    def detections_callback(self, msg) -> None:
        if not ARUCO_INTERFACES_AVAILABLE:
            return
        # Find target marker
        marker = None
        for m in msg.markers:
            if m.id == self.target_marker_id:
                marker = m
                break
        if marker is None:
            return

        # Extract 4 corners
        if len(marker.corner_x) >= 4 and len(marker.corner_y) >= 4:
            corners: Corners4 = [(float(marker.corner_x[i]), float(marker.corner_y[i])) for i in range(4)]
        else:
            return

        self.last_corners = corners
        self.last_center = self._compute_center(corners)

        # Initialize start pose if not set
        if self.start_corners is None:
            self.start_corners = [c for c in corners]
            self.start_center = self._compute_center(self.start_corners)

        # Update trajectories
        for i in range(4):
            self.trajectories[i].append(corners[i])
        
        # Mark that trajectories have been updated (for cached image invalidation)
        self.trajectories_updated = True

    def ibvs_features_callback(self, msg) -> None:
        if not IBVS_INTERFACES_AVAILABLE:
            return
        try:
            # Desired center
            if hasattr(msg, 'aruco_desired_center') and msg.aruco_desired_center is not None:
                self.desired_center = (float(msg.aruco_desired_center.x), float(msg.aruco_desired_center.y))
                self.get_logger().debug(
                    f"Desired center=({self.desired_center[0]:.1f}, {self.desired_center[1]:.1f})"
                )
            # Desired corners (expect 4)
            if hasattr(msg, 'corners') and msg.corners is not None and len(msg.corners) >= 4:
                self.desired_corners = [
                    (float(msg.corners[i].x), float(msg.corners[i].y)) for i in range(4)
                ]
        except Exception:
            # Keep running even if malformed
            pass

    def image_callback(self, msg: Image) -> None:
        # Convert image
        try:
            if msg.encoding == 'mono8':
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
            else:
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Failed to convert image: {e}')
            return

        if not self._got_first_image:
            self._got_first_image = True
            self.get_logger().info('Received first /oak_undistorted_image frame')

        # Always draw overlay on the new image, but mark if trajectories were updated
        annotated = self._create_annotated_image(cv_image)
        
        # Cache the result and reset the update flag
        self.cached_annotated_image = annotated.copy()
        self.trajectories_updated = False

        # Publish (optionally throttled)
        try:
            now_pub = time.time()
            can_publish = True
            if self.publish_rate and self.publish_rate > 0.0:
                min_dt = 1.0 / float(self.publish_rate)
                last = getattr(self, '_last_pub_time', 0.0)
                if (now_pub - last) < min_dt:
                    can_publish = False
                else:
                    self._last_pub_time = now_pub
            if can_publish:
                encoding = 'mono8' if annotated.ndim == 2 else 'bgr8'
                out_msg = self.bridge.cv2_to_imgmsg(annotated, encoding=encoding)
                out_msg.header = msg.header
                self.annotated_pub.publish(out_msg)
        except Exception as e:
            self.get_logger().warn(f'Failed to publish annotated image: {e}')

    # GUI updates and key handling are done in the GUI timer; nothing to do here for the window.

        # Periodic save
        now = time.time()
        if self.save_images and (now - self.last_save_time) >= self.save_interval:
            self.last_save_time = now
            try:
                self._save_output(annotated)
            except Exception as e:
                self.get_logger().warn(f'Failed to save image: {e}')

    # ------------------------ Services ------------------------
    def handle_reset(self, request: Trigger.Request, context) -> Trigger.Response:  # type: ignore[override]
        self._reset_all()
        resp = Trigger.Response()
        resp.success = True
        resp.message = 'Start pose and trajectories reset'
        return resp

    def handle_save_now(self, request: Trigger.Request, context) -> Trigger.Response:  # type: ignore[override]
        resp = Trigger.Response()
        try:
            # Create a simple dummy image if we don't have a live frame to draw on
            if self.last_corners is None and self.desired_corners is None and self.start_corners is None:
                # Minimal blank canvas
                img = np.zeros((720, 1280, 3), dtype=np.uint8)
            else:
                # Use last annotated frame if possible (requires storing last frame)
                img = getattr(self, '_last_frame_for_save', None)
                if img is None:
                    img = np.zeros((720, 1280, 3), dtype=np.uint8)
            self._save_output(img)
            resp.success = True
            resp.message = 'Saved one output'
        except Exception as e:
            resp.success = False
            resp.message = f'Failed to save: {e}'
        return resp

    # ------------------------ Drawing and Saving ------------------------
    def _create_annotated_image(self, image: np.ndarray) -> np.ndarray:
        # Keep a copy for service-based save if needed
        self._last_frame_for_save = image.copy()

        if image.ndim == 2:
            canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            canvas = image.copy()

        thickness = max(1, int(self.draw_thickness))

        # Colors (BGR)
        color_current = (0, 200, 0)       # green
        color_start = (200, 200, 0)       # cyan-ish/yellow
        color_desired = (200, 0, 200)     # magenta
        color_center = (0, 165, 255)      # orange
        traj_colors = [
            (0, 0, 255),   # red TL
            (0, 255, 255), # yellow TR
            (255, 0, 0),   # blue BR
            (0, 255, 0),   # green BL
        ]

        # Current marker
        if self.last_corners is not None:
            self._draw_quad(canvas, self.last_corners, color_current, thickness)
            if self.last_center is None:
                self.last_center = self._compute_center(self.last_corners)
            self._draw_point(canvas, self.last_center, color_center, 6, label='curr')

        # Start marker
        if self.start_corners is not None:
            self._draw_quad(canvas, self.start_corners, color_start, max(1, thickness - 1), dashed=False)
            if self.start_center is None:
                self.start_center = self._compute_center(self.start_corners)
            self._draw_point(canvas, self.start_center, color_start, 5, label='start')

        # Desired marker
        if self.desired_corners is not None:
            self._draw_quad(canvas, self.desired_corners, color_desired, thickness)
            if self.desired_center is not None:
                self._draw_point(canvas, self.desired_center, color_desired, 6, label='des')

        # Trajectories (per corner)
        h, w = canvas.shape[:2]
        for i, traj in enumerate(self.trajectories):
            if len(traj) >= 2:
                pts = np.array(traj, dtype=np.int32)
                pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
                pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
                cv2.polylines(canvas, [pts], isClosed=False, color=traj_colors[i % len(traj_colors)], thickness=thickness)

        # HUD text
        y0 = 24
        line_h = 18
        hud = [
            f'Marker ID: {self.target_marker_id}',
            f'Saving: {self.save_images} ({self.image_format})  dir: {self.out_dir}',
            'Keys: [r]=reset  [s]=save (png+svg+raw)  [p]=toggle save  [q]=close'
        ]
        for i, t in enumerate(hud):
            cv2.putText(canvas, t, (10, y0 + i * line_h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        return canvas

    def _save_output(self, annotated_img: np.ndarray) -> None:
        ts = time.strftime('%Y%m%d_%H%M%S')
        if self.image_format in ('png', 'jpg', 'jpeg'):
            ext = 'png' if self.image_format == 'png' else 'jpg'
            # Save annotated
            ann_path = self.out_dir / f'annotated_{ts}.{ext}'
            ok = cv2.imwrite(str(ann_path), annotated_img)
            if not ok:
                raise RuntimeError(f'cv2.imwrite failed: {ann_path}')
            # Save original/raw if available
            orig = getattr(self, '_last_frame_for_save', None)
            if orig is not None:
                raw_path = self.out_dir / f'original_{ts}.{ext}'
                ok2 = cv2.imwrite(str(raw_path), orig)
                if not ok2:
                    raise RuntimeError(f'cv2.imwrite failed: {raw_path}')
        elif self.image_format == 'svg':
            if annotated_img.ndim == 2:
                h, w = annotated_img.shape
            else:
                h, w = annotated_img.shape[:2]
            svg = self._build_svg_overlay(width=w, height=h)
            path = self.out_dir / f'overlay_{ts}.svg'
            path.write_text(svg)
        else:
            raise ValueError(f'Unsupported image_format: {self.image_format}')

    def _save_both(self, annotated_img: np.ndarray) -> None:
        """Save raster (png/jpg), SVG overlay, and the original/raw frame with a shared timestamp."""
        ts = time.strftime('%Y%m%d_%H%M%S')
        # Raster ext preference: honor parameter when it's png/jpg, otherwise default to png
        if self.image_format in ('jpg', 'jpeg'):
            raster_ext = 'jpg'
        else:
            raster_ext = 'png' if self.image_format == 'png' else 'png'
        # Save raster annotated
        raster_path = self.out_dir / f'annotated_{ts}.{raster_ext}'
        ok = cv2.imwrite(str(raster_path), annotated_img)
        if not ok:
            raise RuntimeError(f'cv2.imwrite failed: {raster_path}')
        # Save original/raw if available
        orig = getattr(self, '_last_frame_for_save', None)
        if orig is not None:
            raw_path = self.out_dir / f'original_{ts}.{raster_ext}'
            ok2 = cv2.imwrite(str(raw_path), orig)
            if not ok2:
                raise RuntimeError(f'cv2.imwrite failed: {raw_path}')
        # Save SVG overlay
        if annotated_img.ndim == 2:
            h, w = annotated_img.shape
        else:
            h, w = annotated_img.shape[:2]
        svg = self._build_svg_overlay(width=w, height=h)
        svg_path = self.out_dir / f'overlay_{ts}.svg'
        svg_path.write_text(svg)

    def _build_svg_overlay(self, width: int, height: int) -> str:
        # Minimal SVG with trajectories, start/current/desired quads and centers
        def color_hex(bgr: Tuple[int, int, int]) -> str:
            b, g, r = bgr
            return f'#{r:02x}{g:02x}{b:02x}'

        color_current = (0, 200, 0)
        color_start = (200, 200, 0)
        color_desired = (200, 0, 200)
        traj_colors = [
            (0, 0, 255),
            (0, 255, 255),
            (255, 0, 0),
            (0, 255, 0),
        ]
        stroke_w = max(1, int(self.draw_thickness))

        parts: List[str] = [
            f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
            # Transparent background
            f"<rect width='{width}' height='{height}' fill='none' />"
        ]

        def svg_poly(points: Corners4, color: Tuple[int, int, int], dashed: bool = False) -> str:
            pts = ' '.join(f"{x:.1f},{y:.1f}" for (x, y) in points)
            dash = " stroke-dasharray='6,4'" if dashed else ''
            return f"<polygon points='{pts}' fill='none' stroke='{color_hex(color)}' stroke-width='{stroke_w}'{dash} />"

        def svg_circle(pt: Corner, color: Tuple[int, int, int], r: float = 3.0) -> str:
            return f"<circle cx='{pt[0]:.1f}' cy='{pt[1]:.1f}' r='{r:.1f}' fill='{color_hex(color)}' />"

        # Start
        if self.start_corners is not None:
            parts.append(svg_poly(self.start_corners, color_start, dashed=False))
            for c in self.start_corners:
                parts.append(svg_circle(c, color_start, r=2.5))
            if self.start_center is None:
                self.start_center = self._compute_center(self.start_corners)
            parts.append(svg_circle(self.start_center, color_start, r=3.5))

        # Desired
        if self.desired_corners is not None:
            parts.append(svg_poly(self.desired_corners, color_desired, dashed=False))
            for c in self.desired_corners:
                parts.append(svg_circle(c, color_desired, r=2.5))
            if self.desired_center is not None:
                parts.append(svg_circle(self.desired_center, color_desired, r=3.5))

        # Current
        if self.last_corners is not None:
            parts.append(svg_poly(self.last_corners, color_current, dashed=False))
            for c in self.last_corners:
                parts.append(svg_circle(c, color_current, r=2.5))
            if self.last_center is None:
                self.last_center = self._compute_center(self.last_corners)
            parts.append(svg_circle(self.last_center, color_current, r=3.5))

        # Trajectories
        for i, traj in enumerate(self.trajectories):
            if len(traj) >= 2:
                d = 'M ' + ' L '.join(f"{x:.1f},{y:.1f}" for (x, y) in traj)
                parts.append(
                    f"<path d='{d}' fill='none' stroke='{color_hex(traj_colors[i % len(traj_colors)])}' stroke-width='{stroke_w}' />"
                )

        parts.append('</svg>')
        return '\n'.join(parts)

    # ------------------------ Helpers ------------------------
    @staticmethod
    def _compute_center(corners: Corners4) -> Corner:
        xs = [c[0] for c in corners]
        ys = [c[1] for c in corners]
        return (float(sum(xs) / 4.0), float(sum(ys) / 4.0))

    @staticmethod
    def _draw_point(img: np.ndarray, pt: Corner, color: Tuple[int, int, int], size: int = 5, label: Optional[str] = None) -> None:
        cv2.circle(img, (int(pt[0]), int(pt[1])), size, color, -1, lineType=cv2.LINE_AA)
        if label:
            cv2.putText(img, label, (int(pt[0]) + 6, int(pt[1]) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    @staticmethod
    def _draw_corners(img: np.ndarray, corners: Corners4, color: Tuple[int, int, int], thickness: int) -> None:
        # No-op: remove 'X' markers at corners
        return

    @staticmethod
    def _draw_quad(img: np.ndarray, corners: Corners4, color: Tuple[int, int, int], thickness: int, dashed: bool = False) -> None:
        pts = [(int(c[0]), int(c[1])) for c in corners]
        # Order TL,TR,BR,BL -> close back to TL
        seq = [0, 1, 2, 3, 0]
        for i in range(4):
            p1 = pts[seq[i]]
            p2 = pts[seq[i + 1]]
            if dashed:
                OakArucoAnnotator._draw_dashed_line(img, p1, p2, color, thickness, dash_len=8, gap_len=6)
            else:
                cv2.line(img, p1, p2, color, thickness, lineType=cv2.LINE_AA)

    @staticmethod
    def _draw_dashed_line(img: np.ndarray, p1: Tuple[int, int], p2: Tuple[int, int], color: Tuple[int, int, int], thickness: int, dash_len: int = 10, gap_len: int = 5) -> None:
        # Draw dashed line between p1 and p2 without changing input types
        p1_arr = np.array(p1, dtype=float)
        p2_arr = np.array(p2, dtype=float)
        d = p2_arr - p1_arr
        length = np.hypot(d[0], d[1])
        if length < 1e-3:
            return
        dir_vec = d / length
        n_dashes = int(length // (dash_len + gap_len)) + 1
        for i in range(n_dashes):
            start = p1_arr + dir_vec * (i * (dash_len + gap_len))
            end = start + dir_vec * dash_len
            s = (int(start[0]), int(start[1]))
            e = (int(min(end[0], p2_arr[0])), int(min(end[1], p2_arr[1])))
            cv2.line(img, s, e, color, thickness, lineType=cv2.LINE_AA)

    def _reset_all(self) -> None:
        self.start_corners = None
        self.start_center = None
        self.last_corners = None
        self.last_center = None
        for t in self.trajectories:
            t.clear()
        self.trajectories_updated = True  # Mark for redraw after reset


def main(args=None) -> None:
    rclpy.init(args=args)
    node = OakArucoAnnotator()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        # Interleave ROS callbacks with GUI updates on the same thread to satisfy Qt/HighGUI constraints
        while rclpy.ok():
            executor.spin_once(timeout_sec=0.01)
            # Drive GUI updates and key handling here (main thread)
            try:
                node.gui_update_callback()
            except Exception:
                # Keep spinning even if GUI fails once
                pass
    except KeyboardInterrupt:
        pass
    except ExternalShutdownException:
        # Shutdown requested elsewhere (e.g., ros2 lifecycle); exit gracefully
        pass
    finally:
        executor.shutdown()
        if getattr(node, 'cv_window_ready', False):
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            # Avoid raising if shutdown was already called elsewhere
            pass


if __name__ == '__main__':
    main()
