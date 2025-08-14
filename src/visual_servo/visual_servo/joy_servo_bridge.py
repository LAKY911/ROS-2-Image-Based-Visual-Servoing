#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import array
import threading

from sensor_msgs.msg import Joy
from geometry_msgs.msg import TwistStamped
from control_msgs.msg import JointJog
from std_srvs.srv import Trigger

# Enums for button names -> axis/button array index
class Axis:
    LEFT_STICK_X = 0
    LEFT_STICK_Y = 1
    LEFT_TRIGGER = 5
    RIGHT_STICK_X = 2
    RIGHT_STICK_Y = 3
    RIGHT_TRIGGER = 4
    D_PAD_X = 6
    D_PAD_Y = 7

class Button:
    A = 0
    B = 1
    X = 3
    Y = 4
    LEFT_BUMPER = 6
    RIGHT_BUMPER = 7
    SELECT = 10
    START = 11
    LEFT_STICK_CLICK = 13
    RIGHT_STICK_CLICK = 14

class JoyServoBridge(Node):
    def __init__(self):
        super().__init__('joy_servo_bridge')

        # Constants
        self.JOY_TOPIC = "/joy"
        self.TWIST_TOPIC = "/lbr/servo_node/delta_twist_cmds"
        self.JOINT_TOPIC = "/lbr/servo_node/delta_joint_cmds"
        self.EEF_FRAME_ID = "lbr_link_ee"
        self.BASE_FRAME_ID = "lbr_link_0"
        
        # Some axes have offsets (default trigger position is 1.0 not 0)
        self.AXIS_DEFAULTS = {
            Axis.LEFT_TRIGGER: 1.0,
            Axis.RIGHT_TRIGGER: 1.0
        }

        # Current frame for publishing commands
        self.frame_to_publish = self.BASE_FRAME_ID

        # Define joint names for the KUKA LBR robot
        self.joint_names = [
            'lbr_A1',
            'lbr_A2',
            'lbr_A3',
            'lbr_A4',
            'lbr_A5',
            'lbr_A6',
            'lbr_A7'
        ]

        # Setup pub/sub
        self.subscription = self.create_subscription(
            Joy,
            self.JOY_TOPIC,
            self.joy_callback,
            10
        )

        self.twist_pub = self.create_publisher(
            TwistStamped,
            self.TWIST_TOPIC,
            10
        )

        self.joint_pub = self.create_publisher(
            JointJog,
            self.JOINT_TOPIC,
            10
        )

        # Create a service client to start the ServoNode
        self.servo_start_client = self.create_client(Trigger, '/lbr/servo_node/start_servo')
        
        # Start the servo node if available (with a timeout)
        if self.servo_start_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Starting servo node...")
            self.servo_start_client.call_async(Trigger.Request())
        else:
            self.get_logger().warn("Servo start service not available")

        self.get_logger().info("Joystick to Servo Bridge Node Started")

    def update_cmd_frame(self, buttons):
        """Updates the command frame based on button presses"""
        # First check if the button indices are within range
        if (Button.SELECT < len(buttons) and 
            Button.START < len(buttons)):
            
            if buttons[Button.SELECT] and self.frame_to_publish == self.EEF_FRAME_ID:
                self.frame_to_publish = self.BASE_FRAME_ID
                self.get_logger().info(f"Command frame changed to {self.frame_to_publish}")
            elif buttons[Button.START] and self.frame_to_publish == self.BASE_FRAME_ID:
                self.frame_to_publish = self.EEF_FRAME_ID
                self.get_logger().info(f"Command frame changed to {self.frame_to_publish}")
        else:
            # Log a warning only once to avoid spamming
            if not hasattr(self, '_button_warning_logged'):
                self.get_logger().warn(f"Button indices out of range. Controller has {len(buttons)} buttons, " 
                                      f"but expected indices SELECT={Button.SELECT} and START={Button.START}")
                self._button_warning_logged = True

    def convert_joy_to_cmd(self, axes, buttons):
        """
        Converts joystick input to twist or joint commands
        Returns: (publish_twist_not_joint, twist_msg, joint_msg)
        """
        # Create the messages we might publish
        twist_msg = TwistStamped()
        joint_msg = JointJog()
        
        # Check if button/axis indices are within range
        max_button_needed = max(Button.A, Button.B, Button.X, Button.Y, 
                                Button.LEFT_BUMPER, Button.RIGHT_BUMPER)
        max_axis_needed = max(Axis.D_PAD_X, Axis.D_PAD_Y, 
                              Axis.RIGHT_STICK_X, Axis.RIGHT_STICK_Y, 
                              Axis.LEFT_STICK_X, Axis.LEFT_STICK_Y,
                              Axis.LEFT_TRIGGER, Axis.RIGHT_TRIGGER)
                              
        if len(buttons) <= max_button_needed or len(axes) <= max_axis_needed:
            # Not enough buttons/axes, return default twist command
            if not hasattr(self, '_controller_warning_logged'):
                self.get_logger().warn(
                    f"Controller doesn't have enough buttons/axes. Has {len(buttons)} buttons and {len(axes)} axes, "
                    f"needs at least {max_button_needed+1} buttons and {max_axis_needed+1} axes."
                )
                self._controller_warning_logged = True
            return True, twist_msg, joint_msg
        
        # Give joint jogging priority because it is only buttons
        if (buttons[Button.A] or buttons[Button.B] or buttons[Button.X] or buttons[Button.Y] or 
           abs(axes[Axis.D_PAD_X]) > 0.1 or abs(axes[Axis.D_PAD_Y]) > 0.1):
            
            # Map the D_PAD to the proximal joints
            joint_msg.joint_names.append(self.joint_names[0])  # lbr_A1
            joint_msg.velocities.append(axes[Axis.D_PAD_X])
            joint_msg.joint_names.append(self.joint_names[1])  # lbr_A2
            joint_msg.velocities.append(axes[Axis.D_PAD_Y])
            
            # Map the diamond to the distal joints
            joint_msg.joint_names.append(self.joint_names[3])  # lbr_A4
            joint_msg.velocities.append(float(buttons[Button.B]) - float(buttons[Button.X]))
            joint_msg.joint_names.append(self.joint_names[5])  # lbr_A6
            joint_msg.velocities.append(float(buttons[Button.Y]) - float(buttons[Button.A]))
            
            return False, twist_msg, joint_msg
        
        # The bread and butter: map buttons to twist commands
        twist_msg.twist.linear.z = axes[Axis.RIGHT_STICK_Y]
        twist_msg.twist.linear.y = axes[Axis.RIGHT_STICK_X]
        
        lin_x_right = -0.5 * (axes[Axis.RIGHT_TRIGGER] - self.AXIS_DEFAULTS.get(Axis.RIGHT_TRIGGER, 0.0))
        lin_x_left = 0.5 * (axes[Axis.LEFT_TRIGGER] - self.AXIS_DEFAULTS.get(Axis.LEFT_TRIGGER, 0.0))
        twist_msg.twist.linear.x = lin_x_right + lin_x_left
        
        twist_msg.twist.angular.y = axes[Axis.LEFT_STICK_Y]
        twist_msg.twist.angular.x = axes[Axis.LEFT_STICK_X]
        
        roll_positive = float(buttons[Button.RIGHT_BUMPER])
        roll_negative = -1.0 * float(buttons[Button.LEFT_BUMPER])
        twist_msg.twist.angular.z = roll_positive + roll_negative
        
        return True, twist_msg, joint_msg
        
    def joy_callback(self, msg: Joy):
        # First update the command frame
        self.update_cmd_frame(msg.buttons)
        
        # Convert joystick input to commands
        publish_twist, twist_msg, joint_msg = self.convert_joy_to_cmd(msg.axes, msg.buttons)
        
        if publish_twist:
            # Publish the TwistStamped
            twist_msg.header.frame_id = self.frame_to_publish
            twist_msg.header.stamp = self.get_clock().now().to_msg()
            self.twist_pub.publish(twist_msg)
        else:
            # Publish the JointJog - make sure we use array.array with 'd' typecode
            joint_msg.header.stamp = self.get_clock().now().to_msg()
            joint_msg.header.frame_id = "lbr_link_3"  # Middle frame of the robot arm
            
            # Convert velocities to the right type
            velocities = array.array('d', joint_msg.velocities)
            joint_msg.velocities = velocities
            
            # Set displacements to NaN (only velocity control)
            joint_msg.displacements = array.array('d', [float('nan')] * len(joint_msg.joint_names))
            
            # Duration of the command
            joint_msg.duration = 0.1
            
            self.joint_pub.publish(joint_msg)

def main(args=None):
    rclpy.init(args=args)
    node = JoyServoBridge()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
