#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <QApplication>
#include <QTimer>
#include "joystick_gui_cpp/joystick_widget.hpp"

class JoystickNode : public rclcpp::Node {
public:
  JoystickNode()
      : Node("joystick_node") {
    // Declare parameters with default values
    this->declare_parameter<std::string>("topic_name", "cmd_vel");
    this->declare_parameter<double>("max_linear_velocity", 1.0);
    this->declare_parameter<double>("max_angular_velocity", 1.0);
    this->declare_parameter<bool>("invert_angular", false);
    this->declare_parameter<std::string>("message_type", "twist");  // New parameter

    // Get parameter values
    std::string topic_name = this->get_parameter("topic_name").as_string();
    max_linear_velocity_ = this->get_parameter("max_linear_velocity").as_double();
    max_angular_velocity_ = this->get_parameter("max_angular_velocity").as_double();
    invert_angular_ = this->get_parameter("invert_angular").as_bool();
    message_type_ = this->get_parameter("message_type").as_string();  // Get message_type

    // Create publishers based on message_type
    if (message_type_ == "twist") {
      twist_publisher_ = this->create_publisher<geometry_msgs::msg::Twist>(topic_name, 10);
    } else if (message_type_ == "twist_stamped") {
      twist_stamped_publisher_ = this->create_publisher<geometry_msgs::msg::TwistStamped>(topic_name, 10);
    } else {
      RCLCPP_ERROR(this->get_logger(), "Invalid message_type parameter. Must be 'twist' or 'twist_stamped'.");
      throw std::runtime_error("Invalid message_type parameter.");
    }

    // Initialize the GUI widget
    widget_ = new JoystickWidget(max_linear_velocity_, max_angular_velocity_, invert_angular_);
    widget_->setWindowTitle("ROS2 Joystick");
    widget_->show();

    // Connect the joystickMoved signal to the publishValues method
    QObject::connect(widget_, &JoystickWidget::joystickMoved,
                     [this](float angular_x, float angular_y) {
                       this->publishValues(angular_x, angular_y);
                     });

    // Connect the new axis values signal for complete control
    QObject::connect(widget_, &JoystickWidget::axisValuesChanged,
                     [this](float linear_x, float linear_y, float linear_z, float angular_x, float angular_y, float angular_z) {
                       this->publishCompleteValues(linear_x, linear_y, linear_z, angular_x, angular_y, angular_z);
                     });

    // Connect signals to update parameters when max velocities change
    QObject::connect(widget_, &JoystickWidget::maxLinearVelocityChanged,
                     [this](double value) {
                       this->set_parameter(rclcpp::Parameter("max_linear_velocity", value));
                     });
    QObject::connect(widget_, &JoystickWidget::maxAngularVelocityChanged,
                     [this](double value) {
                       this->set_parameter(rclcpp::Parameter("max_angular_velocity", value));
                     });

    // Set up parameter callback
    parameter_callback_handle_ = this->add_on_set_parameters_callback(
        std::bind(&JoystickNode::onParameterChanged, this, std::placeholders::_1));

    // Use a QTimer to spin ROS2 periodically
    ros_timer_ = new QTimer(nullptr);
    QObject::connect(ros_timer_, &QTimer::timeout, [this]() {
      rclcpp::spin_some(this->shared_from_this());
    });
    ros_timer_->start(10);  // Spin ROS2 every 10 milliseconds
  }

  ~JoystickNode() {
    twist_publisher_.reset();
    twist_stamped_publisher_.reset();
    delete ros_timer_;
    delete widget_;
  }

  void publishValues(float angular_x, float angular_y) {
    if (message_type_ == "twist") {
      auto msg = geometry_msgs::msg::Twist();
      msg.angular.x = angular_x;
      msg.angular.y = angular_y;
      twist_publisher_->publish(msg);
    } else if (message_type_ == "twist_stamped") {
      auto msg = geometry_msgs::msg::TwistStamped();
      msg.header.stamp = this->get_clock()->now();
      msg.header.frame_id = "lbr_link_ee";  // frame_id
      msg.twist.angular.x = angular_x;
      msg.twist.angular.y = angular_y;
      twist_stamped_publisher_->publish(msg);
    }
  }

  void publishCompleteValues(float linear_x, float linear_y, float linear_z, 
                            float angular_x, float angular_y, float angular_z) {
    if (message_type_ == "twist") {
      auto msg = geometry_msgs::msg::Twist();
      msg.linear.x = linear_x;
      msg.linear.y = linear_y;
      msg.linear.z = linear_z;
      msg.angular.x = angular_x;
      msg.angular.y = angular_y;
      msg.angular.z = angular_z;
      twist_publisher_->publish(msg);
    } else if (message_type_ == "twist_stamped") {
      auto msg = geometry_msgs::msg::TwistStamped();
      msg.header.stamp = this->get_clock()->now();
      msg.header.frame_id = "lbr_link_ee";  // frame_id
      msg.twist.linear.x = linear_x;
      msg.twist.linear.y = linear_y;
      msg.twist.linear.z = linear_z;
      msg.twist.angular.x = angular_x;
      msg.twist.angular.y = angular_y;
      msg.twist.angular.z = angular_z;
      twist_stamped_publisher_->publish(msg);
    }
  }

private:
  // Publishers
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr twist_publisher_;
  rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr twist_stamped_publisher_;

  JoystickWidget *widget_;
  QTimer *ros_timer_;

  double max_linear_velocity_;
  double max_angular_velocity_;
  bool invert_angular_;
  std::string message_type_;

  OnSetParametersCallbackHandle::SharedPtr parameter_callback_handle_;

  rcl_interfaces::msg::SetParametersResult onParameterChanged(const std::vector<rclcpp::Parameter> &parameters) {
    rcl_interfaces::msg::SetParametersResult result;
    result.successful = true;

    for (const auto &param : parameters) {
      if (param.get_name() == "max_linear_velocity") {
        max_linear_velocity_ = param.as_double();
        widget_->setMaxLinearVelocity(max_linear_velocity_);
      } else if (param.get_name() == "max_angular_velocity") {
        max_angular_velocity_ = param.as_double();
        widget_->setMaxAngularVelocity(max_angular_velocity_);
      } else if (param.get_name() == "invert_angular") {
        invert_angular_ = param.as_bool();
        // Update the widget's invert_angular flag
        widget_->invert_angular_ = invert_angular_;
        widget_->emitValues();  // Update values with new inversion
      } else if (param.get_name() == "topic_name") {
        // Topic name change not supported at runtime
        result.successful = false;
        result.reason = "Changing topic_name at runtime is not supported.";
      } else if (param.get_name() == "message_type") {
        // Changing message_type at runtime is not supported
        result.successful = false;
        result.reason = "Changing message_type at runtime is not supported.";
      }
    }

    return result;
  }
};

int main(int argc, char *argv[]) {
  // Initialize ROS2
  rclcpp::init(argc, argv);

  // Initialize Qt Application
  QApplication app(argc, argv);

  // Create the ROS2 node
  auto node = std::make_shared<JoystickNode>();

  // Start the Qt event loop
  int result = app.exec();

  // Shutdown ROS2 after the Qt event loop exits
  rclcpp::shutdown();

  return result;
}
