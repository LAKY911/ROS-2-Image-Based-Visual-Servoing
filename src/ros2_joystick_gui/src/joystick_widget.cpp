#include "joystick_gui_cpp/joystick_widget.hpp"
#include <QPainter>
#include <QMouseEvent>
#include <QGridLayout>
#include <cmath>

JoystickWidget::JoystickWidget(double init_max_linear, double init_max_angular, bool invert_angular, QWidget *parent)
    : QWidget(parent),
      outer_radius_(100.0f),
      inner_radius_(20.0f),
      is_dragging_(false),
      max_linear_velocity_(init_max_linear),
      max_angular_velocity_(init_max_angular),
      invert_angular_(invert_angular),
      linear_x_value_(0.0f),
      linear_y_value_(0.0f),
      linear_z_value_(0.0f),
      angular_x_value_(0.0f),
      angular_y_value_(0.0f),
      angular_z_value_(0.0f),
      step_size_(0.1) {
  setFixedSize(500, 700);  // Increased size for additional controls

  center_ = QPointF(width() / 2, 150);  // Adjusted center position
  handle_pos_ = center_;

  // Create labels
  linear_label_ = new QLabel("Linear Velocity: 0.0 m/s", this);
  angular_label_ = new QLabel("Angular Velocity: 0.0 rad/s", this);

  // Create spin boxes for max velocities
  max_linear_spinbox_ = new QDoubleSpinBox(this);
  max_linear_spinbox_->setRange(0.1, 10.0);
  max_linear_spinbox_->setValue(max_linear_velocity_);
  max_linear_spinbox_->setSingleStep(0.1);
  max_linear_spinbox_->setPrefix("Max Linear: ");
  max_linear_spinbox_->setSuffix(" m/s");

  max_angular_spinbox_ = new QDoubleSpinBox(this);
  max_angular_spinbox_->setRange(0.1, 10.0);
  max_angular_spinbox_->setValue(max_angular_velocity_);
  max_angular_spinbox_->setSingleStep(0.1);
  max_angular_spinbox_->setPrefix("Max Angular: ");
  max_angular_spinbox_->setSuffix(" rad/s");

  // Connect spin boxes to slots and emit signals
  connect(max_linear_spinbox_, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
          this, &JoystickWidget::updateMaxLinear);
  connect(max_angular_spinbox_, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
          this, &JoystickWidget::updateMaxAngular);

  // Create additional axis control buttons
  linear_x_plus_ = new QPushButton("Linear X +", this);
  linear_x_minus_ = new QPushButton("Linear X -", this);
  linear_y_plus_ = new QPushButton("Linear Y +", this);
  linear_y_minus_ = new QPushButton("Linear Y -", this);
  linear_z_plus_ = new QPushButton("Linear Z +", this);
  linear_z_minus_ = new QPushButton("Linear Z -", this);
  angular_x_plus_ = new QPushButton("Angular X +", this);
  angular_x_minus_ = new QPushButton("Angular X -", this);
  angular_y_plus_ = new QPushButton("Angular Y +", this);
  angular_y_minus_ = new QPushButton("Angular Y -", this);
  angular_z_plus_ = new QPushButton("Angular Z +", this);
  angular_z_minus_ = new QPushButton("Angular Z -", this);
  reset_button_ = new QPushButton("Reset All Axes", this);

  // Create labels for current axis values
  linear_x_label_ = new QLabel("Linear X: 0.00 m/s", this);
  linear_y_label_ = new QLabel("Linear Y: 0.00 m/s", this);
  linear_z_label_ = new QLabel("Linear Z: 0.00 m/s", this);
  angular_x_label_ = new QLabel("Angular X: 0.00 rad/s", this);
  angular_y_label_ = new QLabel("Angular Y: 0.00 rad/s", this);
  angular_z_label_ = new QLabel("Angular Z: 0.00 rad/s", this);

  // Connect button signals
  connect(linear_x_plus_, &QPushButton::clicked, this, &JoystickWidget::onLinearXPlus);
  connect(linear_x_minus_, &QPushButton::clicked, this, &JoystickWidget::onLinearXMinus);
  connect(linear_y_plus_, &QPushButton::clicked, this, &JoystickWidget::onLinearYPlus);
  connect(linear_y_minus_, &QPushButton::clicked, this, &JoystickWidget::onLinearYMinus);
  connect(linear_z_plus_, &QPushButton::clicked, this, &JoystickWidget::onLinearZPlus);
  connect(linear_z_minus_, &QPushButton::clicked, this, &JoystickWidget::onLinearZMinus);
  connect(angular_x_plus_, &QPushButton::clicked, this, &JoystickWidget::onAngularXPlus);
  connect(angular_x_minus_, &QPushButton::clicked, this, &JoystickWidget::onAngularXMinus);
  connect(angular_y_plus_, &QPushButton::clicked, this, &JoystickWidget::onAngularYPlus);
  connect(angular_y_minus_, &QPushButton::clicked, this, &JoystickWidget::onAngularYMinus);
  connect(angular_z_plus_, &QPushButton::clicked, this, &JoystickWidget::onAngularZPlus);
  connect(angular_z_minus_, &QPushButton::clicked, this, &JoystickWidget::onAngularZMinus);
  connect(reset_button_, &QPushButton::clicked, this, &JoystickWidget::resetAllAxes);

  // Layout widgets
  QVBoxLayout *main_layout = new QVBoxLayout(this);
  main_layout->setAlignment(Qt::AlignTop);

  // Add the joystick area (we'll leave it as part of the widget's paint event)
  QWidget *joystick_area = new QWidget(this);
  joystick_area->setFixedSize(500, 300);

  main_layout->addWidget(joystick_area);

  // Add labels and spin boxes
  main_layout->addWidget(linear_label_);
  main_layout->addWidget(angular_label_);
  main_layout->addWidget(max_linear_spinbox_);
  main_layout->addWidget(max_angular_spinbox_);

  // Create grid layout for axis control buttons
  QGridLayout *button_layout = new QGridLayout();
  
  // Linear controls
  button_layout->addWidget(new QLabel("Linear X:"), 0, 0);
  button_layout->addWidget(linear_x_minus_, 0, 1);
  button_layout->addWidget(linear_x_plus_, 0, 2);
  button_layout->addWidget(linear_x_label_, 0, 3);
  
  button_layout->addWidget(new QLabel("Linear Y:"), 1, 0);
  button_layout->addWidget(linear_y_minus_, 1, 1);
  button_layout->addWidget(linear_y_plus_, 1, 2);
  button_layout->addWidget(linear_y_label_, 1, 3);
  
  button_layout->addWidget(new QLabel("Linear Z:"), 2, 0);
  button_layout->addWidget(linear_z_minus_, 2, 1);
  button_layout->addWidget(linear_z_plus_, 2, 2);
  button_layout->addWidget(linear_z_label_, 2, 3);
  
  // Angular controls
  button_layout->addWidget(new QLabel("Angular X (Pitch):"), 3, 0);
  button_layout->addWidget(angular_x_minus_, 3, 1);
  button_layout->addWidget(angular_x_plus_, 3, 2);
  button_layout->addWidget(angular_x_label_, 3, 3);
  
  button_layout->addWidget(new QLabel("Angular Y (Roll):"), 4, 0);
  button_layout->addWidget(angular_y_minus_, 4, 1);
  button_layout->addWidget(angular_y_plus_, 4, 2);
  button_layout->addWidget(angular_y_label_, 4, 3);
  
  button_layout->addWidget(new QLabel("Angular Z (Yaw):"), 5, 0);
  button_layout->addWidget(angular_z_minus_, 5, 1);
  button_layout->addWidget(angular_z_plus_, 5, 2);
  button_layout->addWidget(angular_z_label_, 5, 3);

  // Reset button
  button_layout->addWidget(reset_button_, 6, 0, 1, 4); // Span across all columns

  QWidget *button_widget = new QWidget();
  button_widget->setLayout(button_layout);
  main_layout->addWidget(button_widget);

  setLayout(main_layout);
}

JoystickWidget::~JoystickWidget() {}

void JoystickWidget::paintEvent(QPaintEvent *event) {
  Q_UNUSED(event);
  QPainter painter(this);

  // Draw outer circle
  painter.setPen(QPen(Qt::black, 2));
  painter.drawEllipse(center_, outer_radius_, outer_radius_);

  // Draw inner circle (joystick handle)
  painter.setBrush(QBrush(QColor(200, 0, 0)));
  painter.drawEllipse(handle_pos_, inner_radius_, inner_radius_);
}

void JoystickWidget::mousePressEvent(QMouseEvent *event) {
  if (QLineF(event->pos(), handle_pos_).length() < inner_radius_) {
    is_dragging_ = true;
  }
}

void JoystickWidget::mouseMoveEvent(QMouseEvent *event) {
  if (is_dragging_) {
    QPointF offset = event->pos() - center_;
    float distance = std::sqrt(offset.x() * offset.x() + offset.y() * offset.y());
    if (distance > outer_radius_) {
      // Constrain within the outer circle
      float angle = std::atan2(offset.y(), offset.x());
      offset.setX(outer_radius_ * std::cos(angle));
      offset.setY(outer_radius_ * std::sin(angle));
    }
    handle_pos_ = center_ + offset;
    update();
    emitValues();
  }
}

void JoystickWidget::mouseReleaseEvent(QMouseEvent *event) {
  Q_UNUSED(event);
  is_dragging_ = false;
  handle_pos_ = center_;  // Reset to center
  update();
  emitValues();
}

void JoystickWidget::emitValues() {
  // Calculate normalized values from -1 to 1
  float dx = (handle_pos_.x() - center_.x()) / outer_radius_;
  float dy = (center_.y() - handle_pos_.y()) / outer_radius_;

  // Invert angular direction if required
  if (invert_angular_) {
    dx = -dx;
  }

  // Scale with max velocities - these are for the main joystick
  float angular_x_velocity = dy * max_angular_velocity_;  // Pitch (up/down)
  float angular_y_velocity = dx * max_angular_velocity_;  // Roll (left/right)

  // Update the angular_x and angular_y from joystick (main controls)
  angular_x_value_ = angular_x_velocity;
  angular_y_value_ = angular_y_velocity;

  // Update labels
  linear_label_->setText(QString("Angular X (Pitch): %1 rad/s").arg(angular_x_velocity, 0, 'f', 2));
  angular_label_->setText(QString("Angular Y (Roll): %1 rad/s").arg(angular_y_velocity, 0, 'f', 2));

  // Emit the scaled values (legacy signal) - now for angular axes
  emit joystickMoved(angular_x_velocity, angular_y_velocity);
  
  // Emit complete axis values
  emitAxisValues();
}

void JoystickWidget::updateMaxLinear(double value) {
  max_linear_velocity_ = value;
  emit maxLinearVelocityChanged(value);
}

void JoystickWidget::updateMaxAngular(double value) {
  max_angular_velocity_ = value;
  emit maxAngularVelocityChanged(value);
}

void JoystickWidget::setMaxLinearVelocity(double value) {
  max_linear_velocity_ = value;
  max_linear_spinbox_->setValue(value);
}

void JoystickWidget::setMaxAngularVelocity(double value) {
  max_angular_velocity_ = value;
  max_angular_spinbox_->setValue(value);
}

void JoystickWidget::emitAxisValues() {
  updateAxisLabels();
  emit axisValuesChanged(linear_x_value_, linear_y_value_, linear_z_value_, 
                        angular_x_value_, angular_y_value_, angular_z_value_);
}

void JoystickWidget::updateAxisLabels() {
  linear_x_label_->setText(QString("Linear X: %1 m/s").arg(linear_x_value_, 0, 'f', 2));
  linear_y_label_->setText(QString("Linear Y: %1 m/s").arg(linear_y_value_, 0, 'f', 2));
  linear_z_label_->setText(QString("Linear Z: %1 m/s").arg(linear_z_value_, 0, 'f', 2));
  angular_x_label_->setText(QString("Angular X: %1 rad/s").arg(angular_x_value_, 0, 'f', 2));
  angular_y_label_->setText(QString("Angular Y: %1 rad/s").arg(angular_y_value_, 0, 'f', 2));
  angular_z_label_->setText(QString("Angular Z: %1 rad/s").arg(angular_z_value_, 0, 'f', 2));
}

void JoystickWidget::onLinearXPlus() {
  linear_x_value_ += step_size_;
  if (linear_x_value_ > max_linear_velocity_) {
    linear_x_value_ = max_linear_velocity_;
  }
  emitAxisValues();
}

void JoystickWidget::onLinearXMinus() {
  linear_x_value_ -= step_size_;
  if (linear_x_value_ < -max_linear_velocity_) {
    linear_x_value_ = -max_linear_velocity_;
  }
  emitAxisValues();
}

void JoystickWidget::onLinearYPlus() {
  linear_y_value_ += step_size_;
  if (linear_y_value_ > max_linear_velocity_) {
    linear_y_value_ = max_linear_velocity_;
  }
  emitAxisValues();
}

void JoystickWidget::onLinearYMinus() {
  linear_y_value_ -= step_size_;
  if (linear_y_value_ < -max_linear_velocity_) {
    linear_y_value_ = -max_linear_velocity_;
  }
  emitAxisValues();
}

void JoystickWidget::onLinearZPlus() {
  linear_z_value_ += step_size_;
  if (linear_z_value_ > max_linear_velocity_) {
    linear_z_value_ = max_linear_velocity_;
  }
  emitAxisValues();
}

void JoystickWidget::onLinearZMinus() {
  linear_z_value_ -= step_size_;
  if (linear_z_value_ < -max_linear_velocity_) {
    linear_z_value_ = -max_linear_velocity_;
  }
  emitAxisValues();
}

void JoystickWidget::onAngularXPlus() {
  angular_x_value_ += step_size_;
  if (angular_x_value_ > max_angular_velocity_) {
    angular_x_value_ = max_angular_velocity_;
  }
  emitAxisValues();
}

void JoystickWidget::onAngularXMinus() {
  angular_x_value_ -= step_size_;
  if (angular_x_value_ < -max_angular_velocity_) {
    angular_x_value_ = -max_angular_velocity_;
  }
  emitAxisValues();
}

void JoystickWidget::onAngularYPlus() {
  angular_y_value_ += step_size_;
  if (angular_y_value_ > max_angular_velocity_) {
    angular_y_value_ = max_angular_velocity_;
  }
  emitAxisValues();
}

void JoystickWidget::onAngularYMinus() {
  angular_y_value_ -= step_size_;
  if (angular_y_value_ < -max_angular_velocity_) {
    angular_y_value_ = -max_angular_velocity_;
  }
  emitAxisValues();
}

void JoystickWidget::onAngularZPlus() {
  angular_z_value_ += step_size_;
  if (angular_z_value_ > max_angular_velocity_) {
    angular_z_value_ = max_angular_velocity_;
  }
  emitAxisValues();
}

void JoystickWidget::onAngularZMinus() {
  angular_z_value_ -= step_size_;
  if (angular_z_value_ < -max_angular_velocity_) {
    angular_z_value_ = -max_angular_velocity_;
  }
  emitAxisValues();
}

void JoystickWidget::resetAllAxes() {
  linear_x_value_ = 0.0f;
  linear_y_value_ = 0.0f;
  linear_z_value_ = 0.0f;
  angular_x_value_ = 0.0f;
  angular_y_value_ = 0.0f;
  angular_z_value_ = 0.0f;
  
  // Reset joystick position
  handle_pos_ = center_;
  update();
  
  // Update all labels and emit values
  emitAxisValues();
  emitValues();
}
