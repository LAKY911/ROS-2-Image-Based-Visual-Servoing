#ifndef JOYSTICK_GUI_CPP_JOYSTICK_WIDGET_HPP
#define JOYSTICK_GUI_CPP_JOYSTICK_WIDGET_HPP

#include <QWidget>
#include <QPointF>
#include <QLabel>
#include <QDoubleSpinBox>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QPushButton>

class JoystickWidget : public QWidget {
  Q_OBJECT

public:
  explicit JoystickWidget(double init_max_linear = 1.0, double init_max_angular = 1.0, bool invert_angular = false, QWidget *parent = nullptr);
  ~JoystickWidget();

  void setMaxLinearVelocity(double value);
  void setMaxAngularVelocity(double value);

  // Invert angular variable made public for access from JoystickNode
  bool invert_angular_;

  void emitValues();  // Moved to public access specifier

signals:
  void joystickMoved(float linear, float angular);
  void maxLinearVelocityChanged(double value);
  void maxAngularVelocityChanged(double value);
  void axisValuesChanged(float linear_x, float linear_y, float linear_z, float angular_x, float angular_y, float angular_z);

private slots:
  void updateMaxLinear(double value);
  void updateMaxAngular(double value);
  void onLinearXPlus();
  void onLinearXMinus();
  void onLinearYPlus();
  void onLinearYMinus();
  void onLinearZPlus();
  void onLinearZMinus();
  void onAngularXPlus();
  void onAngularXMinus();
  void onAngularYPlus();
  void onAngularYMinus();
  void onAngularZPlus();
  void onAngularZMinus();
  void resetAllAxes();

protected:
  void paintEvent(QPaintEvent *event) override;
  void mousePressEvent(QMouseEvent *event) override;
  void mouseMoveEvent(QMouseEvent *event) override;
  void mouseReleaseEvent(QMouseEvent *event) override;

private:
  // Joystick parameters
  QPointF center_;
  QPointF handle_pos_;
  float outer_radius_;
  float inner_radius_;
  bool is_dragging_;

  // Max velocities
  double max_linear_velocity_;
  double max_angular_velocity_;

  // UI elements
  QLabel *linear_label_;
  QLabel *angular_label_;
  QDoubleSpinBox *max_linear_spinbox_;
  QDoubleSpinBox *max_angular_spinbox_;
  
  // Additional axis control buttons
  QPushButton *linear_x_plus_;
  QPushButton *linear_x_minus_;
  QPushButton *linear_y_plus_;
  QPushButton *linear_y_minus_;
  QPushButton *linear_z_plus_;
  QPushButton *linear_z_minus_;
  QPushButton *angular_x_plus_;
  QPushButton *angular_x_minus_;
  QPushButton *angular_y_plus_;
  QPushButton *angular_y_minus_;
  QPushButton *angular_z_plus_;
  QPushButton *angular_z_minus_;
  QPushButton *reset_button_;
  
  // Labels for current axis values
  QLabel *linear_x_label_;
  QLabel *linear_y_label_;
  QLabel *linear_z_label_;
  QLabel *angular_x_label_;
  QLabel *angular_y_label_;
  QLabel *angular_z_label_;
  
  // Current axis values
  float linear_x_value_;
  float linear_y_value_;
  float linear_z_value_;
  float angular_x_value_;
  float angular_y_value_;
  float angular_z_value_;
  
  // Step size for button increments
  double step_size_;
  
  void emitAxisValues();
  void updateAxisLabels();
};

#endif  // JOYSTICK_GUI_CPP_JOYSTICK_WIDGET_HPP
