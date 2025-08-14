cmake_minimum_required(VERSION 3.5)
project(ros2_joystick_gui)

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(Qt5Core REQUIRED)
find_package(Qt5Widgets REQUIRED)
find_package(Qt5Gui REQUIRED)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

# Include directories
include_directories(
  include
  ${Qt5Widgets_INCLUDE_DIRS}
)

# Specify sources and headers
set(SOURCES
  src/joystick_node.cpp
  src/joystick_widget.cpp
)

set(HEADERS
  include/joystick_gui_cpp/joystick_widget.hpp
)

# Run MOC on headers
qt5_wrap_cpp(MOC_SOURCES ${HEADERS})

# Add the executable
add_executable(joystick_widget_node ${SOURCES} ${MOC_SOURCES})

# Link libraries
target_link_libraries(joystick_widget_node
  Qt5::Widgets
  Qt5::Core
  Qt5::Gui
)

# Add ament dependencies
ament_target_dependencies(joystick_widget_node
  rclcpp
  geometry_msgs
)

# Install the executable
install(TARGETS joystick_widget_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
