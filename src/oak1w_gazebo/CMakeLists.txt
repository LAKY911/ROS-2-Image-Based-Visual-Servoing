cmake_minimum_required(VERSION 3.8)
project(oak1w_gazebo)

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

## find dependencies
find_package(ament_cmake REQUIRED)
#find_package(ament_cmake_python REQUIRED) #for python scripts
#find_package(rclcpp REQUIRED)
#find_package(rclpy REQUIRED) #for python scripts
find_package(ros_gz_sim REQUIRED)

# Install directories
install(
  DIRECTORY launch urdf meshes worlds
  DESTINATION share/${PROJECT_NAME}
)

ament_package()