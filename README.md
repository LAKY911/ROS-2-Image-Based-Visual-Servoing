# ROS 2 Image-Based Visual Servoing

This repository contains a complete ROS 2 implementation of Image-Based Visual Servoing (IBVS) using ArUco markers for robotic manipulation. The project demonstrates autonomous robotic control using visual feedback from camera observations to guide a robotic arm to desired positions.

## Project Overview

This project was developed as part of a Master's thesis, implementing a robust visual servoing system that:
- Detects ArUco markers using OAK-1-W camera
- Computes image feature errors between current and desired marker positions
- Generates camera velocity commands using interaction matrix-based control
- Transforms commands to robot end-effector frame for precise manipulation
- Provides real-time visualization and data logging capabilities

## Workspace Structure

The workspace is organized into the following ROS 2 packages:

### Core Visual Servoing Packages
- **`visual_servo`** - Main IBVS controller implementation with configurable parameters
- **`visual_servo_interfaces`** - Custom message and service definitions for IBVS communication
- **`visual_servo_plotter`** - Real-time plotting and visualization tools for servoing data

### ArUco Detection and Interfaces
- **`oak_aruco_detector`** - OAK-1-W camera integration with ArUco marker detection
- **`oak_aruco_detector_interfaces`** - Message definitions for ArUco marker data
- **`aruco_gazebo`** - Gazebo simulation models for ArUco markers

### Robot Control and Simulation
- **`lbr_fri_ros2_stack`** - KUKA LBR robot integration and control stack
- **`fri`** - Fast Research Interface for KUKA robots
- **`lbr_fri_idl`** - Interface definition language for KUKA FRI communication
- **`oak1w_gazebo`** - Gazebo simulation setup for OAK-1W camera
- **`ros2_joystick_gui`** - GUI-based joystick control for manual robot operation

### Additional Components
- **`bio_ik`** - Bio-inspired inverse kinematics solver for advanced motion planning

## Key Features

- **Real-time IBVS Control**: High-frequency control loop with configurable interaction matrix methods
- **Multi-threaded Architecture**: Concurrent sensor processing and control execution
- **Robust Error Handling**: Timeout detection, convergence monitoring, and graceful shutdown
- **Flexible Configuration**: Runtime parameter adjustment for tuning control behavior
- **Comprehensive Logging**: Data collection and visualization for analysis and debugging
- **Simulation Support**: Complete Gazebo integration for testing and development

## Prerequisites

- ROS 2 Humble or later
- Gazebo simulation environment
- OpenCV for computer vision operations
- NumPy and SciPy for numerical computations
- OAK-1-W camera drivers (for hardware deployment)

## Usage

1. **Build the workspace**:
   ```bash
   cd /path/to/workspace
   colcon build
   source install/setup.bash
   ```

2. **Launch the visual servoing system**:
   ```bash
   # For simulation
   ros2 launch visual_servo ibvs_simulation.launch.py
   
   # For real hardware
   ros2 launch visual_servo ibvs_hardware.launch.py
   ```

3. **Set desired target position**:
   ```bash
   ros2 service call /set_current_as_desired std_srvs/srv/Trigger
   ```

## Author

**Lukas Vitek**  

*This project was developed as part of a Master's thesis at the Czech Technical University (CTU) in Prague.*

## Third-Party Packages

The following packages were integrated from external sources and modified for this project:

- **`lbr_stack`** - KUKA LBR robot ROS 2 integration (includes `fri`, `lbr_fri_idl`, and `lbr_fri_ros2_stack` packages)
  - Source: [lbr-stack/lbr_fri_ros2_stack](https://github.com/lbr-stack/lbr_fri_ros2_stack)
  - Authors: Martin Huber, Christopher E. Mower, Sebastien Ourselin, Tom Vercauteren, Christos Bergeles
  - Citation: Huber et al. (2024) [1]

- **`aruco_gazebo`** - ArUco marker simulation models
  - Source: [joselusl/aruco_gazebo](https://github.com/joselusl/aruco_gazebo)
  - Author: Jose-Luis Sanchez-Lopez

- **`ros2_joystick_gui`** - GUI-based robot control interface
  - Source: [foiegreis/ros2_joystick_gui](https://github.com/foiegreis/ros2_joystick_gui)
  - Author: Greta Russi (foiegreis)

- **`bio_ik`** - Bio-inspired inverse kinematics solver
  - Source: [TAMS-Group/bio_ik](https://github.com/TAMS-Group/bio_ik)
  - Authors: Technical Aspects of Multimodal Systems (TAMS) Group, University of Hamburg

*Note: All third-party packages have been modified and adapted for the specific requirements of this visual servoing implementation.*

## References

[1] Huber, M., Mower, C. E., Ourselin, S., Vercauteren, T., & Bergeles, C. (2024). LBR-Stack: ROS 2 and Python Integration of KUKA FRI for Med and IIWA Robots. *Journal of Open Source Software*, 9(103), 6138. https://doi.org/10.21105/joss.06138

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.