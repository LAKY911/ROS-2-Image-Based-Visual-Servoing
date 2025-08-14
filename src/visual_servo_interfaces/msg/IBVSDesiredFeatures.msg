std_msgs/Header header

# Desired center point of ArUco marker for IBVS
geometry_msgs/Point aruco_desired_center

# Desired corner positions for IBVS (4 corners: top-left, top-right, bottom-right, bottom-left)
geometry_msgs/Point[] corners

# Desired pose for PBVS (Position-Based Visual Servoing)
geometry_msgs/Pose desired_pose
