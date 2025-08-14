from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'oak_aruco_detector'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
    (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*.launch.py'))),
        (os.path.join('share', package_name, 'config'), glob(os.path.join('config', '*'))),
    ],
    install_requires=[
        'setuptools',
        'depthai',
        'opencv-python>=4.5.0,<4.10.0',  # Compatible with NumPy 1.x
        'numpy<2.0',  # Pin to NumPy 1.x for ROS2 cv_bridge compatibility
        'matplotlib',  # For visual servo plotter
    ],
    zip_safe=True,
    maintainer='lukas',
    maintainer_email='lukas.vitek911@gmail.com',
    description='OAK ArUco detector package for ROS2 integration with DepthAI cameras.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'oak_aruco_publisher = oak_aruco_detector.oak_aruco_publisher:main',
            'oak_aruco_annotator = oak_aruco_detector.oak_aruco_annotator:main',
            'gazebo_aruco_detector = oak_aruco_detector.gazebo_aruco_detector:main',
            'visual_servo_plotter = oak_aruco_detector.visual_servo_plotter:main',
            'visual_servo_data_logger = oak_aruco_detector.visual_servo_data_logger:main',
            'image_visual_relay = oak_aruco_detector.image_visual_relay:main',
        ],
    },
)
