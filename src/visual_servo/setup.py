from setuptools import find_packages, setup
import os
from glob import glob


package_name = 'visual_servo'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include all launch files
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='lukas',
    maintainer_email='lukas.vitek911@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'joy_servo_bridge = visual_servo.joy_servo_bridge:main',
            'ibvs_servo_node = visual_servo.ibvs_servo_node:main',
        ],
    },
)
