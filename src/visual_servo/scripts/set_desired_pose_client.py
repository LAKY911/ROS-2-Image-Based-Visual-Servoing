#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
import sys

class SetDesiredPoseClient(Node):
    def __init__(self):
        super().__init__('set_desired_pose_client')
        self.client = self.create_client(Trigger, 'set_current_as_desired')
        
        # Wait for service to be available
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting...')
    
    def call_service(self):
        """Call the set_current_as_desired service"""
        request = Trigger.Request()
        
        self.get_logger().info('Calling set_current_as_desired service...')
        future = self.client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        
        if future.result() is not None:
            response = future.result()
            if response.success:
                self.get_logger().info(f'SUCCESS: {response.message}')
            else:
                self.get_logger().error(f'FAILED: {response.message}')
            return response.success
        else:
            self.get_logger().error('Service call failed!')
            return False

def main(args=None):
    rclpy.init(args=args)
    
    client = SetDesiredPoseClient()
    success = client.call_service()
    
    client.destroy_node()
    rclpy.shutdown()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()
