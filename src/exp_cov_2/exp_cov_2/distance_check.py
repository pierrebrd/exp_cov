# Ported to ROS2
# Due to problem with the node shutdown in ROS2, I had to create a new node to log the final distance.

#!/usr/bin/env python
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import math


class Distance_check(Node):

    def __init__(self):
        super().__init__("distance_check")
        self.sub = self.create_subscription(Odometry, "ground_truth", self.callback, 10)
        self.get_logger().info("Subscribed to ground_truth topic")
        self.total_distance = 0
        self.previous_x = 0
        self.previous_y = 0
        self.first_run = True

    def callback(self, data):
        if self.first_run:
            self.previous_x = data.pose.pose.position.x
            self.previous_y = data.pose.pose.position.y
            self.first_run = False
        x = data.pose.pose.position.x
        y = data.pose.pose.position.y
        d_increment = math.dist((self.previous_x, self.previous_y), (x, y))
        self.total_distance = self.total_distance + d_increment
        # print(f"Total distance traveled is {self.total_distance}")
        self.previous_x = data.pose.pose.position.x
        self.previous_y = data.pose.pose.position.y


class Distance_check_final_logger(Node):
    def __init__(self, total_distance, context):
        super().__init__("distance_check_final_logger", context=context)
        self.get_logger().info(f"Final distance is {total_distance}")
        # Shutdown
        self.destroy_node()


def main():
    rclpy.init(args=None)
    node = Distance_check()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        print(f"Final distance is {node.total_distance}")
        # We also include it in the ROS logs

        # node.get_logger().info(f"Final distance is {node.total_distance}")
        # It will probably fail, lets create a new context to log and then destroy it

        new_context = rclpy.Context()
        rclpy.init(context=new_context)
        new_node = Distance_check_final_logger(node.total_distance, context=new_context)
        rclpy.spin(new_node)

        # Cleanup
        if rclpy.ok():
            node.destroy_node()
            new_node.destroy_node()
            rclpy.try_shutdown()


if __name__ == "__main__":
    main()
