# Ported to ROS2

#!/usr/bin/env python
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan


class Laser_scan_check(Node):

    def __init__(self):
        super().__init__("laser_scan_check")
        self.sub = self.create_subscription(LaserScan, "scan", self.callback, 10)
        self.get_logger().info("Subscribed to scan topic")
        self.currently_triggered = False
        self.total_triggered = 0
        self.distance_threshold = (
            1.0  # Set the distance threshold TODO: set by a parameter
        )
        self.min_distance = float("inf")

    def callback(self, data: LaserScan):
        if not self.currently_triggered:
            # Check if any range in the scan is below the threshold
            if any(r < self.distance_threshold for r in data.ranges):
                self.currently_triggered = True
                self.total_triggered += 1
                self.min_distance = min(data.ranges)  # Get the minimum distance
                angle = (
                    data.angle_min
                    + data.ranges.index(self.min_distance) * data.angle_increment
                )  # Get the angle of the closest obstacle
                self.get_logger().warn(
                    f"Too close to an obstacle, current distance is {self.min_distance:.3f} at angle {angle:.3f}, total triggered: {self.total_triggered} times"
                )

        else:  # currently_triggered is True
            if all(
                r > self.distance_threshold * 1.1 for r in data.ranges
            ):  # 1.1 to avoid oscillation around the threshold
                self.currently_triggered = False
                self.get_logger().warn("No longer too close to an obstacle")
            elif any(r < self.min_distance * 0.95 for r in data.ranges):
                # We are getting closer !
                self.min_distance = min(data.ranges)
                angle = (
                    data.angle_min
                    + data.ranges.index(self.min_distance) * data.angle_increment
                )  # Get the angle of the closest obstacle

                self.get_logger().warn(
                    f"Too close to an obstacle, current distance is {self.min_distance:.3f} at angle {angle:.3f}, total triggered: {self.total_triggered} times"
                )


def main():
    rclpy.init(args=None)
    node = Laser_scan_check()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Cleanup
        if rclpy.ok():
            node.destroy_node()
            rclpy.try_shutdown()


if __name__ == "__main__":
    main()

# TODO: Log location when obstacle is detected? Location of the robot (corrected odom) or the object (from odom and laser scan i guess)
