import rclpy  # renamed rospy to rclpy in ROS2
import rclpy.exceptions
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import Pose, Point, PoseStamped, Quaternion
from nav_msgs.msg import Odometry
from nav2_msgs.action import NavigateToPose
from nav2_msgs.srv import GetCostmap
from tf_transformations import quaternion_from_euler
import argparse
from actionlib_msgs.msg import GoalStatus  # TODO Not sure
from rclpy.duration import Duration

# Not sure it is needed:
import threading
from rclpy.executors import ExternalShutdownException

# class SimpleGoalState:
#     PENDING = 0
#     ACTIVE = 1
#     DONE = 2

# SimpleGoalState.to_string = classmethod(get_name_of_constant)


class WaypointSender(Node):

    def __init__(self, waypoints):
        super().__init__("waypoint_sender")
        self.pose_seq = list()
        x, y, z, w = quaternion_from_euler(
            0, 0, 0
        )  # quaternion_from_euler(roll, pitch, yaw)
        for waypoint in waypoints:
            self.pose_seq.append(Pose(waypoint, Quaternion(x, y, z, w)))
        self.get_logger().info(
            f"Waypoint sender started. Will send up to {len(self.pose_seq)} waypoints."
        )
        self.goal_cnt = 0

        # Create action client
        self.action_client = ActionClient(self, NavigateToPose, "navigate_to_pose")
        self.get_logger().info("Waiting for navigate_to_pose action server...")
        wait = self.action_client.wait_for_server()
        if not wait:
            self.get_logger().error("Action server not available!")
            rclpy.shutdown()
            return
        # rclpy.wait_for_service('/move_base/make_plan')

        # self.get_plan = self.create_client(GetPlan, '/move_base/GlobalPlanner/make_plan')
        self.odom_subscription = self.create_subscription(
            Odometry, "/odom", self.save_odom, 10
        )

        self.get_logger().info("Waypoint sender connected to move base server.")
        self.get_logger().info("Starting waypoint navigation.")
        self.MAX_WAIT = 120
        self.PLAN_CHECK_WAIT = 15
        self.PLAN_CHECK_MAX_RETRIES = 3
        # self.simple_state = SimpleGoalState.DONE
        self.send_goal()
        # rclpy.spin()

    def save_odom(self, msg):
        self.odom = msg.pose.pose

    def active_cb(self):
        self.get_logger().info(
            f"Goal pose {self.goal_cnt} is now being processed by the Action Server..."
        )

    def get_current_goal(self):
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = "map"
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose = self.pose_seq[self.goal_cnt]
        return goal_msg

    def send_goal(self):
        goal = self.get_current_goal()
        self.get_logger().info(f"Sending goal pose {self.goal_cnt} to Action Server")
        self.get_logger().info(str(self.pose_seq[self.goal_cnt]))

        # ROS2 action client API
        self.goal_future = self.action_client.send_goal_async(
            goal, feedback_callback=self.feedback_callback
        )
        self.goal_future.add_done_callback(self.goal_response_callback)

        self.goal_start = self.get_clock().now().nanoseconds / 1e9
        self.last_plan_check = self.goal_start
        self.plan_check_retries = 1

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info("Goal rejected")
            return

        self.get_logger().info("Goal accepted")
        self.goal_handle = goal_handle
        self.result_future = goal_handle.get_result_async()
        self.result_future.add_done_callback(self.get_result_callback)

    def feedback_callback(self, feedback_msg):
        """Complete ROS2 feedback callback with timeout and plan checking"""
        self.get_logger().info(f"Feedback for goal pose {self.goal_cnt} received")

        current_time = self.get_clock().now().nanoseconds / 1e9

        # Check timeout
        if abs(current_time - self.goal_start) >= self.MAX_WAIT:
            if self.goal_cnt < len(self.pose_seq):
                self.get_logger().info(
                    f"Timeout reached for current goal. Skipping to next one."
                )
                self.goal_handle.cancel_goal_async()
                self.move_to_next_goal()
            else:
                self.get_logger().info(
                    f"Timeout reached for last goal. Waypoint navigation ended."
                )
                self.goal_handle.cancel_goal_async()
                rclpy.shutdown()
                return

        # Check if we need to validate the plan
        elif abs(current_time - self.last_plan_check) > self.PLAN_CHECK_WAIT:
            start = PoseStamped()
            start.header.frame_id = "map"
            start.header.stamp = self.get_clock().now().to_msg()
            start.pose = self.odom

            goal = PoseStamped()
            goal.header.frame_id = "map"
            goal.header.stamp = self.get_clock().now().to_msg()
            goal.pose = self.pose_seq[self.goal_cnt]

            tolerance = 0.5
            no_plan = True

            try:
                if hasattr(self, "get_plan") and self.get_plan:
                    # For Nav2, this would need to be updated to use ComputePathToPose service
                    plan = self.get_plan(start, goal, tolerance)
                    no_plan = len(plan.plan.poses) == 0
                else:
                    # Skip plan checking if service not available
                    no_plan = False
            except Exception as e:
                no_plan = True
                self.get_logger().info(
                    f"Plan check {self.plan_check_retries} out of {self.PLAN_CHECK_MAX_RETRIES} failed with exception."
                )
                self.get_logger().info(
                    f"start:\n{start}\ngoal:\n{goal}\ntolerance:\n{tolerance}\nexception:\n{e}"
                )
            finally:
                if no_plan:
                    self.get_logger().info(
                        f"Plan check {self.plan_check_retries} out of {self.PLAN_CHECK_MAX_RETRIES} failed without plan."
                    )
                    self.plan_check_retries += 1
                else:
                    self.get_logger().info(
                        f"Plan check {self.plan_check_retries} out of {self.PLAN_CHECK_MAX_RETRIES} ok."
                    )
                    self.plan_check_retries = 1
                self.last_plan_check = current_time

    def get_result_callback(self, future):
        result = future.result().result
        status = future.result().status

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info(f"Goal pose {self.goal_cnt} reached!")
            self.goal_cnt += 1
            if self.goal_cnt < len(self.pose_seq):
                self.send_goal()
            else:
                self.get_logger().info("All waypoints completed!")
                rclpy.shutdown()
        elif status == GoalStatus.STATUS_ABORTED:
            self.get_logger().info(f"Goal pose {self.goal_cnt} aborted")
            self.move_to_next_goal()
        elif status == GoalStatus.STATUS_CANCELED:
            self.get_logger().info(f"Goal pose {self.goal_cnt} canceled")
            self.move_to_next_goal()

    def move_to_next_goal(self):
        self.goal_cnt += 1
        if self.goal_cnt < len(self.pose_seq):
            self.send_goal()
        else:
            self.get_logger().info("All waypoints processed!")
            rclpy.shutdown()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Send a list of waypoints to move_base."
    )
    parser.add_argument(
        "-p", "--path", required=True, help="Path to the waypoints csv file."
    )
    return parser.parse_known_args()[0]


def read_waypoints(file_path):
    waypoints = list()
    with open(file_path, "r") as file:
        pos = [l.strip().split(",") for l in file.readlines()]
        [
            waypoints.append(Point(float(x.strip()), float(y.strip()), 0))
            for (x, y) in pos
        ]  # on the ground, so height 0
    return waypoints


def main():
    rclpy.init()  # Needed in ros2

    args = parse_args()
    csv_file_path = args.path

    waypoints = read_waypoints(csv_file_path)

    node = WaypointSender(waypoints)  # Create a node in ROS2
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    try:
        main()
    except rclpy.exceptions.ROSInterruptException:
        rclpy.get_logger().info("Navigation finished.")
