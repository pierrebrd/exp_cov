#!/usr/bin/env python
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from rcl_interfaces.msg import Log
from rclpy.action import ActionClient, GoalResponse, CancelResponse
from nav2_msgs.action import NavigateToPose
from nav2_msgs.action._navigate_to_pose import (
    NavigateToPose_GetResult_Response,
    NavigateToPose_Result,
    NavigateToPose_FeedbackMessage,
)
from nav2_msgs.action._back_up import BackUp_FeedbackMessage
from nav2_msgs.action._follow_path import FollowPath_FeedbackMessage
from nav2_msgs.action._spin import Spin_FeedbackMessage
from nav2_msgs.action._wait import Wait_FeedbackMessage
from action_msgs.msg import GoalStatusArray, GoalStatus, GoalInfo
import nav2_msgs.msg
from geometry_msgs.msg import PoseStamped
from builtin_interfaces.msg import Duration


class GoalInfo:
    """Class to hold information about a navigation goal."""

    def __init__(self, goal_id):
        self.goal_id = goal_id
        self.status = 0  # UNKNOWN
        self.recoveries = 0
        self.distance_remaining = None
        self.estimated_time_remaining = None
        self.navigation_time = None
        self.current_pose = PoseStamped()  # Initialize with an empty PoseStamped
        self.compute_path_status = 0  # UNKNOWN


class BackupGoalInfo:
    """Class to hold information about a backup navigation goal."""

    def __init__(self, goal_id):
        self.goal_id = goal_id
        self.status = 0  # UNKNOWN
        self.distance_traveled = None


class FollowPathGoalInfo:
    """Class to hold information about a follow path goal."""

    def __init__(self, goal_id):
        self.goal_id = goal_id
        self.status = 0  # UNKNOWN
        self.distance_to_goal = None
        self.speed = None


class SpinGoalInfo:
    """Class to hold information about a spin goal."""

    def __init__(self, goal_id):
        self.goal_id = goal_id
        self.status = 0  # UNKNOWN
        self.angular_distance_traveled = None


class WaitGoalInfo:
    """Class to hold information about a wait goal."""

    def __init__(self, goal_id):
        self.goal_id = goal_id
        self.status = 0  # UNKNOWN
        self.time_left = -1.0
        self.clock = None
        self.initial_time_left = None
        self.initial_clock = None


class Nav_stack_listener(Node):
    def __init__(self):
        super().__init__("nav_stack_listener")
        # self.logs_sub = self.create_subscription(Log, "rosout", self.logs_callback, 10)
        # self.get_logger().info("Subscribed to rosout topic")
        self.goals_info = {}  # Dictionary to hold goal information
        self.backup_goals_info = {}  # Dictionary to hold backup goal information
        self.follow_path_goals_info = (
            {}
        )  # Dictionary to hold follow path goal information
        self.spin_goals_info = {}  # Dictionary to hold spin goal information
        self.wait_goals_info = {}  # Dictionary to hold wait goal information

        # Subscribe to NavigateToPose action feedback and status
        self.nav_feedback_sub = self.create_subscription(
            NavigateToPose_FeedbackMessage,
            "/navigate_to_pose/_action/feedback",
            self.nav_feedback_callback,
            10,
        )

        self.nav_status_sub = self.create_subscription(
            GoalStatusArray,
            "/navigate_to_pose/_action/status",
            self.nav_status_callback,
            10,
        )

        self.nav_backup_feedback_sub = self.create_subscription(
            BackUp_FeedbackMessage,
            "/backup/_action/feedback",
            self.nav_backup_feedback_callback,
            10,
        )

        self.nav_backup_status_sub = self.create_subscription(
            GoalStatusArray,
            "/backup/_action/status",
            self.nav_backup_status_callback,
            10,
        )

        # FollowPath Goals are probably intermediary goals when navigating to a goal, not really useful
        # self.nav_follow_path_feedback_sub = self.create_subscription(
        #     FollowPath_FeedbackMessage,
        #     "/follow_path/_action/feedback",
        #     self.nav_follow_path_feedback_callback,
        #     10,
        # )

        # self.nav_follow_path_status_sub = self.create_subscription(
        #     GoalStatusArray,
        #     "/follow_path/_action/status",
        #     self.nav_follow_path_status_callback,
        #     10,
        # )

        self.nav_spin_feedback_sub = self.create_subscription(
            Spin_FeedbackMessage,
            "/spin/_action/feedback",
            self.nav_spin_feedback_callback,
            10,
        )

        self.nav_spin_status_sub = self.create_subscription(
            GoalStatusArray,
            "/spin/_action/status",
            self.nav_spin_status_callback,
            10,
        )

        self.nav_wait_feedback_sub = self.create_subscription(
            Wait_FeedbackMessage,
            "/wait/_action/feedback",
            self.nav_wait_feedback_callback,
            10,
        )

        self.nav_wait_status_sub = self.create_subscription(
            GoalStatusArray,
            "/wait/_action/status",
            self.nav_wait_status_callback,
            10,
        )

        # Not really useful (the goals id are not linked to the main goals, so it's hard to track them)

        # self.nav_compute_path_to_pose_status_sub = self.create_subscription(
        #     GoalStatusArray,
        #     "/compute_path_to_pose/_action/status",
        #     self.nav_compute_path_to_pose_status_callback,  # Reusing the same callback for simplicity
        #     10,
        # )

        self.get_logger().info("Subscribed to nav2 topics")

    def logs_callback(self, data):
        self.get_logger().info(f"Log message received: {data.msg}")

    def nav_feedback_callback(self, msg: NavigateToPose_FeedbackMessage):
        feedback = msg.feedback

        # Convert goal id bytes to hex string for logging
        goal_id_str = msg.goal_id.uuid.tobytes().hex()

        if goal_id_str not in self.goals_info:
            # If this is a new goal, initialize its info
            self.goals_info[goal_id_str] = GoalInfo(goal_id_str)

        goal_info = self.goals_info[goal_id_str]
        new_current_pose = feedback._current_pose

        if feedback.number_of_recoveries > goal_info.recoveries:
            goal_info.recoveries = feedback.number_of_recoveries
            self.get_logger().warn(
                f"Goal {goal_id_str[:8]}... new recovery: total {feedback.number_of_recoveries} "
            )

        goal_info.current_pose = new_current_pose
        goal_info.distance_remaining = feedback.distance_remaining
        goal_info.estimated_time_remaining = feedback.estimated_time_remaining
        goal_info.navigation_time = feedback.navigation_time

        # TODO: distance remaining : maybe if the distance remaining doesnt change for a while, we can assume the robot struggles to find a path -> bad goal
        # same for estimated time remaining or navigation time

        # TODO : is pose the corrected pose of the robot ? maybe it could be used for logging the current position

    def nav_status_callback(self, msg: GoalStatusArray):
        for status in msg.status_list:
            assert isinstance(status, GoalStatus)  # TODO: remove
            goal_id_str = status.goal_info.goal_id.uuid.tobytes().hex()

            if goal_id_str not in self.goals_info:
                # If this is a new goal, initialize its info
                self.goals_info[goal_id_str] = GoalInfo(goal_id_str)

            goal_info = self.goals_info[goal_id_str]
            assert isinstance(goal_info, GoalInfo)  # TODO: remove
            assert isinstance(goal_info.current_pose, PoseStamped)  # TODO: remove

            if goal_info.status != status.status:
                # Log the status change

                if status.status == GoalStatus.STATUS_SUCCEEDED:
                    self.get_logger().info(f"Goal {goal_id_str[:8]}... has SUCCEEDED")

                elif status.status == GoalStatus.STATUS_ABORTED:
                    self.get_logger().warn(
                        f"Goal {goal_id_str[:8]}... has been ABORTED, curent_pose: {goal_info.current_pose.pose}"
                    )

                elif status.status == GoalStatus.STATUS_CANCELED:
                    self.get_logger().warn(
                        f"Goal {goal_id_str[:8]}... has been CANCELED, curent_pose: {goal_info.current_pose.pose}"
                    )

                else:
                    self.get_logger().info(
                        f"Goal {goal_id_str[:8]}... status changed from {self.get_status_text(goal_info.status)} to {self.get_status_text(status.status)}"
                    )
                # We update the goal status
                goal_info.status = status.status

    def nav_backup_feedback_callback(self, msg: BackUp_FeedbackMessage):
        feedback = msg.feedback

        # Convert goal id bytes to hex string for logging
        goal_id_str = msg.goal_id.uuid.tobytes().hex()

        if goal_id_str not in self.backup_goals_info:
            # If this is a new goal, initialize its info
            self.backup_goals_info[goal_id_str] = BackupGoalInfo(goal_id_str)

        goal_info = self.backup_goals_info[goal_id_str]

        # We just update the distance traveled
        goal_info.distance_traveled = feedback.distance_traveled

    def nav_backup_status_callback(self, msg: GoalStatusArray):
        for status in msg.status_list:
            assert isinstance(status, GoalStatus)  # TODO: remove
            goal_id_str = status.goal_info.goal_id.uuid.tobytes().hex()

            if goal_id_str not in self.backup_goals_info:
                # If this is a new goal, initialize its info
                self.backup_goals_info[goal_id_str] = BackupGoalInfo(goal_id_str)

            goal_info = self.backup_goals_info[goal_id_str]

            if goal_info.status != status.status:
                # Log the status change

                if status.status == GoalStatus.STATUS_SUCCEEDED:
                    self.get_logger().info(
                        f"Backup Goal {goal_id_str[:8]}... has SUCCEEDED, distance traveled: {goal_info.distance_traveled}"
                    )

                elif status.status == GoalStatus.STATUS_ABORTED:
                    self.get_logger().warn(
                        f"Backup Goal {goal_id_str[:8]}... has been ABORTED, distance traveled: {goal_info.distance_traveled}"
                    )

                elif status.status == GoalStatus.STATUS_CANCELED:
                    self.get_logger().warn(
                        f"Backup Goal {goal_id_str[:8]}... has been CANCELED, distance traveled: {goal_info.distance_traveled}"
                    )

                else:
                    self.get_logger().info(
                        f"Backup Goal {goal_id_str[:8]}... status changed from {self.get_status_text(goal_info.status)} to {self.get_status_text(status.status)}"
                    )
                # We update the goal status
                goal_info.status = status.status

    def nav_follow_path_feedback_callback(self, msg: FollowPath_FeedbackMessage):
        feedback = msg.feedback

        # Convert goal id bytes to hex string for logging
        goal_id_str = msg.goal_id.uuid.tobytes().hex()

        if goal_id_str not in self.follow_path_goals_info:
            # If this is a new goal, initialize its info
            self.follow_path_goals_info[goal_id_str] = FollowPathGoalInfo(goal_id_str)

        goal_info = self.follow_path_goals_info[goal_id_str]

        # Update the feedback data
        goal_info.distance_to_goal = feedback.distance_to_goal
        goal_info.speed = feedback.speed

    def nav_follow_path_status_callback(self, msg: GoalStatusArray):
        for status in msg.status_list:
            assert isinstance(status, GoalStatus)  # TODO: remove
            goal_id_str = status.goal_info.goal_id.uuid.tobytes().hex()

            if goal_id_str not in self.follow_path_goals_info:
                # If this is a new goal, initialize its info
                self.follow_path_goals_info[goal_id_str] = FollowPathGoalInfo(
                    goal_id_str
                )

            goal_info = self.follow_path_goals_info[goal_id_str]

            if goal_info.status != status.status:
                # Log the status change

                if status.status == GoalStatus.STATUS_SUCCEEDED:
                    self.get_logger().info(
                        f"FollowPath Goal {goal_id_str[:8]}... has SUCCEEDED, distance to goal: {goal_info.distance_to_goal}, speed: {goal_info.speed}"
                    )

                elif status.status == GoalStatus.STATUS_ABORTED:
                    self.get_logger().warn(
                        f"FollowPath Goal {goal_id_str[:8]}... has been ABORTED, distance to goal: {goal_info.distance_to_goal}, speed: {goal_info.speed}"
                    )

                elif status.status == GoalStatus.STATUS_CANCELED:
                    self.get_logger().warn(
                        f"FollowPath Goal {goal_id_str[:8]}... has been CANCELED, distance to goal: {goal_info.distance_to_goal}, speed: {goal_info.speed}"
                    )

                else:
                    self.get_logger().info(
                        f"FollowPath Goal {goal_id_str[:8]}... status changed from {self.get_status_text(goal_info.status)} to {self.get_status_text(status.status)}"
                    )
                # We update the goal status
                goal_info.status = status.status

    def nav_spin_feedback_callback(self, msg: Spin_FeedbackMessage):
        feedback = msg.feedback

        # Convert goal id bytes to hex string for logging
        goal_id_str = msg.goal_id.uuid.tobytes().hex()

        if goal_id_str not in self.spin_goals_info:
            # If this is a new goal, initialize its info
            self.spin_goals_info[goal_id_str] = SpinGoalInfo(goal_id_str)

        goal_info = self.spin_goals_info[goal_id_str]

        # Update the feedback data
        goal_info.angular_distance_traveled = feedback.angular_distance_traveled

    def nav_spin_status_callback(self, msg: GoalStatusArray):
        for status in msg.status_list:
            assert isinstance(status, GoalStatus)  # TODO: remove
            goal_id_str = status.goal_info.goal_id.uuid.tobytes().hex()

            if goal_id_str not in self.spin_goals_info:
                # If this is a new goal, initialize its info
                self.spin_goals_info[goal_id_str] = SpinGoalInfo(goal_id_str)

            goal_info = self.spin_goals_info[goal_id_str]

            if goal_info.status != status.status:
                # Log the status change

                if status.status == GoalStatus.STATUS_SUCCEEDED:
                    self.get_logger().info(
                        f"Spin Goal {goal_id_str[:8]}... has SUCCEEDED, angular distance traveled: {goal_info.angular_distance_traveled}"
                    )

                elif status.status == GoalStatus.STATUS_ABORTED:
                    self.get_logger().warn(
                        f"Spin Goal {goal_id_str[:8]}... has been ABORTED, angular distance traveled: {goal_info.angular_distance_traveled}"
                    )

                elif status.status == GoalStatus.STATUS_CANCELED:
                    self.get_logger().warn(
                        f"Spin Goal {goal_id_str[:8]}... has been CANCELED, angular distance traveled: {goal_info.angular_distance_traveled}"
                    )

                else:
                    self.get_logger().info(
                        f"Spin Goal {goal_id_str[:8]}... status changed from {self.get_status_text(goal_info.status)} to {self.get_status_text(status.status)}"
                    )
                # We update the goal status
                goal_info.status = status.status

    def nav_wait_feedback_callback(self, msg: Wait_FeedbackMessage):

        # TODO: clean the mess about clock and time left.
        # Maybe we can analyse the time in log_from_bag.py
        feedback = msg.feedback

        # Convert goal id bytes to hex string for logging
        goal_id_str = msg.goal_id.uuid.tobytes().hex()

        if goal_id_str not in self.wait_goals_info:
            # If this is a new goal, initialize its info
            self.wait_goals_info[goal_id_str] = WaitGoalInfo(goal_id_str)

        goal_info = self.wait_goals_info[goal_id_str]

        if goal_info.initial_time_left is None:
            # If this is the first feedback, initialize the initial time left
            goal_info.initial_time_left = (
                feedback.time_left.sec + feedback.time_left.nanosec * 1e-9
            )
            self.get_logger().info(
                f"Started Waiting, estimated time left: {goal_info.initial_time_left}s, Goal {goal_id_str[:8]}..."
            )

        # Update the feedback data
        goal_info.time_left = feedback.time_left.sec + feedback.time_left.nanosec * 1e-9

    def nav_wait_status_callback(self, msg: GoalStatusArray):
        # TODO: clean the mess about clock and time left.
        for status in msg.status_list:
            assert isinstance(status, GoalStatus)  # TODO: remove
            goal_id_str = status.goal_info.goal_id.uuid.tobytes().hex()

            if goal_id_str not in self.wait_goals_info:
                # If this is a new goal, initialize its info
                self.wait_goals_info[goal_id_str] = WaitGoalInfo(goal_id_str)

            current_clock = (
                status.goal_info.stamp.sec + status.goal_info.stamp.nanosec * 1e-9
            )

            goal_info = self.wait_goals_info[goal_id_str]
            if goal_info.initial_clock is None:
                goal_info.initial_clock = current_clock

            if goal_info.status != status.status:
                # Log the status change

                if status.status == GoalStatus.STATUS_SUCCEEDED:
                    self.get_logger().info(
                        f"Wait Goal {goal_id_str[:8]}... has SUCCEEDED, time left: {goal_info.time_left}s"
                    )

                elif status.status == GoalStatus.STATUS_ABORTED:
                    self.get_logger().warn(
                        f"Wait Goal {goal_id_str[:8]}... has ABORTED, time left: {goal_info.time_left}s"
                    )
                elif status.status == GoalStatus.STATUS_CANCELED:
                    self.get_logger().warn(
                        f"Wait Goal {goal_id_str[:8]}... has CANCELED, time left: {goal_info.time_left}s"
                    )

                else:
                    self.get_logger().info(
                        f"Wait Goal {goal_id_str[:8]}... status changed from {self.get_status_text(goal_info.status)} to {self.get_status_text(status.status)}"
                    )
                # We update the goal status
                goal_info.status = status.status

    def nav_compute_path_to_pose_status_callback(self, msg: GoalStatusArray):
        """Handle status updates for the ComputePathToPose action."""
        for status in msg.status_list:
            assert isinstance(status, GoalStatus)  # TODO: remove
            goal_id_str = status.goal_info.goal_id.uuid.tobytes().hex()

            if goal_id_str not in self.goals_info:
                # If this is a new goal, initialize its info
                self.goals_info[goal_id_str] = GoalInfo(goal_id_str)

            goal_info = self.goals_info[goal_id_str]
            assert isinstance(goal_info, GoalInfo)  # TODO: remove

            if goal_info.compute_path_status != status.status:
                # Log the status change

                if status.status == GoalStatus.STATUS_SUCCEEDED:
                    self.get_logger().info(
                        f"ComputePathToPose Goal {goal_id_str[:8]}... has SUCCEEDED"
                    )

                elif status.status == GoalStatus.STATUS_ABORTED:
                    self.get_logger().warn(
                        f"ComputePathToPose Goal {goal_id_str[:8]}... has been ABORTED"
                    )

                elif status.status == GoalStatus.STATUS_CANCELED:
                    self.get_logger().warn(
                        f"ComputePathToPose Goal {goal_id_str[:8]}... has been CANCELED"
                    )

                else:
                    self.get_logger().info(
                        f"ComputePathToPose Goal {goal_id_str[:8]}... status changed from {self.get_status_text(goal_info.compute_path_status)} to {self.get_status_text(status.status)}"
                    )
                # We update the goal status
                goal_info.compute_path_status = status.status

    def get_status_text(self, status_code):
        """Convert status code to human-readable text"""
        status_map = {
            0: "UNKNOWN",
            1: "ACCEPTED",
            2: "EXECUTING",
            3: "CANCELING",
            4: "SUCCEEDED",
            5: "CANCELED",
            6: "ABORTED",
        }
        return status_map.get(status_code, f"UNKNOWN_STATUS_{status_code}")


def main():
    rclpy.init(args=None)
    node = Nav_stack_listener()

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

# Todo : should log current estimated position when it founds a problem
