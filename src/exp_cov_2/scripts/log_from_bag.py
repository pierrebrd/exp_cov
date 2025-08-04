"""
Read a ROS2 bag file and extract information about the run.
For now, the information extracted is similar to the one in logged_run.py in the initial exp_cov in ROS1,
but due to the differences in ROS2, it is easier to parse the bag file than to monitor the run logs live.
"""

# Required imports
import ast
from logging import ERROR, INFO
import subprocess as sp
import argparse
import cv2
from PIL import Image
import numpy as np
import os
import yaml
import rclpy  # ROS2
from rclpy.serialization import deserialize_message
import rosbag2_py
from rosidl_runtime_py.utilities import get_message
import tf2_msgs.msg
import rcl_interfaces.msg  # To deal with rcl_interfaces/msg/Log messages
import json
import zipfile
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import TransformStamped
from tf2_geometry_msgs import do_transform_pose_stamped, do_transform_pose
from geometry_msgs.msg import PoseStamped, Pose
from nav_msgs.msg import Odometry
import matplotlib.pyplot as plt  # For plotting the robot path

# from time import gmtime, strftime, sleep
import time
import sys

# Import functions from benchmark.py in exploration_benchmarking

sys.path.append(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "..",
        "exploration_benchmarking",
        "benchmark_utils",
    )
)

from benchmark import read_rosbag2, evo_metrics

# Global variables

verbose = False
DEBUG_LEVEL = 10  # Debug level for logging, can be adjusted as needed
INFO_LEVEL = 20  # Info level for logging, can be adjusted as needed
WARN_LEVEL = 30  # Warning level for logging, can be adjusted as needed
ERROR_LEVEL = 40  # Error level for logging, can be adjusted as needed
FATAL_LEVEL = 50  # Critical level for logging, can be adjusted as needed

time.gmtime


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark a ROS2 simulation run")
    parser.add_argument(
        "run_path", help="Path to the run folder (should contain a 'rosbags' subfolder)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    return parser.parse_args()


def format_time(timestamp=None):
    """
    Gets timestamp (current time if not provided) in readable format.
    Args:
        timestamp (float, optional): Timestamp in seconds
    Returns:
        str: Current time in format YYYY-MM-DD HH:MM:SS
    """
    if timestamp is None:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    else:
        return time.strftime(
            "%Y-%m-%d %H:%M:%S", time.gmtime(timestamp / 1_000_000_000)
        )


def format_log(timestamp, sim_time, name, msg):
    "format a log message"
    return (
        f"{format_time(timestamp)}, sim_time: {sim_time:.1f}s [{name}] {msg.strip()}\n"
    )


def should_log_message(msg, name):
    # List of important messages to log
    important_messages = [  # Problem: the ExploreNode logs are not included into /rosout, because they are creazted with a custom name logger, not the node logger...
        "waypoint sender started",
        "connected to move_base nav2 server",  # ROS2
        "sending goal",
        "wait goal has succeeded",
        "wait goal has been aborted",
        "wait goal has been canceled",
        "spin goal has succeeded",
        "spin goal has been aborted",
        "spin goal has been canceled",
        "computepathtopose goal has succeeded",
        "computepathtopose goal has been aborted",
        "computepathtopose goal has been canceled",
        "followpath goal has succeeded",
        "followpath goal has been aborted",
        "followpath goal has been canceled",
        "backup goal has succeeded",
        "backup goal has been aborted",
        "backup goal has been canceled",
        "nav goal has succeeded",
        "nav goal has been aborted",
        "nav goal has been canceled",
        "was successful",
        "was aborted by the navigation stack",
        "failed to create",
        "failed to generate a valid path",
        "collision",
        "for spin behavior",
        "running wait",
        "wait completed successfully",
        "running backup",
        "backup completed successfully",
        "blacklist",
        "black list",
        "found frontiers",  # Not working
        "waiting for costmap",
        "final distance",  # distance_check
        "no longer too close to an obstacle",  # laser_scan_check
        "too close to an obstacle",  # laser_scan_check
        "aborted",
        "canceled",
        "succeeded",
        "failed",
    ]

    # Completely ignore certain types of messages
    ignore_messages = [
        "worldtomap failed",
    ]

    msg_lower = msg.lower()
    name_lower = name.lower()

    # If the message contains one of the ignore strings, do not log it
    if any(x in msg_lower for x in ignore_messages) or any(
        x in name_lower for x in ignore_messages
    ):
        return (False, "")

    # If the message contains one of the important strings, log it
    for x in important_messages:
        if x in msg_lower or x in name_lower:
            return (True, x)

    return (False, "")


def add_issue(time, pose, message, additional_info, goal, issues, issue_ongoing):
    """
    Add an issue to the issues list.
    If the issue is ongoing, append to the last issue.
    Otherwise, create a new issue.
    """
    if issue_ongoing:
        issues[-1].append((time, pose, message, additional_info, goal))
    else:
        issues.append([(time, pose, message, additional_info, goal)])
        issue_ongoing = True
    return issues, issue_ongoing


def process_messages(messages, logfile_path, error_log_path):
    """Read the messages and select the appropriate action for each message"""

    global verbose
    # List of goal, the time at which they were sent, and the time at which they were reached
    goals = []

    ground_truth_path = []
    current_time = 0.0  # Initialize current time
    start_time = None
    end_time = None
    current_map_to_odom = None
    current_odom_to_base_link = None
    pose_initialized = False
    initial_pose = None
    current_pose = None
    current_pose_tuple = None
    estimated_path = []
    # The issues list contains the difficult situations that lead to errors.
    # It is a list of list, each list containing tuples with : (timestamp,estimated_pose, issue, additional_info(such as a value), goal )
    # We will then be able to use this issues list to learn and understand what happens to the robot.
    issues = []
    issue_ongoing = False

    with open(logfile_path, mode="+a", encoding="utf-8") as info_log, open(
        error_log_path, mode="+a", encoding="utf-8"
    ) as error_log:
        try:
            for topic, data, timestamp in messages:
                if topic == "/clock":
                    # Adjust current_time
                    current_time = data.clock.sec + data.clock.nanosec * 1e-9
                    if verbose:
                        print(f"Current time updated: {current_time}s")
                elif not pose_initialized and topic == "/odom":
                    # Initialize the pose from the first odom message
                    assert isinstance(data, Odometry)
                    initial_pose = data.pose.pose
                    pose_initialized = True
                    if verbose:
                        print(
                            f"Initial pose set at {current_time}s: {initial_pose.position.x}, {initial_pose.position.y}"
                        )
                elif topic == "/tf":
                    assert isinstance(data, TFMessage)
                    for transform in data.transforms:
                        assert isinstance(transform, TransformStamped)
                        # We only consider map->odom->base_link transform
                        if (
                            transform.header.frame_id == "map"
                            and transform.child_frame_id == "odom"
                        ):
                            current_map_to_odom = transform
                        elif (
                            transform.header.frame_id == "odom"
                            and transform.child_frame_id == "base_link"
                        ):
                            current_odom_to_base_link = transform
                        if (
                            pose_initialized
                            and current_map_to_odom
                            and current_odom_to_base_link
                        ):
                            current_pose = do_transform_pose(
                                do_transform_pose(initial_pose, current_map_to_odom),
                                current_odom_to_base_link,
                            )
                            current_pose_tuple = (
                                current_pose.position.x,
                                current_pose.position.y,
                                current_pose.position.z,
                                current_pose.orientation.x,
                                current_pose.orientation.y,
                                current_pose.orientation.z,
                                current_pose.orientation.w,
                            )
                            estimated_path.append(current_pose_tuple)

                elif topic == "/goal_sent":
                    # Add the goal and the timestamp to the message.
                    goals.append([(data.x, data.y), current_time, None])
                    if not start_time:
                        start_time = current_time
                        info_log.write(
                            format_log(
                                timestamp,
                                start_time,
                                "info",
                                f"Starting run at sim time {start_time}.",
                            )
                        )
                    info_log.write(
                        format_log(
                            timestamp,
                            current_time,
                            "info",
                            f"Goal sent at {current_time}s: {data.x}, {data.y}",
                        )
                    )
                    if verbose:
                        print(f"Goal sent at {current_time}s: {data.x}, {data.y}")

                elif topic == "/goal_reached":
                    # Find the goal that was reached and update its time
                    # It should be last goal of the list
                    # With explore-lite, goals are often cancelled before being reached;
                    # so the information of a goal being reached is not really useful.
                    goal_point = (data.x, data.y)
                    goal_found = False
                    for i in range(len(goals) - 1, -1, -1):
                        if goals[i][0] == goal_point:
                            # Set the current time as the reached time
                            goals[i][2] = current_time
                            goal_found = True
                            info_log.write(
                                format_log(
                                    timestamp,
                                    current_time,
                                    "info",
                                    f"Goal reached at {current_time}s, after {current_time-goals[i][1]}s: {goals[i][0]}",
                                )
                            )
                            break
                    if not goal_found and verbose:
                        print(
                            f"Warning: Goal reached signal received at {current_time} but no matching goal found"
                        )
                    # We update the end_time of the exploration
                    end_time = current_time

                elif topic == "/ground_truth":
                    # We save the ground truth position of the robot
                    # TODO: data is a nav_msgs/msg/Odometry msg, maybe we should convert it to a PoseStamped?
                    ground_truth_path.append(data)
                    if verbose:
                        print(
                            f"Ground truth position at {data.header.stamp.sec + data.header.stamp.nanosec * 1e-9}s: "
                            f"{data.pose.pose.position.x}, {data.pose.pose.position.y}"
                        )
                elif topic == "/rosout":
                    # Parse the logs
                    assert isinstance(data, rcl_interfaces.msg.Log)
                    msg = data.msg
                    name = data.name
                    level = data.level

                    should_log = should_log_message(msg, name)
                    if should_log[0]:
                        info_log.write(format_log(timestamp, current_time, name, msg))
                        # Depending of the reason why we log the message, we save some information, to create the tree that lead to difficult situations
                        reason = should_log[1]

                        if reason == "nav goal has succeeded":
                            issues[-1].append(
                                (
                                    current_time,
                                    current_pose_tuple,
                                    reason,
                                    None,
                                    msg[:8],
                                )
                            )
                            issue_ongoing = False

                        elif reason == "was successful":
                            issues[-1].append(
                                (
                                    current_time,
                                    current_pose_tuple,
                                    reason,
                                    ast.literal_eval(
                                        msg.split()[1] + msg.split()[2],
                                    ),  # Goal destination
                                    None,
                                )
                            )
                            issue_ongoing = False
                        elif (
                            reason
                            in [
                                "spin goal has been aborted",
                                "spin goal has been canceled",
                                "backup goal has been aborted",
                                "backup goal has been canceled",
                                "computepathtopose goal has been aborted",
                                "computepathtopose goal has been canceled",
                                "followpath goal has been aborted",
                                "followpath goal has been canceled",
                            ]
                            and issue_ongoing
                        ):
                            issues, issue_ongoing = add_issue(
                                current_time,
                                current_pose_tuple,
                                reason,
                                None,
                                msg[:8],
                                issues,
                                issue_ongoing,
                            )

                        elif reason == "backup goal has succeeded" and issue_ongoing:
                            issues, issue_ongoing = add_issue(
                                current_time,
                                current_pose_tuple,
                                reason,
                                float(msg.split()[-1]),  # Distance traveled
                                msg[:8],
                                issues,
                                issue_ongoing,
                            )

                        elif reason == "spin goal has succeeded" and issue_ongoing:
                            issues, issue_ongoing = add_issue(
                                current_time,
                                current_pose_tuple,
                                reason,
                                float(msg.split()[-1]),  # Angular distance traveled
                                msg[:8],
                                issues,
                                issue_ongoing,
                            )

                        elif (
                            reason
                            in [
                                "nav goal has been aborted",
                                "nav goal has been canceled",
                            ]
                            and issue_ongoing
                        ):
                            issues, issue_ongoing = add_issue(
                                current_time,
                                current_pose_tuple,
                                reason,
                                None,
                                msg[:8],
                                issues,
                                issue_ongoing,
                            )

                        elif reason == "too close to an obstacle":
                            issues, issue_ongoing = add_issue(
                                current_time,
                                current_pose_tuple,
                                reason,
                                (float(msg.split()[8]), float(msg.split()[11][:-1])),
                                None,
                                issues,
                                issue_ongoing,
                            )

                        elif (
                            reason
                            in [
                                "no longer too close to an obstacle",
                                "for spin behavior",
                                "running wait",
                                "wait completed successfully",
                                "running backup",
                                "backup completed successfully",
                            ]
                            and issue_ongoing
                        ):
                            # For things we wan't to log only if there is an ongoing issue
                            issues, issue_ongoing = add_issue(
                                current_time,
                                current_pose_tuple,
                                msg,
                                None,
                                None,
                                issues,
                                issue_ongoing,
                            )

                        elif reason in ["failed to create", "collision"]:
                            issues, issue_ongoing = add_issue(
                                current_time,
                                current_pose_tuple,
                                msg,
                                None,
                                None,
                                issues,
                                issue_ongoing,
                            )

                        elif reason == "failed to generate a valid path":
                            issues, issue_ongoing = add_issue(
                                current_time,
                                current_pose_tuple,
                                reason,
                                ast.literal_eval(
                                    msg.split()[-2] + msg.split()[-1]
                                ),  # Goal destination
                                None,
                                issues,
                                issue_ongoing,
                            )

                        elif reason == "was aborted by the navigation stack":
                            issues, issue_ongoing = add_issue(
                                current_time,
                                current_pose_tuple,
                                reason,
                                ast.literal_eval(
                                    msg.split()[1] + msg.split()[2]
                                ),  # Goal destination
                                None,
                                issues,
                                issue_ongoing,
                            )

                    # Check for error conditions in the output
                    # TODO : not sure if this is really useful
                    if "error" in msg.lower():
                        error_log.write(
                            f"{format_time(timestamp)}: Exploration Error - [{name}] {msg.strip()}\n"
                        )
                    elif "abort" in msg.lower():  #
                        error_log.write(
                            f"{format_time(timestamp)}: Exploration Aborted - [{name}] {msg.strip()}\n"
                        )
                    elif "stuck" in msg.lower():
                        error_log.write(
                            f"{format_time(timestamp)}: Robot Stuck - [{name}] {msg.strip()}\n"
                        )
                    elif "timeout" in msg.lower():
                        error_log.write(
                            f"{format_time(timestamp)}: Operation Timeout - [{name}] {msg.strip()}\n"
                        )
                    elif level >= WARN_LEVEL:
                        error_log.write(
                            f"{format_time(timestamp)}: Geneneric error - [{name}] {msg.strip()}\n"
                        )

                    if "exploration stopped." in msg.lower():
                        info_log.write(
                            format_log(
                                timestamp,
                                current_time,
                                "info",
                                f"Finished exploration, lasting {(current_time - start_time):.1f}s : [{name}] {msg.strip()}",
                            )
                        )
                        break

        except KeyboardInterrupt as e:
            error_msg = f"Logging Interrupted by user.\n"
            info_log.write(error_msg)
            error_log.write(error_msg)

    return (ground_truth_path, estimated_path, goals, start_time, end_time, issues)


def main():

    global verbose

    args = parse_args()
    run_path = os.path.abspath(args.run_path)
    verbose = args.verbose

    bag_path = os.path.join(run_path, "rosbags/")

    # Create the logs folder and paths
    logs_folder = os.path.join(run_path, "logs")
    if not os.path.exists(logs_folder):
        os.makedirs(logs_folder)
    logfile_path = os.path.join(logs_folder, "log.txt")
    error_log_path = os.path.join(logs_folder, "error_log.txt")

    messages = read_rosbag2(bag_path)

    robot_path, estimated_path, goals, start_time, end_time, issues = process_messages(
        messages, logfile_path, error_log_path
    )

    # Save the issues to a JSON file
    issues_path = os.path.join(logs_folder, "issues.json")
    with open(issues_path, "w") as issues_file:
        json.dump(issues, issues_file, indent=4)


if __name__ == "__main__":
    main()
