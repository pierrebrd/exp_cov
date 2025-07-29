"""
Read a ROS2 bag file and extract information about the run.
For now, the information extracted is similar to the one in logged_run.py in the initial exp_cov in ROS1,
but due to the differences in ROS2, it is easier to parse the bag file than to monitor the run logs live.
"""

# Required imports
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
        "was successful",
        "was aborted",
        "failed to create",
        "failed to generate a valid path",
        "collision",
        "for spin behavior",
        "running wait",
        "wait completed successfully",
        "running backup",
        "blacklist",
        "black list",
        "found frontiers",  # Not working
        "waiting for costmap",
        "final distance",  # distance_check
        "too close to an obstacle",  # laser_scan_check
        "aborted",
        "cancelled",
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
        return False

    # If the message contains one of the important strings, log it
    return any(x in msg_lower for x in important_messages) or any(
        x in name_lower for x in important_messages
    )


def process_messages(messages, logfile_path, error_log_path):
    """Read the messages and select the appropriate action for each message"""

    global verbose
    # List of goal, the time at which they were sent, and the time at which they were reached
    goals = []

    robot_path = []
    current_time = 0.0  # Initialize current time
    start_time = None
    end_time = None

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

                    robot_path.append(
                        (
                            data.header.stamp.sec + data.header.stamp.nanosec * 1e-9,
                            (data.pose.pose.position.x, data.pose.pose.position.y),
                            (
                                data.pose.pose.orientation.z,
                                data.pose.pose.orientation.w,
                            ),
                        )
                    )
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

                    if should_log_message(msg, name):
                        info_log.write(format_log(timestamp, current_time, name, msg))

                    # Check for error conditions in the output
                    if "error" in msg.lower():
                        error_log.write(
                            f"{format_time(timestamp)}: Exploration Error - [{name}] {msg.strip()}\n"
                        )
                    elif "abort" in msg.lower():  # TODO Not sure it is useful
                        error_log.write(
                            f"{format_time(timestamp)}: Exploration Aborted - [{name}] {msg.strip()}\n"
                        )
                    elif "stuck" in msg.lower():  # TODO Not sure it is useful
                        error_log.write(
                            f"{format_time(timestamp)}: Robot Stuck - [{name}] {msg.strip()}\n"
                        )
                    elif "timeout" in msg.lower():  # TODO Not sure it is useful
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
        except Exception as e:
            error_msg = f"Unexpected error during exploration: {str(e)}\n"
            error_log.write(error_msg)

    return (robot_path, goals, start_time, end_time)


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

    robot_path, goals, start_time, end_time = process_messages(
        messages, logfile_path, error_log_path
    )

    # TODO: add ape and rpe stats to the log files
    # ape_stats, rpe_stats = evo_metrics(
    #     run_path, "/ground_truth", "/tf:map.base_link", plot=False, align_origin=False
    # )
    # if ape_stats:
    #     print("APE stats:")
    #     print(ape_stats)
    # else:
    #     print("No APE stats available or an error occurred.")

    # if rpe_stats:
    #     print("RPE stats:")
    #     print(rpe_stats)
    # else:
    #     print("No RPE stats available or an error occurred.")


if __name__ == "__main__":
    main()
