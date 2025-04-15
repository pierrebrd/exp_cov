"""
This script executes exploration vs coverage navigation comparisons using ROS and Stage simulator.

It runs multiple experiments comparing two robot navigation strategies:
1. Exploration using explore_lite
2. Coverage navigation using predefined waypoints

For each run, it:
- Creates a new directory for the run's data
- Executes exploration and measures time/area covered
- Executes waypoint coverage and measures time/area covered 
- Saves maps and logs for both strategies
- Compares and records the differences in time and area coverage

The script uses ROS nodes for:
- Stage simulation
- SLAM (slam_toolbox)
- Distance checking
- Map saving

Requirements:
- ROS with stage_ros
- explore_lite package
- slam_toolbox package
- Proper ROS workspace setup with exp_cov package
"""

import subprocess as sp
from time import gmtime, strftime, sleep
import argparse
import cv2
from PIL import Image
import numpy as np
import os
import rospy

def parse_args():
    """
    Parse and validate command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments containing:
            - waypoints: Path to CSV file with waypoints
            - world: Path to Stage world file
            - runs: Number of test runs to execute (default: 1)
    """
    parser = argparse.ArgumentParser(description='Start exploration, logging time info to file and saving the final map.')
    parser.add_argument('--waypoints', required=True, help="Path to the waypoints csv file.")
    parser.add_argument('--world', required=True, help="Path to the stage world file.")
    parser.add_argument('-r', '--runs', required=False, default=1,  type=check_positive, help="Number of tests to run.", metavar="RUNS")
    parser.add_argument('-d', '-dir', required=False, default="", help="Directory to save the run data.", metavar="DIR")
    return parser.parse_args()

def now():
    """
    Get current timestamp in readable format.

    Returns:
        str: Current time in YYYY-MM-DD HH:MM:SS format
    """
    return strftime("%Y-%m-%d %H:%M:%S", gmtime())

def run_expl(logfile_path, run_subfolder = ""):
    """
    Run exploration using explore_lite and save the resulting map.

    Args:
        logfile_path (str): Path where to save exploration logs
        run_subfolder (str): Subfolder for the current run's data

    Returns:
        int: Time taken for exploration in seconds
    """
    start = None
    args = ["roslaunch", "exp_cov", "explore_lite2.launch"]
    error_log_path = os.path.join(run_subfolder, "exploreErrorLog.txt")
    
    with sp.Popen(args, stdout=sp.PIPE, stderr=sp.STDOUT) as process:
        with open(logfile_path, mode="+a", encoding="utf-8") as logfile, \
             open(error_log_path, mode="+a", encoding="utf-8") as error_log:
            try:
                start = rospy.get_rostime().secs
                logfile.write(f"{now()}: Starting exploration.\n")
                for line in process.stdout:
                    line = line.decode('utf8')
                    # Check for error conditions in the output
                    if "error" in line.lower():
                        error_log.write(f"{now()}: Exploration Error - {line.strip()}\n")
                    if "abort" in line.lower():
                        error_log.write(f"{now()}: Exploration Aborted - {line.strip()}\n")
                    if "stuck" in line.lower():
                        error_log.write(f"{now()}: Robot Stuck - {line.strip()}\n")
                    if "timeout" in line.lower():
                        error_log.write(f"{now()}: Operation Timeout - {line.strip()}\n")
                        
                    if line.strip()[1:].startswith("["):
                        if "exploration stopped." in line.lower():
                            logfile.write(f"{now()}: Finished exploration.\n")
                            break
            except KeyboardInterrupt as e:
                error_msg = f"{now()}: Exploration Interrupted by user.\n"
                logfile.write(error_msg)
                error_log.write(error_msg)
            except Exception as e:
                error_msg = f"{now()}: Unexpected error during exploration: {str(e)}\n"
                error_log.write(error_msg)
            finally:
                time = rospy.get_rostime().secs - start
                logfile.write(f"{now()}: Exploration ros time is {strftime('%H:%M:%S', gmtime(time))}.\n")
                process.kill()
                map_name = os.path.join(run_subfolder, "Map_exploration")
                save_map = ["rosrun", "map_server", "map_saver", "-f", map_name]
                sp.run(save_map)
                try:
                    Image.open(f"{map_name}.pgm").save(f"{map_name}.png")
                    os.remove(f"{map_name}.pgm")
                except IOError:
                    print("Cannot convert pgm map to png.")
                finally:
                    return time

def run_cov(waypoints, logfile_path="./coverage.log", run_subfolder = ""):
    """
    Run waypoint coverage navigation and save the resulting map.

    Args:
        waypoints (str): Path to waypoints CSV file
        logfile_path (str): Path where to save coverage logs
        run_subfolder (str): Subfolder for the current run's data

    Returns:
        int: Time taken for coverage navigation in seconds
    """
    start = None
    args = ["rosrun", "exp_cov", "waypoint_navigation.py", "-p", waypoints]
    error_log_path = os.path.join(run_subfolder, "coverageErrorLog.txt")
    
    with sp.Popen(args, stdout=sp.PIPE, stderr=sp.STDOUT) as process:
        with open(logfile_path, mode="+a", encoding="utf-8") as logfile, \
             open(error_log_path, mode="+a", encoding="utf-8") as error_log:
            try:
                start = rospy.get_rostime().secs
                logfile.write(f"{now()}: Starting waypoint navigation.\n")
                for line in process.stdout:
                    line = line.decode('utf8')
                    # Check for error conditions in the output
                    if "error" in line.lower():
                        error_log.write(f"{now()}: Navigation Error - {line.strip()}\n")
                    if "abort" in line.lower():
                        error_log.write(f"{now()}: Navigation Aborted - {line.strip()}\n")
                    if "stuck" in line.lower():
                        error_log.write(f"{now()}: Robot Stuck - {line.strip()}\n")
                    if "timeout" in line.lower():
                        error_log.write(f"{now()}: Operation Timeout - {line.strip()}\n")
                        
                    if "final goal" in line.lower():
                        logfile.write(f"{now()}: Finished waypoint navigation.\n")
                        break
            except KeyboardInterrupt as e:
                error_msg = f"{now()}: Waypoint navigation Interrupted by user.\n"
                logfile.write(error_msg)
                error_log.write(error_msg)
            except Exception as e:
                error_msg = f"{now()}: Unexpected error during navigation: {str(e)}\n"
                error_log.write(error_msg)
            finally:
                time = rospy.get_rostime().secs - start
                logfile.write(f"{now()}: Waypoint navigation ros time is {strftime('%H:%M:%S', gmtime(time))}.\n")
                process.kill()
                map_name = os.path.join(run_subfolder, "Map_coverage")
                save_map = ["rosrun", "map_server", "map_saver", "-f", map_name]
                sp.run(save_map)
                try:
                    Image.open(f"{map_name}.pgm").save(f"{map_name}.png")
                    os.remove(f"{map_name}.pgm")
                except IOError:
                    print("Cannot convert pgm map to png.")
                finally:
                    return time

def run_exploration(cmd_args, logfile_path, run_subfolder):
    """
    Set up and execute the complete exploration process.
    
    Launches required ROS nodes (Stage, SLAM, distance checker) and
    runs the exploration strategy.

    Args:
        cmd_args: Command line arguments
        logfile_path (str): Path for logging
        run_subfolder (str): Subfolder for the current run's data

    Returns:
        int: Total exploration time in seconds
    """
    print("starting exploration.")
    stage_args = ["roslaunch", "exp_cov", "stage_init.launch", f"worldfile:={cmd_args.world}"]
    slam_args = ["roslaunch", "exp_cov", "slam_toolbox_no_rviz.launch"]
    dist_args = ["rosrun", "exp_cov", "distance_check.py"]
    with sp.Popen(stage_args, stdout=sp.DEVNULL, stderr=sp.DEVNULL) as stage_process:
        sleep(3)
        print("started stage.")
        with sp.Popen(slam_args, stdout=sp.DEVNULL, stderr=sp.DEVNULL) as slam_process:
            sleep(10)
            print("started slam.")
            with open(logfile_path, mode="+a", encoding="utf-8") as logfile:
                with sp.Popen(dist_args, stdout=logfile, stderr=logfile) as dist_process:
                    time = run_expl(logfile_path, run_subfolder)
                    print("exploration finished.")
                    dist_process.terminate()
                    slam_process.terminate()
                    stage_process.terminate()
                    return time

def run_coverage(cmd_args, logfile_path, run_subfolder):
    """
    Set up and execute the complete coverage navigation process.
    
    Launches required ROS nodes (Stage, SLAM, distance checker) and
    runs the waypoint coverage strategy.

    Args:
        cmd_args: Command line arguments
        logfile_path (str): Path for logging
        run_subfolder (str): Subfolder for the current run's data

    Returns:
        int: Total coverage time in seconds
    """
    print("starting coverage.")
    stage_args = ["roslaunch", "exp_cov", "stage_init.launch", f"worldfile:={cmd_args.world}"]
    slam_args = ["roslaunch", "exp_cov", "waypoint_slam.launch"]
    dist_args = ["rosrun", "exp_cov", "distance_check.py"]
    with sp.Popen(stage_args, stdout=sp.DEVNULL, stderr=sp.DEVNULL) as stage_process:
        sleep(3)
        print("started stage.")
        with sp.Popen(slam_args, stdout=sp.DEVNULL, stderr=sp.DEVNULL) as slam_process:
            sleep(10)
            print("started slam.")
            with open(logfile_path, mode="+a", encoding="utf-8") as logfile:
                with sp.Popen(dist_args, stdout=logfile, stderr=logfile) as dist_process:
                    time = run_cov(cmd_args.waypoints, logfile_path, run_subfolder)
                    print("coverage finished.")
                    dist_process.terminate()
                    slam_process.terminate()
                    stage_process.terminate()
                    return time

def check_positive(value):
    """
    Validate that a string represents a positive integer.

    Args:
        value: Value to check

    Returns:
        int: The validated positive integer

    Raises:
        argparse.ArgumentTypeError: If value is not a positive integer
        Exception: If value cannot be converted to integer
    """
    try:
        value = int(value)
        if value <= 0:
            raise argparse.ArgumentTypeError("{} is not a positive integer".format(value))
    except ValueError:
        raise Exception("{} is not an integer".format(value))
    return value

def main(cmd_args):
    """
    Main execution logic for the exploration vs coverage comparison.

    For each run:
    1. Creates a new run directory (within parent directory if specified)
    2. Executes exploration and coverage tests
    3. Compares and logs:
        - Time differences between strategies
        - Area coverage differences between strategies
        - Saves exploration and coverage maps

    Args:
        cmd_args: Parsed command line arguments containing run parameters
    """
    logfile_path_exploration = "explore.log"
    logfile_path_coverage = "coverage.log"
    logfile_path_result = "result.log"

    # Crea la directory parent se specificata e non esiste
    parent_dir = cmd_args.d if cmd_args.d else "."
    if parent_dir != "." and not os.path.exists(parent_dir):
        os.makedirs(parent_dir)

    # Trova il numero massimo di run nella directory parent
    maxrun = 0
    for i in os.listdir(parent_dir):
        try:
            if int(i[3:]) >= maxrun:
                maxrun = int(i[3:]) + 1
        except:
            continue

    time_deltas = list()
    area_deltas = list()
    for r in range(int(cmd_args.runs)):
        print(f"run {r+1}/{cmd_args.runs} starting.")
        run_subfolder = os.path.join(parent_dir, f"run{maxrun+r}")
        os.mkdir(run_subfolder)
        logfile_path_exploration_run = os.path.join(run_subfolder, logfile_path_exploration)
        logfile_path_coverage_run = os.path.join(run_subfolder, logfile_path_coverage)
        logfile_path_result_run = os.path.join(run_subfolder, logfile_path_result)
        exploration_time = run_exploration(cmd_args, logfile_path_exploration_run, run_subfolder)
        sleep(2)
        coverage_time = run_coverage(cmd_args, logfile_path_coverage_run, run_subfolder)
        with open(logfile_path_result_run, mode="+a", encoding="utf-8") as logfile:
            time_delta = exploration_time-coverage_time
            time_deltas.append(exploration_time-coverage_time)
            msg = f"{now()}: Coverage time: {coverage_time}; Exploration time: {exploration_time}. Exploration - Coverage: {time_delta}. Unit is seconds."
            print(msg)
            logfile.write(f"{msg}\n")
            expl_map = cv2.imread(os.path.join(run_subfolder, "Map_exploration.png"), cv2.IMREAD_GRAYSCALE)
            cov_map = cv2.imread(os.path.join(run_subfolder, "Map_coverage.png"), cv2.IMREAD_GRAYSCALE)
            expl_map_area = np.sum(expl_map >= 250)
            cov_map_area = np.sum(cov_map >= 250)
            area_delta = expl_map_area-cov_map_area
            area_deltas.append(area_delta)
            msg = f"{now()}: Coverage mapped area: {cov_map_area}; Exploration mapped area: {expl_map_area}. Exploration - Coverage: {area_delta}. Unit is 0.05 meters, a pixel in the map."
            print(msg)
            logfile.write(f"{msg}\n")
        print(f"run {r+1}/{cmd_args.runs} finished.")
    print(f"time_deltas (exploration-coverage): {time_deltas}\n;\narea_deltas (exploration-coverage): {area_deltas}\n;\n")

if __name__ == "__main__":

    cmd_args = parse_args()
    with sp.Popen(["roscore"], stdout=sp.DEVNULL, stderr=sp.DEVNULL) as roscore_process:
        sleep(3)
        rospy.set_param('use_sim_time', True)
        rospy.init_node('just_for_time', anonymous=True)
        sleep(3)
        main(cmd_args)
        roscore_process.kill()

