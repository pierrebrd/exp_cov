# NOTE: This script has not been ported!!

import argparse
import os
import subprocess
import sys
from pathlib import Path
from amplpy import AMPL


def resolve_path(path_str):
    """
    Resolves a relative or absolute path to an absolute path.

    Args:
        path_str (str): Relative or absolute path

    Returns:
        Path: Resolved absolute path
    """
    path = Path(path_str)
    if path.is_absolute():
        return path
    return Path.cwd() / path


def check_pose(value):
    """
    Checks that the pose value is in the correct format "X Y"

    Args:
        value (str): String containing two numbers separated by space

    Returns:
        tuple: Pair of floats (x, y)
    """
    try:
        ret_value = value.split()
        if len(ret_value) != 2:
            raise argparse.ArgumentTypeError(
                f"Pose '{value}' must be made of 2 numbers"
            )
        return (float(ret_value[0]), float(ret_value[1]))
    except ValueError:
        raise Exception(f"{value} is not made of 2 numbers")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the complete exploration and coverage workflow"
    )

    # Directory arguments
    parser.add_argument(
        "--dest-dir",
        required=True,
        type=str,
        help="Destination directory for all outputs (relative or absolute path)",
    )
    parser.add_argument(
        "--runs-dir",
        required=False,
        type=str,
        help="Directory for explore and waypoint runs (relative or absolute path)",
    )

    # map_rgb_simul arguments
    parser.add_argument(
        "--map",
        required=True,
        type=str,
        help="Map rgb image (relative or absolute path)",
    )
    parser.add_argument(
        "--mask",
        required=True,
        type=str,
        help="Movement mask image (relative or absolute path)",
    )
    parser.add_argument(
        "--n-maps", type=int, default=10, help="Number of maps to generate"
    )
    parser.add_argument(
        "--n-worlds", type=int, default=10, help="Number of worlds to generate"
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=0.035888,
        help="Number of meters per pixel in png map.",
    )
    parser.add_argument(
        "--pose",
        type=check_pose,
        default=(0.0, 0.0),
        help="Robot pose X and Y coordinates as two numbers (e.g., '-5.0 -5.0')",
    )

    # fuse_maps_floorplan arguments

    # optimization_data arguments
    parser.add_argument(
        "--max-guards",
        default=100,
        type=int,
        help="Number of guards.",
        metavar="GUARDS",
    )
    parser.add_argument(
        "--witnesses",
        default=100,
        type=int,
        help="Number of witnesses.",
        metavar="WITNESSES",
    )

    # AMPL solver arguments
    parser.add_argument(
        "--model",
        required=True,
        type=str,
        help="Path to AMPL model file (relative or absolute path)",
    )

    # tsp arguments

    # Additional arguments for run_explore_and_waypoint
    parser.add_argument("--world", required=True, type=str, help="World file number")
    parser.add_argument("-r", "--runs", type=int, default=1, help="Number of runs")

    args = parser.parse_args()

    # Resolve all paths to absolute paths
    args.dest_dir = resolve_path(args.dest_dir)
    if args.runs_dir:
        args.runs_dir = resolve_path(args.runs_dir)
    args.map = resolve_path(args.map)
    args.mask = resolve_path(args.mask)
    args.model = resolve_path(args.model)
    args.world = os.path.join(args.dest_dir, f"world{args.world}.world")

    return args


def run_command(command, error_msg):
    try:
        print(f"\nExecuting command: {' '.join(command)}\n")
        subprocess.run(command, check=True)
        print(f"\nCompleted successfully\n\n")
    except subprocess.CalledProcessError as e:
        print(f"\nError: {error_msg}")
        print(f"Command failed: {' '.join(command)}")
        print(f"Exit code: {e.returncode}")
        sys.exit(1)


def run_ampl_solver(model_path, data_path, solution_dir):
    """
    Solves the optimization problem using AMPL with CPLEX via amplpy

    Args:
        model_path: Path to the .mod file
        data_path: Path to the .dat file
        solution_dir: Directory where results will be saved
    """

    try:
        # Initialize AMPL
        ampl = AMPL()

        # Load model and data
        ampl.read(str(model_path))  # Load .mod file
        ampl.readData(str(data_path))  # Load .dat file

        # Set solver to CPLEX
        ampl.setOption("solver", "cplex")

        # Solve the model
        ampl.solve()

        # Verify that the solution is optimal
        if ampl.getValue("solve_result") != "solved":
            raise Exception(f"Error in solution: {ampl.getValue('solve_result')}")

        # Get the guard_choice variable
        guard_choice = ampl.getVariable("guard_choice")

        # Save only non-zero values
        output_path = os.path.join(solution_dir, "chosen_guards.txt")
        with open(output_path, "w") as f:
            f.write("guard_choice [*] :=\n")
            # Iterate over instances correctly
            for i in range(1, guard_choice.numInstances() + 1):
                try:
                    val = guard_choice[i].value()
                    if val > 0.5:  # To handle rounding errors in binary variables
                        f.write(f"{i} 1\n")
                except KeyError:
                    continue
            f.write(";\n")

        print(f"Solution saved to {output_path}")

    except Exception as e:
        print(f"Error during AMPL execution: {str(e)}")
        raise


def main():
    args = parse_args()

    # Check that input files exist
    required_files = {
        "Map file": args.map,
        "Mask file": args.mask,
        "Model file": args.model,
    }

    for name, path in required_files.items():
        if not path.exists():
            print(f"Error: {name} not found at: {path}")
            sys.exit(1)

    # Create directories
    dest_dir = args.dest_dir
    solution_dir = Path(os.path.join(dest_dir, "solution"))
    runs_dir = (
        Path(args.runs_dir)
        if args.runs_dir
        else Path(os.path.join(solution_dir, "runs"))
    )

    os.makedirs(dest_dir, exist_ok=True)
    os.makedirs(solution_dir, exist_ok=True)
    os.makedirs(runs_dir, exist_ok=True)

    # Script execution
    # Script execution
    print(f"Creating maps")
    map_cmd = [
        "python3",
        "map_rgb_simul.py",
        "--map",
        str(args.map),
        "--mask",
        str(args.mask),
        "--batch",
        str(args.n_maps),
        "--worlds",
        str(args.n_worlds),
        "--dir",
        str(dest_dir),
        "--pose",
        f"{args.pose[0]} {args.pose[1]}",
        "--scale",
        str(args.scale),
    ]
    run_command(map_cmd, "Map generation failed")

    print(f"Merging maps")
    fuse_cmd = [
        "python3",
        "fuse_maps_floorplan.py",
        "--dir",
        str(dest_dir),
        "--output-dir",
        str(solution_dir),
    ]
    run_command(fuse_cmd, "Map fusion failed")

    print(f"Generating optimization data")
    opt_data_cmd = [
        "python3",
        "optimization_data.py",
        "--img",
        str(os.path.join(solution_dir, "otsu.png")),
        "--output",
        str(os.path.join(solution_dir, "data.dat")),
        "--costs",
        str(os.path.join(solution_dir, "costs.txt")),
        "--max-guards",
        str(args.max_guards),
        "--witnesses",
        str(args.witnesses),
    ]
    run_command(opt_data_cmd, "Optimization data generation failed")

    print(f"Solving linear problem with cplex")
    try:
        run_ampl_solver(
            args.model, os.path.join(solution_dir, "data.dat"), solution_dir
        )
    except Exception as e:
        print(f"AMPL optimization error: {e}")
        sys.exit(1)

    print(f"Calculating path")
    tsp_cmd = [
        "python3",
        "tsp.py",
        "--data",
        str(os.path.join(solution_dir, "data.dat")),
        "--guards",
        str(os.path.join(solution_dir, "chosen_guards.txt")),
        "--img",
        str(os.path.join(solution_dir, "otsu.png")),
        "--scale",
        str(args.scale),
        "--output",
        str(os.path.join(solution_dir, "waypoints.csv")),
    ]
    run_command(tsp_cmd, "TSP calculation failed")

    print(f"Running exploration")
    explore_cmd = [
        "python3",
        "logged_run.py",
        "--waypoints",
        str(os.path.join(solution_dir, "waypoints.csv")),
        "--world",
        str(args.world),
        "--runs",
        str(args.runs),
        "--dir",
        str(runs_dir),
    ]
    run_command(explore_cmd, "Exploration and waypoints failed")

    print("\nWorkflow completed successfully!")
    print(f"Results saved in: {dest_dir}")
    print(f"Solutions in: {solution_dir}")
    print(f"Exploration runs in: {runs_dir}")


if __name__ == "__main__":
    main()
