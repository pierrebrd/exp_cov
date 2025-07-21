import os
import sys
from amplpy import AMPL
import argparse
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(
        description="Launches a solver with AMPL, and logs the results."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="model file to use",
    )
    parser.add_argument("--data", required=True, help="Data file path")
    parser.add_argument("--solver", default="cplex", help="Solver to use")

    parser.add_argument("--output-dir", default=None, help="Output directory")
    return parser.parse_args()


def write_results(
    output_file,
    ampl,
):
    with open(output_file, "w") as f:
        solve_result = ampl.get_value("solve_result")
        objective = ampl.get_value("z")
        nG = ampl.get_parameter("nG").value()
        nW = ampl.get_parameter("nW").value()
        guards_chosen = sum(
            ampl.get_variable("guard_choice").get_values().to_dict().values()
        )
        witnesses_covered = sum(
            ampl.get_variable("covered").get_values().to_dict().values()
        )
        guard_choice = ampl.get_variable("guard_choice").get_values().to_dict()
        guard_cost = ampl.get_parameter("guard_cost").get_values().to_dict()
        coverage = ampl.get_parameter("coverage").get_values().to_dict()
        covered = ampl.get_variable("covered").get_values().to_dict()
        Guards = ampl.get_set("Guards").to_list()
        Witnesses = ampl.get_set("Witnesses").to_list()
        distance = ampl.get_parameter("distance").get_values().to_dict()

        f.write(f"=== COVERAGE OPTIMIZATION RESULTS ===\n")
        f.write(f"Solve status: {solve_result}\n")
        f.write(f"Objective value: {objective:.6f}\n")

        f.write(f"Number of guards chosen: {guards_chosen:.0f} out of {nG}\n")
        f.write(f"Number of witnesses covered: {witnesses_covered:.0f} out of {nW}\n")
        f.write(f"Coverage percentage: {100 * witnesses_covered / nW:.2f}%\n")
        f.write(
            f"Total guard cost: {sum(guard_cost[g] * guard_choice[g] for g in Guards):.6f}\n"
        )
        f.write(
            f"Total distance cost: {sum(guard_choice[g1] * guard_choice[g2] * distance[g1, g2] for g1 in Guards for g2 in Guards):.6f}\n"
        )

        f.write(f"\n=== CHOSEN GUARDS ===\n")
        for g, choice in guard_choice.items():
            if choice > 0.5:
                f.write(f"Guard {g}: cost = {guard_cost[g]:.6f}\n")

        f.write(f"\n=== COVERAGE DETAILS ===\n")
        f.write("Witness -> Covering Guards\n")
        for w in Witnesses:
            f.write(f"Witness {w}: ")
            covering_guards = [
                str(g)
                for g in Guards
                if coverage.get((w, g), False) and guard_choice.get(g, 0) > 0.5
            ]
            f.write(" ".join(covering_guards) + " ")
            status = "YES" if covered.get(w, 0) > 0.5 else "NO"
            f.write(f"(covered: {status})\n")


def main():

    args = parse_args()
    model = os.path.abspath(args.model)
    data = os.path.abspath(args.data)
    solver = args.solver
    output_dir = args.output_dir

    if args.output_dir is None:
        output_dir = os.path.dirname(data)
    else:
        output_dir = os.path.abspath(args.output_dir)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Solve the problem

    ampl = AMPL()

    # Load model and data
    print(f"Loading model: {model}")
    ampl.read(model)

    print(f"Loading data: {data}")
    ampl.read_data(data)

    # Set solver
    ampl.set_option("solver", solver)

    # Solve
    print("Solving...")
    ampl.solve()

    # Get results
    solve_result = ampl.get_value("solve_result")

    if solve_result == "solved":

        # Generate output filename based on model type and timestamp
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
        output_file = os.path.join(output_dir, f"solving_results_{timestamp}.txt")

        # Write results
        write_results(
            output_file,
            ampl,
        )

        print(f"Solution found and saved to: {output_file}")

    else:
        print(f"Optimization failed: {solve_result}")

    ampl.close()


if __name__ == "__main__":
    main()
