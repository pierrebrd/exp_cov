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
        help="Coordinate X e Y della posa del robot come due numeri (es: '-5.0 -5.0')",
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
        print(f"\nEsecuzione comando: {' '.join(command)}\n")
        subprocess.run(command, check=True)
        print(f"\nCompletato con successo\n\n")
    except subprocess.CalledProcessError as e:
        print(f"\nErrore: {error_msg}")
        print(f"Comando fallito: {' '.join(command)}")
        print(f"Codice di uscita: {e.returncode}")
        sys.exit(1)


def run_ampl_solver(model_path, data_path, solution_dir):
    """
    Risolve il problema di ottimizzazione usando AMPL con CPLEX attraverso amplpy

    Args:
        model_path: Percorso al file .mod
        data_path: Percorso al file .dat
        solution_dir: Directory dove salvare i risultati
    """

    try:
        # Inizializza AMPL
        ampl = AMPL()

        # Carica il modello e i dati
        ampl.read(str(model_path))  # Carica il file .mod
        ampl.readData(str(data_path))  # Carica il file .dat

        # Imposta il solver su CPLEX
        ampl.setOption("solver", "cplex")

        # Risolvi il modello
        ampl.solve()

        # Verifica che la soluzione sia ottima
        if ampl.getValue("solve_result") != "solved":
            raise Exception(f"Errore nella soluzione: {ampl.getValue('solve_result')}")

        # Ottieni la variabile guard_choice
        guard_choice = ampl.getVariable("guard_choice")

        # Salva solo i valori non nulli
        output_path = os.path.join(solution_dir, "chosen_guards.txt")
        with open(output_path, "w") as f:
            f.write("guard_choice [*] :=\n")
            # Itera sulle istanze in modo corretto
            for i in range(1, guard_choice.numInstances() + 1):
                try:
                    val = guard_choice[i].value()
                    if (
                        val > 0.5
                    ):  # Per gestire errori di arrotondamento nelle variabili binarie
                        f.write(f"{i} 1\n")
                except KeyError:
                    continue
            f.write(";\n")

        print(f"Soluzione salvata in {output_path}")

    except Exception as e:
        print(f"Errore durante l'esecuzione di AMPL: {str(e)}")
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
            print(f"Errore: {name} non trovato in: {path}")
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
    run_command(map_cmd, "Generazione mappe fallita")

    print(f"Merging maps")
    fuse_cmd = [
        "python3",
        "fuse_maps_floorplan.py",
        "--dir",
        str(dest_dir),
        "--output-dir",
        str(solution_dir),
    ]
    run_command(fuse_cmd, "Fusione mappe fallita")

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
    run_command(opt_data_cmd, "Generazione dati ottimizzazione fallita")

    print(f"Solving linear problem with cplex")
    try:
        run_ampl_solver(
            args.model, os.path.join(solution_dir, "data.dat"), solution_dir
        )
    except Exception as e:
        print(f"Errore nell'ottimizzazione AMPL: {e}")
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
    run_command(tsp_cmd, "Calcolo TSP fallito")

    print(f"Running runs")
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
    run_command(explore_cmd, "Esplorazione e waypoints falliti")

    print("\nWorkflow completed successfully!")
    print(f"Results saved in: {dest_dir}")
    print(f"Soluzioni in: {solution_dir}")
    print(f"Exploration runs in: {runs_dir}")


if __name__ == "__main__":
    main()
