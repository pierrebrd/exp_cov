# Python script not linked to ROS, doesn't need to be ported to ROS2

import cv2
import os
import argparse
import shapely
import shapely.validation
import matplotlib.pyplot as plt
import concurrent.futures
import numpy as np
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder


def print_simple_param(file, name, value):
    file.write(f"param {name} := {value};\n")


def print_vector_param(file, name, pairs):
    file.write(f"param {name} := \n")
    for index, value in pairs:
        file.write(f"{index} {value}\n")
    file.write(";\n")


def print_bidimensional_param(file, name, rows, cols, vals):
    file.write(f"param {name} : ")
    for col in range(1, cols + 1):
        file.write(f" {col} ")
    file.write(" := \n")
    for row in range(1, rows + 1):
        file.write(f"{row} ")
        for col in range(0, cols):
            file.write(f"{vals[row-1][col]} ")
        file.write("\n")
    file.write(";\n")


def coverage_w(coverage_w_data):
    MAX_VISIBILITY_RANGE = 10000
    i, poly, w, guards = coverage_w_data
    coverage_w = []
    for j, g in enumerate(guards):
        seg = shapely.LineString([g, w])
        if poly.contains(seg) and seg.length < MAX_VISIBILITY_RANGE:
            coverage_w.insert(j, 1)
        else:
            coverage_w.insert(j, 0)
    return (i, coverage_w)


def distance_gg(distance_gg_data):
    # TODO: I dont know if it's a normal behavior, but the distance between a guard and itself is 1. Also, we compute distances for each pair twice...
    dist_gg = []
    i, g1, guards, pathfinding_matrix = distance_gg_data
    finder = AStarFinder(diagonal_movement=DiagonalMovement.always)
    pathfinding_grid = Grid(matrix=pathfinding_matrix)
    g1_x, g1_y = g1
    for i2, g2 in enumerate(guards):
        pathfinding_grid.cleanup()
        start = pathfinding_grid.node(g1_x, g1_y)
        g2_x, g2_y = g2
        end = pathfinding_grid.node(g2_x, g2_y)
        path, _ = finder.find_path(start, end, pathfinding_grid)
        dist_gg.insert(i2, len(path))
    return (i, dist_gg)


def min_distance_to_holes(poly_with_holes, point):
    if len(poly_with_holes.interiors) == 0:
        return -1
    return min([hole.distance(point) for hole in poly_with_holes.interiors])


def print_dat(
    output_file,
    poly,
    costs_path,
    pathfinding_matrix,
    max_guards,
    witnesses_num,
    coverage,
    obs_dist,
    guard_cost_mult,
):
    output_file.write("\ndata;\n")

    print_simple_param(output_file, "coverage_coeff", 1)
    print_simple_param(output_file, "distance_coeff", 1)

    witnesses = [
        poly.exterior.line_interpolate_point(d, normalized=True)
        for d in np.linspace(0, 1, witnesses_num)
    ]
    nW = len(witnesses)
    guards = []
    GUARD_RESOLUTION = 1
    while True:
        guards.clear()
        # poly.bounds is the minimum bounding region given as (min_x, min_y, max_x, max_y)
        for x in range(round(poly.bounds[0]), round(poly.bounds[2]), GUARD_RESOLUTION):
            for y in range(
                round(poly.bounds[1]), round(poly.bounds[3]), GUARD_RESOLUTION
            ):
                guard_candidate = shapely.Point([x, y])
                if (
                    poly.contains(guard_candidate)
                    and poly.exterior.distance(guard_candidate) >= obs_dist
                    and (
                        min_distance_to_holes(poly, guard_candidate) >= obs_dist
                        or min_distance_to_holes(poly, guard_candidate) == -1
                    )
                ):
                    guards.append((x, y))
        GUARD_RESOLUTION += 1
        if len(guards) <= max_guards:
            break
    nG = len(guards)
    print_simple_param(output_file, "nW", nW)
    print_simple_param(output_file, "nG", nG)

    coverage = []
    coverable = 0
    with concurrent.futures.ProcessPoolExecutor() as executor:
        coverage_w_data = [(i, poly, w, guards) for i, w in enumerate(witnesses)]
        for i, cov_w in executor.map(coverage_w, coverage_w_data):
            coverage.insert(i, cov_w)
            if any(cov_w):
                coverable += 1

    min_coverage = min(coverage, (coverable / nW) - 0.1)
    print_simple_param(output_file, "min_coverage", min_coverage)

    guard_costs = []
    with open(costs_path, "r") as costs_file:
        costs = dict()
        line = costs_file.readline()
        while line != "":
            x, y, cost = (
                float(line.strip().split()[0]),
                float(line.strip().split()[1]),
                float(line.strip().split()[2]),
            )
            costs[(x, y)] = cost
            line = costs_file.readline()
        for i, (x, y) in enumerate(guards):
            guard_costs.append((i + 1, costs[x, y] * guard_cost_mult))

    print_vector_param(output_file, "guard_cost", guard_costs)

    print_bidimensional_param(output_file, "coverage", nW, nG, coverage)

    distances = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        distance_gg_data = [
            (i, g1, guards, pathfinding_matrix) for i, g1 in enumerate(guards)
        ]
        for i, dist_gg in executor.map(distance_gg, distance_gg_data):
            distances.insert(i, dist_gg)
    print_bidimensional_param(output_file, "distance", nG, nG, distances)

    output_file.write("\nend;\n")

    # Save guard location outside problem data since it's of no use for the optimization (distances are precalculated)
    print_vector_param(
        output_file,
        "guard_position",
        [(i + 1, (f"{x} {y}")) for (i, (x, y)) in enumerate(guards)],
    )
    print_vector_param(
        output_file,
        "witness_position",
        [(i + 1, (f"{p.x} {p.y}")) for (i, p) in enumerate(witnesses)],
    )


def check_fraction(value):
    try:
        value = float(value)
        if value <= 0 or value > 1:
            raise argparse.ArgumentTypeError("{} is not > 0 and <= 1.".format(value))
    except ValueError:
        raise Exception(f"{value} is not a float.")
    return value


def check_positive(value):
    try:
        value = int(value)
        if value <= 0:
            raise argparse.ArgumentTypeError(
                "{} is not a positive integer.".format(value)
            )
    except ValueError:
        raise Exception(f"{value} is not an integer.")
    return value


def check_positive_or_zero(value):
    try:
        value = int(value)
        if value < 0:
            raise argparse.ArgumentTypeError(
                "{} is not a positive integer nor zero.".format(value)
            )
    except ValueError:
        raise Exception(f"{value} is not an integer.")
    return value


def check_positive_float(value):
    try:
        value = float(value)
        if value <= 0:
            raise argparse.ArgumentTypeError(
                "{} is not a positive float.".format(value)
            )
    except ValueError:
        raise Exception(f"{value} is not a float.")
    return value


def parse_args():
    parser = argparse.ArgumentParser(description="Compute data to run optimization.")
    parser.add_argument(
        "--img",
        default=os.path.join(os.getcwd(), "image.png"),
        help="Path to the map png image file.",
        metavar="IMG_PATH",
    )
    parser.add_argument(
        "--costs",
        default=os.path.join(os.getcwd(), "costs.txt"),
        help="Path to the map txt costs file.",
        metavar="COSTS_PATH",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to output data file. If not specified, will use same directory as costs file.",
        metavar="OUTPUT_PATH",
    )
    parser.add_argument(
        "--max-guards",
        default=300,
        type=check_positive,
        help="Maximum number of guards.",
        metavar="GUARDS",
    )
    parser.add_argument(
        "--witnesses",
        default=300,
        type=check_positive,
        help="Number of witnesses.",
        metavar="WITNESSES",
    )
    parser.add_argument(
        "--coverage",
        default=0.80,
        type=check_fraction,
        help="Minimum coverage desired. Must be > 0 and <= 1. Note that it may be lowered due to guard visibility.",
        metavar="COVERAGE",
    )
    parser.add_argument(
        "--obs-dist",
        default=5,
        type=check_positive_or_zero,
        help="Minimum distance from guards to obstacles.",
        metavar="DISTANCE",
    )
    parser.add_argument(
        "--guard-cost-mult",
        default=1,
        type=check_positive_float,
        help="Multiplier for guard cost.",
        metavar="MULTIPLIER",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    costs_path = args.costs
    img_path = args.img
    max_guards = args.max_guards
    witnesses = args.witnesses
    coverage = args.coverage
    obs_dist = args.obs_dist
    guard_cost_mult = args.guard_cost_mult
    MIN_HOLE_AREA = 10
    DEBUG_HOLES = False
    DEBUG_CONTOUR = False

    # If output path is not specified, use the same directory as costs file
    if args.output is None:
        output_path = os.path.join(os.path.dirname(costs_path), "data.dat")
    else:
        output_path = args.output

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    contours, _ = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=lambda cnt: cv2.contourArea(cnt))
    if DEBUG_CONTOUR:
        debug = np.zeros_like(img)
        cv2.drawContours(debug, [max_contour], -1, (255, 255, 255), cv2.FILLED)
        _ = (
            plt.subplot(111),
            plt.imshow(cv2.cvtColor(debug, cv2.COLOR_BGR2RGB)),
            plt.title("max_contour"),
        )
        plt.show()
    height, width = img.shape
    pathfinding_matrix = []
    for h in range(height):
        row = []
        for w in range(width):
            if img[h, w] < 10:
                row.append(0)
            else:
                row.append(1)
        pathfinding_matrix.append(row)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    holes = []
    for c in contours:
        if (
            cv2.contourArea(c) != cv2.contourArea(max_contour)
            and cv2.contourArea(c) >= MIN_HOLE_AREA
        ):
            holes.append(c)
            if DEBUG_HOLES:
                debug = np.zeros_like(img)
                cv2.drawContours(debug, [c], -1, (255, 255, 255), cv2.FILLED)
                _ = (
                    plt.subplot(111),
                    plt.imshow(cv2.cvtColor(debug, cv2.COLOR_BGR2RGB)),
                    plt.title("hole"),
                )
                plt.show()
    poly_no_holes = shapely.Polygon([[p[0][0], p[0][1]] for p in max_contour])
    poly_no_holes = poly_no_holes.buffer(0)
    # If poly isn't a well defined polygon, we try to fix it
    if poly_no_holes.geom_type == "MultiPolygon":
        poly = max(poly_no_holes.geoms, key=lambda a: a.area)
    holes_fixed = []
    for h in holes:
        hole = shapely.Polygon([[p[0][0], p[0][1]] for p in h])
        if hole.geom_type == "MultiPolygon":
            hole = max(hole.geoms, key=lambda a: a.area)
        if poly_no_holes.contains(hole):
            holes_fixed.append(hole.buffer(0).exterior)
    poly = shapely.Polygon([[p[0][0], p[0][1]] for p in max_contour], holes=holes_fixed)
    poly = poly.buffer(0)
    if poly.geom_type == "MultiPolygon":
        poly = max(poly.geoms, key=lambda a: a.area)
    if shapely.validation.explain_validity(poly) == "Valid Geometry":
        with open(output_path, "w") as output_file:
            print_dat(
                output_file,
                poly,
                costs_path,
                pathfinding_matrix,
                max_guards,
                witnesses,
                coverage,
                obs_dist,
                guard_cost_mult,
            )
        print(f"Data written to {output_path}")
    else:
        print(shapely.validation.explain_validity(poly))


if __name__ == "__main__":
    main()
