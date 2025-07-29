# Python script not linked to ROS, doesn't need to be ported to ROS2

import cv2
import matplotlib.pyplot as plt
import argparse
import os


def parse_args():

    parser = argparse.ArgumentParser(
        description="Visualize optmization data before or after. "
    )
    parser.add_argument(
        "--img",
        default=os.path.join(os.getcwd(), "image.png"),
        help="Path to the png image file.",
        metavar="IMG_PATH",
    )
    parser.add_argument(
        "--data",
        default=os.path.join(os.getcwd(), "data.dat"),
        help="Path to the text data file.",
        metavar="DATA_PATH",
    )
    parser.add_argument(
        "--guards",
        default="",
        help="Path to the text guards file. If not given, all guard candidates will be shown.",
        metavar="GUARDS_PATH",
    )
    return parser.parse_args()


def read_param(
    file,
    param_start,
    param_end,
    param_name,
    value_read,
    filter,
    data_on_single_line=False,
):
    param_values = []
    line = file.readline()
    while (not line.strip().startswith(param_start)) and line != "":
        line = file.readline()
    if line == "":
        raise Exception(
            f'Start of param {param_name} "{param_start}" not found in file {file}.'
        )
    if not data_on_single_line:
        line = file.readline()
        while line.strip() != param_end and line != "":
            if filter(line):
                param_values.append(value_read(line))
            line = file.readline()
        if line == "":
            print(
                f'Warning: End of param {param_name} "{param_end}" not found in file {file}.'
            )
    else:
        param_values.append(value_read(line))
    return param_values


def main():

    args = parse_args()
    img_path = args.img
    img = cv2.imread(img_path)
    data_file_path = args.data
    chosen_guards_file_path = args.guards

    guard_position = []
    guard_choice = []
    witness_position = []
    guard_cost = []
    param_end = ";"

    if chosen_guards_file_path != "":
        with open(chosen_guards_file_path, "r") as file:
            guard_choice = read_param(
                file,
                "guard_choice [*] :=",
                param_end,
                "guard_choice",
                lambda line: line.strip().split()[0],
                lambda line: True,
            )
        with open(data_file_path, "r") as file:
            total_guards_number = read_param(
                file,
                "param nG :=",
                param_end,
                "nG",
                lambda line: line.strip().split()[-1].rstrip(param_end),
                lambda line: True,
                data_on_single_line=True,
            )[0]
            guard_position = read_param(
                file,
                "param guard_position :=",
                param_end,
                "guard_position",
                lambda line: (line.strip().split()[1], line.strip().split()[2]),
                lambda line: line.split()[0] in guard_choice,
            )
            file.seek(0, 0)
            witness_position = read_param(
                file,
                "param witness_position :=",
                param_end,
                "witness_position",
                lambda line: (line.strip().split()[1], line.strip().split()[2]),
                lambda line: True,
            )

        print(f"Total number of guards: {total_guards_number}")
        print(f"Number of chosen guards: {len(guard_choice)}")
        print(f"Number of witnesses: {len(witness_position)}")
        for x, y in guard_position:
            img = cv2.circle(
                img, (int(x), int(y)), radius=1, color=(0, 0, 255), thickness=-1
            )
        for x, y in witness_position:
            img = cv2.circle(
                img,
                (round(float(x)), round(float(y))),
                radius=1,
                color=(255, 0, 0),
                thickness=-1,
            )
        _ = (
            plt.subplot(111),
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)),
            plt.title("Chosen Guards"),
        )
        plt.show()
    else:
        with open(data_file_path, "r") as file:
            guard_cost = read_param(
                file,
                "param guard_cost :=",
                param_end,
                "guard_cost",
                lambda line: float(line.split()[1]),
                lambda line: True,
            )
            file.seek(0, 0)
            guard_position = read_param(
                file,
                "param guard_position :=",
                param_end,
                "guard_position",
                lambda line: (line.strip().split()[1], line.strip().split()[2]),
                lambda line: True,
            )
            file.seek(0, 0)
            witness_position = read_param(
                file,
                "param witness_position :=",
                param_end,
                "witness_position",
                lambda line: (line.strip().split()[1], line.strip().split()[2]),
                lambda line: True,
            )

        print(f"Total number of guards: {len(guard_position)}")
        print(f"Number of witnesses: {len(witness_position)}")
        for i, (x, y) in enumerate(guard_position):
            img = cv2.circle(
                img,
                (int(x), int(y)),
                radius=1,
                color=(0, 0, 255 - (255 * (guard_cost[i] * 4))),
                thickness=-1,
            )
        for x, y in witness_position:
            img = cv2.circle(
                img,
                (round(float(x)), round(float(y))),
                radius=1,
                color=(255, 0, 0),
                thickness=-1,
            )
        _ = (
            plt.subplot(111),
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)),
            plt.title("guards and witnesses"),
        )
        plt.show()


if __name__ == "__main__":
    main()
