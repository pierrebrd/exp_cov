#!/bin/bash


set -e

source /opt/ros/humble/setup.bash

source src/exploration_benchmarking/ros2_simulation/install/setup.bash

source src/exp_cov_2/install/setup.bash


# Execute the main command (probably bash)
exec "$@"