#!/bin/bash

# same as initial_build.sh, but without removing what is already built
set -e

source /opt/ros/humble/setup.bash

cd src/exp_cov_2/
colcon build --symlink-install 
source install/setup.bash

cd ../exploration_benchmarking/ros2_simulation/
colcon build --symlink-install
source install/setup.bash

cd ../../../

# Execute the main command (probably bash)
exec "$@"