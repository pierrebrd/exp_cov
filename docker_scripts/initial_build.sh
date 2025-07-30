#!/bin/bash
set -e


source /opt/ros/humble/setup.bash

cd src/exp_cov_2/
rm -rf build/ log/ install/
colcon build --symlink-install 
source install/setup.bash

cd ../exploration_benchmarking/ros2_simulation/
rm -rf build/ log/ install/
colcon build --symlink-install
source install/setup.bash

cd ../../../

# Execute the main command (probably bash)
exec "$@"