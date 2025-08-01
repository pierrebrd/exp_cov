# Experience-Based Coverage

This repository is a fork of [d-ber/exp_cov](https://github.com/d-ber/exp_cov), and containes the additions of [paoloterragno/exp_cov](https://github.com/paoloterragno/exp_cov) in the `pter-develop` branch.

The original work of d-ber (Davide Betalero) is made to work on ROS1 Noetic on Ubuntu 20.04. This fork aims to port the project to ROS2 Humble on Ubuntu 22.04, and to add some features.

## Cloning

This project uses recursive submodules, so you need to clone it with the `--recurse-submodules` option:

```bash
git clone --recurse-submodules <repo_url>
```

You can install the conda environment with the following command

```bash
conda env create -f environment-minimal.yml
```



## Content

The project provides some utilities for experience-based coverage.

The main files are in `src/exp_cov`.

### Workflow

A quick rundown of the main files follows:

#### map_rgb_simul.py

This script simulates the dynamics of a semi-static environment using two images: one for the environment and the objects, and one for the movement areas. It produces N configurations of the same environment and world files for [stage](https://wiki.ros.org/stage_ros). The worlds can be explored using [explore_lite](https://wiki.ros.org/explore_lite).

#### fuse_maps_floorplan.py

Once you have a sufficient number of maps of the same environment, you can fuse them with this script.

#### optimization_data.py

This script computes the data for the optimization phase of the method. As of v1.0.2, it prints to stdout, and you'll likely want to redirect the output to a file.

After this, you can run a solver on the data using one of the models found in the optimization folder.

#### opt_viz.py

This script allows you to view the positions of guards and witnesses produced by optimization_data.py and to view the optimization result.

#### tsp.py

This script computes the coverage navigation order by solving a TSP instance using [python-tsp](https://github.com/fillipe-gsm/python-tsp).

#### run_explore_and_waypoint.py

This script executes N explorations versus coverage runs on stage.

#### waypoint_navigation.py

This script implements waypoint coverage navigation. It's used by run_explore_and_waypoint.py.
