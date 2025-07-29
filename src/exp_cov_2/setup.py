from setuptools import find_packages, setup

package_name = "exp_cov_2"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="pierre",
    maintainer_email="pierrebrd7@gmail.com",
    description="ROS2 port of exp_cov by Davide Bertalero (d-ber)",
    license="MIT",
    entry_points={
        "console_scripts": [
            "distance_check = exp_cov_2.distance_check:main",
            "laser_scan_check = exp_cov_2.laser_scan_check:main",
            "nav_stack_listener = exp_cov_2.nav_stack_listener:main",
        ],
    },
)
