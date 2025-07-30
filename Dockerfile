FROM ros:humble

# Install apt dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-pip \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-vcstool \
    git \
    nano \
    wget \
    xvfb \
    wmctrl \
    x11-utils \
    libfltk1.3-dev \
    ros-humble-ament-cmake \
    curl \
    gnupg2 \
    lsb-release \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip3 install pip --upgrade
RUN pip3 install --no-cache-dir --break-system-packages \
    matplotlib \
    opencv-python \
    pillow \
    evo \
    amplpy \
    pathfinding \
    python-tsp \
    shapely 

# Copy source code to workspace
ENV WORKDIR=/root/exp_cov
RUN mkdir -p ${WORKDIR}
COPY . ${WORKDIR}

# Initialize rosdep
RUN [ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ] || rm /etc/ros/rosdep/sources.list.d/20-default.list && \
    rosdep init && \
    rosdep update

RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add - && \
    echo "deb http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" > /etc/apt/sources.list.d/ros2-latest.list && \
    apt update

# We start by exploration_benchmarking
ENV ROS_WS=/root/exp_cov/src/exploration_benchmarking/ros2_simulation
WORKDIR ${ROS_WS}
RUN rm -rf build log install

RUN rosdep install --from-paths src --ignore-src -r -y --rosdistro humble


# And then exp_cov_2
ENV ROS_WS=/root/exp_cov/src/exp_cov_2
WORKDIR ${ROS_WS}
RUN rm -rf build log install

RUN rosdep install --from-paths exp_cov_2/ --ignore-src -r -y --rosdistro humble


# Build packages
# RUN /bin/bash -c "source /opt/ros/humble/setup.bash && \
#     colcon build --symlink-install --packages-select stage && \
#     source install/setup.bash && \
#     colcon build --symlink-install"


# Set up environment
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc

# Environment variables needed for stage
ENV DISPLAY=:1
ENV screen=0
ENV resolution=800x600x24
RUN echo "Xvfb ${DISPLAY} -screen ${screen} ${resolution} &" >> ~/.bashrc

CMD ["bash"]