# Use an official Ubuntu base image
FROM ubuntu:20.04

# Set non-interactive installation to avoid user prompts
ENV DEBIAN_FRONTEND=noninteractive

# Update package list and install dependencies
RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y \
    curl \
    gnupg2 \
    lsb-release \
    wget \
    build-essential \
    software-properties-common \
    python3-pip \
    x11-apps \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    libx11-dev \
    tmux \
    && rm -rf /var/lib/apt/lists/*

# Set up ROS repository (for ROS Noetic)
RUN curl -sSL http://packages.ros.org/ros.key | apt-key add - \
    && echo "deb [arch=amd64] http://packages.ros.org/ros/ubuntu $(lsb_release -c | awk '{print $2}') main" | tee /etc/apt/sources.list.d/ros1.list


ENV ROS_DISTRO noetic
# Install ROS 1 Noetic (Desktop version)
RUN apt-get update \
    && apt-get install -y \
    ros-${ROS_DISTRO}-desktop-full \
    ros-${ROS_DISTRO}-gazebo-ros-pkgs \
    ros-${ROS_DISTRO}-joy \
    ros-${ROS_DISTRO}-rqt \
    gazebo11 \
    python3-rosdep \
    python3-wstool \
    python3-catkin \
    python3-catkin-tools \
    ros-${ROS_DISTRO}-libfranka \
    ros-${ROS_DISTRO}-franka-ros \
    ros-${ROS_DISTRO}-panda-moveit-config \
    ros-${ROS_DISTRO}-gazebo-ros-control \
    ros-${ROS_DISTRO}-rospy-message-converter \
    ros-${ROS_DISTRO}-effort-controllers \
    ros-${ROS_DISTRO}-joint-state-controller \
    ros-${ROS_DISTRO}-moveit \
    ros-${ROS_DISTRO}-moveit-commander \
    ros-${ROS_DISTRO}-moveit-visual-tools \
    && rm -rf /var/lib/apt/lists/*

# Initialize rosdep
RUN rosdep init \
    && rosdep update

# Set up ROS environment
RUN echo "source /opt/ros/noetic/setup.bash" >> /etc/bash.bashrc \
    && echo "export ROS_DOMAIN_ID=0" >> /etc/bash.bashrc \
    && echo "export GAZEBO_MODEL_PATH=/opt/ros/${ROS_DISTRO}/share/gazebo_models" >> /etc/bash.bashrc

ENV LD_LIBRARY_PATH /opt/ros/${ROS_DISTRO}/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}

# Install python requirements
RUN ln -s /usr/bin/python3 /usr/bin/python
COPY ./requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

# Set up the default command (Bash shell)
CMD ["/bin/bash"]