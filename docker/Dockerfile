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
    && rm -rf /var/lib/apt/lists/*

# Set up ROS repository (for ROS Noetic)
RUN curl -sSL http://packages.ros.org/ros.key | apt-key add - \
    && echo "deb [arch=amd64] http://packages.ros.org/ros/ubuntu $(lsb_release -c | awk '{print $2}') main" | tee /etc/apt/sources.list.d/ros1.list


# Install ROS 1 Noetic (Desktop version)
RUN apt-get update \
    && apt-get install -y \
    ros-noetic-desktop-full \
    ros-noetic-gazebo-ros-pkgs \
    ros-noetic-joy \
    ros-noetic-rqt \
    gazebo11 \
    python3-rosdep \
    && rm -rf /var/lib/apt/lists/*

# Initialize rosdep
RUN rosdep init \
    && rosdep update

# Set up ROS environment
RUN echo "source /opt/ros/noetic/setup.bash" >> /etc/bash.bashrc \
    && echo "export ROS_DOMAIN_ID=0" >> /etc/bash.bashrc \
    && echo "export GAZEBO_MODEL_PATH=/opt/ros/noetic/share/gazebo_models" >> /etc/bash.bashrc

# Set up the default command (Bash shell)
CMD ["/bin/bash"]