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
    sudo \
    wget \
    build-essential \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Set up ROS repository
RUN curl -sSL http://packages.ros.org/ros.key | sudo apt-key add - \
    && echo "deb [arch=amd64] http://packages.ros.org/ros2/ubuntu $(lsb_release -c | awk '{print $2}') main" | sudo tee /etc/apt/sources.list.d/ros2.list

# Install ROS 2 (you can change it to ROS 1 if needed, here we install ROS 2 Foxy)
RUN apt-get update \
    && apt-get install -y \
    ros-foxy-desktop \
    ros-foxy-gazebo-ros-pkgs \
    ros-foxy-joy \
    ros-foxy-rqt \
    && rm -rf /var/lib/apt/lists/*

# Install Gazebo
RUN sudo apt-get update \
    && sudo apt-get install -y \
    gazebo11 \
    ros-foxy-gazebo-ros-pkgs \
    && rm -rf /var/lib/apt/lists/*

# Set up ROS environment
RUN echo "source /opt/ros/foxy/setup.bash" >> ~/.bashrc \
    && echo "export ROS_DOMAIN_ID=0" >> ~/.bashrc \
    && echo "export GAZEBO_MODEL_PATH=/opt/ros/foxy/share/gazebo_models" >> ~/.bashrc

# Install other dependencies (optional)
RUN sudo apt-get update \
    && sudo apt-get install -y \
    python3-pip \
    python3-rosdep \
    && sudo apt-get clean

RUN sudo apt-get update \
    && sudo apt-get install -y \
    x11-apps \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    libx11-dev \
    && sudo apt-get clean

# Initialize rosdep
RUN sudo rosdep init \
    && rosdep update


# Set up the default command (Bash shell)
CMD ["/bin/bash"]
