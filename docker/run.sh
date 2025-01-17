#!/usr/bin/env bash

parent_dir=$(dirname "$(pwd)")
workspace_dir="${parent_dir}/workspace"
cache_dir="${parent_dir}/gazebo_cache"
mkdir -p $cache_dir

# X11 forwarding: https://www.baeldung.com/linux/docker-container-gui-applications 
touch /tmp/.docker.xauth
xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f /tmp/.docker.xauth nmerge - 

docker run -it --rm  \
    --name ros_gazebo \
    --gpus all \
    --shm-size=2gb \
    --privileged \
    -v $workspace_dir:/workspace \
    -v $cache_dir:/root/.gazebo \
    --volume /tmp/.docker.xauth:/tmp/.docker.xauth \
    --volume /tmp/.X11-unix:/tmp/.X11-unix \
    --env DISPLAY=$DISPLAY \
    --env XAUTHORITY=/tmp/.docker.xauth \
    --net=host \
    -w /workspace \
    ros-gazebo