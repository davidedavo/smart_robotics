#!/usr/bin/env bash

parent_dir=$(dirname "$(pwd)")
code_dir="${parent_dir}/src"
cache_dir="${parent_dir}/gazebo_cache"
mkdir -p $cache_dir

# X11 forwarding: https://www.baeldung.com/linux/docker-container-gui-applications 
touch /tmp/.docker.xauth
xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f /tmp/.docker.xauth nmerge - 

docker run -it --rm  \
    --gpus all \
    --privileged \
    -v $code_dir:/src \
    -v $cache_dir:/root/.gazebo \
    --volume /tmp/.docker.xauth:/tmp/.docker.xauth \
    --volume /tmp/.X11-unix:/tmp/.X11-unix \
    --env DISPLAY=$DISPLAY \
    --env XAUTHORITY=/tmp/.docker.xauth \
    --net=host \
    -w /src \
    ros-gazebo