#!/usr/bin/env bash

parent_dir=$(dirname "$(pwd)")
workspace_dir="${parent_dir}/workspace"
cache_dir="${parent_dir}/gazebo_cache"
mkdir -p $cache_dir

# X11 forwarding: https://www.baeldung.com/linux/docker-container-gui-applications 
touch /tmp/.docker.xauth
xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f /tmp/.docker.xauth nmerge - 

command_exists () {
    type "$1" &> /dev/null ;
}

if command_exists nvidia-smi; then
      extra_params="--gpus all --runtime nvidia"
      echo -e "\t[INFO] nvidia gpus exists"
else
      extra_params=""
      echo -e "\t[INFO] nvidia gpus does not exist (falling back to docker). Rviz and Gazebo most likely will not work!"
fi

docker run -it --rm  \
    --name ros_gazebo \
    $extra_params \
    --env="QT_X11_NO_MITSHM=1" \
    --shm-size=2gb \
    --privileged \
    -v $workspace_dir:/workspace \
    -v $cache_dir:/root/.gazebo \
    --volume /tmp/.docker.xauth:/tmp/.docker.xauth \
    --volume /tmp/.X11-unix:/tmp/.X11-unix \
    --env DISPLAY=$DISPLAY \
    --env XAUTHORITY=/tmp/.docker.xauth \
    --device /dev/dri:/dev/dri -v /dev/dri:/dev/dri \
    --net=host \
    -w /workspace \
    ros-gazebo