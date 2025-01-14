parent_dir=$(dirname "$(pwd)")
code_dir="${parent_dir}/src"
cache_dir="${parent_dir}/gazebo_cache"
mkdir -p $cache_dir


docker run -it --rm  \
    --gpus all \
    --privileged \
    --env DISPLAY=$DISPLAY \
    -v $code_dir:/src \
    -v $cache_dir:/root/.gazebo \
    -v "$HOME/.Xauthority:/root/.Xauthority:rw" \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    --net=host \
    -w /src \
    ros-gazebo