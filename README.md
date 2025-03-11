# Smart Robotics project

## Installation
```bash
git clone https://github.com/davidedavo/smart_robotics.git
cd smart_robotics
git submodule update --init
cd docker
./build.sh
./run.sh
```

Now you are inside the container.

## Build panda simulator package
Run the following code at the first run, after the docker build:
```bash
cd src/panda_simulator
./build_ws.sh
cd ../..
catkin build
```

## Run gazebo with panda robot
```bash
source devel/setup.bash
roslaunch smart_robotics smart_robotics.launch
```
