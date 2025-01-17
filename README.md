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
Run the following code:
```bash
cd src/panda_simulator
./build_ws.sh

cd ../..
source devel/setup.bash
```

## Run gazebo with panda robot
```bash
roslaunch panda_gazebo panda_world.launch
```
