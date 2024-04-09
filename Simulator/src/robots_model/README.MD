### Turtlebot3 
compile:
```
sudo apt-get install ros-noetic-turtlebot3-msgs

catkin_make --only-pkg-with-deps gazebo_env turtlebot3_simulations
```
run:
```
roslaunch gazebo_env gazebo_turtlebot3_launch.launch
```


### Jackal
compile:
```
sudo apt-get install ros-noetic-sick-tim ros-noetic-lms1xx ros-noetic-velodyne-description ros-noetic-pointgrey-camera-description ros-noetic-flir-camera-description ros-noetic-twist-mux ros-noetic-robot-localization ros-noetic-interactive-marker-twist-server 

catkin_make --only-pkg-with-deps gazebo_env jackal_gazebo
```
run:
```
roslaunch gazebo_env gazebo_jackal_launch.launch
```