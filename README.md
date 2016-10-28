# udacity-steering-prediction
prediction of steering wheel using real driving data, Udacity challenge 2


## first install ROS kinetic

## run build_catkin.sh to setup environment

## to start with ros

~~~
# start core
roscore

# go to data folder
cd data

# play back data

rosbag play --clock *.bag roslaunch

roslaunch udacity_launch bag_play.launch

~~~