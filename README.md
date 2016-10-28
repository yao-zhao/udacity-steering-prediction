# udacity-steering-prediction
prediction of steering wheel using real driving data, Udacity challenge 2


## first install ROS kinetic

## run build_catkin.sh to setup environment

## Running

### start core
~~~
roscore
~~~

### play back data
~~~
rosbag play --clock *.bag 
~~~

### visualize
~~~
roslaunch udacity_launch rviz.launch
~~~

### log
~~~
roslaunch udacity_launch logging.launch republish_raw2compressed_images:=true bagPath:="/media/yz/Data/udacity_SDC/"

roslaunch udacity_launch logging.launch republish_raw2compressed_images:=true bagPath:="/media/yz/Data/udacity_SDC/"

~~~


<!-- roslaunch udacity_launch bag_play.launch -->

## useful commands

- check if a package exist
~~~
rospack find udacity_launch
~~~