cmake_minimum_required(VERSION 2.8.3)
project(aruco_pose_estimation)

find_package(OpenCV REQUIRED)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  dynamic_reconfigure
  geometry_msgs
  image_transport
  roscpp
  rospy
  tf
  aruco
  aruco_msgs
  aruco_ros
  sensor_msgs
  visualization_msgs
)


catkin_package(
  #INCLUDE_DIRS include
  #LIBRARIES aruco_ros_utils
  CATKIN_DEPENDS aruco aruco_ros sensor_msgs tf 
)


