# Passing custom messages through ROS1-ROS2 bridge

There are a number of steps needed to build and pass custom messages between ROS1 and ROS2.

This project contains 3 folder:
1. ROS1 workspace: **ros1_ws/** 
2. ROS2 workspace: **ros2_ws/**
3. Custom message: **msg/**
(Custom message from **msg/** folder is referenced from both ROS1 and ROS2 workspaces)

This guide assumes that project cloned to folder **~/ros2_experiments/bridge_msgs/**. To source those workspaces following commands are used:
+ ros1_ws: **source ~/ros2_experiments/bridge_msgs/ros1_ws/devel/setup.bash**
+ ros2_ws: **. ~/ros2_experiments/bridge_msgs/ros2_ws/install/local_setup.bash**


## Prerequisites

To compile successfully these projects require ROS1 and ROS2 to be installed in the system.
This guide assumes that ROS2 installed in folder **~/ros2_ws** and ROS1 installed in folder **/opt/ros/lunar/**. To source ROS core workspaces following commands are used:
+ ROS1: **source /opt/ros/lunar/setup.bash**
+ ROS2: **. ~/ros2_ws/install/local_setup.bash**

If you need to install ROS1/ROS2, reference these links:
* https://github.com/ros2/ros2/wiki/Installation
* https://andrasta.atlassian.net/wiki/display/EN/ROS1-ROS2+bridge
* http://wiki.ros.org/lunar/Installation/Ubuntu


## Project compilation

#### Compile ros1_ws
Open new terminal and run:
```commandline
$ cd ~/ros2_experiments/bridge_msgs/ros1_ws/
$ source /opt/ros/lunar/setup.bash
$ catkin_make
$ source devel/setup.bash
```

#### Make sure compilation is successful
Start roscore. Open new terminal and run:
```commandline
$ source /opt/ros/lunar/setup.bash
$ roscore
```
Open new terminal and run:
```commandline
$ devel/lib/bridge_test_msgs/my_subscriber &
$ devel/lib/bridge_test_msgs/my_publisher
```
Messages should be passed inside ROS1 successfully at this point. After you saw message kill my_subscriber process.

#### Compile ros2_ws
Open new terminal and run:
```commandline
$ cd ~/ros2_experiments/bridge_msgs/ros2_ws
$ . ~/ros2_ws/install/local_setup.bash
$ ament build
$ . install/local_setup.bash
```

#### Make sure compilation is successfull
```commandline
$ my_subscriber &
$ my_publisher
```
Messages should be passed inside ROS2 successfully at this point, After you saw message kill my_subscriber process.

### Compile ros1_bridge
#### Fix ros1_bridge configuration files

Note: This step maybe not needed if you use ROS2 beta2 or higher.

Open new terminal and run:
```commandline
cd ~/ros2_ws/src/ros2/ros1_bridge
```
Fix files **CMakeLists.txt** and **package.xml** as shown in this diff
```git
diff --git a/CMakeLists.txt b/CMakeLists.txt
index b3867d5..2d720d2 100644
--- a/CMakeLists.txt
+++ b/CMakeLists.txt
@@ -12,7 +12,7 @@ find_package(ament_cmake REQUIRED)
 find_package(rclcpp REQUIRED)
 find_package(rmw_implementation_cmake REQUIRED)
 find_package(std_msgs REQUIRED)
+find_package(bridge_test_msgs REQUIRED)
 
 # find ROS 1 packages
 include(cmake/find_ros1_package.cmake)
@@ -34,7 +34,7 @@ if(NOT ros1_roscpp_FOUND)
 
 find_ros1_package(std_msgs REQUIRED)
+find_ros1_package(bridge_test_msgs REQUIRED)
 
 set(TEST_ROS1_BRIDGE FALSE)
 if(BUILD_TESTING)
@@ -92,7 +92,7 @@ function(custom_executable target)
     ament_target_dependencies(${target}${target_suffix}
       "ros1_roscpp"
       "ros1_std_msgs"
+      "ros1_bridge_test_msgs"
     )
   endif()
   if(ARG_DEPENDENCIES)
@@ -110,14 +110,14 @@ macro(targets)
 
   custom_executable(simple_bridge_1_to_2 "src/simple_bridge_1_to_2.cpp"
     ROS1_DEPENDENCIES
+    TARGET_DEPENDENCIES "std_msgs" "bridge_test_msgs")
   custom_executable(simple_bridge_2_to_1 "src/simple_bridge_2_to_1.cpp"
     ROS1_DEPENDENCIES
+    TARGET_DEPENDENCIES "std_msgs" "bridge_test_msgs")
 
   custom_executable(simple_bridge "src/simple_bridge.cpp"
     ROS1_DEPENDENCIES
+    TARGET_DEPENDENCIES "std_msgs" "bridge_test_msgs")
 
   add_library(${PROJECT_NAME}${target_suffix} SHARED
     "src/convert_builtin_interfaces.cpp"
@@ -127,7 +127,7 @@ macro(targets)
     ${message_packages}
     "ros1_roscpp"
     "ros1_std_msgs"
+    "ros1_bridge_test_msgs"    
   )
 
   install(TARGETS ${PROJECT_NAME}${target_suffix}
diff --git a/package.xml b/package.xml
index 5d6a83c..0a6b7e8 100644
--- a/package.xml
+++ b/package.xml
@@ -15,11 +15,11 @@
   <build_depend>rclcpp</build_depend>
   <build_depend>rmw_implementation_cmake</build_depend>
   <build_depend>std_msgs</build_depend>
+  <build_depend>bridge_test_msgs</build_depend>
 
   <exec_depend>rclcpp</exec_depend>
   <exec_depend>std_msgs</exec_depend>
+  <exec_depend>bridge_test_msgs</exec_depend>
 
   <test_depend>ament_cmake_nose</test_depend>
   <test_depend>ament_lint_auto</test_depend>
```

#### Rebuild ros1_bridge with sourced workspaces

Open new terminal and run:
```commandline
$ source /opt/ros/lunar/setup.bash
$ . ~/ros2_ws/install/local_setup.bash
$ source ~/ros2_experiments/bridge_msgs/ros1_ws/devel/setup.bash
$ . ~/ros2_experiments/bridge_msgs/ros2_ws/install/local_setup.bash
$ cd ~/ros2_ws
$ rm -rf build/ros1_bridge
$ src/ament/ament_tools/scripts/ament.py build --build-tests --symlink-install --only ros1_bridge
$ . install/local_setup.bash
```
#### Test ros1_bridge
 
Make sure **roscore** is still working in separate terminal
 
Run:
```commandline
$ dynamic_bridge
```
In new terminal start ROS1 subscriber
```commandline
$ source /opt/ros/lunar/setup.bash
$ source ~/ros2_experiments/bridge_msgs/ros1_ws/devel/setup.bash
~/ros2_experiments/bridge_msgs/ros1_ws/devel/lib/bridge_test_msgs/my_subscriber
```
In new terminal start ROS2 publisher
```commandline
$ . ~/ros2_ws/install/local_setup.bash
$ . ~/ros2_experiments/bridge_msgs/ros2_ws/install/local_setup.bash
$ my_publisher
```

At this point messages published from ROS2 should be visible in subscriber on ROS1.

Terminal with dynamic_bridge prints following text for each bridged message:
```
created 2to1 bridge for topic 'bridge_test_topic' with ROS 2 type 'bridge_test_msgs/BridgeTestMessage' and ROS 1 type ''
  Passing message from ROS 2 to ROS 1
  Passing message from ROS 2 to ROS 1
  Passing message from ROS 2 to ROS 1
```

You also can stop **my_subscriber** in ROS1 and **my_publisher** in ROS2 and try it vice versa.


### Useful links
https://andrasta.atlassian.net/wiki/display/EN/ROS1-ROS2+bridge

https://github.com/ros2/ros2/wiki/Defining-custom-interfaces-(msg-srv)

http://wiki.ros.org/ROS/Tutorials/DefiningCustomMessages

http://wiki.ros.org/ROS/Tutorials/CreatingMsgAndSrv#Creating_a_msg

https://github.com/ros2/ros1_bridge

https://github.com/ros2/ros1_bridge/blob/master/doc/index.rst

https://github.com/ros2/ros1_bridge/issues/77

https://discourse.ros.org/t/ros2-how-to-use-custom-message-in-project-where-its-declared/2071


