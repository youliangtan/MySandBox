# ROS Aruco Pose Estimation

## Getting Started
- Opencv is installed with ROS, check if it's install via [here](https://stackoverflow.com/questions/8804064/find-opencv-version-installed-on-ubuntu)

- ROS Pkg `aruco_ros` (gitpull and compile from src), link is [here](https://github.com/pal-robotics/aruco_ros)
- ROS Pkg `usb_cam`   (apt install), doc is [here](http://wiki.ros.org/usb_cam)
  
```
sudo apt-get install ros-melodic-usb-cam
```

## Run Code
Drag the `marker_publisher.launch` to aruco_ros launch folder, then replace the original one (fornow, TODO)

```
roslaunch aruco_ros marker_publisher.launch
```

**if cam not working**
- use command `v4l2-ctl -d /dev/video0 --list-formats-ext` to check cam specs
- configure it in launch file

## notes
- Refer to `rqt_graph.png` to checkout the nodes
- Use `webcam.sh` to change default cam on /dev/video0.original, not use here, but useful
- Configure the launch file to suit the use
- or can build from scratch based of raw open_cv aruco detector, [here](https://github.com/fdcl-gwu/aruco-markers/)
