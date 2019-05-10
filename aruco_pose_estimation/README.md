# ROS Aruco Pose Estimation
To Find the pose (tf) of detected Marker.

<p align="center">
  <img width="600" height="400" src="rviz_display.gif">
</p>

---

## Getting Started
- Opencv is installed with ROS, check if it's install via [here](https://stackoverflow.com/questions/8804064/find-opencv-version-installed-on-ubuntu)

- ROS Pkg `aruco_ros` (gitpull and compile from src), link is [here](https://github.com/pal-robotics/aruco_ros)
- ROS Pkg `usb_cam`   (apt install), doc is [here](http://wiki.ros.org/usb_cam)
  
```
sudo apt-get install ros-melodic-usb-cam
```

---

## Camera Calibration

Refer to OpenCV Camera Calibration code ([here](https://docs.opencv.org/2.4/doc/tutorials/calib3d/camera_calibration/camera_calibration.html#results)). This tool has provided a chessboard.png for calibration.

Compile by:
```
cd /build
cmake ..
make -j4
./camera_calibration   #this will take in 'default.xml' as calibration param
```

Follow the calibration steps, `distortion matrix` and `camera matrix` will be provided. Fill in the vals into `/config/usb_cam.yaml`

---

## USB Cam Package

Assume that normal webcam or usb cam is used, via `usb_cam` pkg. Some useful commands:

```
v4l2-ctl -d /dev/video0 --list-formats-ext  # to check cam specs and configure into usb_cam launch file
webcam.sh                                   # to change default cam on /dev/video0.original, not use here, but useful
```

---

## Run Aruco Pose Estimation Code

This assume that the cam specs above is configured to the launch file below.

**Multi-markers**

```
roslaunch aruco_pose_estimation multi_markers.launch
```

After test, node will publish markers pose array to topic `/aruco_marker_publisher/markers`. `/tf` topic is not updated


**Single Marker**
```
roslaunch aruco_pose_estimation single_marker.launch
# on another terminal
rostopic echo /tf
```

Pose will be published to `/aruco_single/pose`. Both can try use Rviz to check it's image, for single user can check the tf on rviz. Tf_tree is as such:   map -> camera -> marker.


## Notes
- Refer to `rqt_graph.png` to checkout the nodes
- or can build from scratch based of raw open_cv aruco detector, [here](https://github.com/fdcl-gwu/aruco-markers/)
