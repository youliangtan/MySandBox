# My Sand Box
This is just a graveyard for random codes. Nah... juz feel sad of deleting these, so a backup here might be a good idea. =)

### 1) Vehicle Count
- Using OpenCV background subtraction method to do simple vehicle count via a video footage
- User need to manually select the `in_line` and `out_line` on the road with 10m distance. then script will calculate the speed and num of vehicles
- This testing script is quite reliable in finding contour and centroid motion tracking.

### 2) Startathon 2018
- This is a failed fall detection python script. 
- Via OpenCV try to find the center mass of the detected object, and find its sudden change in z-motion.

### 3) SmartBoard
- This is a hackathon code for 2017. We built a huge advertisement board with the usage of web server, and arduino for a vending machine feature.
- Uses arduino serial and web server, flask to conduct communication

### 4) PcdTools
- with pcd/ply viewer
- downsample .pcd file
- pcl icp matcher

### 5) Google Cartographer launch file backup
- carto_rmf package
- config files and ROS launch for cartographer


### 6) Mock ROS2 ATM
- ROS2 Python Script 
- Pub and Sub custom defined msg, mock a AGV Task Manager
- 2 ros2 packages, need to compile `colcon symlink.....`


### 7) ROS2 Payload
- Testing pkg with cpp and py in one single pkg


### 8) PCL OBJ Pose Estimation (100% ready to use)
- line segmentation via ransac and clustering
- filter then identify target line, then find its pose
- refer to code for more info

### 9) ROS Aruco Pose Estimation (raw)
- raw launch file which documented what I have tested
....

### 10) Simple VRP
- using Google OR Tools to test out constraint and optimization problem


### 11) Baby Bloomer
 - Simple GUI and Backend flask server for user to upload an image, and get object detection with hazardous object with YOLOv8
 - Use `ngrok http 5000` to expose the local flask server to public testing

MORE TO BURY HERE, RIP Codes!!
