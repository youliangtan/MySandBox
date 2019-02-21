# PCL Pose Estimation
2D PCL as lib for finding target line, and pose estimation of the target line. Hokoyu collected .pcd file was used in this experiment.

![alt text](/ransac.png?)


## Build

```
mkdir build
cd build
cmake ..
make
```

## Run Code
```
./object_pose_estimation  -input SavedCloud0.pcd
```

## Additional Info / notes
A class `ObjectPoseEstimate2D` was created to perform this identification of target line. User can change the `config.yaml` to configure the selection of target line. Please make good use!!!
