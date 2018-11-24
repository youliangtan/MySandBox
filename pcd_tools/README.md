# Pcd Tools
Some tools used while working on pcl project....

pcd_viewer, downsampler (voxel_grid), pcl's icp code

```
mkdir build
cd build
cmake ..
make
```

Run Code
```
./voxel_grid {pcd path} -s {voxel leaf size}
./pcd_viewer {pcd ply path} {optional second pointcloud} -s {point size}
./icp_matcher
```
