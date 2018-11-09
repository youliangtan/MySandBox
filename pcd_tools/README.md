#Pcd Tools

pcd_viewer and downsampler (voxel_grid)

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
```
