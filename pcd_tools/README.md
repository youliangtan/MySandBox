# Pcd Tools
Some tools used while working on pcl project....

## Build

pcd_viewer, downsampler (voxel_grid), pcl's icp code

```
mkdir build
cd build
cmake ..
make
```

## Run Code
```
./voxel_grid {pcd path} -s {voxel leaf size}
./pcd_viewer {pcd ply path} {optional second pointcloud} -s {point size}
./icp_matcher   # pcl icp
```

## Additional Info / notes

[libpointmatcher icp](libpointmatcher.readthedocs.io)
```
cd ~/libpointmatcher/build/examples
./icp_simple [{1}.csv]  [{2}.csv]

```

* difficutly is to chg pcd to csv, how?
* effectiveness?
