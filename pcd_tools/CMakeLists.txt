cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(pcd_viewer)

find_package(PCL 1.7 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (pcd_viewer pcd_viewer.cpp)
target_link_libraries (pcd_viewer ${PCL_LIBRARIES})

add_executable (voxel_grid voxel_grid.cpp)
target_link_libraries (voxel_grid ${PCL_LIBRARIES})

add_executable (icp_matcher icp_matcher.cpp)
target_link_libraries (icp_matcher ${PCL_LIBRARIES})
