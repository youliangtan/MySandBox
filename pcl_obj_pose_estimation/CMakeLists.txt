cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
set (CMAKE_CXX_STANDARD 11)


project(object_pose_estimation)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
SET(BUILD_SHARED_LIBS ON)


find_package(PCL 1.8 REQUIRED)
find_package(YAML-CPP REQUIRED yaml-cpp)

include_directories(${PCL_INCLUDE_DIRS} ${YAML_CPP_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS} ${YAML_CPP_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS} ${YAML_CPP_DEFINITIONS})

add_executable (object_pose_estimation object_pose_estimation.cpp)
target_link_libraries (object_pose_estimation ${PCL_LIBRARIES} ${YAML_CPP_LIBRARIES})
