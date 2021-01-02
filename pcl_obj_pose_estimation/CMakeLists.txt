cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set (CMAKE_CXX_STANDARD 17)

project(object_pose_estimation)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
SET(BUILD_SHARED_LIBS ON)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -lboost_thread")

set(BOOST_LIBS date_time system)
find_package(Boost COMPONENTS ${BOOST_LIBS} REQUIRED)

find_package(PCL 1.8 REQUIRED)
find_package(YAML-CPP REQUIRED yaml-cpp)

include_directories(${PCL_INCLUDE_DIRS} ${YAML_CPP_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS} ${YAML_CPP_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS} ${YAML_CPP_DEFINITIONS})

add_executable (object_pose_estimation object_pose_estimation.cpp)
target_link_libraries (object_pose_estimation ${PCL_LIBRARIES} ${YAML_CPP_LIBRARIES} ${Boost_LIBRARIES})
