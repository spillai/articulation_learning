cmake_minimum_required(VERSION 2.6.0)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -Wno-deprecated-declarations -Wno-write-strings")
# pull in the pods macros. See cmake/pods.cmake for documentation
set(POD_NAME perception-utilities)
include(cmake/pods.cmake)

#tell cmake to build these subdirectories
add_subdirectory(fs-utils)
add_subdirectory(opencv-utils)
add_subdirectory(pcl-utils)
add_subdirectory(lcm-utils)
add_subdirectory(vis-utils)

