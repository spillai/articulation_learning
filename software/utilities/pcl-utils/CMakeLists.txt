cmake_minimum_required(VERSION 2.6.0)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -Wno-deprecated-declarations -Wno-write-strings")

# =========== inits ==============
set(POD_NAME pcl-utils)
set(LCM_PACKAGES lcm)
set(LIBBOT_PACKAGES bot2-core bot2-lcmgl-client bot2-param-client bot2-frames)
set(LCMTYPES_KINECT lcmtypes_kinect)
set(LCM_UTILS lcm-utils)
set(OPENCV_PACKAGES opencv perception-opencv-utils)
set(PCL_PACKAGES pcl_common-1.7 pcl_features-1.7 pcl_filters-1.7 pcl_geometry-1.7 pcl_io-1.7 pcl_kdtree-1.7 pcl_keypoints-1.7 pcl_octree-1.7 pcl_outofcore-1.7 pcl_people-1.7 pcl_recognition-1.7 pcl_registration-1.7 pcl_sample_consensus-1.7 pcl_search-1.7 pcl_segmentation-1.7 pcl_tracking-1.7 pcl_visualization-1.7)

set(Boost_USE_STATIC_LIBS OFF) 
set(Boost_USE_MULTITHREADED ON)  
set(Boost_USE_STATIC_RUNTIME OFF) 
find_package(Boost 1.46.1 COMPONENTS system thread date_time iostreams) 

if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS}) 
endif()

# pull in the pods macros. See cmake/pods.cmake for documentation
include(cmake/pods.cmake)

# =========== source files ==============
set(cpp_files 
  pcl_utils.cpp
  pose_utils.cpp
  frame_utils.cpp
)

set(hpp_files 
  pcl_utils.hpp
  pose_utils.hpp
  frame_utils.hpp
)

# =========== pcl-utils Library ==============
add_library(pcl-utils SHARED 
  ${cpp_files}
)

# Requirements
set(REQUIRED_PACKAGES ${LIBBOT_PACKAGES} ${OPENCV_PACKAGES} ${PCL_PACKAGES} eigen3)

# Use requirements
pods_use_pkg_config_packages(pcl-utils ${REQUIRED_PACKAGES} )

# Target link libraries 
target_link_libraries(pcl-utils ${Boost_LIBRARIES})

# Target version
set_target_properties(pcl-utils PROPERTIES SOVERSION 1)

# Install libararies
pods_install_libraries(pcl-utils)

# Install headers
pods_install_headers( ${hpp_files} DESTINATION pcl-utils)

# Install pkg_config
pods_install_pkg_config_file(pcl-utils
  LIBS -lpcl-utils
  REQUIRES ${REQUIRED_PACKAGES}
  VERSION 0.0.1)
