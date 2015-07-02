cmake_minimum_required(VERSION 2.6.0)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -Wno-deprecated-declarations -Wno-write-strings")

# =========== inits ==============
set(POD_NAME vis-utils)
set(LCM_PACKAGES lcm)
set(LIBBOT_PACKAGES bot2-core bot2-lcmgl-client bot2-param-client bot2-frames)
set(LCMTYPES_KINECT lcmtypes_kinect)
set(OPENCV_PACKAGES opencv) 
set(PCL_PACKAGES pcl_common-1.7 pcl_features-1.7 pcl_filters-1.7 pcl_io-1.7 pcl_kdtree-1.7 pcl_keypoints-1.7 pcl_octree-1.7 pcl_recognition-1.7 pcl_registration-1.7 pcl_sample_consensus-1.7 pcl_search-1.7 pcl_tracking-1.7)
# set(PCL_PACKAGES pcl_common-1.7 pcl_features-1.7 pcl_filters-1.7 pcl_geometry-1.7 pcl_io-1.7 pcl_kdtree-1.7 pcl_keypoints-1.7 pcl_octree-1.7 pcl_outofcore-1.7 pcl_people-1.7 pcl_recognition-1.7 pcl_registration-1.7 pcl_sample_consensus-1.7 pcl_search-1.7 pcl_segmentation-1.7 pcl_tracking-1.7 pcl_visualization-1.7)

# pull in the pods macros. See cmake/pods.cmake for documentation
include(cmake/pods.cmake)

# =========== source files ==============
set(cpp_files 
  vis_utils.cpp
)

set(hpp_files 
  vis_utils.hpp
)  
# =========== lcm-utils Library ==============
add_library(vis-utils SHARED ${cpp_files})

# Requirements
set(REQUIRED_PACKAGES ${LIBBOT_PACKAGES} ${LCMTYPES_KINECT} ${PCL_PACKAGES} 
  visualization)

# Use requirements
pods_use_pkg_config_packages(vis-utils ${REQUIRED_PACKAGES})  

# Target link libraries 
target_link_libraries(vis-utils)

# Target version
set_target_properties(vis-utils PROPERTIES SOVERSION 1)

# Install libararies
pods_install_libraries(vis-utils)

# Install headers
pods_install_headers(${hpp_files} DESTINATION vis-utils)

# Install pkg_config
pods_install_pkg_config_file(vis-utils
    CFLAGS
    LIBS -lvis-utils -lpthread
    REQUIRES ${REQUIRED_PACKAGES}
    VERSION 0.0.1)
