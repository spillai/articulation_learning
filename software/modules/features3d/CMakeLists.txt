cmake_minimum_required(VERSION 2.6.0)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-deprecated-declarations -Wno-write-strings -Wno-unused-result")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -Wno-deprecated-declarations -Wno-write-strings -Wno-unused-result")

# =========== Features3D Library ==============
add_library(features3d SHARED 
    keypoint_tracker.cpp
    track_manager.cpp
    feature_types.cpp
)

# Requirements
set(REQUIRED_PACKAGES ${OPENCV_PACKAGES} ${FS_UTILS})

# Use requirements
pods_use_pkg_config_packages(features3d ${REQUIRED_PACKAGES})

# Target version
set_target_properties(features3d PROPERTIES SOVERSION 1)

target_link_libraries(features3d 
  boost_system
)

# Install libararies
pods_install_libraries(features3d)

# Install headers
pods_install_headers( 
  feature_types.hpp
  keypoint_tracker.hpp 
  track_manager.hpp
  DESTINATION features3d)

# Install pkg_config
pods_install_pkg_config_file(features3d
  LIBS -lfeatures3d
  REQUIRES ${REQUIRED_PACKAGES} 
  VERSION 0.0.1)
