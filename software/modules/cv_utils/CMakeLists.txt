cmake_minimum_required(VERSION 2.6.0)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-deprecated-declarations -Wno-write-strings -Wno-unused-result")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -Wno-deprecated-declarations -Wno-write-strings -Wno-unused-result")

# =========== cvuilts Library ==============
file(GLOB h_files "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
file(GLOB c_files "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

add_library(cv_utils SHARED 
    Distance.cpp 
    Triangulation.cpp 
    FindCameraMatrices.cpp
    # MultiCameraDistance.cpp
    # IDistance.cpp
    # MultiCameraPnP.cpp
    Common.cpp
    # IFeatureMatcher.cpp
    RichFeatureMatcher.cpp
    # OFFeatureMatcher.cpp
    # BundleAdjuster.cpp
    # GPUSURFFeatureMatcher.cpp
    AbstractFeatureMatcher.cpp
    # SfMUpdateListener.cpp
)

# Requirements
set(REQUIRED_PACKAGES ${OPENCV_PACKAGES}) # ${FS_UTILS}

# Use requirements
pods_use_pkg_config_packages(cv_utils ${REQUIRED_PACKAGES})

# Target version
set_target_properties(cv_utils PROPERTIES SOVERSION 1)

# Install libararies
pods_install_libraries(cv_utils)

# Install headers
pods_install_headers( ${h_files} DESTINATION cv_utils)

# Install pkg_config
pods_install_pkg_config_file(cv_utils
  LIBS -lcv_utils
  REQUIRES ${REQUIRED_PACKAGES} 
  VERSION 0.0.1)
