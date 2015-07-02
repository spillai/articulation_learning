cmake_minimum_required(VERSION 2.6.0)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -Wno-deprecated-declarations -Wno-write-strings")

# =========== inits ==============
set(POD_NAME lcm-utils)
set(LCM_PACKAGES lcm)
set(LIBBOT_PACKAGES bot2-core bot2-lcmgl-client bot2-param-client bot2-frames)
set(LCMTYPES_KINECT lcmtypes_kinect)

# pull in the pods macros. See cmake/pods.cmake for documentation
include(cmake/pods.cmake)

# =========== source files ==============
set(cpp_files 
  lcm-reader-util.cpp
)

set(hpp_files 
  lcm-reader-util.hpp
)  

# =========== lcm-utils Library ==============
add_library(lcm-utils SHARED ${cpp_files})

# Requirements
set(REQUIRED_PACKAGES ${LIBBOT_PACKAGES} ${LCMTYPES_KINECT})

# Use requirements
pods_use_pkg_config_packages(lcm-utils ${REQUIRED_PACKAGES})  

# Target link libraries 
target_link_libraries(lcm-utils)

# Target version
set_target_properties(lcm-utils PROPERTIES SOVERSION 1)

# Install libararies
pods_install_libraries(lcm-utils)

# Install headers
pods_install_headers(${hpp_files} DESTINATION lcm-utils)

# Install pkg_config
pods_install_pkg_config_file(lcm-utils
    CFLAGS
    LIBS -llcm-utils
    REQUIRES ${REQUIRED_PACKAGES}
    VERSION 0.0.1)
