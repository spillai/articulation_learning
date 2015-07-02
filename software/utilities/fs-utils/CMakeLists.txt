cmake_minimum_required(VERSION 2.6.0)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -Wno-deprecated-declarations -Wno-write-strings")

# =========== inits ==============
set(POD_NAME fs-utils)
set(LCM_PACKAGES lcm)
set(LIBBOT_PACKAGES bot2-core bot2-lcmgl-client bot2-param-client bot2-frames)

# pull in the pods macros. See cmake/pods.cmake for documentation
include(cmake/pods.cmake)

# =========== source files ==============
set(cpp_files 
  profiler.cpp
  quaternion.cpp
  rigid_transform.cpp
)

set(hpp_files 
  profiler.hpp
  quaternion.hpp
  rigid_transform.hpp
  thread_safe_queue.hpp
  thread_safe_grabber.hpp
)


# =========== fs-utils Library ==============
add_library(fs-utils SHARED 
  ${cpp_files}
)

# Requirements
set(REQUIRED_PACKAGES ${LIBBOT_PACKAGES})

# Use requirements
pods_use_pkg_config_packages(fs-utils ${REQUIRED_PACKAGES})

# Target version
set_target_properties(fs-utils PROPERTIES SOVERSION 1)

# Install libararies
pods_install_libraries(fs-utils)

# Install headers
pods_install_headers( ${hpp_files} DESTINATION fs-utils)

# Install pkg_config
pods_install_pkg_config_file(fs-utils
  LIBS -lfs-utils
  REQUIRES ${REQUIRED_PACKAGES} 
  VERSION 0.0.1)

