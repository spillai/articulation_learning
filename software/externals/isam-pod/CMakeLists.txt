# =========== cmake stuff  ==============
# setting cmake module path
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
cmake_minimum_required(VERSION 2.6.0)
set(POD_NAME isam)

# Pods cmake
include(cmake/pods.cmake)
add_subdirectory(isam)
pods_install_pkg_config_file(isam
    LIBS -lisam
    VERSION 0.0.1)

