cmake_minimum_required(VERSION 2.6.0)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-deprecated-declarations -Wno-write-strings -Wno-unused-result")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations -Wno-write-strings -Wno-unused-result")

# =========== DBoW2 Library ==============
set(REQUIRED_PACKAGES ${OPENCV_PACKAGES} ${FS_UTILS} ${PCL_PACKAGES})
include_directories( ${CMAKE_CURRENT_SOURCE_DIR}/DUtils ${CMAKE_CURRENT_SOURCE_DIR}/DUtilsCV ${CMAKE_CURRENT_SOURCE_DIR}/DVision ${CMAKE_CURRENT_SOURCE_DIR}/DBoW2)
file(GLOB dbow2_h_files 
    "${CMAKE_CURRENT_SOURCE_DIR}/DBoW2/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/DUtils/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/DUtils/*.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/DUtilsCV/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/DVision/*.h"
)
file(GLOB dbow2_c_files 
    "${CMAKE_CURRENT_SOURCE_DIR}/DBoW2/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/DUtils/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/DUtilsCV/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/DVision/*.cpp"
)
add_library(DBoW2  SHARED 
    dbow2_wrapper.cpp
    ${dbow2_c_files}
)
pods_use_pkg_config_packages(DBoW2
    ${REQUIRED_PACKAGES} 
)
set_target_properties(DBoW2 PROPERTIES SOVERSION 1)
pods_install_libraries(DBoW2)
pods_install_headers( ${dbow2_h_files} dbow2_wrapper.hpp  
    DESTINATION fs_perception_wrappers/DBoW2)
pods_install_pkg_config_file(DBoW2
  LIBS -lDBoW2
  REQUIRES ${REQUIRED_PACKAGES} 
  VERSION 0.0.1)
