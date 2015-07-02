cmake_minimum_required(VERSION 2.6.0)

# add_subdirectory(wjbf_depth_filling)

# pull in the pods macros. See cmake/pods.cmake for documentation
set(POD_NAME perception-opencv-utils)
include(cmake/pods.cmake)

find_package(PkgConfig REQUIRED)
# add_subdirectory(matlab)

# FIND_PACKAGE(Qt)
FIND_PACKAGE(OpenGL)
# find_package(CGAL COMPONENTS Core )

# if ( CGAL_FOUND )
#   include( ${CGAL_USE_FILE} )
#   # include( CGAL_CreateSingleSourceCGALProgram )
#   include_directories(${CGAL_INCLUDE_DIR})
# else()
#     message(STATUS "This program requires the CGAL library, and will not be compiled.")
# endif()

# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c++0x -Wno-deprecated-declarations -Wno-write-strings")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -Wno-deprecated-declarations -Wno-write-strings")

set(cpp_files 
  # opencv_utils.cpp
  math_utils.cpp
  # imshow_utils.cpp
  color_utils.cpp
  imgproc_utils.cpp
  plot_utils.cpp
  calib_utils.cpp
  #time_utils.cpp
  kinect_utils.cpp
)


set(hpp_files 
  opencv_utils.hpp
  math_utils.hpp
  # imshow_utils.hpp
  color_utils.hpp
  imgproc_utils.hpp
  plot_utils.hpp
  calib_utils.hpp
  #time_utils.hpp
  kinect_utils.hpp
)  

# Create a shared library libhello.so with a single source file
add_library(perception-opencv-utils SHARED
    ${cpp_files})

set_target_properties(perception-opencv-utils PROPERTIES SOVERSION 1)

pods_use_pkg_config_packages(perception-opencv-utils opencv
    bot2-core lcmtypes_kinect ${FS_UTILS}
    #kinect-utils Qt5OpenGL
)

target_link_libraries(perception-opencv-utils
    pthread z                 
    ${OPENGL_LIBRARIES} 
    # ${QT_QTCORE_LIBRARY_RELEASE}
    # ${QT_QTGUI_LIBRARY_RELEASE}
    # ${QT_QTOPENGL_LIBRARY_RELEASE}
    # ${CGAL_LIBRARIES} ${CGAL_3RD_PARTY_LIBRARIES}
)


# make the header public
pods_install_libraries(perception-opencv-utils)
pods_install_headers(${hpp_files} DESTINATION perception_opencv_utils)

# uncomment these lines to link against another library via pkg-config
set(REQUIRED_PACKAGES bot2-core opencv image-utils) # kinect-utils 
#pods_use_pkg_config_packages(hello ${REQUIRED_PACKAGES})

# create a pkg-config file for the library, to make it easier for other
# software to use.
pods_install_pkg_config_file(perception-opencv-utils
    CFLAGS
    LIBS -lperception-opencv-utils -lpthread -lz
    REQUIRES ${REQUIRED_PACKAGES} 
    VERSION 0.0.1)
