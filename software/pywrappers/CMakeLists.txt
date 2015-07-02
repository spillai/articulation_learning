cmake_minimum_required(VERSION 2.6.0)
set(POD_NAME pywrappers)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations -Wno-write-strings -Wl,-no-undefined")

# =========== cmake stuff  ==============
# setting cmake module path
SET(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Pods cmake
include(cmake/pods.cmake)

# Boost python
SET(PYTHON_LIBRARY "/usr/lib/python2.7/config-x86_64-linux-gnu/libpython2.7.so")
include(cmake/boost-python.cmake)

# Cholmod, Eigen3
# include(cmake/FindCholmod.cmake)
include(cmake/FindEigen3.cmake)

# pull in the pods macros. See cmake/pods.cmake for documentation
set(LCM_PACKAGES lcm)
set(LIBBOT_PACKAGES bot2-core bot2-lcmgl-client bot2-param-client bot2-frames)
set(LCMTYPES_KINECT lcmtypes_kinect)
set(PERCEPTION-UTILS lcm-utils pcl-utils vis-utils perception-opencv-utils)
set(OPENCV_PACKAGES opencv)

set(PCL_PACKAGES pcl_common-1.7 pcl_features-1.7 pcl_filters-1.7 pcl_geometry-1.7 pcl_io-1.7 pcl_kdtree-1.7 
    pcl_keypoints-1.7 pcl_octree-1.7 pcl_outofcore-1.7 pcl_people-1.7 pcl_recognition-1.7 pcl_registration-1.7 
    pcl_sample_consensus-1.7 pcl_search-1.7 pcl_segmentation-1.7 pcl_tracking-1.7 pcl_visualization-1.7)

# =========== fs_type_conversions Python wrapper ==============
boost_python_module(fs_type_conversions fs_types.cpp 
    utils/opencv_numpy_conversion.cpp 
    # utils/eigen_numpy_conversion.cpp
)
pods_use_pkg_config_packages(fs_type_conversions
    opencv 
)
target_link_libraries(fs_type_conversions boost_system boost_python)
pods_install_libraries(fs_type_conversions)
pods_install_pkg_config_file(fs_type_conversions
    CFLAGS
    LIBS -lfs_type_conversions
    VERSION 0.0.0
)


# =========== fs_utils Python wrapper ==============
boost_python_module(fs_utils fs_utils.cpp)
pods_use_pkg_config_packages(fs_utils
  opencv features3d # mser3d # wjbf
  fs-utils lcm-utils vis-utils 
  # gpft idiap-mser 
  perception-opencv-utils 
  bot2-core bot2-lcmgl-client bot2-param-client bot2-frames 
  pcl-utils DBoW2
  ${PCL_PACKAGES} ${LCMTYPES_KINECT} # birchfield-klt  idiap-mser slic
  # lear-dense-trajectories fs-dense-trajectories 
  cv_utils
)
target_link_libraries(fs_utils boost_system fs_type_conversions)
pods_install_libraries(fs_utils)
pods_install_pkg_config_file(fs_utils
    CFLAGS
    LIBS -lfs_utils
    VERSION 0.0.1)

# =========== fs_isam Python wrapper ==============
pkg_check_modules(EIGEN3 eigen3)
find_package(Cholmod REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIR} ${CHOLMOD_INCLUDES} 
)
boost_python_module(fs_isam fs_isam.cpp)
pods_use_pkg_config_packages(fs_isam
  opencv isam 
)
target_link_libraries(fs_isam boost_system 
    ${EIGEN3_LIBRARIES} ${CHOLMOD_LIBRARIES} # ${GTSAM_LIBRARIES}
    fs_type_conversions)
pods_install_libraries(fs_isam)
pods_install_pkg_config_file(fs_isam
    CFLAGS
    LIBS -lfs_isam
    VERSION 0.0.1)


# =========== pcl_utils Python wrapper ==============
boost_python_module(fs_pcl_utils fs_pcl_utils.cpp)
pods_use_pkg_config_packages(fs_pcl_utils
  bot2-core bot2-lcmgl-client bot2-param-client bot2-frames opencv
  pcl-utils ${PCL_PACKAGES} pcl-utils 
  )
target_link_libraries(fs_pcl_utils boost_system fs_type_conversions)
pods_install_libraries(fs_pcl_utils)
pods_install_pkg_config_file(fs_pcl_utils
    LIBS -lfs_pcl_utils
    VERSION 0.0.1)


# =========== Apriltags Python wrapper ==============
boost_python_module(fs_apriltags fs_apriltags.cpp)
pods_use_pkg_config_packages(fs_apriltags 
  apriltags opencv glib-2.0 eigen3 pcl-utils features3d 
  fs-utils pcl-utils perception-opencv-utils
  ${PCL_PACKAGES} 
  )
target_link_libraries(fs_apriltags boost_system fs_type_conversions)
pods_install_libraries(fs_apriltags)
pods_install_pkg_config_file(fs_apriltags
    LIBS -lfs_apriltags
    REQUIRES apriltags
    VERSION 0.0.1)


# =========== Articulation Python wrapper ==============
find_package(PkgConfig REQUIRED)
pkg_check_modules(BULLET bullet)
pkg_check_modules(BOOST boost)
pkg_check_modules(GSL REQUIRED gsl)

include_directories(
  ${BULLET_INCLUDE_DIRS} 
  ${BOOST_INCLUDE_DIRS} 
  ${GSL_INCLUDE_DIRS}
)

boost_python_module(fs_articulation fs_articulation.cpp)
pods_use_pkg_config_packages(fs_articulation 
  articulation-structure-learner opencv glib-2.0 eigen3 
  lcm glib-2.0 lcmtypes_kinect bot2-core bot2-lcmgl-client bot2-frames
  bot2-param-client opencv perception-opencv-utils eigen3 vis-utils
  # pcl-utils 
  # features3d 
  # fs-utils pcl-utils perception-opencv-utils
  # ${PCL_PACKAGES} 
  )
target_link_libraries(fs_articulation boost_system fs_type_conversions)
pods_install_libraries(fs_articulation)
pods_install_pkg_config_file(fs_articulation
    LIBS -lfs_articulation
    REQUIRES articulation
    VERSION 0.0.1)
