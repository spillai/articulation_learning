cmake_minimum_required(VERSION 2.6.0)

# find_package(PkgConfig REQUIRED)
# find_package(PCL 1.2 REQUIRED)
# #find_package(Flann)
# find_package(GLUT REQUIRED)

# pkg_check_modules(EIGEN REQUIRED eigen3)
# pkg_check_modules(OPENGL REQUIRED gl)
# pkg_check_modules(GLU REQUIRED glu)
# #pkg_check_modules(PCL_COMMON REQUIRED pcl_common-1.2)
# pkg_check_modules(GSL REQUIRED gsl)


# include_directories(
#   ${PCL_INCLUDE_DIRS} 
#   ${GSL_INCLUDE_DIRS}
#   ${GLUT_INCLUDE_DIRS}
#   ${GLU_INCLUDE_DIRS}
#   ${EIGEN_INCLUDE_DIRS}
#   ${FLANN_INCLUDE_DIRS}
# )

#link_directories(${PCL_LIBRARY_DIRS})
#link_directories(${GSL_LIBRARY_DIRS})

# add_subdirectory(tld)
# add_subdirectory(cvblobs)
# include_directories(tld tld/tracker)
# link_libraries(tld)

# create an executable, and make it public
add_executable(articulation-detector
  detector.cc
)

pods_use_pkg_config_packages ( articulation-detector 
  lcm lcmtypes_kinect apriltags 
  bot2-core bot2-lcmgl-client bot2-param-client bot2-frames
  opencv perception-opencv-utils lcm-utils pcl-utils
  eigen3
  visualization

 )


target_link_libraries(articulation-detector boost_system
  ${LCMTYPES_LIBS} 
)

pods_install_executables(articulation-detector)

# install all python files in the python/ subdirectory
#pods_install_python_packages(${CMAKE_CURRENT_SOURCE_DIR})

pods_install_python_script(articulation-collector collector.py)
