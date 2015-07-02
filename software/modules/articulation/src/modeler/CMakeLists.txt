cmake_minimum_required(VERSION 2.6.0)

find_package(PkgConfig REQUIRED)
pkg_check_modules(BULLET bullet)
pkg_check_modules(BOOST boost)
pkg_check_modules(GSL REQUIRED gsl)


add_subdirectory(structure/hogman_minimal)
include_directories(structure/hogman_minimal/include)
include_directories(models)

include_directories(
  ${BULLET_INCLUDE_DIRS} 
  ${BOOST_INCLUDE_DIRS} 
  ${GSL_INCLUDE_DIRS}
)

# =========== Articulation Learner ==============
file(GLOB gp_h_files "${CURRENT_SOURCE_DIR}/models/gaussian_process/*.h"
    "${CURRENT_SOURCE_DIR}/models/gaussian_process/*.hpp"
    "${CURRENT_SOURCE_DIR}/models/gaussian_process/*.hxx")

add_library(articulation-structure-learner SHARED
  models/factory.cpp
  models/generic_model.cpp
  models/pca_gp_model.cpp
  models/prismatic_model.cpp
  models/rigid_model.cpp
  models/rotational_model.cpp
  models/SingleGP.cpp
  models/covarianceFunctions.cpp

  structure/ArticulatedObject.cpp
  structure/structs.cpp
  structure/hogman_wrapper.cpp

  utils.cpp
)

pods_use_pkg_config_packages ( articulation-structure-learner 
  lcm glib-2.0 lcmtypes_kinect bot2-core bot2-lcmgl-client bot2-frames
  bot2-param-client opencv perception-opencv-utils eigen3 vis-utils
)

pods_install_pkg_config_file(articulation-structure-learner
  CFLAGS
  LIBS -larticulation-structure-learner
  VERSION 0.0.1)

target_link_libraries(articulation-structure-learner 
  ${BULLET_LIBRARIES} 
  ${BOOST_LIBRARIES} 
  ${LCMTYPES_LIBS} 
  ${GSL_LIBRARIES}   
  boost_system
  hogman_csparse 
  #hogman_stuff 
  hogman_graph 
  hogman_graph_optimizer 
  hogman_graph_optimizer_hogman
)

pods_install_headers(
  structure/ArticulatedObject.hpp 
  structure/structs.h
  structure/hogman_wrapper.h
  
  DESTINATION articulation)

pods_install_headers(
  models/factory.h 
  models/generic_model.h 
  models/pca_gp_model.h 
  models/prismatic_model.h 
  models/rigid_model.h 
  models/rotational_model.h 

  ${gp_h_files}

  DESTINATION articulation)

pods_install_headers(utils.hpp 
  DESTINATION articulation)


# make the library public
pods_install_libraries(articulation-structure-learner)

find_package(OpenGL REQUIRED)
set(GLUT_LIBRARIES -lglut)
# =========== Articulation Learner ==============
# create an executable, and make it public
add_executable(articulation-learner
  structure_learner.cc
)
pods_install_pkg_config_file(articulation-learner
    LIBS -l${LCMTYPES_LIBS}
    REQUIRES bot2-vis bot2-frames
    VERSION 0.0.1)

pods_use_pkg_config_packages ( articulation-learner 
  lcm glib-2.0 lcmtypes_kinect bot2-core bot2-lcmgl-client bot2-frames
  bot2-param-client opencv perception-opencv-utils eigen3 vis-utils
)

target_link_libraries(articulation-learner
  articulation-structure-learner 
  boost_system
  hogman_csparse 
  #hogman_stuff 
  hogman_graph 
  hogman_graph_optimizer 
  hogman_graph_optimizer_hogman
)

pods_install_executables(articulation-learner)

# # =========== Articulation Generator ==============
# add_executable(articulation-generator
#   ground_truth_generator.cc
#   utils.cpp
# )

# pods_install_pkg_config_file(articulation-generator
#     LIBS -larticulation-generator -l${LCMTYPES_LIBS}
#     ${LCMTYPES_LIBS} 
#     REQUIRES bot2-vis bot2-frames
#     VERSION 0.0.1)

# pods_use_pkg_config_packages ( articulation-generator lcm glib-2.0 bot2-core bot2-lcmgl-client lcmtypes_er-lcmtypes opencv   bot2-core bot2-lcmgl-client bot2-param-client bot2-frames)

# target_link_libraries(articulation-generator
#   ${LCMTYPES_LIBS} 
# )

# pods_install_executables(articulation-generator)



