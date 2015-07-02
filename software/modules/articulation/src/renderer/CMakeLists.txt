find_package(PkgConfig REQUIRED)
find_package(OpenGL REQUIRED)
pkg_check_modules(BULLET bullet)
pkg_check_modules(BOOST boost)
pkg_check_modules(GSL REQUIRED gsl)

set(GLUT_LIBRARIES -lglut)

pkg_check_modules(BOT2_VIS bot2-vis)
if(NOT BOT2_VIS_FOUND)
    message("bot2-vis not found.  Not building libbot2 renderer")
    return()
endif(NOT BOT2_VIS_FOUND)

include_directories(
  ${BULLET_INCLUDE_DIRS} 
  ${BOOST_INCLUDE_DIRS} 
  ${GSL_INCLUDE_DIRS}
)

include_directories(../modeler/structure/hogman_minimal/include)
# =========== Articulation Renderer ==============
# create an executable, and make it public
add_library(articulation-renderer SHARED
  articulation_renderer.cc
)

# target_link_libraries(articulation-renderer)

pods_use_pkg_config_packages ( articulation-renderer 
  articulation-structure-learner 
  bot2-vis bot2-frames bot2-core bot2-lcmgl-client 
  lcm glib-2.0 lcmtypes_kinect bot2-param-client opencv perception-opencv-utils eigen3
  ${LCMTYPES_LIBS}
)

pods_install_headers(articulation_renderer.h 
  DESTINATION articulation)

pods_install_pkg_config_file(articulation-renderer
        CFLAGS
        LIBS -larticulation-renderer 
        REQUIRES bot2-vis bot2-frames 
	# DEPENDS -larticulation-renderer -l
        VERSION 0.0.1)

pods_install_libraries(articulation-renderer)

# add_executable(articulation-viewer main.c)

# target_link_libraries(articulation-viewer
#   ${LCMTYPES_LIBS} 
#   ${GLUT_LIBRARIES})

# pods_use_pkg_config_packages(articulation-viewer
#     glib-2.0
#     bot2-vis bot2-core bot2-lcmgl-client bot2-param-client bot2-frames
#     lcm lcmtypes_kinect lcmtypes_er-lcmtypes lcmtypes_articulation 
#     articulation-renderer
#     #lcmtypes_er-articulation 
#     )
# pods_install_executables(articulation-viewer)
