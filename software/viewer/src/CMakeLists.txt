add_definitions(
    -std=gnu99)

#build the viewer

add_executable(al-viewer 
    udp_util.c
    main.cpp
    )

include_directories(
    ${GTK2_INCLUDE_DIRS}
    ${GLUT_INCLUDE_DIR})

target_link_libraries(al-viewer
    ${GTK2_LDFLAGS}
    ${GLUT_LIBRARIES})

pods_use_pkg_config_packages(al-viewer
  bot2-vis 
  bot2-lcmgl-renderer 
  bot2-frames-renderers 
  bot2-frames
  # er-renderers
  image-util-renderer
  kinect-renderer
  collections_renderer  
  articulation-renderer
  articulation-structure-learner  
)

pods_install_executables(al-viewer)
