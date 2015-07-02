add_library(collections_renderer SHARED collections_renderer.cpp)

set (REQUIRED_LIBS
    lcm
    bot2-vis
    bot2-frames
    eigen3)

pods_use_pkg_config_packages(collections_renderer ${REQUIRED_LIBS})

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    target_link_libraries(collections_renderer boost_system-mt boost_thread-mt)
endif(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")

# set the library API version.  Increment this every time the public API
# changes.
set_target_properties(collections_renderer PROPERTIES SOVERSION 1)

pods_install_libraries(collections_renderer)

pods_install_headers(collections_renderer.hpp DESTINATION visualization)

pods_install_pkg_config_file(collections_renderer
    LIBS -lcollections_renderer
    REQUIRES ${REQUIRED_LIBS}
    VERSION 0.0.1)

