find_package(PkgConfig REQUIRED)
pkg_check_modules(BOT2_VIS bot2-vis)
if(NOT BOT2_VIS_FOUND)
    message("bot2-vis not found.  Not building libbot2 renderer")
    return()
endif(NOT BOT2_VIS_FOUND)

#add_definitions(-Wall -std=gnu99)

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    link_libraries(boost_thread-mt)
else(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    link_libraries(boost_thread)
endif(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")

add_executable(collections_example collections_example.cpp)
pods_use_pkg_config_packages(collections_example lcm)

# make executables public
pods_install_executables(collections_example)

add_executable(collections_viewer collections_viewer.cpp)
pods_use_pkg_config_packages(collections_viewer bot2-vis 
  bot2-lcmgl-renderer lcm collections_renderer)
target_link_libraries(collections_viewer boost_system boost_system-mt boost_thread)
pods_install_executables(collections_viewer)
