cmake_minimum_required(VERSION 2.6.0)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -Wno-deprecated-declarations -Wno-write-strings")
set(POD_NAME articulation)
include(cmake/pods.cmake)

# automatically build LCM types.  This also defines a number of CMake
# variables, see cmake/lcmtypes.cmake for details
include(cmake/lcmtypes.cmake)
lcmtypes_build()

include_directories(${LCMTYPES_INCLUDE_DIRS})

add_subdirectory(src/detector)
add_subdirectory(src/modeler)
add_subdirectory(src/renderer)
