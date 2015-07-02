cmake_minimum_required(VERSION 2.6.0)

# pull in the pods macros. See cmake/pods.cmake for documentation
set(POD_NAME jpeg-utils)
include(cmake/pods.cmake)

find_package (JPEG REQUIRED)

add_subdirectory(src)
