add_definitions(-Wall)

# Create a shared library 
add_library(jpeg-utils SHARED jpeg-utils.c)

# set the library API version.  Increment this every time the public API
# changes.
set_target_properties(jpeg-utils PROPERTIES SOVERSION 1)

# JPEG includes and libraries
include_directories(${JPEG_INCLUDE_DIR})
target_link_libraries(jpeg-utils ${JPEG_LIBRARIES})

# make the header public
# install it to include/hello
pods_install_headers(jpeg-utils.h DESTINATION jpeg-utils)

# make the library public
pods_install_libraries(jpeg-utils)

# create a pkg-config file for the library, to make it for other software to
# use it.
pods_install_pkg_config_file(jpeg-utils
    CFLAGS
    LIBS -ljpeg-utils -ljpeg
    VERSION 0.0.1)
