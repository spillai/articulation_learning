# Create a shared library libhello.so with a single source file
add_library(qhull SHARED
   user.c
   global.c
   stat.c 
   io.c
   geom2.c
   poly2.c
   merge.c
   qhull.c
   geom.c
   poly.c 
   qset.c 
   mem.c

#  geom2.c  geom.c  global.c  io.c  mem.c  merge.c  poly2.c  poly.c  qconvex.c  qdelaun.c  qhalf.c  qhull.c  qset.c  qvoronoi.c  rbox.c  stat.c  unix.c  user.c  user_eg2.c  user_eg.c

    #geom2.c  geom.c  global.c  io.c  mem.c  merge.c  poly2.c  poly.c  qconvex.c  qdelaun.c  qhalf.c  qhull.c  qset.c  qvoronoi.c  rbox.c  stat.c  unix.c  user.c  user_eg2.c  user_eg.c
)

# make the header public
# install it to include/hello
pods_install_headers(geom.h  io.h  mem.h  merge.h  poly.h  qhull_a.h  qhull.h  qset.h  stat.h  user.h DESTINATION qhull)

# make the library public
pods_install_libraries(qhull)

# uncomment these lines to link against another library via pkg-config
#set(REQUIRED_PACKAGES pkg_a pkg_b)
#pods_use_pkg_config_packages(hello ${REQUIRED_PACKAGES})

# create a pkg-config file for the library, to make it easier for other
# software to use.
pods_install_pkg_config_file(qhull
    CFLAGS
    LIBS -lqhull
    REQUIRES ${REQUIRED_PACKAGES}
    VERSION 0.0.1)
