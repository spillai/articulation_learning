# =========== cmake stuff  ==============
# setting cmake module path
cmake_minimum_required(VERSION 2.6.0)
set(POD_NAME fs_perception_modules)

# Pods cmake
include(cmake/pods.cmake)

# =========== inits ==============
set(LCM_PACKAGES lcm)
set(LIBBOT_PACKAGES bot2-core bot2-lcmgl-client bot2-param-client bot2-frames)
set(LCMTYPES_KINECT lcmtypes_kinect)
set(LCM_UTILS lcm-utils)
set(OPENCV_PACKAGES opencv perception-opencv-utils)
set(FS_UTILS fs-utils lcm-utils vis-utils pcl-utils) 
set(PCL_PACKAGES pcl_common-1.7 pcl_features-1.7 pcl_filters-1.7 pcl_io-1.7 pcl_kdtree-1.7 pcl_keypoints-1.7 pcl_octree-1.7 pcl_recognition-1.7 pcl_registration-1.7 pcl_sample_consensus-1.7 pcl_search-1.7 pcl_segmentation-1.7 pcl_tracking-1.7)

find_package(Boost COMPONENTS python REQUIRED)

# Authors: Sudeep (spillai@csail.mit.edu)
add_subdirectory(features3d)
add_subdirectory(articulation)
add_subdirectory(DBoW2)
add_subdirectory(cv_utils)

# Externals
# add_subdirectory(mser3d)
# add_subdirectory(gpft)
# add_subdirectory(fs_dense_trajectories)
# add_subdirectory(lear_dense_trajectories)
# add_subdirectory(birchfield_klt)
# add_subdirectory(slic)
# add_subdirectory(idiap_mser)

# add_subdirectory(lear_gist)



