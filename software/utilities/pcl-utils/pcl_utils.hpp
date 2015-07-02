// 
// Author: Sudeep Pillai (spillai@csail.mit.edu) 
// Updates: Aug 07, 2013
// 

#ifndef PCL_UTILS_HPP_
#define PCL_UTILS_HPP_

// Standard includes
#include <unistd.h>
#include <stdio.h>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <map>
#include <set> 

// opencv includes
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>

// kinect include
#include <lcmtypes/kinect.hpp>

// pcl includes
// #include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/search/organized.h>
#include <pcl/search/kdtree.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/filters/conditional_removal.h>
// #include <pcl/segmentation/extract_clusters.h>
#include <pcl/features/don.h>

#include <pcl/range_image/range_image.h>
#include <pcl/range_image/range_image_planar.h>

#include <pcl/filters/fast_bilateral.h>
#include <pcl/filters/fast_bilateral_omp.h>
#include <pcl/filters/median_filter.h>

// namespace fs { namespace python {

namespace pcl_utils { 

#define NaN std::numeric_limits<float>::quiet_NaN()

// To PCL
template <typename PointT>
void convert_mat_to_pcl(const cv::Mat_<float>& cloud, pcl::PointCloud<PointT>& pcl);
void convert_mat_to_pclnormal(const cv::Mat_<float>& cloud, pcl::PointCloud<pcl::PointNormal>& pcl);

void convert_img_with_cloudmat_to_pcl(const cv::Mat& img, 
                                      const cv::Mat_<cv::Vec3f>& cloud, 
                                      pcl::PointCloud<pcl::PointXYZRGB>::Ptr& pcl); 

template <typename PointT>
void convert_cloudmat_to_pcl(const cv::Mat3f& cloud, pcl::PointCloud<PointT>& pcl);


void convert_depthmat_to_rangeimageplanar(const cv::Mat& depth, 
                            pcl::RangeImagePlanar& range);

// To cv::Mat
template <typename PointT>
void convert_pcl_to_cloudmat(const pcl::PointCloud<PointT>& pcl, cv::Mat_<cv::Vec3f>& cloud, 
             const cv::Vec3f& NaNvalue = 
             cv::Vec3f(NaN,NaN,NaN)); 

void convert_pcl_to_normalsmat(const pcl::PointCloud<pcl::Normal>& pcl, 
                               cv::Mat3f& normals, 
                               const cv::Vec3f& NaNvalue = 
                               cv::Vec3f(NaN,NaN,NaN));
    
// Other conversions
void project_points(const pcl::PointCloud<pcl::InterestPoint>::Ptr& cloud, 
                    std::vector<cv::Point2f>& points);


// Unpack utilities
void unpack_kinect_frame_with_cloud(const kinect::frame_msg_t* msg, 
                                    cv::Mat& img, 
                                    pcl::PointCloud<pcl::PointXYZRGB>::Ptr& pcl, 
                                    float scale=1.f);

void unpack_kinect_frame_with_range(const kinect::frame_msg_t* msg, 
                                    cv::Mat& img, 
                                    pcl::RangeImagePlanar& range, 
                                    float scale=1.f);

void compute_normals(const cv::Mat3f& cloud, cv::Mat3f& normals,
                     float depth_change_factor=0.5, float smoothing_size=10.f);

template <typename PointT>
void compute_normals(const pcl::PointCloud<PointT>& cloud, 
                     pcl::PointCloud<pcl::Normal>& normals, 
                     float depth_change_factor=0.5, 
                     float smoothing_size=10.f);

template <typename PointT>
void fast_bilateral_filter(pcl::PointCloud<PointT>& cloud,
                           float sigmaS=30.f, float sigmaR=0.05);

template <typename PointT>
void median_filter(pcl::PointCloud<PointT>& cloud,
                   int win_size=5);


template <typename PointT>
void compute_difference_of_normals(const pcl::PointCloud<PointT>& cloud,
                                     pcl::PointCloud<PointT>& out, 
                                     const int& small_scale=3, const int& large_scale=7);


}

// } // namespace vision
// } // namespace fs


#endif
