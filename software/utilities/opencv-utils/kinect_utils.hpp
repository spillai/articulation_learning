#ifndef KINECT_UTILS_H
#define KINECT_UTILS_H

#include <lcm/lcm.h>
#include <bot_core/bot_core.h>
#include <lcmtypes/kinect.hpp>
#include <opencv2/opencv.hpp>

#include <limits>
#include <image_utils/jpeg.h>
#include <kinect/kinect-utils.h>

#include <zlib.h>

using namespace std;

namespace opencv_utils { 

void scale_image(cv::Mat& img, int interp, const float scale);
void unpack_rgb(const kinect::frame_msg_t* msg, cv::Mat& img);
void unpack_depth(const kinect::frame_msg_t* msg, cv::Mat_<uint16_t>& img);
void unpack_rgb_scaled(const kinect::frame_msg_t* msg, 
                       cv::Mat& img, const float scale = 1.f); 
void upack_depth_scaled(const kinect::frame_msg_t* msg, 
                        cv::Mat_<uint16_t>& img, const float scale = 1.f);
void unpack_cloud_scaled(const cv::Mat_<uint16_t>& depth,
                         cv::Mat_<cv::Vec3f>& cloud, float scale = 1.f);
void unpack_kinect_frame(const kinect::frame_msg_t* msg, 
                         cv::Mat& img, float scale = 1.f); 
void unpack_kinect_frame_with_depth(const kinect::frame_msg_t* msg, 
                                    cv::Mat& img, 
                                    cv::Mat_<uint16_t>& depth, float scale = 1.f); 
void unpack_kinect_frame_with_cloud(const kinect::frame_msg_t* msg, 
                                    cv::Mat& img, cv::Mat_<cv::Vec3f>& cloud, float scale = 1.f);
// void depth_to_cloud(cv::Mat& depth, cv::Mat_<cv::Vec3f>& cloud, float scale = 1.f);
cv::Mat3b color_depth_map(const cv::Mat& depth, float scale = 30.f/1000, bool min_max = false);

cv::Vec3f estimate_noise(float depth); // depth in m

cv::Point3f
get_xyz(const cv::Mat_<uint16_t>& depth, cv::Point2f& pt);

std::vector<cv::Point3f>
get_xyz(const cv::Mat_<uint16_t>& depth, std::vector<cv::Point2f>& pts);
}

#endif
