#ifndef OPENCV_COLOR_UTILS_H__
#define OPENCV_COLOR_UTILS_H__

#include <math.h>
#include <stdio.h>
#include <opencv2/opencv.hpp>

namespace opencv_utils { 
cv::Vec3b hsv_to_bgrvec(const cv::Vec3b& hsv);
cv::Vec3b hsv_to_bgrvec(const cv::Point3f& hsv);
cv::Scalar hsv_to_bgr(const cv::Point3f& hsv);
cv::Vec3b bgr_to_hsvvec(const cv::Vec3b& bgr);
cv::Vec3b bgr_to_hsvvec(const cv::Point3f& bgr);
cv::Scalar bgr_to_hsv(const cv::Point3f& bgr);
cv::Scalar val_to_bgr(const float& val);
}

#endif // OPENCV_COLOR_UTILS_H
