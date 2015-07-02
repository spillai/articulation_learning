#ifndef OPENCV_PLOT_UTILS_H__
#define OPENCV_PLOT_UTILS_H__

#include <math.h>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "color_utils.hpp"
#include <fs-utils/thread_safe_queue.hpp>

namespace opencv_utils { 
    cv::Scalar heatmap_bgr( float val );
    cv::Vec3b heatmap_bgrvec( float val );
void fillColors( std::vector<cv::Scalar>& colors, bool normalize=false, bool hsv=false );
    void fillColors( std::vector<cv::Vec3b>& colors );
    cv::Mat plot_1D_hist(std::vector<float>& hist);
    void putText(const std::string& text, cv::Mat& img);
    static cv::Vec3b computeColor(float fx, float fy);
    inline bool isFlowCorrect(cv::Point2f u);
    void drawOpticalFlow(const cv::Mat_<cv::Point2f>& flow, cv::Mat& dst, 
                                float maxmotion = -1);
    void drawFlow(const cv::Mat_<cv::Vec2f>& flow, cv::Mat& dst, 
                         float maxmotion = -1);
cv::Mat plot_1D(const std::string& ch, std::vector<std::pair<int, float> >& data);
}

#endif // OPENCV_PLOT_UTILS_H
