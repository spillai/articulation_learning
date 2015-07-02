#include <stdio.h>
#include "color_utils.hpp"
#include <sys/time.h>

namespace opencv_utils { 

    cv::Vec3b hsv_to_bgrvec(const cv::Vec3b& hsv) { 
        cv::Mat hsv_mat(1,1,CV_8UC3);
        hsv_mat = cv::Scalar(int(hsv[0]), int(hsv[1]), int(hsv[2]));
        cv::Mat bgr_mat;
        cv::cvtColor(hsv_mat, bgr_mat, cv::COLOR_HSV2BGR);
        return bgr_mat.at<cv::Vec3b>(0,0);
    }

    cv::Vec3b hsv_to_bgrvec(const cv::Point3f& hsv) { 
        cv::Mat hsv_mat(1,1,CV_8UC3);
        hsv_mat = cv::Scalar(int(hsv.x), int(hsv.y), int(hsv.z));
        cv::Mat bgr_mat;
        cv::cvtColor(hsv_mat, bgr_mat, cv::COLOR_HSV2BGR);
        return bgr_mat.at<cv::Vec3b>(0,0);
    }

    cv::Scalar hsv_to_bgr(const cv::Point3f& hsv) { 
        cv::Vec3b bgr = hsv_to_bgrvec(hsv);
        return cv::Scalar(bgr[0], bgr[1], bgr[2]);
    }

    cv::Vec3b bgr_to_hsvvec(const cv::Vec3b& bgr) { 
        cv::Mat bgr_mat(1,1,CV_8UC3);
        bgr_mat = cv::Scalar(int(bgr[0]), int(bgr[1]), int(bgr[1]));
        cv::Mat hsv_mat;
        cv::cvtColor(bgr_mat, hsv_mat, cv::COLOR_BGR2HSV);
        return hsv_mat.at<cv::Vec3b>(0,0);
    }

    cv::Vec3b bgr_to_hsvvec(const cv::Point3f& bgr) { 
        cv::Mat bgr_mat(1,1,CV_8UC3);
        bgr_mat = cv::Scalar(int(bgr.x), int(bgr.y), int(bgr.z));
        cv::Mat hsv_mat;
        cv::cvtColor(bgr_mat, hsv_mat, cv::COLOR_BGR2HSV);
        return hsv_mat.at<cv::Vec3b>(0,0);
    }

    cv::Scalar bgr_to_hsv(const cv::Point3f& bgr) { 
        cv::Vec3b hsv = bgr_to_hsvvec(bgr);
        return cv::Scalar(hsv[0], hsv[1], hsv[2]);
    }


    cv::Scalar val_to_bgr(const float& val) {
      cv::Vec3b bgr = hsv_to_bgrvec(cv::Vec3b(val*180.f,255,255)); 
      return cv::Scalar(bgr[0], bgr[1], bgr[2]);
    }


}
