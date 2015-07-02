#ifndef __FRAME_UTILS_HPP__
#define __FRAME_UTILS_HPP__

#include <limits>
#include <opencv2/opencv.hpp>
#include <bot_core/bot_core.h>

// lcm
#include <lcm/lcm-cpp.hpp>
#include <lcmtypes/bot2_param.h>

// opencv-utils includes
#include <perception_opencv_utils/opencv_utils.hpp>

// kinect include
#include <lcmtypes/kinect.hpp>

// pcl_utils include
#include <pcl-utils/pcl_utils.hpp>

// pcl includes
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

// // weighted-joint-bilateral filter
// #include <fs_perception_wrappers/wjbf/wjbf_wrapper.hpp>

namespace opencv_utils {

class Frame {
 
 public:
  uint64_t utime_;
  cv::Mat img_, gray_;
  cv::Mat_<uint16_t> depth_;
  cv::Mat1b img_mask_, cloud_mask_, normals_mask_, combined_mask_;
  cv::Mat3f cloud_, normals_; 

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcl_cloud_;
  pcl::PointCloud<pcl::Normal>::Ptr pcl_normals_;  

  bool recompute_normals, recompute_bilateral;
  
  void internalReset();

 public: 

  // Copy constructor for Frame
  Frame();
  // Frame(const Frame& frame);
  Frame(const double utime, const cv::Mat& img, const cv::Mat& depth, const float scale=1.f, const bool fill_depth=false);
  Frame(const kinect::frame_msg_t* msg, const float scale=1.f, const bool fill_depth=false);
  ~Frame();
  
  bool ready() const ;
  uint64_t getTimestamp() const;

  // void publish(lcm::LCM& lcm) const ;
  // void update(const kinect::frame_msg_t *msg, const float scale, const bool fill_depth=false);
  void computeNormals(float scale = 1.f, float depth_change_factor=0.5, float smoothing_size=10.f,
                      const cv::Vec3f& NaN_value = cv::Vec3f(NAN, NAN, NAN));
  void fastBilateralFilter(float sigmaS = 3.f, float sigmaR = 0.05);
  void medianFilter(int win_size = 5);
  
  void pruneDepth(const float depth_in_m, const cv::Vec3f& NaN_value = cv::Vec3f(NAN, NAN, NAN));
  
  void computeDepthMask(const cv::Vec3f& NaN_value);
  void computeNormalsMask(const cv::Vec3f& NaN_value);
  void computeMask(const cv::Mat3f& mat, cv::Mat1b& mask, const cv::Vec3f& NaN_value);
  // void computeNormalsMask(const cv::Vec3f& normals_NaN = cv::Vec3f(0,0,1));

  cv::Size size() const { return img_.size(); }
  
  void setRGB(const cv::Mat& img);
  void setGray(const cv::Mat& img);
  void setImageMask(const cv::Mat1b& mask);

  // cv::Mat getRGBRef() ;
  const cv::Mat& getRGBRef() const ;
  const cv::Mat& getGrayRef() const ;
  const cv::Mat_<uint16_t>& getDepthRef() const ;
  const cv::Mat1b& getImageMaskRef() const;
  
  const cv::Mat3f& getCloudRef() const;
  const cv::Mat1b& getCloudMaskRef() const;
  
  const cv::Mat3f& getNormalsRef() const;
  const cv::Mat1b& getNormalsMaskRef() const; 

  const cv::Mat1b& getCombinedMaskRef() const; 

  void plot();
  // std::vector<cv::Point3f> getXYZ(const std::vector<cv::Point2f>& pts);
};
}


#endif
