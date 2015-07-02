// 
// Author: Sudeep Pillai (spillai@csail.mit.edu) Aug 27, 2013
// General Purpose class for dealing with Kinect-like sensors
// Frame encapsulates all the relevant data, and chooses to
// retain information that may be required elsewhere in the
// processing pipeline (normals, cloud, rgb, gray etc)
// or any other arbitrary map 

// Ideas: Sensor class for publishing sensor frames
// Plot sensor-specific objects using class?
// 
#include <stdio.h>
#include <sys/time.h>
#include "frame_utils.hpp"

using namespace std;

namespace opencv_utils { 

Frame::Frame() {
  internalReset();
}

// Default copy constructor is sufficient (copy mat ref)
// Frame::Frame(const Frame& frame)  {

// }
Frame::Frame(const double utime, const cv::Mat& img, const cv::Mat& depth, const float scale, const bool fill_depth) {
// Reset all data on update
  internalReset();

  // Propagate utime
  utime_ = utime;

  // Populate img, and cloud
  // opencv_utils::scale_image(im, cv::INTER_CUBIC, scale);
  // opencv_utils::scale_image(depth, cv::INTER_NEAREST, scale);

  // Copy over img
  if (img.channels() == 1)
    setGray(img);
  else if (img.channels() == 3)
    setRGB(img);
  else { 
    std::cerr << "Unknown number of channels " << std::endl;
    assert(0);
  }

  // Copy depth (16bit)
  depth_ = depth.clone();
  
  // Unpack cloud from depth
  assert(!depth_.empty());
  unpack_cloud_scaled(depth_, cloud_, 1.f);

  return;

}

Frame::Frame(const kinect::frame_msg_t *msg, const float scale, const bool fill_depth) {
  // Reset all data on update
  internalReset();

  // Propagate utime
  utime_ = msg->timestamp;

  // Populate img, and cloud
  cv::Mat img;
  opencv_utils::unpack_kinect_frame_with_depth(msg, img, depth_, scale);

  // if (fill_depth) {
  //   fs::vision::weighted_joint_bilateral_filter(img, depth_, depth_);
  // }
  
  // opencv_utils::unpack_kinect_frame_with_cloud(msg, img, cloud_, scale);

  if (img.channels() == 1)
    setGray(img);
  else if (img.channels() == 3)
    setRGB(img);
  else { 
    std::cerr << "Unknown number of channels " << std::endl;
    assert(0);
  }

  // Unpack cloud from depth
  assert(!depth_.empty());
  unpack_cloud_scaled(depth_, cloud_, 1.f);
  
  // Compute depth mask
  // computeDepthMask(cv::Vec3f(NAN, NAN, NAN));
  return;
}


Frame::~Frame() {
}

void Frame::internalReset() {

  // reset
  recompute_normals = true, recompute_bilateral = true; 
  
  img_ = cv::Mat();
  gray_ = cv::Mat();
  depth_ = cv::Mat_<uint16_t>();
  img_mask_ = cv::Mat();

  cloud_ = cv::Mat3f();
  cloud_mask_ = cv::Mat1b();
  
  normals_ = cv::Mat3f();
  normals_mask_ = cv::Mat1b();

  combined_mask_ = cv::Mat1b();
}

bool Frame::ready() const { 
  return (!img_.empty() || !gray_.empty());
}

// void Frame::publish(lcm::LCM& lcm) const {
//   if (msg != NULL)
//     lcm.publish("KINECT_FRAME", msg);
// }

// // Available after update()
// // utime, img_, cloud_, cloud_mask_
// void Frame::update(const kinect::frame_msg_t *msg, const float scale, const bool fill_depth) {
// }

// void Frame::setTimestamp(const int64_t& utime, bool update = true) { 
//   utime_ = utime;
// }

void Frame::setRGB(const cv::Mat& img) {
  assert(img.channels() == 3);
  img_ = img.clone();
  cv::cvtColor(img_, gray_, cv::COLOR_BGR2GRAY);
}

void Frame::setGray(const cv::Mat& gray) {
  assert(gray.channels() == 1);
  gray_ = gray.clone();
  // cv::cvtColor(gray_, img_, cv::COLOR_GRAY2BGR);
}

void Frame::setImageMask(const cv::Mat1b& mask) {
  // set Image mask
  if (img_mask_.empty())
    img_mask_ = mask.clone();
  else
    cv::bitwise_and(img_mask_, mask, img_mask_);
  
  // Compute combined mask
  if (combined_mask_.empty()) 
    combined_mask_ = img_mask_.clone();
  else {
    // Bitwise AND
    cv::bitwise_and(combined_mask_, img_mask_, combined_mask_);
  }
}

uint64_t Frame::getTimestamp() const { 
  return utime_;
}

// cv::Mat Frame::getRGBRef() {
//   cv::Mat img;
//   cv::cvtColor(img_, img, cv::COLOR_BGR2RGB);
//   return img;
// }

const cv::Mat& Frame::getRGBRef() const {
  return img_;
}

const cv::Mat& Frame::getGrayRef() const {
  return gray_;
}

const cv::Mat_<uint16_t>& Frame::getDepthRef() const {
  return depth_;
}

const cv::Mat1b& Frame::getImageMaskRef() const {
  return img_mask_;
}

// cv::Mat Frame::getDepth() { 
//   return depth_.clone();
// }

const cv::Mat3f& Frame::getNormalsRef() const { 
  return normals_;
}

const cv::Mat1b& Frame::getNormalsMaskRef() const {
  return normals_mask_;
}

const cv::Mat3f& Frame::getCloudRef() const {
  return cloud_;
}

const cv::Mat1b& Frame::getCloudMaskRef() const {
  return cloud_mask_;
}

const cv::Mat1b& Frame::getCombinedMaskRef() const {
  return combined_mask_;
}

// params?
void Frame::computeNormals(const float scale,
                           const float depth_change_factor, const float smoothing_size,
                           const cv::Vec3f& NaN_value) {
  if (depth_.empty() || cloud_.empty()) {
    std::cerr << "Cannot compute normals: Depth/Cloud unavailable!" << std::endl;
    return;
  }
  
  // Get cloud and convert to pcl
  cv::Mat3f scloud;
  if (scale != 1.f)
    cv::resize(cloud_, scloud, cv::Size(depth_.cols*scale, depth_.rows*scale),0, 0, cv::INTER_AREA);
  else
    scloud = cloud_;

  // PCL cloud
  // if (!pcl_cloud_) { 
  pcl_cloud_ = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>());
  pcl_utils::convert_cloudmat_to_pcl(scloud, *pcl_cloud_);
    // }

  // Compute PCL normals
  pcl::PointCloud<pcl::Normal>::Ptr
      normals(new pcl::PointCloud<pcl::Normal>());
  pcl_utils::compute_normals(*pcl_cloud_, *normals, depth_change_factor, smoothing_size);

  // Convert PCL normals to cv::Vec3f
  cv::Mat3f snormals;
  pcl_utils::convert_pcl_to_normalsmat(*normals, snormals, NaN_value);

  // Write back to cloud, and normals with original size
  if (scale != 1.f) {
    cv::resize(snormals, normals_, depth_.size(), 0, 0, cv::INTER_AREA);
  } else {
    normals_ = snormals.clone();
  }
  
  // Compute the normals mask 
  computeNormalsMask(NaN_value);

  // set normals to computed
  recompute_normals = false; 
  
}

void Frame::fastBilateralFilter(float sigmaS, float sigmaR) {
  if (cloud_.empty()) {
    std::cerr << "Cannot perform fast bilateral filter : Cloud unavailable!" << std::endl;
    return;
  }

  if (!recompute_bilateral) {
    std::cerr << "Bilateral already computed, skipping!" << std::endl;
    return;
  }
  
  // Convert to PCL 
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr
      cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
  pcl_utils::convert_cloudmat_to_pcl(cloud_, *cloud);

  // Fast bilateral filter
  pcl_utils::fast_bilateral_filter(*cloud, sigmaS, sigmaR);

  // Convert PCL normals to cv::Vec3f
  pcl_utils::convert_pcl_to_cloudmat(*cloud, cloud_);

  // set bilateral to computed, but normals can be improved
  recompute_bilateral = false, recompute_normals = true; 
  
  return;
}


void Frame::medianFilter(int win_size) {
  if (cloud_.empty()) {
    std::cerr << "Cannot perform fast bilateral filter : Cloud unavailable!" << std::endl;
    return;
  }

  // Convert to PCL 
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr
      cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
  pcl_utils::convert_cloudmat_to_pcl(cloud_, *cloud);

  // Median filter
  pcl_utils::median_filter(*cloud, win_size);

  // Convert PCL normals to cv::Vec3f
  pcl_utils::convert_pcl_to_cloudmat(*cloud, cloud_);

  return;
}


void Frame::pruneDepth(const float depth_in_m, const cv::Vec3f& NaN_value) {
  if (cloud_.empty()) {
    std::cerr << "Cannot prune: Cloud unavailable!" << std::endl;
    return;
  }

  cv::Vec3f* v = cloud_.ptr<cv::Vec3f>(0);
  for (int j=0; j<cloud_.rows * cloud_.cols; j++) {
    cv::Vec3f& val = v[j];
    if (val[2] > depth_in_m)
      val = NaN_value;
  }

  // Recompute depth mask
  computeDepthMask(NaN_value);
  
}

void Frame::computeMask(const cv::Mat3f& mat, cv::Mat1b& mask, const cv::Vec3f& NaN_value) {
  if (mat.empty()) {
    std::cerr << "Cannot compute mask: parent unavailable!" << std::endl;
    return;
  }
  
  // Build mask
  mask = cv::Mat1b(mat.size());
  for (int j=0; j<mat.cols * mat.rows; j++)  { 
    const cv::Vec3f& v = mat.at<cv::Vec3f>(j);
    mask.at<uchar>(j) = ((v == v) && v != NaN_value) ? 255 : 0;
  }

  // Compute combined mask
  if (combined_mask_.empty()) 
    combined_mask_ = mask.clone();
  else {
    // Bitwise AND
    cv::bitwise_and(combined_mask_, mask, combined_mask_);
  }
}

void Frame::computeDepthMask(const cv::Vec3f& NaN_value) {
  computeMask(cloud_, cloud_mask_, NaN_value);   
}

void Frame::computeNormalsMask(const cv::Vec3f& NaN_value) {
  computeMask(normals_, normals_mask_, NaN_value);   
}

void Frame::plot() {
  if (!img_.empty())
    cv::imshow("RGB", getRGBRef());
  if (!cloud_.empty())
    cv::imshow("Cloud", getCloudRef());
  if (!normals_.empty()) {
    // For visualization purposes
    cv::Mat3f normals = getNormalsRef().clone();
    normals += cv::Mat3f::ones(normals.size());
    normals *= 0.5;
    cv::imshow("Normals", normals);
  }
  if (!cloud_mask_.empty())
    cv::imshow("Cloud Mask", getCloudMaskRef());
  if (!normals_mask_.empty())
    cv::imshow("Normals Mask", getNormalsMaskRef());
  if (!combined_mask_.empty())
    cv::imshow("Combined Mask", getCombinedMaskRef());
  cv::waitKey(10);
}

// std::vector<cv::Point3f>
// Frame::getXYZ(const std::vector<cv::Point2f>& pts) {
//   return get_xyz(depth_, pts);
// }

}
