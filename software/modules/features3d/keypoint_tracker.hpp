// 
// Author: Sudeep Pillai (spillai@csail.mit.edu) Aug 26, 2013
// 

#ifndef KEYPOINT_TRACKER_HPP_
#define KEYPOINT_TRACKER_HPP_

// opencv 
#include <opencv2/opencv.hpp>

// opencv-utils includes
#include <perception_opencv_utils/opencv_utils.hpp>

// frame-utils includes
#include <pcl-utils/frame_utils.hpp>

// Profiler
#include <fs-utils/profiler.hpp>

// feature types
#include "feature_types.hpp"
#include "track_manager.hpp"

namespace fsvision
{

class KeyPointTracker {

 private:

  // Constants
  float dist_th_, angle_th_, max_features_;
  
  //--------------------------------------------
  // Tracklet manager
  //--------------------------------------------
  TrackManager<fsvision::Feature3D> track_manager_;
  Profiler profiler_;
  
  public:
  KeyPointTracker();
  KeyPointTracker(float dist_th, float angle_th);
  ~KeyPointTracker();
  bool needFeatures() { return track_manager_.ready(max_features_); }
  void setDistanceThreshold(float dist_th) { dist_th_ = dist_th; }
  void setAngleThreshold(float angle_th) { angle_th_ = angle_th; }
  void setMaxFeatures(int max_features) { max_features_ = max_features; }
  void update(const opencv_utils::Frame& frame, std::vector<fsvision::Feature3D>& fpts);
  void addFeatures (const std::vector<fsvision::Feature3D>& fpts);
  std::vector<fsvision::Feature3D> getStableFeatures(int min_track_size=1);
  void plot();
    
};

}

#endif 
