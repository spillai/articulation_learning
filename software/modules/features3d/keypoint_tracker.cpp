// 
// Author: Sudeep Pillai (spillai@csail.mit.edu) Aug 26, 2013
// 

#include "keypoint_tracker.hpp"

namespace fsvision
{

KeyPointTracker::KeyPointTracker() {
  // Setup profiler
  profiler_.setName("KeyPointTracker");
  profiler_.debug("KeyPointTracker:CTOR");
}

KeyPointTracker::KeyPointTracker(float dist_th, float angle_th)
    : dist_th_(dist_th), angle_th_(angle_th)  {
  // Setup profiler
  profiler_.setName("KeyPointTracker");
  profiler_.debug("KeyPointTracker:CTOR");
}
KeyPointTracker::~KeyPointTracker() {

}

void
KeyPointTracker::plot() {
  profiler_.enter("plot");
  // track_manager_.plot();
  profiler_.leave("plot");

}

void 
KeyPointTracker::addFeatures (const std::vector<fsvision::Feature3D>& fpts) { 
  //--------------------------------------------
  // Profiler
  //--------------------------------------------
  profiler_.enter("addFeatures(fpts)");

  //--------------------------------------------
  // Add Features and describe them
  //--------------------------------------------
  track_manager_.addFeatures(fpts);

  profiler_.leave("addFeatures(fpts)");
}

std::vector<fsvision::Feature3D>
KeyPointTracker::getStableFeatures(int min_track_size) {

  std::vector<fsvision::Feature3D> fpts;
  track_manager_.getAllFeatures(fpts, min_track_size);
  return fpts;
 
}

void 
KeyPointTracker::update(const opencv_utils::Frame& frame, 
                        std::vector<fsvision::Feature3D>& c_fpts) {

  // Matching by finding disjoint union sets  ===============================
  // Cluster keypoints that are within radius ball, and angle diff
  profiler_.enter("update");

  //--------------------------------------------
  // Retrieve previous stable features
  //--------------------------------------------
  int allowed_skips = 5;
  std::vector<fsvision::Feature3D> p_fpts;
  track_manager_.getStableFeatures(p_fpts);

  //--------------------------------------------
  // Determine correspondences 
  // 1. Prop. corr from p_fpts to c_fpts
  //    c_fpts are appropriately ID'd
  // 2. Add new features (id=-1) to c_tpts 
  //    track_manager takes care of IDing
  //--------------------------------------------
  // Perform correspondence matching
  if (p_fpts.size()) {
            
    // Convert points
    std::vector<cv::Point3f> p_points;
    fsvision::Feature3D::convert(p_fpts, p_points);
    std::vector<cv::Point3f> c_points;
    fsvision::Feature3D::convert(c_fpts, c_points);

    // Reshape to 1-channel matrix
    cv::Mat data = cv::Mat(p_points).reshape(1);
    cv::Mat query = cv::Mat(c_points).reshape(1);
    // std::cerr << data << std::endl;

    // Build index
    cv::flann::Index index(data, cv::flann::KDTreeIndexParams(4));

    // Query
    cv::Mat indices(c_points.size(), 10, CV_32SC1);
    cv::Mat dists(c_points.size(), 10, CV_32FC1);
    index.knnSearch(query, indices, dists, 10,
                    cv::flann::SearchParams(64));

    // NN matching
    for (int j=0; j<indices.rows; j++) {
      int* p = indices.ptr<int>(j);
      float* d = dists.ptr<float>(j);

      const fsvision::Feature3D& c_fj = c_fpts[j];
      const cv::KeyPoint& c_kpj = c_fj.get_keypoint();

      int bestIdx = -1;
      for (int k=0; k<indices.cols; k++) {
        // Skip random indices
        int nidx = p[k];
        if (nidx < 0 || nidx >= p_fpts.size())
          continue;
        
        const fsvision::Feature3D& p_fk = p_fpts[nidx];
        const cv::KeyPoint& p_kpk = p_fk.get_keypoint();

        // 3D Matching
        // 1. Angle between normals/tangents within threshold
        if ((c_fj.normal().dot(p_fk.normal()) < cos(angle_th_)) ||
            (c_fj.tangent().dot(p_fk.tangent()) < cos(angle_th_)))
          continue;

        // Do 2D matching ??
        
        // 2. L2 Distance between R^3 points
        if (d[k] < dist_th_ * dist_th_) {
          bestIdx = nidx;
          break;
        }
      }
      if (bestIdx < 0) { 
        c_fpts[j].set_id(-1); 
      } else {
        assert(bestIdx >=0 && bestIdx < p_fpts.size());
        c_fpts[j].set_id(p_fpts[bestIdx].get_id());
      }
    }
    
    // Add features
    addFeatures(c_fpts);

  }
  //--------------------------------------------
  // Add Features if previous frame didn't exist
  // 1. Add new features (id=-1) to c_fpts
  //    track_manager takes care of IDing
  //--------------------------------------------
  else {
    if (!c_fpts.size()) return;

    // Unique IDing at tracklet_manager
    for (auto& fpt: c_fpts)
      fpt.set_id(-1);
    
    // Add features
    addFeatures(c_fpts);
  }

  profiler_.leave("update");
  return;
}



// void 
// KeyPointTracker::update(opencv_utils::Frame& frame, 
//                         const std::vector<Feature3D>& c_kpts) {

//   // Matching by finding disjoint union sets  ===============================
//   // Cluster keypoints that are within radius ball, and angle diff
//   profiler_.enter("update");

//   //--------------------------------------------
//   // Retrieve previous stable features
//   //--------------------------------------------
//   std::vector<Feature3D> p_tpts;
//   track_manager_.getStableFeatures(p_tpts);

//   //--------------------------------------------
//   // Determine correspondences 
//   // 1. Prop. corr from p_tpts to c_tpts
//   //    c_tpts are appropriately ID'd
//   // 2. Add new features (id=-1) to c_tpts 
//   //    track_manager takes care of IDing
//   //--------------------------------------------
//   // Perform correspondence matching
//   if (p_tpts.size()) {
            
//     // Convert points
//     std::vector<cv::Point2f> p_points(p_tpts.size());
//     std::vector<cv::Point2f> c_points(c_kpts.size());
//     for (int j=0; j<p_tpts.size(); j++)
//       p_points[j] = p_tpts[j].get_point();
//     for (int j=0; j<c_kpts.size(); j++)
//       c_points[j] = c_kpts[j].pt;
    
//     cv::Mat data = cv::Mat(p_points).reshape(1);
//     cv::Mat query = cv::Mat(c_points).reshape(1);
//     // std::cerr << data << std::endl;

//     // Build index
//     cv::flann::Index index(data, cv::flann::KDTreeIndexParams(4));

//     // Query
//     cv::Mat indices(c_points.size(), 10, CV_32SC1);
//     cv::Mat dists(c_points.size(), 10, CV_32FC1);
//     index.knnSearch(query, indices, dists, 10,
//                     cv::flann::SearchParams(64));


//     std::vector<T> c_tpts(c_kpts.size()); 
//     for (int j=0; j<indices.rows; j++) {
//       int* p = indices.ptr<int>(j);
//       float* d = dists.ptr<float>(j);

//       int bestIdx = -1;
//       for (int k=0; k<indices.cols; k++) {
//         int nidx = p[k];
//         if (nidx < 0 || nidx >= p_tpts.size()) continue;
        
//         // Don't add if beyond radius^2
//         if (d[k] < dist_th_ * dist_th_ &&
//             fabs(cos((c_kpts[j].angle - p_tpts[nidx].get_keypoint().angle) * CV_PI / 180.f)) > cos(angle_th_)) {
//           bestIdx = nidx;
//           break;
//         }
//       }
//       if (bestIdx < 0) 
//         c_tpts[j] = Feature3D(now, -1, c_kpts[j]);
//       else {
//         assert(bestIdx >=0 && bestIdx < p_tpts.size());
//         c_tpts[j] = Feature3D(now, p_tpts[bestIdx].get_id(), c_kpts[j]);
//       }
//     }
    
//     // Add features
//     addFeatures(c_tpts);

//   }
//   //--------------------------------------------
//   // Add Features if previous frame didn't exist
//   // 1. Detect features with ones mask
//   // 2. Add new features (id=-1) to c_tpts
//   //    track_manager takes care of IDing
//   //--------------------------------------------
//   else {
//     if (!c_kpts.size()) return;

//     // Unique IDing at tracklet_manager
//     std::vector<Feature3D> c_tpts(c_kpts.size());
//     for (int j=0; j<c_kpts.size(); j++) { 
//       const int32_t id = -1; 
//       c_tpts[j] = T(now, id, c_kpts[j]);
//     }

//     // Add features
//     addFeatures(c_tpts);
//   }

//   profiler_.leave("update");
//   return;
// }


}
