#include "feature_types.hpp"

namespace fsvision {

//============================================
// Feature class
//============================================
Feature::Feature() : id_(-1), utime_(0) {
  status_ = UNKNOWN;
}

Feature::Feature (int64_t utime, int64_t id) : utime_(utime), id_(id) {
  status_ = UNKNOWN;  
}

int64_t Feature::get_id() const {
  return id_;
}

void Feature::set_id(const int64_t& id) { 
  id_ = id; 
}

int64_t Feature::get_utime() const {
  return utime_;
}

void Feature::set_utime(const int64_t& utime) { 
  utime_ = utime;
}

int Feature::get_status() const {
  return status_;
}

void Feature::set_status(const uint8_t& status) { 
  status_ = status;
}

Feature::~Feature () {
}


//============================================
// Feature2D class
//============================================
Feature2D::Feature2D () : Feature(), point_(cv::KeyPoint(cv::Point2f(0,0), 0)) {
}

Feature2D::Feature2D (int64_t utime, int64_t id) : Feature(utime, id) { 
  desc_ = cv::Mat();
}

Feature2D::Feature2D (int64_t utime, int64_t id,
                      const cv::KeyPoint& point) : Feature(utime, id) { 
  point_ = point;
}

bool Feature2D::point_valid() const {
  return (point_.pt.x != -1 || point_.pt.y != -1);
}
  
cv::Point2f Feature2D::get_point() const {
  return point_.pt;
}

void Feature2D::set_point(const cv::Point2f& point) { 
  point_.pt = point;
}

cv::Mat Feature2D::get_desc() const {
  return desc_;
}

void Feature2D::set_desc(const cv::Mat& desc) { 
  desc_ = desc.clone();
}

cv::KeyPoint Feature2D::get_keypoint() const {
  return point_;
}

void Feature2D::set_keypoint(const cv::KeyPoint& kpoint) {
  point_ = kpoint;
}


Feature2D::~Feature2D () {
  
}

// If tpts has size, then only populate the point_ field, 
// else init, and populate
void
Feature2D::convert(const std::vector<cv::Point2f>& pts, std::vector<Feature2D>& tpts) { 
  // if (tpts.size() != pts.size()) 
  //   tpts = std::vector<Feature2D>(pts.size());
  assert(tpts.size() == pts.size());
  for (int j=0; j<pts.size(); j++) 
    tpts[j].set_point(pts[j]);
}


void 
Feature2D::convert(const std::vector<Feature2D>& tpts, std::vector<cv::Point2f>& pts) { 
  pts = std::vector<cv::Point2f>(tpts.size());
  for (int j=0; j<tpts.size(); j++) 
    pts[j] = tpts[j].get_point();
}

void 
Feature2D::convert(const std::vector<Feature2D>& tpts, std::vector<cv::KeyPoint>& kpts) { 
  kpts = std::vector<cv::KeyPoint>(tpts.size());
  for (int j=0; j<tpts.size(); j++) {
    kpts[j].pt = tpts[j].get_point();
    kpts[j].class_id = j;
    kpts[j].response = 1;
    kpts[j].angle = -1;
    kpts[j].size = 1;
  }
}


//============================================
// Feature3D class
//============================================

Feature3D::Feature3D () : Feature2D() {
  xyz_ = cv::Point3f(NAN, NAN, NAN);
  normal_ = cv::Point3f(NAN, NAN, NAN);
  tangent_ = cv::Point3f(NAN, NAN, NAN);
}
Feature3D::Feature3D (int64_t utime, int64_t id) : Feature2D(utime, id) {
  xyz_ = cv::Point3f(NAN, NAN, NAN);
  normal_ = cv::Point3f(NAN, NAN, NAN);
  tangent_ = cv::Point3f(NAN, NAN, NAN);
}
    
void Feature3D::setFeature3D(const cv::Point3f& xyz,
                             const cv::Point3f& normal, 
                             const cv::Point3f& tangent) {
  xyz_ = xyz;
  normal_ = normal;
  tangent_ = tangent;
}

void Feature3D::setCovariance3D(const float xx, const float xy, const float xz,
                                const float yy, const float yz, const float zz) {
  covs_[0] = xx, covs_[1] = xy, covs_[2] = xz,
      covs_[3] = yy, covs_[4] = yz, covs_[5] = zz;
}
cv::Vec6f Feature3D::covs() const {
  return covs_;
}
  
cv::Point3f Feature3D::xyz() const {
  return xyz_;
}

bool Feature3D::xyz_valid() const {
  return (!(xyz_.x != xyz_.x) && (cv::norm(xyz_) != 0));
}
  
bool Feature3D::normal_valid() const {
  return (!(normal_.x != normal_.x) && (cv::norm(normal_) != 0));
}

cv::Point3f Feature3D::normal() const {
  return normal_;
}

bool Feature3D::tangent_valid() const {
  return (!(tangent_.x != tangent_.x) && (cv::norm(tangent_) != 0));
}

cv::Point3f Feature3D::tangent() const {
  return tangent_;
}

Feature3D::~Feature3D () {}

// // If tpts has size, then only populate the point_ field, 
// // else init, and populate
// static void
// convert(const std::vector<cv::Point3f>& pts, std::vector<Feature3D>& tpts) { 
//   if (tpts.size() != pts.size()) 
//     tpts = std::vector<Feature3D>(pts.size());
//   for (int j=0; j<pts.size(); j++) 
//     tpts[j].set_xyz(pts[j]);
// }


// Propagate pts, handling NaNs
void 
Feature3D::convert(const std::vector<Feature3D>& tpts, std::vector<cv::Point3f>& pts) { 
  pts = std::vector<cv::Point3f>(tpts.size());
  for (int j=0; j<tpts.size(); j++) { 
    cv::Point3f p = tpts[j].xyz();
    if (p.x != p.x)
      pts[j] = cv::Point3f(0.,0.,0.);
    else 
      pts[j] = tpts[j].xyz();
  }
}


// If tpts has size, then only populate the point_ field, 
// else init, and populate
void
Feature3D::convert(const std::vector<cv::Point2f>& pts, std::vector<Feature3D>& tpts) { 
  // if (tpts.size() != pts.size()) 
  // tpts = std::vector<Feature3D>(pts.size());
  assert(tpts.size() == pts.size());
  for (int j=0; j<pts.size(); j++) 
    tpts[j].set_point(pts[j]);
}


void 
Feature3D::convert(const std::vector<Feature3D>& tpts, std::vector<cv::Point2f>& pts) { 
  pts = std::vector<cv::Point2f>(tpts.size());
  for (int j=0; j<tpts.size(); j++) 
    pts[j] = tpts[j].get_point();
}

void 
Feature3D::convert(const std::vector<Feature3D>& tpts, std::vector<cv::KeyPoint>& kpts) { 
  kpts = std::vector<cv::KeyPoint>(tpts.size());
  for (int j=0; j<tpts.size(); j++) {
    kpts[j].pt = tpts[j].get_point();
    kpts[j].class_id = j;
    kpts[j].response = 1;
    kpts[j].angle = -1;
    kpts[j].size = 1;
  }
}


} // namespace fsvision


// class Track { 
//   enum TrackStatus { 
//     GOOD, 
//     WAITING, 
//     BAD, 
//     INVALID, 
//     OUT_OF_BOUNDS, 
//     UNKNOWN
//   };

//   int status_;
//   std::vector<Feature2D> track_;

//   Track2D () {
//     status_ = TrackStatus::UNKNOWN;
//   }
        
//   int size() { 
//     return track_.size();
//   }

//   void add(const Feature2D& feature) { 
//     track_.push_back(feature);
//     // assert(!feature.desc_.empty());
//   }
        
//   void prune_by_size(const int size) { 
//     if (track_.size() < size) return;
//     track_.erase(track_.begin(), track_.end() - size);
//   }

//   void prune_by_timespan_ms(const int timespan_ms) { 
//     int64_t latest_utime = track_.end()[-1].get_utime();
//     int64_t first_utime = track_.begin()[0].get_utime();
//     if (latest_utime - first_utime > timespan_ms * 1e3) { 
//       int idx = track_.size()-1; 
//       for (; idx>=0; idx--) 
//         if (latest_utime - track_[idx].get_utime() > timespan_ms * 1e3)
//           break;
//       track_.erase(track_.begin(), track_.begin() + idx);
//     }
//   }

//   void set_status(const int& status) { 
//     status_ = status;
//   }

// };
// typedef std::map<int64_t, Track2D> Track2DMap; 
// typedef std::map<int64_t, Track2D>::iterator Track2DMapIt; 
    
// struct Feature3D : public Feature { 
//     cv::Point3f point_; 
//     cv::Vec3f normal_;
//     Feature3D () : Feature(), point_(cv::Point3f(0,0,0)), normal_(cv::Vec3f(0,0,1)) { }
//     Feature3D (int64_t utime, int64_t id, 
//                const cv::Point3f& point, const cv::Vec3f& normal) : Feature(utime, id) { 
//         point_ = point; 
//         normal_ = normal;
//     }
//     virtual ~Feature3D () {}
// };


    
    // def compute_depth_mask(self):
    //     assert self.valid

    //     # Depth mask
    //     self.depth_mask = np.bitwise_not(self.depth <= 0)

    // def get_rgb_with_depth_mask(self):
    //     pass
    //     # # Img with NaN mask
    //     # img_with_depth_mask = np.empty_like(img)
    //     # for j in range(3):
    //     # 	img_with_depth_mask[:,:,j] = np.bitwise_and(img[:,:,j], depth_mask);
        
    // def compute_normals(self, smoothing_size=10, depth_change_factor=0.5):
    //     # Integral normal estimation (%timeit ~52ms per loop)
    //     self.normals = pcl_utils.integral_normal_estimation(self.X,
    //                                                         smoothing_size=smoothing_size,
    //                                                         depth_change_factor=depth_change_factor);
    //     self.normals_mask = np.bitwise_not(np.any(np.isnan(self.normals), axis=2))

    // def compute_normals(self, smoothing_size=10, depth_change_factor=0.5):
    //     # Integral normal estimation (%timeit ~52ms per loop)
    //     self.normals = pcl_utils.integral_normal_estimation(self.X,
    //                                                         smoothing_size=smoothing_size,
    //                                                         depth_change_factor=depth_change_factor);
    //     self.normals_mask = np.bitwise_not(np.any(np.isnan(self.normals), axis=2))
        
    // def visualize_normals(self):
    //     # Normalize to range [0,1]
    //     normals_img = 0.5 * (self.normals + np.ones_like(self.normals));
    //     # Plot
