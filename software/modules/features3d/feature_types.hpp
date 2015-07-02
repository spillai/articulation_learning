#ifndef FSVISION_FEATURE_TYPES_HPP_
#define FSVISION_FEATURE_TYPES_HPP_

// opencv includes
#include <opencv2/opencv.hpp>

// opencv-utils includes
#include <perception_opencv_utils/opencv_utils.hpp>

namespace fsvision { 

// static const int FEAT_BLOCK_SIZE = 7;
// static const int NUM_FEATS = 1500;
// static const int MIN_ADD_RADIUS = 10;

static cv::Point3f NaN_pt(NAN, NAN, NAN);

//============================================
// Base Feature class
//============================================
class Feature { 
  int64_t id_; 
  int64_t utime_;
  uint8_t status_;
 public: 
  Feature ();
  Feature (int64_t utime, int64_t id); 
  enum status { UNKNOWN=0, DETECTED, PREDICTED, MATCHED };
  int64_t get_id() const ; 
  void set_id(const int64_t& id) ; 
  int64_t get_utime() const ;
  void set_utime(const int64_t& utime) ;
  int get_status() const ;
  void set_status(const uint8_t& status) ;

  virtual ~Feature ();
};

//============================================
// Feature2D class
//============================================
class Feature2D : public Feature { 
 public:
  cv::KeyPoint point_;
  cv::Mat desc_;

  Feature2D (); 
  Feature2D (int64_t utime, int64_t id); 
  Feature2D (int64_t utime, int64_t id, const cv::KeyPoint& point); 
  virtual ~Feature2D () ; 

  bool point_valid() const;
  cv::Point2f get_point() const; 
  void set_point(const cv::Point2f& point) ;

  cv::Mat get_desc() const; 
  void set_desc(const cv::Mat& desc) ;

  cv::KeyPoint get_keypoint() const ; 
  void set_keypoint(const cv::KeyPoint& kpoint); 
  
  // If tpts has size, then only populate the point_ field, 
  // else init, and populate
  static void
  convert(const std::vector<cv::Point2f>& pts, std::vector<Feature2D>& tpts); 
  static void 
  convert(const std::vector<Feature2D>& tpts, std::vector<cv::Point2f>& pts); 
  static void 
  convert(const std::vector<Feature2D>& tpts, std::vector<cv::KeyPoint>& kpts); 
};

//============================================
// Feature3D class
//============================================
class Feature3D : public Feature2D { 
  cv::Point3f xyz_;
  cv::Point3f normal_;
  cv::Point3f tangent_;
  cv::Vec6f covs_;
  
 public: 
  Feature3D (); 
  Feature3D (int64_t utime, int64_t id);

  // for boost python vector_indexing_suite
  bool operator==(const Feature3D& rhs) { return false; }
  bool operator!=(const Feature3D& rhs) { return true; }
  
  void setFeature3D(const cv::Point3f& xyz,
                    const cv::Point3f& normal = NaN_pt,
                    const cv::Point3f& tangent = NaN_pt); 
  void setCovariance3D(const float xx, const float xy, const float xz,
                       const float yy, const float yz, const float zz);

  cv::Vec6f covs() const; 
  
  cv::Point3f xyz() const; 
  bool xyz_valid() const; 
  
  bool normal_valid() const; 
  cv::Point3f normal() const; 

  bool tangent_valid() const; 
  cv::Point3f tangent() const; 

  virtual ~Feature3D (); 

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
  static void 
  convert(const std::vector<Feature3D>& tpts, std::vector<cv::Point3f>& pts); 

  static void
  convert(const std::vector<cv::Point2f>& pts, std::vector<Feature3D>& tpts); 
  static void 
  convert(const std::vector<Feature3D>& tpts, std::vector<cv::Point2f>& pts); 
  static void 
  convert(const std::vector<Feature3D>& tpts, std::vector<cv::KeyPoint>& kpts); 

  
};



}
#endif


    
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
