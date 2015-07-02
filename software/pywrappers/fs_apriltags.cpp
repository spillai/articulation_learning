#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

// opencv include
#include <opencv2/opencv.hpp>

// frame utils
#include <pcl-utils/frame_utils.hpp>
#include <pcl-utils/pose_utils.hpp>

// Features3d
#include <features3d/feature_types.hpp>

// April tags detector and various families that can be selected by command line option
#include <AprilTags/TagDetector.h>
#include <AprilTags/Tag16h5.h>
#include <AprilTags/Tag25h7.h>
#include <AprilTags/Tag25h9.h>
#include <AprilTags/Tag36h9.h>
#include <AprilTags/Tag36h11.h>

// opencv-utils includes
#include <perception_opencv_utils/opencv_utils.hpp>

// Profiler
#include <fs-utils/profiler.hpp>

#include "fs_types.hpp"

namespace py = boost::python;

namespace fs { namespace python {

#ifndef PI
  const double PI = 3.14159265358979323846;
#endif
  const double TWOPI = 2.0*PI;

static inline bool in_bounds(const cv::Mat& cloud, const cv::Point2f& p) {
  return (p.x >= 0 && p.x < cloud.cols &&
          p.y >= 0 && p.y < cloud.rows);
}

class AprilTag {
 public: 
  AprilTag(): id(-1) {
    pose = std::vector<double>(7,0);
    pose[3] = 1; // pose[3,4,5,6] = 1, 0, 0, 0
  }

  // // for boost python vector_indexing_suite
  // bool operator==(const AprilTag& rhs) { return false; }
  // bool operator!=(const AprilTag& rhs) { return true; }

  std::vector<double> getPose() const {
    return pose;
  }

  std::vector<fsvision::Feature3D> getFeatures() const {
    return features;
  }
  
  int id;
  std::vector<double> pose;
  std::vector<fsvision::Feature3D> features;
};

class AprilTag2D {
 public: 
  AprilTag2D(): id(-1) {}

  std::vector<cv::Point2f> getFeatures() const {
    return features;
  }
  int id;
  std::vector<cv::Point2f> features;
};

class AprilTagsWrapper {
 public:
  AprilTagsWrapper() :
      m_tagDetector(NULL),
      m_tagCodes(AprilTags::tagCodes36h11),
      m_width(640),
      m_height(480),
      m_tagSize(0.079),
      // m_tagSize(0.1715),
      m_fx(528.49404721),
      // m_fx(576.09757860),
      // m_fy(576.09757860), 
      m_fy(528.49404721),
      m_px(m_width/2-0.5),
      m_py(m_height/2-0.5) 

  {
    profiler.setName("AprilTags_utils");
    m_tagDetector = new AprilTags::TagDetector(m_tagCodes);
  }
  
  ~AprilTagsWrapper() {
    delete m_tagDetector;
  }

  // purely processes the frame
  // getTags retrieves the tags, and its features
  void processFrame(opencv_utils::Frame& frame_) {
    frame = frame_;
    const cv::Mat& gray = frame.getGrayRef();
    detections = m_tagDetector->extractTags(gray);

    std::cerr << "DETECTIONS: " << detections.size() << std::endl;

    // Compute normals if not computed
    if (frame.getNormalsRef().empty()) { 
      frame.computeNormals(1.f);
      cv::Mat3f normals = frame.getNormalsRef().clone();
      cv::imshow("normals", normals * 0.5 + 0.5);
      cv::waitKey(1);
    }
  }

  std::vector< AprilTag2D > processImage(const cv::Mat& gray) {
    detections = m_tagDetector->extractTags(gray);    
    std::cerr << "DETECTIONS: " << detections.size() << std::endl;

    std::vector<AprilTag2D> tags(detections.size());
    for (int j=0; j<detections.size(); j++) {
      // Get Features
      tags[j].id = detections[j].id;
      tags[j].features = std::vector<cv::Point2f>(4);
      for (int k=0; k<4; k++) {
        tags[j].features[k] = cv::Point2f(detections[j].p[k].first, detections[j].p[k].second);
      }
    }
    return tags;

    
    // std::vector< cv::Mat_<float> > all_pts(detections.size());    
    // for (int j=0; j<detections.size(); j++) {
    //   cv::Mat_<float> pts(4, 2);
    //   for (int k=0; k<4; k++)
    //     pts(k,0) = detections[j].p[k].first, pts(k,1) = detections[j].p[k].second;
    //   all_pts[j] = pts;      
    // }
    // return all_pts;
  }

  /**
   * Normalize angle to be within the interval [-pi,pi].
   */
  inline double standardRad(double t) {
    if (t >= 0.) {
      t = fmod(t+PI, TWOPI) - PI;
    } else {
      t = fmod(t-PI, -TWOPI) + PI;
    }
    return t;
  }

  void wRo_to_euler(const Eigen::Matrix3d& wRo, double& yaw, double& pitch, double& roll) {
    yaw = standardRad(atan2(wRo(1,0), wRo(0,0)));
    double c = cos(yaw);
    double s = sin(yaw);
    pitch = standardRad(atan2(-wRo(2,0), wRo(0,0)*c + wRo(1,0)*s));
    roll  = standardRad(atan2(wRo(0,2)*s - wRo(1,2)*c, -wRo(0,1)*s + wRo(1,1)*c));
  }

  // getTags, call after processFrame
 std::vector<AprilTag> getTags() {

    const cv::Mat_<uint16_t>& depth = frame.getDepthRef();
    const cv::Mat3f& cloud = frame.getCloudRef();
    const cv::Mat3f& normals = frame.getNormalsRef();
    std::vector<AprilTag> tags(detections.size());
    for (int j=0; j<detections.size(); j++) {
     
      // Get Features
      std::vector<fsvision::Feature3D> features(4);
      for (int k=0; k<4; k++) {
        
        // Propagate timestamp, and ID
        cv::Point2f p(detections[j].p[k].first, detections[j].p[k].second);
        cv::Point2f p2(detections[j].p[(k+1)%4].first, detections[j].p[(k+1)%4].second);
        cv::Point2f p3(detections[j].p[(k+2)%4].first, detections[j].p[(k+2)%4].second);

        // Initialize feature3d, and propagate keypoint data
        fsvision::Feature3D& fpt = features[k];
        fpt = fsvision::Feature3D(frame.getTimestamp(), detections[j].id * 4 + k);
        fpt.set_point(p);

        // 3D extraction
        if (!depth.empty() && in_bounds(depth, p)) { 
          cv::Point3f xyz = opencv_utils::get_xyz(depth, p);

          if (!normals.empty() && !cloud.empty()) { 
            cv::Point3f nvec = normals(p);
            cv::Point3f tvec = cloud(p2)-cloud(p);
            cv::Point3f t2vec = cloud(p3)-cloud(p);
            tvec *= 1.f / cv::norm(tvec);
            t2vec *= 1.f / cv::norm(t2vec);
            fpt.setFeature3D(xyz, tvec.cross(t2vec), tvec);
            // fpt.setFeature3D(xyz, nvec, tvec);
          } else { 
            fpt.setFeature3D(xyz);
          }
        }
      }

      // Get pose estimates either from RGB and camera params, or from Depth if available
      std::vector<double> pose(7);
      if (normals.empty()) { 
        // Get Pose estimates
        Eigen::Vector3d translation;
        Eigen::Matrix3d rotation;
        detections[j].getRelativeTranslationRotation(m_tagSize, m_fx, m_fy, m_px, m_py,
                                                     translation, rotation);

        // Weird set of transformations (TODO: validate)
        Eigen::Matrix3d F; F << 1, 0, 0, 0, -1, 0, 0, 0, 1;
        Eigen::Matrix3d fixed_rot = F*rotation;
        double yaw, pitch, roll;
        wRo_to_euler(fixed_rot, yaw, pitch, roll);

        double rpy[3], quat[4];
        rpy[0] = roll, rpy[1] = pitch, rpy[2] = yaw;
        bot_roll_pitch_yaw_to_quat(rpy, quat);
     
        // Eigen::Quaterniond quat(rotation);
        pose[0] = translation(0), pose[1] = translation(1), pose[2] = translation(2);
        // pose[3] = quat.w(), pose[4] = quat.x(), pose[5] = quat.y(), pose[6] = quat.z();
        pose[3] = quat[0], pose[4] = quat[1], pose[5] = quat[2], pose[6] = quat[3];

        // std::cerr << "quat" << quat[0] << " " << quat[1] << " " << quat[2] << " " << quat[3] << std::endl;
      
        cout << "  distance=" << translation.norm() << "m, x=" << translation(0)
             << ", y=" << translation(1) << ", z=" << translation(2)
             << ", yaw=" << yaw << ", pitch=" << pitch
             << ", roll=" << roll << endl;
      } else {
        const fsvision::Feature3D& f = features[0];
        // Compute pose from observation
        pose_utils::pose_t pt_pose(0, f.xyz(), f.normal(), f.tangent());
        pose[0] = pt_pose.pos[0], pose[1] = pt_pose.pos[1], pose[2] = pt_pose.pos[2];
        pose[3] = pt_pose.orientation[0], pose[4] = pt_pose.orientation[1],
            pose[5] = pt_pose.orientation[2], pose[6] = pt_pose.orientation[3];
        // std::cerr << pt_pose << std::endl;
      }
      
      
      // Write to tags
      tags[j].id = detections[j].id;
      tags[j].pose = pose;
      tags[j].features = features;
    }
    return tags;
  }

 //  // getTags, call after processFrame
 // std::vector<AprilTag2D> getTags2D() {
 //   detections = m_tagDetector->extractTags(gray);    
 //    std::vector<AprilTag2D> tags(detections.size());
 //    for (int j=0; j<detections.size(); j++) {
 //      // Get Features
 //      tags[j].id = detections[j].id;
 //      tags[j].features = std::vector<cv::Point2f>(4);
 //      for (int k=0; k<4; k++) {
 //        tags[j].features[k] = cv::Point2f(detections[j].p[k].first, detections[j].p[k].second);
 //      }
 //    }
 //    return tags;
 //  }

  
  cv::Mat1b getMask(float scale) {
    std::vector<std::vector<cv::Point> > mask_pts;
    for (int j=0; j<detections.size(); j++) {
      AprilTags::TagDetection& detection = detections[j];

      std::vector<cv::Point> pts(4);
      for (int k=0; k<4; k++)
        pts[k] = cv::Point(detection.p[k].first, detection.p[k].second);

      // Scale up points before masking
      pts = opencv_utils::scale_contour(pts, scale);

      // Add to mask
      mask_pts.push_back(pts);
    }

    mask = cv::Mat1b(frame.getRGBRef().size()); mask = 255;
    cv::fillPoly(mask, mask_pts, cv::Scalar(0), CV_AA, 0);
    return mask;
  }

  
  // std::vector<fsvision::Feature3D> extractFeatures() {
  //     profiler.enter("ExtractFeatures()");
  //     int64_t now = frame.getTimestamp();
  //     std::cerr << "now: " << now << " " << detections.size() << std::endl;

  //     const cv::Mat_<uint16_t>& depth = frame.getDepthRef();
  //     std::vector<fsvision::Feature3D> ret;
  //     for (int j=0; j<detections.size(); j++) {

  //       std::vector<fsvision::Feature3D> tags;
  //       for (int k=0; k<4; k++) {
          
  //         // Propagate timestamp, and ID
  //         cv::Point2f p(detections[j].p[k].first, detections[j].p[k].second);
  //         cv::Point2f p2(detections[j].p[(k+1)%4].first, detections[j].p[(k+1)%4].second);
  //         cv::Point2f p3(detections[j].p[(k+2)%4].first, detections[j].p[(k+2)%4].second);

  //         // Initialize feature3d, and propagate keypoint data
  //         fsvision::Feature3D fpt(now, detections[j].id * 4 + k);
  //         fpt.set_point(p);

  //         // // 3D extraction
  //         // if (!depth.empty()) { 
  //         //   // Handle NaN downstream
  //         //   cv::Point3f xyz(mean_val(cloud, p));
  //         //   if (xyz.x != xyz.x) continue;

  //         //   // cv::Point3f t1(cloud(p2)-cloud(p)), t2(cloud(p3)-cloud(p));
  //         //   cv::Point3f t1(mean_val(cloud, p2)-mean_val(cloud, p)),
  //         //       t2(mean_val(cloud, p3)-mean_val(cloud, p));
  //         //   float t1norm = cv::norm(t1), t2norm = cv::norm(t2);
  //         //   // std::cerr << "t1norm: " << t2 << " " << t1 << std::endl;
  //         //   if (t1norm == 0 || t2norm == 0 || t1.x != t1.x || t2.x != t2.x) continue;
  //         //   t1 *= 1.f / t1norm, t2 *= 1.f / t2norm;

  //         //   // cv::Point3f normal(normals(p)); // = mean_val(normals, p);
  //         //   cv::Point3f normal = t1.cross(t2);
  //         //   float nnorm = cv::norm(normal);
  //         //   if (nnorm == 0 || normal.x != normal.x) continue;
  //         //   normal *= 1.f / nnorm;
  //         //   // std::cerr << "normal: " << normal << std::endl;
  //         //   if (normal.x != normal.x) continue;
  //         //   fpt.setFeature3D(xyz, normal, t1);

  //         //   if (detections[j].id == 0 && k == 0)
  //         //     std::cerr << detections[j].id * 4 + k
  //         //               << " Normal: " << normal << " tangent: " << t1 << std::endl;
  //         }

  //         // Push Feature3D (xyz/normal/tangent could be NaN)
  //         tags.push_back(fpt);
  //       }
  //       // if (tags.size() == 4)
  //       ret.insert(ret.end(), tags.begin(), tags.end());
  //     }
  //     std::cerr << "features: " << ret.size() << std::endl;
  //     profiler.leave("ExtractFeatures()");

  // }
  
  std::vector<AprilTags::TagDetection> detections;
  fsvision::Profiler profiler; 
  AprilTags::TagCodes m_tagCodes; 
  AprilTags::TagDetector* m_tagDetector;

  opencv_utils::Frame frame;
  cv::Mat1b mask;
  
  int m_width; // image size in pixels
  int m_height;

  double m_tagSize; // April tag side length in meters of square black frame
  double m_fx; // camera focal length in pixels
  double m_fy;
  double m_px; // camera principal point
  double m_py;

};

BOOST_PYTHON_MODULE(fs_apriltags)
{
  // Py Init and Main types export
  init_and_export_converters();

  expose_template_type<std::vector<fsvision::Feature3D> >();
  
  // Class def
  py::class_<fs::python::AprilTag>("AprilTag")
      .def_readonly("id", &fs::python::AprilTag::id)
      .def("getPose", &fs::python::AprilTag::getPose)
      .def("getFeatures", &fs::python::AprilTag::getFeatures)
      ;
  expose_template_type<std::vector<fs::python::AprilTag> >();

  // Class def
  py::class_<fs::python::AprilTag2D>("AprilTag2D")
      .def_readonly("id", &fs::python::AprilTag2D::id)
      .def("getFeatures", &fs::python::AprilTag2D::getFeatures)
      ;
  expose_template_type<std::vector<cv::Point2f> >();
  expose_template_type<std::vector<fs::python::AprilTag2D> >();

  
  // Class def
  py::class_<fs::python::AprilTagsWrapper>("AprilTagDetection")
      .def("processFrame", &fs::python::AprilTagsWrapper::processFrame)
      .def("processImage", &fs::python::AprilTagsWrapper::processImage)
      .def("getTags", &fs::python::AprilTagsWrapper::getTags)
      // .def("getTags2D", &fs::python::AprilTagsWrapper::getTags2D)
      .def("getMask", &fs::python::AprilTagsWrapper::getMask, (py::arg("scale")=1.0))
      // .def("getMotionEstimate", &fsvision::FovisWrapper::getMotionEstimate)
      ;
 
}

} // namespace python
} // namespace fs
