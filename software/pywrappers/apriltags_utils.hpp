// std includes
#include <iostream>

// lcm
#include <lcm/lcm-cpp.hpp>
#include <lcmtypes/bot2_param.h>

// opencv include
#include <opencv2/opencv.hpp>

// libbot/lcm includes
#include <bot_core/bot_core.h>
#include <lcmtypes/bot_core.hpp>
#include <bot_frames/bot_frames.h>
// #include <bot_param/param_client.h>
// #include <bot_param/param_util.h>
#include <bot_lcmgl_client/lcmgl.h>

// April tags detector and various families that can be selected by command line option
#include <AprilTags/TagDetector.h>
#include <AprilTags/Tag16h5.h>
#include <AprilTags/Tag25h7.h>
#include <AprilTags/Tag25h9.h>
#include <AprilTags/Tag36h9.h>
#include <AprilTags/Tag36h11.h>

#include <utility>
#include <boost/tuple/tuple.hpp>

// frame utils
#include <features3d/feature_types.hpp>
#include <pcl-utils/frame_utils.hpp>

// opencv-utils includes
#include <perception_opencv_utils/opencv_utils.hpp>

// Profiler
#include <fs-utils/profiler.hpp>


namespace py = boost::python;

namespace fsvision {


inline cv::Vec3f mean_val(const cv::Mat3f& normals, const cv::Point2f& p) {
  // Ensure keypoint is valid
  assert(p.x >= 0 && p.x < normals.cols &&
         p.y >= 0 && p.y < normals.rows);

  const int sz = 5; 
  int pxmin = std::max(0, int(p.x - sz/2));
  int pymin = std::max(0, int(p.y - sz/2));
  int pxmax = std::min(normals.cols-1, int(p.x + sz/2)); 
  int pymax = std::min(normals.rows-1, int(p.y + sz/2)); 

  int count = 0; 
  cv::Vec3f normal(0.f,0.f,0.f);
  for (int y=pymin; y<=pymax; y++) {
    for (int x=pxmin; x<=pxmax; x++) {
      const cv::Point3f& n = normals(y,x);
      if (n.x != n.x) continue;
      normal[0] += n.x, normal[1] += n.y, normal[2] += n.z; 
      count++;
    }
  }
  if (count == 0)
    return cv::Vec3f(std::numeric_limits<float>::quiet_NaN(),
                     std::numeric_limits<float>::quiet_NaN(), 
                     std::numeric_limits<float>::quiet_NaN());
  normal *= 1.f / count; 
  return normal;
  
}

// std::vector<AprilTag>
// apriltags_extractTags(PyObject* img) { 

//   profiler.enter("ExtractTags()");
//   // Do the conversion
//   NDArrayConverter cvt;
//   cv::Mat gray = cvt.toMat(img);
//   if (gray.type() == CV_8UC3)
//     cv::cvtColor(gray, gray, cv::COLOR_RGB2GRAY);
  
//   // Detect April tags (requires a gray scale image)
//   std::vector<AprilTags::TagDetection> detections = m_tagDetector.extractTags(gray);
//   std::vector<AprilTag> ret;

//   for (int j=0; j<detections.size(); j++) {
//     fsvision::AprilTag tag;
//     tag.id = detections[j].id;

//     cv::Mat_<float> pts(4,2);
//     for (int k=0; k<4; k++) pts(k,0) = detections[j].p[k].first, pts(k,1) = detections[j].p[k].second;
//     tag.p = cvt.toNDArray(pts.clone());
//     ret.push_back(tag);
//   }
//   profiler.leave("ExtractTags()");
//   return ret;
// }


// // Image gets modified
// PyObject* 
// apriltags_extractMask(opencv_utils::Frame& frame, float scale) {
//   // detect April tags (requires a gray scale image)
//   std::vector<AprilTags::TagDetection> detections = m_tagDetector.extractTags(frame.getGrayRef());

//   std::vector<std::vector<cv::Point> > mask_pts;
//   for (int j=0; j<detections.size(); j++) {
//     AprilTags::TagDetection& detection = detections[j];

//     std::vector<cv::Point> pts(4);
//     for (int k=0; k<4; k++)
//       pts[k] = cv::Point(detection.p[k].first, detection.p[k].second);

//     // Scale up points before masking
//     pts = opencv_utils::scale_contour(pts, scale);

//     // Add to mask
//     mask_pts.push_back(pts);
//   }

//   cv::Mat1b mask(frame.getRGBRef().size()); mask = 255;
//   cv::fillPoly(mask, mask_pts, cv::Scalar(0), CV_AA, 0);

//   NDArrayConverter cvt;
//   PyObject* ret = cvt.toNDArray(mask);
//   return ret;
// }


}
