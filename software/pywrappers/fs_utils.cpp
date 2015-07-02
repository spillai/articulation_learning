// Wrapper for most external modules, and trackers

#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

// frame utils
#include <pcl-utils/frame_utils.hpp>
#include <perception_opencv_utils/opencv_utils.hpp>

// features3d
#include <features3d/feature_types.hpp>

// // // mser3d
// #include <mser3d/mser3d.hpp>

// // gpft
// #include <gpft/general_purpose_feature_tracker.hpp>

// pcl-utils includes
#include <pcl-utils/pcl_utils.hpp>

// vis-utils includes
#include <vis-utils/vis_utils.hpp>

// kinect frame
#include <lcmtypes/kinect.hpp>
#include <kinect/kinect-utils.h>

// lcm log player wrapper
#include <lcm-utils/lcm-reader-util.hpp>

// fs_types
#include "fs_types.hpp"

// // birchfield wrapper
// #include <fs_perception_wrappers/birchfield_klt/klt_wrapper.hpp>

// DBoW2 wrapper
#include <fs_perception_wrappers/DBoW2/dbow2_wrapper.hpp>

// // GIST wrapper
// #include <fs_perception_wrappers/lear_gist/gist_wrapper.hpp>

// // Dense trajectories wrapper
// #include <fs_perception_wrappers/lear_dense_trajectories/lear_dense_trajectories.hpp>
// #include <fs_perception_wrappers/fs_dense_trajectories/fs_dense_trajectories.hpp>

// // idiap_mser
// #include <fs_perception_wrappers/idiap_mser/idiap_mser_wrapper.hpp>

// // SLIC 
// #include <fs_perception_wrappers/slic/slic_generic.hpp>
// #include <fs_perception_wrappers/slic/slic_utils.hpp>

// cv_utils
#include <cv_utils/FindCameraMatrices.h>
#include <cv_utils/RichFeatureMatcher.h>

namespace py = boost::python;

namespace fs { namespace vision { 

// Wrapper to the isam::Slam class, with opencv support, and simple batch
// optimization routine 
class RichFeatureMatcherWrapper {
 private:

  std::vector<cv::Mat> imgs;
  std::vector<cv::Mat> masks;
  std::vector<std::vector<cv::KeyPoint> > imgpts;
  cv::Mat F;

  // cv::Ptr<cv::GridAdaptedFeatureDetector> detector;
  cv::Ptr<cv::FeatureDetector> detector;
  cv::Ptr<cv::DescriptorExtractor> extractor;
  cv::Ptr<cv::DescriptorMatcher> matcher;
  std::vector<cv::Mat> descriptors;
  
 public:
  RichFeatureMatcherWrapper() {
    matcher = cv::Ptr<cv::DescriptorMatcher>(cv::DescriptorMatcher::create( "FlannBased" ));
  }
  
  RichFeatureMatcherWrapper(const std::string& det_str, const std::string& desc_str,
                            const cv::Mat& im1, const cv::Mat& im2,
                            const cv::Mat& mask1, const cv::Mat& mask2) {
    imgs.push_back(im1), imgs.push_back(im2);
    masks.push_back(mask1), masks.push_back(mask2);

    detector = cv::FeatureDetector::create(det_str);
    extractor = cv::DescriptorExtractor::create(desc_str);
    matcher = cv::Ptr<cv::DescriptorMatcher>(cv::DescriptorMatcher::create( "FlannBased" ));
    
    std::cout << " -------------------- extract feature points for all images -------------------\n";

    //-- Step 1: Detect the keypoints using FAST Detector
    detector->detect(imgs, imgpts);

    //-- Step 2: Calculate descriptors (feature vectors)
    extractor->compute(imgs, imgpts, descriptors);
  std::cout << " ------------------------------------- done -----------------------------------\n";
  }

  RichFeatureMatcherWrapper(const RichFeatureMatcherWrapper& rfmw) {
  }

  
  
  static void crossCheckMatching( cv::Ptr<cv::DescriptorMatcher>& descriptorMatcher,
                                  const cv::Mat& descriptors1, const cv::Mat& descriptors2,
                                  std::vector<cv::DMatch>& filteredMatches12, int knn=1 )
  {
    filteredMatches12.clear();
    vector<vector<cv::DMatch> > matches12, matches21;
    descriptorMatcher->knnMatch( descriptors1, descriptors2, matches12, knn );
    descriptorMatcher->knnMatch( descriptors2, descriptors1, matches21, knn );
    for( size_t m = 0; m < matches12.size(); m++ )
    {
      bool findCrossCheck = false;
      for( size_t fk = 0; fk < matches12[m].size(); fk++ )
      {
        cv::DMatch forward = matches12[m][fk];

        for( size_t bk = 0; bk < matches21[forward.trainIdx].size(); bk++ )
        {
          cv::DMatch backward = matches21[forward.trainIdx][bk];
          if( backward.trainIdx == forward.queryIdx )
          {
            filteredMatches12.push_back(forward);
            findCrossCheck = true;
            break;
          }
        }
        if( findCrossCheck ) break;
      }
    }
  }

  
  ~RichFeatureMatcherWrapper() {
  }

  cv::Mat_<int> getMatchInds(const cv::Mat& pts1, const cv::Mat& descriptors1,
                             const cv::Mat& pts2, const cv::Mat& descriptors2) {

    // Matching descriptor vectors using FLANN matcher
    std::vector< cv::DMatch > matches;
    crossCheckMatching( matcher, descriptors1, descriptors2, matches, 1 );

    // Align 2D matches
    vector<cv::Point2f> points1(matches.size()), points2(matches.size());
    for( size_t i = 0; i < matches.size(); i++ )
    {
      const int qidx = matches[i].queryIdx;
      assert(qidx >= 0 && qidx < descriptors1.rows);
      points1[i] = cv::Point2f(pts1.row(qidx));      
      
      const int tidx = matches[i].trainIdx;
      assert(tidx >= 0 && tidx < descriptors2.rows);
      points2[i] = cv::Point2f(pts2.row(tidx));
    }
    
    // Geometric verification
    std::vector<uchar> status;
    findFundamentalMat(points1, points2, CV_FM_RANSAC, 1.0, 0.995, status); // CV_FM_RANSAC, 3.0, 0.99

    // Count inliers, remove outliers
    int inliers = 0;
    for (int j=0; j<status.size(); j++)
      if (status[j])
        inliers++;

    // Return mapping between inliers
    cv::Mat_<int> inliers_mapping(inliers, 2);
    for (int j=0, k=0; j<matches.size(); j++) { 
      if (status[j]) {
        inliers_mapping(k,0) = matches[j].queryIdx;
        inliers_mapping(k,1) = matches[j].trainIdx;
        
        k++;
      }
    }

    return inliers_mapping;
  }
  
  std::vector< cv::Mat_<float> > getMatches() {
    const cv::Mat& im1 = imgs[0];
    const cv::Mat& im2 = imgs[1];
    
    const std::vector<cv::KeyPoint>& kp1 = imgpts[0];
    const std::vector<cv::KeyPoint>& kp2 = imgpts[1];

    const cv::Mat& descriptors1 = descriptors[0];
    const cv::Mat& descriptors2 = descriptors[1];
    if(descriptors1.empty()) {
      CV_Error(0,"descriptors_1 is empty");
    }
    if(descriptors2.empty()) {
      CV_Error(0,"descriptors_2 is empty");
    }
    
    // // matching descriptor vectors using Brute Force matcher
    // // allow cross-check. use Hamming distance for binary descriptor (ORB)
    // std::vector< DMatch > matches;
    // BFMatcher matcher(NORM_HAMMING,true); 
    // matcher.match( descriptors1, descriptors2, matches );
    // assert(matches.size() > 0);

    //-- Step 3: Matching descriptor vectors using FLANN matcher
    std::vector< cv::DMatch > matches;
    crossCheckMatching( matcher, descriptors1, descriptors2, matches, 1 );

    vector<int> queryIdxs( matches.size() ), trainIdxs( matches.size() );
    for( size_t i = 0; i < matches.size(); i++ )
    {
      queryIdxs[i] = matches[i].queryIdx;
      trainIdxs[i] = matches[i].trainIdx;
    }

    std::vector<cv::Point2f> points1; cv::KeyPoint::convert(kp1, points1, queryIdxs);
    std::vector<cv::Point2f> points2; cv::KeyPoint::convert(kp2, points2, trainIdxs);

    std::vector<uchar> status;
    cv::findFundamentalMat(points1, points2, CV_FM_RANSAC, 1.0, 0.995, status); // CV_FM_RANSAC, 3.0, 0.99
    
    // Prune keypoints
    std::vector<char> status_(status.size(), 0);
    std::vector<cv::KeyPoint> kp1_good, kp2_good;
    for (int j=0; j<status.size(); j++) {
      if (status[j]) { 
        kp1_good.push_back(cv::KeyPoint(points1[j], 1.f));
        kp2_good.push_back(cv::KeyPoint(points2[j], 1.f));
        status_[j] = 1; 
      }
    }

    
    // FlannBasedMatcher matcher;
    // std::vector< DMatch > matches;
    // matcher.match( descriptors1, descriptors2, matches );

    // //-- Quick calculation of max and min distances between keypoints
    // double max_dist = 0; double min_dist = 100;
    // for( int i = 0; i < descriptors1.rows; i++ )
    // { double dist = matches[i].distance;
    //   if( dist < min_dist ) min_dist = dist;
    //   if( dist > max_dist ) max_dist = dist;
    // }

    // printf("-- Max dist : %f \n", max_dist );
    // printf("-- Min dist : %f \n", min_dist );


    // //-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist )
    // //-- PS.- radiusMatch can also be used here.
    // std::vector< DMatch > matches_good;
    // for( int i = 0; i < descriptors1.rows; i++ )
    // { if( matches[i].distance <= 3 * min_dist )
    //   {

    //     // Check bounds
    //     cv::Point2f pt1 = kp1[matches[i].queryIdx].pt;
    //     cv::Point2f pt2 = kp2[matches[i].trainIdx].pt;

    //     if (pt1.x < 0 || pt1.x >= im1.cols ||
    //         pt2.x < 0 || pt2.x >= im2.cols ||
    //         pt1.y < 0 || pt1.y >= im1.rows ||
    //         pt2.y < 0 || pt2.y >= im2.rows)
    //       continue;

    //     // std::cerr << matches[i].queryIdx << "->" << matches[i].trainIdx << ": "
    //     //           << matches[i].distance << std::endl;
        
    //     matches_good.push_back( matches[i] );
    //   }
    // }

    // {
    //   Mat vis;
    //   drawMatches( im1, kp1, im2, kp2,
    //                matches_good, vis, Scalar::all(-1), Scalar::all(-1),
    //                vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    //   imshow("Corr" , vis );
    // }
    
    // // Localize
    // std::vector<cv::Point2f> kp1_good, kp2_good;
    // for (int j=0; j<matches_good.size(); j++) {
    //   kp1_good.push_back(kp1[matches[j].queryIdx].pt);
    //   kp2_good.push_back(kp1[matches[j].trainIdx].pt);
    // }
    

    // F = cv::findHomography(kp1_good, kp2_good, CV_RANSAC);
    
    
    // std::vector<cv::DMatch> matches_good;
    // // std::vector<cv::KeyPoint> kp1_valid,kp2_valid;
    // for(unsigned int i = 0; i < matches.size(); i++ ) { 
    //   if (matches[i].trainIdx <= 0) {
    //     matches[i].trainIdx = matches[i].imgIdx;
    //   }

    //   // if (matches[i].trainIdx < 0 || matches[i].queryIdx < 0)
    //   //   continue;

    //   matches_good.push_back( matches[i] );
    //   // kp1_valid.push_back(kp1[matches[i].queryIdx]);
    //   // kp2_valid.push_back(kp2[matches[i].trainIdx]);
    // }

    // {
    //   //-- Draw only "good" matches
    //   cv::Mat vis;
    //   drawMatches( im1, kp1_good, im2, kp2_good,
    //                vector<DMatch>(), vis, Scalar::all(-1), Scalar::all(-1),
    //                vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );        

    //   //-- Show detected matches
    //   // stringstream ss; ss << "Feature Matches " << idx_i << "-" << idx_j;
    //   imshow("Corr" , vis );
    //   waitKey(10);
    //   // destroyWindow(ss.str());
    // }

    

   
    {
      //-- Draw only "good" matches
      cv::Mat vis;
      drawMatches( im1, kp1, im2, kp2,
                   matches, vis, cv::Scalar::all(-1), cv::Scalar::all(-1),
                   status_, cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

      //-- Show detected matches
      // stringstream ss; ss << "Feature Matches " << idx_i << "-" << idx_j;
      cv::imshow("Match" , vis );
      cv::waitKey(10);
      // destroyWindow(ss.str());
    }

    // Convert to pts
    assert(kp1_good.size() == kp2_good.size());

    cv::Mat_<float> kpmat1(kp1_good.size(), 2);
    cv::Mat_<float> kpmat2(kp1_good.size(), 2);
    for (int j=0; j<kp1_good.size(); j++) {
      kpmat1(j,0) = kp1_good[j].pt.x;
      kpmat1(j,1) = kp1_good[j].pt.y;

      kpmat2(j,0) = kp2_good[j].pt.x;
      kpmat2(j,1) = kp2_good[j].pt.y;

    }
    std::vector<cv::Mat_<float> > pts_matched;
    pts_matched.push_back(kpmat1);
    pts_matched.push_back(kpmat2);
    
    return pts_matched;
    
  }

  cv::Mat getFundamentalMat() {
    return F;
  }
  
};


} // namespace vision
} // namespace fs


namespace fs { namespace python { 

// Convert kinect_frame_msg hack!
struct kinect_frame_msg_to_pyobject {
  static PyObject* convert(const kinect::frame_msg_t& msg){
    if (msg.timestamp == 0) return py::object().ptr();
    opencv_utils::Frame frame(&msg, msg.tilt_radians);
    return py::incref(py::object(frame).ptr());
  }
};

void publish_cloud(std::string channel, opencv_utils::Frame& frame) {
  vis_utils::publish_cloud(channel, frame.getCloudRef(), frame.getRGBRef());
}


BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(computeNormals_overloads,
                                       opencv_utils::Frame::computeNormals, 0, 4);
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(fastBilateralFilter_overloads,
                                      opencv_utils::Frame::fastBilateralFilter, 0, 2);
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(pruneDepth_overloads,
                                       opencv_utils::Frame::pruneDepth, 1, 2);
BOOST_PYTHON_FUNCTION_OVERLOADS(publish_camera_frame_overloads,
                                vis_utils::publish_camera_frame, 0, 1);
// BOOST_PYTHON_FUNCTION_OVERLOADS(botframes_get_trans_overloads,
//                                 vis_utils::botframes_get_trans_overloads, 2, 3);

// BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(slic_forGivenSuperpixelSize_overloads,
//                                        fs::vision::SLICGeneric::forGivenSuperpixelSize, 3, 5);

BOOST_PYTHON_MODULE(fs_utils)
{
  // Main types export
  fs::python::init_and_export_converters();
    
  py::to_python_converter<kinect::frame_msg_t, fs::python::kinect_frame_msg_to_pyobject>();

  // // kinect::frame_msg_t
  // py::class_<kinect::frame_msg_t>("kinect_msg_t")
  //     .def_readwrite("accel", &kinect::frame_msg_t::accel)

  py::scope scope = py::scope();
  
  // Frame
  py::class_<opencv_utils::Frame>("Frame")
      .def(py::init< double, cv::Mat, cv::Mat, py::optional<float, bool> >
           (py::args("utime", "img", "depth", "scale", "fill_depth")))
      .def_readwrite("utime", &opencv_utils::Frame::utime_)
      .def("getRGB", &opencv_utils::Frame::getRGBRef,
           py::return_value_policy<py::copy_const_reference>())
      // .def("getBGR", &opencv_utils::Frame::getRGBRef,
      //      py::return_value_policy<py::copy_const_reference>())
      .def("getGray", &opencv_utils::Frame::getGrayRef,
           py::return_value_policy<py::copy_const_reference>())
      .def("getDepth", &opencv_utils::Frame::getDepthRef,
           py::return_value_policy<py::copy_const_reference>())
      .def("getCloud", &opencv_utils::Frame::getCloudRef,
           py::return_value_policy<py::copy_const_reference>())
      // .def("getXYZ", &opencv_utils::Frame::getXYZ)           
      .def("getNormals", &opencv_utils::Frame::getNormalsRef,
           py::return_value_policy<py::copy_const_reference>())
      .def("plot", &opencv_utils::Frame::plot)
      //.def("getCloudMask", &opencv_utils::Frame::getCloudMaskRef)
      // .def("getNormalsMask", &opencv_utils::Frame::getNormalsMaskRef)
      // .def("getCombinedMask", &opencv_utils::Frame::getCombinedMaskRef)

      .def("ready", &opencv_utils::Frame::ready)
      .def("computeNormals", &opencv_utils::Frame::computeNormals, computeNormals_overloads())
      .def("fastBilateralFilter", &opencv_utils::Frame::fastBilateralFilter,
           (py::arg("sigmaS")=3.f, py::arg("sigmaR")=0.05))
      .def("medianFilter", &opencv_utils::Frame::medianFilter, (py::arg("win_size")=5))
      .def("pruneDepth", &opencv_utils::Frame::pruneDepth, pruneDepth_overloads())
      ;
  
  py::class_<cv::KeyPoint>("KeyPoint")
      // .def_readwrite("pt", &cv::KeyPoint::pt)
      .def_readwrite("size", &cv::KeyPoint::size)
      .def_readwrite("angle", &cv::KeyPoint::angle)
      .def_readwrite("response", &cv::KeyPoint::response)
      .def_readwrite("octave", &cv::KeyPoint::octave)
      .def_readwrite("class_id", &cv::KeyPoint::class_id)
      ;

  py::class_<fsvision::Feature, boost::noncopyable>("Feature")
      // .def_readwrite("utime", &fsvision::Feature::utime_)
      .add_property("id", &fsvision::Feature::get_id, &fsvision::Feature::set_id)
      .add_property("utime", &fsvision::Feature::get_utime, &fsvision::Feature::set_utime)
      ;

  py::class_<fsvision::Feature2D, py::bases<fsvision::Feature> >("Feature2D")
      .add_property("keypoint", &fsvision::Feature2D::get_keypoint, &fsvision::Feature2D::set_keypoint)
      .add_property("point", &fsvision::Feature2D::get_point, &fsvision::Feature2D::set_point)
      .add_property("desc", &fsvision::Feature2D::get_desc, &fsvision::Feature2D::set_desc)
      ;

  py::class_<fsvision::Feature3D, py::bases<fsvision::Feature2D> >("Feature3D")
      .def("xyz", &fsvision::Feature3D::xyz)
      // .def("covs", &fsvision::Feature3D::covs)
      .def("normal", &fsvision::Feature3D::normal)
      .def("tangent", &fsvision::Feature3D::tangent)

      .def("xyz_valid", &fsvision::Feature3D::xyz_valid)
      .def("normal_valid", &fsvision::Feature3D::normal_valid)
      .def("tangent_valid", &fsvision::Feature3D::tangent_valid)

      // .def("construct_pose", &fsvision::Feature3D::construct_pose)
      // .def("setFeature3D", &fsvision::Feature3D::setFeature3D)
      // .def("setCovariance3D", &fsvision::Feature3D::setCovariance3D)
      ;
  
  // LCMLogReader wrapper
  py::class_<LCMLogReader>("LCMLogReader")
      .def(py::init< std::string, py::optional<double, bool> >
           (py::args("filename", "scale", "sequential_read")))
      .def("reset", &LCMLogReader::reset)
      .def("getNextFrame", &LCMLogReader::getNextKinectFrame)
      .def("getFrame", &LCMLogReader::getKinectFrame)
      .def("getFrameWithTimestamp", &LCMLogReader::getKinectFrameWithTimestamp)
      .def("getNumFrames", &LCMLogReader::getNumFrames)
      .def_readonly("frame_num", &LCMLogReader::frame_num)
      ;
  
  // DBoW2 result wrapper
  py::class_<DBoW2::Result>("DBow2Result")
      .def_readonly("Id", &DBoW2::Result::Id)
      .def_readonly("Score", &DBoW2::Result::Score)
      ;
  expose_template_type<std::vector<DBoW2::Result> >();
  
  // DBoW2 wrapper
  py::class_<fs::vision::DBoW2>("DBoW2")
      // .def(py::init< py::optional<int, int, int, int> >(
      //     py::args("num_feats", "affine_consistency_check", "window_width", "mindist")))
      .def("reset", &fs::vision::DBoW2::reset)
      .def("addFrame", &fs::vision::DBoW2::addFrame)
      .def("addImage", &fs::vision::DBoW2::addImage)
      .def("addDescription", &fs::vision::DBoW2::addDescription)
      .def("queryFrame", &fs::vision::DBoW2::queryFrame)
      .def("queryImage", &fs::vision::DBoW2::queryImage)
      .def("queryDescription", &fs::vision::DBoW2::queryDescription)
      .def("load", &fs::vision::DBoW2::load)
      .def("save", &fs::vision::DBoW2::save)
      .def("build", &fs::vision::DBoW2::build,
           (py::arg("k")=9, py::arg("L")=3, py::arg("use_direct_index")=true))
      .def("buildVOC", &fs::vision::DBoW2::buildVOC,
           (py::arg("k")=9, py::arg("L")=3))
      .def("buildDB", &fs::vision::DBoW2::buildDB,
           (py::arg("use_direct_index")=true))
      // .def("getStableFeatures", &fs::vision::DBoW2::getStableFeatures)
      ;

    // // GPFT wrapper
  // py::class_<fsvision::GeneralPurposeFeatureTracker>("GPFT")
  //     .def(py::init< py::optional<bool, bool, int, int, int, int, int, int, int> >(
  //         py::args("use_gftt", "enable_subpixel_refinement",
  //                  "num_feats", "min_add_radius", "feat_block_size",
  //                  "feature_match_threshold", "feature_distance_threshold",
  //                  "allowed_skips", "allowed_predictions")))
  //     .def("processFrame", &fsvision::GeneralPurposeFeatureTracker::processFrame)
  //     .def("getStableFeatures", &fsvision::GeneralPurposeFeatureTracker::getStableFeatures)
  //     ;

  // // Birchfield KLT wrapper
  // py::class_<fs::vision::BirchfieldKLT>("BirchfieldKLT")
  //     .def(py::init< py::optional<int, int, int, int> >(
  //         py::args("num_feats", "affine_consistency_check", "window_width", "mindist")))
  //     .def("processFrame", &fs::vision::BirchfieldKLT::processFrame)
  //     .def("getStableFeatures", &fs::vision::BirchfieldKLT::getStableFeatures)
  //     ;

  // // LEAR Dense Trajectories wrapper
  // py::class_<fs::vision::LearDenseTrajectories>("LearDenseTrajectories")
  //     .def("processFrame", &fs::vision::LearDenseTrajectories::processFrame)
  //     .def("getStableFeatures", &fs::vision::LearDenseTrajectories::getStableFeatures)
  //     ;

  // // Dense Trajectories wrapper
  // py::class_<fs::vision::DenseTrajectories>("DenseTrajectories")
  //     .def("processFrame", &fs::vision::DenseTrajectories::processFrame,
  //          (py::arg("frame"), py::arg("mask")=cv::Mat()))
  //     .def("getStableFeatures", &fs::vision::DenseTrajectories::getStableFeatures)
  //     ;

  // // SLIC
  // // scoped to add enums
  // {
  //   py::scope scope = 
  //       py::class_<fs::vision::SLICGeneric>("SLIC")
  //       // .def(py::init<>())
  //       .def("forGivenSuperpixelSize", &fs::vision::SLICGeneric::forGivenSuperpixelSize, 
  //            boost::python::args("img", "superpixelsize", "compactness", "color_conv", "mask"))
  //       .def("forGivenNumberOfSuperpixels", &fs::vision::SLICGeneric::forGivenNumberOfSuperpixels,
  //            boost::python::args("img", "k", "compactness", "color_conv", "mask"))
  //       ;

  //   // SLIC enums as attrs, need to cast to avoid conversion of enum type
  //   scope.attr("SLIC_RGB") =
  //       static_cast<int>(fs::vision::SLICGeneric::SLIC_RGB);
  //   scope.attr("SLIC_HSV") =
  //       static_cast<int>(fs::vision::SLICGeneric::SLIC_HSV);
  //   scope.attr("SLIC_LAB") =
  //       static_cast<int>(fs::vision::SLICGeneric::SLIC_LAB);
    
  // }

    // // Seperate module for slic_utils
    // py::def("build_node_graph",
    //         &fs::vision::build_node_graph);
    // py::def("unique_label_img",
    //         &fs::vision::unique_label_img);
    // py::def("extract_contour_mask_from_labels",
    //         &fs::vision::extract_contour_mask_from_labels);
    // py::def("draw_contours_from_labels",
    //         &fs::vision::draw_contours_from_labels);
    // py::def("my_stddev_from_labels",
    //         &fs::vision::mu_stddev_from_labels);
    // py::def("apply_label_values_to_img",
    //         &fs::vision::apply_label_values_to_img);
    // py::def("mean_from_labels",
    //         &fs::vision::mean_from_labels);
    // py::def("median_from_labels",
    //         &fs::vision::median_from_labels);
    // py::def("attrs_from_labels",
    //         &fs::vision::attrs_from_labels,
    //         py::args("img", "labels", "mask"));
  
  // General-purpose utilities
  py::def("publish_camera_frame", &vis_utils::publish_camera_frame,
          fs::python::publish_camera_frame_overloads());
  
  py::def("botframes_get_trans", &vis_utils::botframes_get_pose); 
  // botframes_get_trans_overloads()); 

  py::def("publish_cloud", &fs::python::publish_cloud);
  py::def("publish_image_t", &vis_utils::publish_image_t);

  // py::def("estimate_plane", &fs::python::estimate_plane);


  expose_template_type< cv::KeyPoint >();
  expose_template_type< std::vector<cv::KeyPoint> >();
  expose_template_type< std::vector< std::vector<cv::KeyPoint> > >();

  py::class_<fs::vision::RichFeatureMatcherWrapper>("RichFeatureMatcher")
      .def(py::init<std::string, std::string, cv::Mat, cv::Mat, cv::Mat, cv::Mat>())
      .def("getFundamentalMat", &fs::vision::RichFeatureMatcherWrapper::getFundamentalMat)
      .def("getMatches", &fs::vision::RichFeatureMatcherWrapper::getMatches)
      .def("getMatchInds", &fs::vision::RichFeatureMatcherWrapper::getMatchInds)
      ;

  
  // py::def("weighted_joint_bilateral_filter", &fs::vision::weighted_joint_bilateral_filter);
  
  // // opencv_utils::MSER wrapper
  // py::class_<opencv_utils::MSER>("CVMSER")
  //     .def(py::init< py::optional<int, int, int, double , double, int, double, double, int> >
  //          py::args("delta", "min_area", "max_area", "max_variation", "min_diversity", 
  //                   "max_evolution", "area_threshold", "min_margin", "edge_blur_size"))
  //     .def("update", &fsvision::MSER3D::update)
  //     ;
  
  // // MSER3D wrapper
  // py::class_<fsvision::MSER3D>("MSER3D")
  //     .def(py::init< py::optional<int, int, int, int, double , double, int, double, double, int> >
  //          (py::args("pyramid_levels", "delta", "min_area", "max_area", "max_variation", "min_diversity", 
  //                    "max_evolution", "area_threshold", "min_margin", "edge_blur_size")))
  //     .def("update", &fsvision::MSER3D::update)
  //     ;

  // // IDIAP MSER wrapper
  // py::class_<fs::vision::IDIAPMSERWrapper>("IDIAPMSER")
  //     .def(py::init< py::optional<int, double, double, double, double , bool> >
  //          ( py::args("delta", "min_area", "max_area", "max_variation", "min_diversity", "eight") ))
  //     .def("processFrame", &fs::vision::IDIAPMSERWrapper::processFrame)
  //     ;

  // // LEAR GIST wrapper
  // py::class_<fs::vision::GIST>("LearGIST")
  //     // .def(py::init< py::optional<int, int, int, int> >(
  //     //     py::args("num_feats", "affine_consistency_check", "window_width", "mindist")))
  //     .def("processFrame", &fs::vision::GIST::processFrame)
  //     ;
  

  
  
}

} // namespace fs
} // namespace python



