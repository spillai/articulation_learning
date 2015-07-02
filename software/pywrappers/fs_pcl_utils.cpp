// Wrapper for particular pcl functions
// The cython pcl module is incomplete

#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

// fs_types
#include "fs_types.hpp"

// Eigen
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>

// pcl-utils
#include <pcl-utils/pcl_utils.hpp>

#include <pcl/io/pcd_io.h>
#include <pcl/octree/octree.h>
#include <pcl/correspondence.h>
#include <pcl/registration/icp.h>

#include <pcl/registration/correspondence_rejection_surface_normal.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>

#include <pcl/registration/transformation_estimation_svd.h>
#include <pcl/registration/transformation_estimation_point_to_plane.h>

#include <pcl/ModelCoefficients.h>
#include <pcl/segmentation/sac_segmentation.h>

namespace py = boost::python;

namespace fs { namespace python { 

Eigen::Matrix4f icp(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_in,
                    pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_out) {
  pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
  icp.setInputCloud(cloud_in);
  icp.setInputTarget(cloud_out);
  pcl::PointCloud<pcl::PointXYZ>::Ptr Final(new pcl::PointCloud<pcl::PointXYZ>());
  icp.align(*Final);
  Eigen::Matrix4f T = icp.getFinalTransformation();
  return T;
}

std::vector<int>
change_detection(const cv::Mat_<float>& from_cloud, const cv::Mat_<float>& to_cloud,
                 const float resolution, const int noise_filter,
                 const bool return_inverse) {

  // Octree resolution - side length of octree voxels
  std::cerr << "Resolution: " << resolution << std::endl;
  std::cerr << "Noise filter: " << noise_filter << std::endl;
  
  // Convert to pcl
  pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_from_cloud(new pcl::PointCloud<pcl::PointXYZ>());
  pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_to_cloud(new pcl::PointCloud<pcl::PointXYZ>()); 

  pcl_utils::convert_mat_to_pcl<pcl::PointXYZ>(from_cloud, *pcl_from_cloud);
  pcl_utils::convert_mat_to_pcl<pcl::PointXYZ>(to_cloud, *pcl_to_cloud); 
  
  // Instantiate octree-based point cloud change detection class
  pcl::octree::OctreePointCloudChangeDetector<pcl::PointXYZ> octree (resolution);

  // Add points from cloudA to octree
  octree.setInputCloud (pcl_from_cloud);
  octree.addPointsFromInputCloud ();
  
  // Switch octree buffers: This resets octree but keeps previous tree structure in memory.
  octree.switchBuffers ();

  // Add points from cloudB to octree
  octree.setInputCloud (pcl_to_cloud);
  octree.addPointsFromInputCloud ();
  
  // cloud2[inds]: Get vector of point indices from octree voxels
  // which did not exist in previous buffer
  std::vector<int> newPointIdxVector;
  octree.getPointIndicesFromNewVoxels (newPointIdxVector); // , noise_filter);

  // Return pts from cloud2 that is consistent with cloud1
  if (return_inverse) {
    std::vector<int> all_inds(pcl_to_cloud->points.size(), 0);

    // Set original inds
    for (int j=0; j<all_inds.size(); j++)
      all_inds[j] = j;
    
    // Set values at locations for masking
    for (int j=0; j<newPointIdxVector.size(); j++)
      all_inds[newPointIdxVector[j]] = -1;

    for (std::vector<int>::iterator it = all_inds.begin(); it != all_inds.end(); )
      if (*it == -1)
        it = all_inds.erase(it);
      else
        it++;

    // copy inds back 
    newPointIdxVector = all_inds;
  }
  
  // to_cloud[inds]
  return newPointIdxVector;
}


py::tuple
correspondence_rejection_SAC(const cv::Mat_<float>& from_cloud,
                             const cv::Mat_<float>& to_cloud,
                             const cv::Mat_<float>& from_cloud_dense=cv::Mat_<float>(),
                             const cv::Mat_<float>& to_cloud_dense=cv::Mat_<float>(),
                             float inlier_threshold=0.05,
                             int max_iterations=20) {

  pcl::registration::CorrespondenceRejectorSampleConsensus<pcl::PointXYZ> sac;

  // Convert to pcl
  pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_from_cloud(new pcl::PointCloud<pcl::PointXYZ>());
  pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_to_cloud(new pcl::PointCloud<pcl::PointXYZ>()); 

  pcl_utils::convert_mat_to_pcl<pcl::PointXYZ>(from_cloud, *pcl_from_cloud);
  pcl_utils::convert_mat_to_pcl<pcl::PointXYZ>(to_cloud, *pcl_to_cloud); 

  // Convert correspondences
  pcl::CorrespondencesPtr corrs(new pcl::Correspondences());
  for (int j=0; j<from_cloud.rows; j++) {
    corrs->push_back(pcl::Correspondence(j, j, 0.f));
    // cv::norm(from_cloud.row(j) - to_cloud.row(j))));
  }

  // Set input, and target clouds
  sac.setInputCloud(pcl_from_cloud);
  sac.setTargetCloud(pcl_to_cloud);

  // Set params
  sac.setInlierThreshold(inlier_threshold);
  sac.setMaxIterations(max_iterations);

  // Set correspondences, and refine params
  sac.setRefineModel(true);
  sac.setInputCorrespondences(corrs);

  // Get inliers, and refine model
  pcl::CorrespondencesPtr inliers_(new pcl::Correspondences());
  sac.getCorrespondences(*inliers_);

  // Convert Inliers
  std::vector<int> inliers(inliers_->size());
  for (int j=0; j<inliers_->size(); j++)
    inliers[j] = (*inliers_)[j].index_query; 

  // Point-Point: Get least sq. rigid body transformation via SVD
  Eigen::Matrix4f T_ = sac.getBestTransformation();
  std::cerr << "tf: " << T_ << std::endl;

  // Optionally employ dense registration
  if (!from_cloud_dense.empty() && !to_cloud_dense.empty()) {
    std::cerr << "ICP with dense point clouds " << std::endl;
    // Convert to pcl
    pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_from_cloud_dense
        (new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_to_cloud_dense
        (new pcl::PointCloud<pcl::PointXYZ>()); 

    pcl_utils::convert_mat_to_pcl<pcl::PointXYZ>(from_cloud_dense, *pcl_from_cloud_dense);
    pcl_utils::convert_mat_to_pcl<pcl::PointXYZ>(to_cloud_dense, *pcl_to_cloud_dense); 

    // std::cerr << "ICP aligning  " << pcl_from_cloud_dense->points.size() << " "
    //           << pcl_to_cloud_dense->points.size() << std::endl;
    T_ = icp(pcl_from_cloud_dense, pcl_to_cloud_dense);
    std::cerr << "tf icp: " << T_ << std::endl;
  }
  
  // Convert tf
  cv::Mat T;
  cv::eigen2cv(T_, T);

  // Return tf and inliers
  return py::make_tuple(T, inliers);
}

cv::Mat
transformation_estimationSVD(const cv::Mat_<float>& from_cloud,
                             const cv::Mat_<float>& to_cloud,
                             const cv::Mat_<float>& from_cloud_dense=cv::Mat_<float>(),
                             const cv::Mat_<float>& to_cloud_dense=cv::Mat_<float>()) {

  // Convert to pcl
  pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_from_cloud(new pcl::PointCloud<pcl::PointXYZ>());
  pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_to_cloud(new pcl::PointCloud<pcl::PointXYZ>()); 

  pcl_utils::convert_mat_to_pcl<pcl::PointXYZ>(from_cloud, *pcl_from_cloud);
  pcl_utils::convert_mat_to_pcl<pcl::PointXYZ>(to_cloud, *pcl_to_cloud); 

  // Convert correspondences
  pcl::CorrespondencesPtr corrs(new pcl::Correspondences());
  for (int j=0; j<from_cloud.rows; j++) {
    corrs->push_back(pcl::Correspondence(j, j, 0.f));
  }

  // Point-Point: Get least sq. rigid body transformation via SVD
  Eigen::Matrix4f T_;
  pcl::registration::TransformationEstimationSVD<pcl::PointXYZ, pcl::PointXYZ> svd;
  svd.estimateRigidTransformation(*pcl_from_cloud, *pcl_to_cloud, *corrs, T_);
  std::cerr << "tf: " << T_ << std::endl;

  // Optionally employ dense registration
  if (!from_cloud_dense.empty() && !to_cloud_dense.empty()) {
    std::cerr << "ICP with dense point clouds " << std::endl;
    // Convert to pcl
    pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_from_cloud_dense
        (new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_to_cloud_dense
        (new pcl::PointCloud<pcl::PointXYZ>()); 

    pcl_utils::convert_mat_to_pcl<pcl::PointXYZ>(from_cloud_dense, *pcl_from_cloud_dense);
    pcl_utils::convert_mat_to_pcl<pcl::PointXYZ>(to_cloud_dense, *pcl_to_cloud_dense); 

    // std::cerr << "ICP aligning  " << pcl_from_cloud_dense->points.size() << " "
    //           << pcl_to_cloud_dense->points.size() << std::endl;
    T_ = icp(pcl_from_cloud_dense, pcl_to_cloud_dense);
    std::cerr << "tf icp: " << T_ << std::endl;
  }
  
  // Convert tf
  cv::Mat T;
  cv::eigen2cv(T_, T);

  // Return tf and inliers
  return T;
}

py::tuple
plane_estimation_SAC(const cv::Mat_<float>& cloud,
                     float inlier_threshold=0.05,
                     int max_iterations=100, std::string method_str="SAC_RANSAC") {

  pcl::registration::CorrespondenceRejectorSampleConsensus<pcl::PointXYZ> sac;

  // Convert to pcl
  pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZ>());
  pcl_utils::convert_mat_to_pcl<pcl::PointXYZ>(cloud, *pcl_cloud);

  // Setup plane estimation
  pcl::ModelCoefficients::Ptr coefficients_(new pcl::ModelCoefficients);
  pcl::PointIndices::Ptr inliers_(new pcl::PointIndices);
  
  // Create the segmentation object
  pcl::SACSegmentation<pcl::PointXYZ> seg;

  // Optional
  seg.setOptimizeCoefficients (true);
  seg.setMaxIterations(max_iterations);
  
  // Mandatory
  seg.setModelType (pcl::SACMODEL_PLANE);

  int method =
      (method_str == "SAC_LMEDS") ? pcl::SAC_LMEDS :
      (method_str == "SAC_MSAC") ? pcl::SAC_MSAC :
      (method_str == "SAC_RANSAC") ? pcl::SAC_RANSAC :
      (method_str == "SAC_MLESAC") ? pcl::SAC_MLESAC :
      (method_str == "SAC_PROSAC") ? pcl::SAC_PROSAC :
      pcl::SAC_RANSAC;

  seg.setMethodType (method);
  seg.setDistanceThreshold (inlier_threshold);

  seg.setInputCloud (pcl_cloud);
  seg.segment (*inliers_, *coefficients_);

  std::vector<int> inliers = inliers_->indices; 
  std::vector<float> coeffs = coefficients_->values;

  // Return coeffs and inliers
  return py::make_tuple(coeffs, inliers);
}


cv::Mat3f compute_normals_wrapper(const cv::Mat3f& cloud,
                                  float depth_change_factor=0.5, float smoothing_size=10.f) {
  cv::Mat3f normals;
  pcl_utils::compute_normals(cloud, normals, depth_change_factor, smoothing_size);
  return normals;
}


BOOST_PYTHON_MODULE(fs_pcl_utils)
{
  // Py Init and Main types export
  fs::python::init_and_export_converters();

  py::def("compute_normals",
          &fs::python::compute_normals_wrapper, 
          py::args("cloud", "depth_change_factor", "smoothing_size"));

  py::def("CorrespondenceRejectionSAC",
          &fs::python::correspondence_rejection_SAC, 
          (py::arg("source"), py::arg("target"),
           py::arg("source_dense"), py::arg("target_dense"),
           py::arg("inlier_threshold")=0.05, py::arg("max_iterations")=20));

    py::def("TransformationEstimationSVD",
          &fs::python::transformation_estimationSVD, 
          (py::arg("source"), py::arg("target"),
           py::arg("source_dense"), py::arg("target_dense")));
  
    // py::def("CorrespondenceRejectionPointToPlaneSAC",
    //       &fs::python::correspondence_rejection_point_to_plane_SAC, 
    //       (py::arg("source"), py::arg("target"),
    //        py::arg("inlier_threshold")=0.05, py::arg("max_iterations")=20));

  py::def("PlaneEstimationSAC",
          &fs::python::plane_estimation_SAC, 
          (py::arg("cloud"),
           py::arg("inlier_threshold")=0.05, py::arg("max_iterations")=20, py::arg("method")="SAC_RANSAC"));

  py::def("change_detection",
          &fs::python::change_detection, 
          (py::arg("source"), py::arg("target"),
           py::arg("resolution")=0.01, py::arg("noise_filter")=7,
           py::arg("return_inverse")=false));

}


} // namespace python
} // namespace fs
