// 
// Author: Sudeep Pillai (spillai@csail.mit.edu) 
// Updates: Aug 07, 2013
// 

// pcl_utils include
#include "pcl_utils.hpp"

// opencv-utils includes
#include <perception_opencv_utils/opencv_utils.hpp>

namespace pcl_utils { 

// To PCL
template <typename PointT>
void convert_mat_to_pcl(const cv::Mat_<float>& cloud, pcl::PointCloud<PointT>& pcl) { 
  
  pcl.width = cloud.rows;
  pcl.height = 1;
  pcl.points.resize(cloud.rows);
  
  // Row-wise data
  for (int j=0; j<pcl.points.size(); j++)
    pcl.points[j] = PointT(cloud.at<float>(j,0), cloud.at<float>(j,1), cloud.at<float>(j,2));
  
}
template void convert_mat_to_pcl<pcl::Normal>(const cv::Mat_<float>& cloud, pcl::PointCloud<pcl::Normal>& pcl);
template void convert_mat_to_pcl<pcl::PointXYZ>(const cv::Mat_<float>& cloud, pcl::PointCloud<pcl::PointXYZ>& pcl);
template void convert_mat_to_pcl<pcl::PointXYZRGB>(const cv::Mat_<float>& cloud, pcl::PointCloud<pcl::PointXYZRGB>& pcl);

void convert_mat_to_pclnormal(const cv::Mat_<float>& cloud, pcl::PointCloud<pcl::PointNormal>& pcl) {
      
  // Check if cloud is [N x 6]
  assert(cloud.cols == 6);

  // Copy points
  pcl.width = cloud.rows;
  pcl.height = 1;
  pcl.points.resize(cloud.rows);
  
  // Row-wise data
  for (int j=0; j<pcl.points.size(); j++)
    pcl.points[j].x = cloud.at<float>(j,0), pcl.points[j].y = cloud.at<float>(j,1), pcl.points[j].z = cloud.at<float>(j,2);
 
  // Copy normals
  for (int j=0; j<pcl.points.size(); j++) {
    pcl.points[j].normal[0] = cloud.at<float>(j,3);
    pcl.points[j].normal[1] = cloud.at<float>(j,4);
    pcl.points[j].normal[2] = cloud.at<float>(j,5);
  }
  
}

void convert_img_with_cloudmat_to_pcl(const cv::Mat& img, 
                                      const cv::Mat_<cv::Vec3f>& cloud, 
                                      pcl::PointCloud<pcl::PointXYZRGB>::Ptr& pcl) { 

  if (!pcl)
    pcl = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl->width = img.cols;
  pcl->height = img.rows;
  pcl->points.resize(pcl->width * pcl->height);

  for (int j=0; j<pcl->points.size(); j++) { 
    const cv::Vec3b& pbgr = img.at<cv::Vec3b>(j);
    const cv::Vec3f& pxyz = cloud.at<cv::Vec3f>(j);
    if (pxyz[0] == 0 && pxyz[1] == 0 && pxyz[2] == 0) { 
      pcl->points[j].x = std::numeric_limits<float>::quiet_NaN(), 
          pcl->points[j].y = std::numeric_limits<float>::quiet_NaN(), 
          pcl->points[j].z = std::numeric_limits<float>::quiet_NaN();
    } else {
      pcl->points[j].x = pxyz[0], 
          pcl->points[j].y = pxyz[1], pcl->points[j].z = pxyz[2];
    }
    pcl->points[j].r = pbgr[2], 
        pcl->points[j].g = pbgr[1], pcl->points[j].b = pbgr[0];
  }
}

template <typename PointT>
void convert_cloudmat_to_pcl(const cv::Mat3f& cloud, pcl::PointCloud<PointT>& pcl) { 

  assert(cloud.channels() == 3);
  
  pcl.width = cloud.cols;
  pcl.height = cloud.rows;
  pcl.points.resize(pcl.width * pcl.height);

  for (int j=0; j<pcl.points.size(); j++) {
    const cv::Vec3f& pxyz = cloud.at<cv::Vec3f>(j);
    if (pxyz[0] == 0 && pxyz[1] == 0 && pxyz[2] == 0) { // float/double
      pcl.points[j].x = std::numeric_limits<float>::quiet_NaN(), 
          pcl.points[j].y = std::numeric_limits<float>::quiet_NaN(), 
          pcl.points[j].z = std::numeric_limits<float>::quiet_NaN();
    } else {
      pcl.points[j].x = pxyz[0], 
          pcl.points[j].y = pxyz[1], pcl.points[j].z = pxyz[2];
    }
  }
}
template void convert_cloudmat_to_pcl<pcl::PointXYZ>(const cv::Mat3f& cloud, pcl::PointCloud<pcl::PointXYZ>& pcl);
template void convert_cloudmat_to_pcl<pcl::PointXYZRGB>(const cv::Mat3f& cloud, pcl::PointCloud<pcl::PointXYZRGB>& pcl);




void convert_depthmat_to_rangeimageplanar(const cv::Mat_<uint16_t>& depth, 
                                          pcl::RangeImagePlanar& range) { 

  cv::Mat depthf;
  if (depth.type() != CV_32F)
    depth.convertTo(depthf, CV_32F, 1.f / 1000);
  else 
    depthf = depth;
        
  // Convert 0 depth to NaN
  for (int i=0; i<depthf.rows; i++) 
    for (int j=0; j<depthf.cols; j++) 
      if (depthf.at<float>(i,j) == 0) 
        depthf.at<float>(i,j) = std::numeric_limits<float>::quiet_NaN();

  // Set range image 
  range.setDepthImage((float*) depthf.data, depthf.cols, depthf.rows, 
                      320, 240, 576.09757860, 576.09757860);

  // assert(range.points.size() == depthf.rows * depthf.cols);
  std::cerr << " range points: " << range.points.size() << std::endl;
        
  // Speedup
  // Suggested by http://www.pcl-users.org/Narf-Keypoints-in-Kinect-data-speed-td4019395.html
  range.setUnseenToMaxRange();
}


// To cv::Mat
template <typename PointT>
void convert_pcl_to_cloudmat(const pcl::PointCloud<PointT>& pcl, cv::Mat3f& cloud, const cv::Vec3f& NaNvalue ) { 
  cloud = cv::Mat_<cv::Vec3f>(pcl.height, pcl.width, NaNvalue);
  for (int j=0; j<pcl.points.size(); j++) { 
    cv::Vec3f& pxyz = cloud.at<cv::Vec3f>(j);
    if ((pcl.points[j].x != pcl.points[j].x) &&
        (pcl.points[j].y != pcl.points[j].y) && 
        (pcl.points[j].z != pcl.points[j].z)) { 
      continue;
    } else {
      pxyz[0] = pcl.points[j].x, 
          pxyz[1] = pcl.points[j].y, pxyz[2] = pcl.points[j].z;
    }
  }
}
template void convert_pcl_to_cloudmat<pcl::PointXYZ>(const pcl::PointCloud<pcl::PointXYZ>& pcl, cv::Mat3f& cloud,
                                     const cv::Vec3f& NaNvalue );
template void convert_pcl_to_cloudmat<pcl::PointXYZRGB>(const pcl::PointCloud<pcl::PointXYZRGB>& pcl, cv::Mat3f& cloud,
                                        const cv::Vec3f& NaNvalue );


void convert_pcl_to_normalsmat(const pcl::PointCloud<pcl::Normal>& pcl, 
                     cv::Mat3f& normals, const cv::Vec3f& NaNvalue ) { 

  normals = cv::Mat3f(pcl.height, pcl.width, NaNvalue);
  for (int j=0; j<pcl.points.size(); j++) { 
    cv::Vec3f& pxyz = normals.at<cv::Vec3f>(j);
    if ((pcl.points[j].normal[0] != pcl.points[j].normal[0]) &&
        (pcl.points[j].normal[1] != pcl.points[j].normal[1]) && 
        (pcl.points[j].normal[2] != pcl.points[j].normal[2])) { 
    } else {
      pxyz[0] = pcl.points[j].normal[0], 
          pxyz[1] = pcl.points[j].normal[1], pxyz[2] = pcl.points[j].normal[2];
    }
  }
}

void project_points(const pcl::PointCloud<pcl::InterestPoint>::Ptr& cloud, 
                    std::vector<cv::Point2f>& points) { 
  points = std::vector<cv::Point2f>(cloud->points.size());

  float constant = 1.f / 576.09757860; // 1.0f / kcal->intrinsics_rgb.fx ;
  for(int j=0; j<cloud->points.size(); j++) {
    float d = cloud->points[j].z; 
    float u = cloud->points[j].x / (constant * d) + 319.50;
    float v = cloud->points[j].y / (constant * d) + 239.50;
    points[j].x = u, points[j].y = v;;            
  }
}

void unpack_kinect_frame_with_cloud(const kinect::frame_msg_t* msg, 
                                    cv::Mat& img, 
                                    pcl::PointCloud<pcl::PointXYZRGB>::Ptr& pcl, 
                                    float scale) { 
  cv::Mat_<cv::Vec3f> cloud;
  opencv_utils::unpack_kinect_frame_with_cloud(msg, img, cloud, scale);

  convert_img_with_cloudmat_to_pcl(img, cloud, pcl);
}


void unpack_kinect_frame_with_range(const kinect::frame_msg_t* msg, 
                                    cv::Mat& img, 
                                    pcl::RangeImagePlanar& range, 
                                    float scale) { 
  cv::Mat_<uint16_t> depth;
  opencv_utils::unpack_kinect_frame_with_depth(msg, img, depth, scale);

  convert_depthmat_to_rangeimageplanar(depth, range);        
}


void compute_normals(const cv::Mat3f& cloud, cv::Mat3f& normals,
                     float depth_change_factor, float smoothing_size) {

  // Convert to pcl
  pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZ>()); 
  pcl_utils::convert_cloudmat_to_pcl<pcl::PointXYZ>(cloud, *pcl_cloud);
  
  // Compute normals
  pcl::PointCloud<pcl::Normal>::Ptr pcl_normals(new pcl::PointCloud<pcl::Normal>()); 
  pcl_utils::compute_normals<pcl::PointXYZ>(*pcl_cloud, *pcl_normals, depth_change_factor, smoothing_size);

  // Convert to mat
  pcl_utils::convert_pcl_to_normalsmat(*pcl_normals, normals); // filled with NaN

  return;
}

template <typename PointT>
void compute_normals(const pcl::PointCloud<PointT>& cloud, pcl::PointCloud<pcl::Normal>& normals, 
                     float depth_change_factor, float smoothing_size) { 
 
  pcl::IntegralImageNormalEstimation<PointT, pcl::Normal> ne;
  // typename pcl::PointCloud<PointT>::ConstPtr cloud_ptr(&cloud);
  
  // options: COVARIANCE_MATRIX, AVERAGE_3D_GRADIENT, AVERAGE_DEPTH_CHANGE, SIMPLE_3D_GRADIENT
  ne.setNormalEstimationMethod (ne.AVERAGE_3D_GRADIENT); // AVERAGE_3D_GRADIENT);
  ne.setMaxDepthChangeFactor(depth_change_factor); // 0.5f
  if (smoothing_size != 0.f) { 
    ne.setNormalSmoothingSize(smoothing_size);    // 10.f
    ne.setDepthDependentSmoothing(true);
  } else {
    ne.setDepthDependentSmoothing(false);
    ne.setNormalSmoothingSize(1.f);    // 10.f
  }
  // ne.setRectSize(21,21);

  // options: BORDER_POLICY_MIRROR, BORDER_POLICY_IGNORE
  ne.setBorderPolicy(ne.BORDER_POLICY_IGNORE); // BORDER_POLICY_MIRROR);

  ne.setInputCloud(cloud.makeShared());
  ne.compute(normals);

 
  return;
}
template void compute_normals<pcl::PointXYZ>(const pcl::PointCloud<pcl::PointXYZ>& cloud,
                                                pcl::PointCloud<pcl::Normal>& normals, 
                                                float depth_change_factor, float smoothing_size);
template void compute_normals<pcl::PointXYZRGB> (const pcl::PointCloud<pcl::PointXYZRGB>& cloud,
                                                 pcl::PointCloud<pcl::Normal>& normals, 
                                                 float depth_change_factor, float smoothing_size);

template <typename PointT>
void fast_bilateral_filter(pcl::PointCloud<PointT>& cloud, float sigmaS, float sigmaR) {

  pcl::FastBilateralFilterOMP<PointT> fbf(4);
  fbf.setSigmaS(sigmaS); // 15.f (pcl)
  fbf.setSigmaR(sigmaR); // 0.05 defaults
  fbf.setInputCloud(cloud.makeShared());
  fbf.applyFilter(cloud);
  return;
}

template void fast_bilateral_filter<pcl::PointXYZ>(pcl::PointCloud<pcl::PointXYZ>& cloud,
                                    float sigmaS, float sigmaR);
template void fast_bilateral_filter<pcl::PointXYZRGB>(pcl::PointCloud<pcl::PointXYZRGB>& cloud,
                                                      float sigmaS, float sigmaR);


template <typename PointT>
void median_filter(pcl::PointCloud<PointT>& cloud, int win_size) {

  pcl::MedianFilter<PointT> mf;
  mf.setWindowSize(win_size); // 5 (pcl)
  mf.setInputCloud(cloud.makeShared());
  mf.applyFilter(cloud);
  return;
}

template void median_filter<pcl::PointXYZ>(pcl::PointCloud<pcl::PointXYZ>& cloud,
                                    int win_size);
template void median_filter<pcl::PointXYZRGB>(pcl::PointCloud<pcl::PointXYZRGB>& cloud,
                                                      int win_size);


template <typename PointT>
void compute_difference_of_normals(const pcl::PointCloud<PointT>& cloud,
                                   pcl::PointCloud<PointT>& out,
                                   const int& small_scale, const int& large_scale) {

  // int scale1 = 3, scale2 = 7;
  typename pcl::PointCloud<PointT>::ConstPtr cloud_ptr(&cloud);
  std::cerr << "cloud_ptr: " << cloud_ptr << std::endl;
  
  typename pcl::search::OrganizedNeighbor<PointT>::Ptr
      tree (new pcl::search::OrganizedNeighbor<PointT>());
  tree->setInputCloud(cloud_ptr);
  
  // Compute normals using both small and large scales at each point
  pcl::NormalEstimationOMP<PointT, pcl::PointNormal> ne;
  ne.setInputCloud (cloud_ptr);
  ne.setSearchMethod (tree);
  ne.setViewPoint (std::numeric_limits<float>::max (), 
                   std::numeric_limits<float>::max (), 
                   std::numeric_limits<float>::max ());  

  // calculate normals with the small scale
  cout << "Calculating normals for scale..." << small_scale << endl;
  pcl::PointCloud<pcl::PointNormal>::Ptr
      normals_small_scale (new pcl::PointCloud<pcl::PointNormal>);

  ne.setRadiusSearch (small_scale);
  ne.compute (*normals_small_scale);
    
  // calculate normals with the large scale
  cout << "Calculating normals for scale..." << large_scale << endl;
  pcl::PointCloud<pcl::PointNormal>::Ptr
      normals_large_scale (new pcl::PointCloud<pcl::PointNormal>);

  ne.setRadiusSearch (large_scale);
  ne.compute (*normals_large_scale);

  // Create output cloud for DoN results
  pcl::PointCloud<pcl::PointNormal>::Ptr
      doncloud (new pcl::PointCloud<pcl::PointNormal>);
  pcl::copyPointCloud<PointT, pcl::PointNormal>(*cloud_ptr, *doncloud);

  std::cout << "Calculating DoN... " << std::endl;
  // Create DoN operator
  pcl::DifferenceOfNormalsEstimation<PointT, pcl::PointNormal, pcl::PointNormal> don;
  don.setInputCloud (cloud_ptr);
  don.setNormalScaleLarge (normals_large_scale);
  don.setNormalScaleSmall (normals_small_scale);

  if (!don.initCompute ()) {
    std::cerr << "Error: Could not intialize DoN feature operator" << std::endl;
    std::cerr << "Exiting! " << std::endl;
    // exit (EXIT_FAILURE);
  }

  // Compute DoN
  don.computeFeature (*doncloud);
}

template void compute_difference_of_normals<pcl::PointXYZ>
(const pcl::PointCloud<pcl::PointXYZ>& cloud, pcl::PointCloud<pcl::PointXYZ>& out, 
 const int& small_scale, const int& large_scale);

template void compute_difference_of_normals<pcl::PointXYZRGB>
(const pcl::PointCloud<pcl::PointXYZRGB>& cloud, pcl::PointCloud<pcl::PointXYZRGB>& out,
 const int& small_scale, const int& large_scale);

}
