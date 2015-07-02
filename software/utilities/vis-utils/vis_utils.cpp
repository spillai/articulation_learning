#include "vis_utils.hpp"

namespace vis_utils { 

static VisualizationWrapper vis;

void 
populate_viz_cloud(vs::point3d_list_t& viz_list, 
                   const cv::Mat_<cv::Vec3f>& cloud, const cv::Mat& img, 
                   int64_t object_id) { 

  if (!vis.find_id("KINECT_POSE"))
    publish_camera_frame(bot_timestamp_now());

  viz_list.nnormals = viz_list.normals.size(); 
  viz_list.npointids = viz_list.pointids.size(); 

  viz_list.id = bot_timestamp_now(); 
  viz_list.collection = vis.get_id("KINECT_POSE"); 
  viz_list.element_id = 1; 

  viz_list.npoints = cloud.rows * cloud.cols;
  viz_list.points = std::vector<vs::point3d_t>(cloud.rows * cloud.cols);

  viz_list.ncolors = img.rows * img.cols; 
  viz_list.colors = std::vector<vs::color_t>(img.rows * img.cols);

  const cv::Vec3f* pptr = cloud.ptr<cv::Vec3f>(0);
  const cv::Vec3b* cptr = img.ptr<cv::Vec3b>(0);
  for (int k=0; k<cloud.rows * cloud.cols; k++) { 
    viz_list.points[k].x = (*pptr)[0];
    viz_list.points[k].y = (*pptr)[1];
    viz_list.points[k].z = (*pptr)[2];

    viz_list.colors[k].r = (*cptr)[2] * 1.f / 255.f;
    viz_list.colors[k].g = (*cptr)[1] * 1.f / 255.f;
    viz_list.colors[k].b = (*cptr)[0] * 1.f / 255.f;

    pptr++;
    cptr++;
  }
    
  return;
}


void 
populate_viz_cloud(vs::point3d_list_t& viz_list, 
                   const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud, 
                   int64_t object_id) { 

  if (!vis.find_id("KINECT_POSE"))
    publish_camera_frame(bot_timestamp_now());
  
  viz_list.nnormals = viz_list.normals.size(); 
  viz_list.npointids = viz_list.pointids.size(); 

  viz_list.id = bot_timestamp_now(); 
  viz_list.collection = vis.get_id("KINECT_POSE"); 
  viz_list.element_id = 1; 

  viz_list.npoints = cloud->points.size();
  viz_list.points = std::vector<vs::point3d_t>(cloud->points.size());

  viz_list.ncolors = cloud->points.size(); 
  viz_list.colors = std::vector<vs::color_t>(cloud->points.size());

  for (int k=0; k<cloud->points.size(); k++) { 
    viz_list.points[k].x = cloud->points[k].x;
    viz_list.points[k].y = cloud->points[k].y;
    viz_list.points[k].z = cloud->points[k].z;

    viz_list.colors[k].r = cloud->points[k].r * 1.f / 255;
    viz_list.colors[k].g = cloud->points[k].g * 1.f / 255;
    viz_list.colors[k].b = cloud->points[k].b * 1.f / 255;
  }
  return;
}


    
void publish_cloud(const std::string& channel, 
                   const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud) { 

  //----------------------------------
  // Viz inits
  //----------------------------------
  vs::point3d_list_collection_t viz_cloud_msg;
  viz_cloud_msg.id = vis.get_id(channel); 
  viz_cloud_msg.name = channel; 
  viz_cloud_msg.type = VS_POINT3D_LIST_COLLECTION_T_POINT; 
  viz_cloud_msg.reset = true; 

  viz_cloud_msg.point_lists.resize(1);
  populate_viz_cloud(viz_cloud_msg.point_lists[0], cloud);

  viz_cloud_msg.nlists = viz_cloud_msg.point_lists.size();
  vis.lcm.publish("POINTS_COLLECTION", &viz_cloud_msg);

  // convert(pcl, cloud)

}

void 
publish_cloud(const std::string& channel, 
              const cv::Mat_<cv::Vec3f>& cloud, const cv::Mat& img) { 

  //----------------------------------
  // Viz inits
  //----------------------------------
  vs::point3d_list_collection_t viz_cloud_msg;
  viz_cloud_msg.id = vis.get_id(channel);
  viz_cloud_msg.name = channel; 
  viz_cloud_msg.type = VS_POINT3D_LIST_COLLECTION_T_POINT; 
  viz_cloud_msg.reset = true; 

  viz_cloud_msg.point_lists.resize(1);
  populate_viz_cloud(viz_cloud_msg.point_lists[0], cloud, img);

  viz_cloud_msg.nlists = viz_cloud_msg.point_lists.size();
  vis.lcm.publish("POINTS_COLLECTION", &viz_cloud_msg);

  return;

}

// typically from="KINECT", to="local"
std::vector<double>
botframes_get_pose(const std::string& from, const std::string& to) { 
                    // double utime) {
  BotTrans from_to;
  int ret = bot_frames_get_trans
      (vis.b_frames, from.c_str(), to.c_str(), &from_to);

  // x,y,z,r,p,y
  std::vector<double> vec(6);
  
  if (ret) {
    memcpy (&vec[0], from_to.trans_vec, 3 * sizeof(double));
    bot_quat_to_roll_pitch_yaw(from_to.rot_quat, &vec[3]);
    std::cerr << "Get Trans: " << cv::Mat(vec) << std::endl;
  } else {
    std::cerr << "Failed to retrieve relative pose between "
              << from << " and " << to << std::endl;
  }

  return vec;
}

void publish_image_t(const std::string& channel, const cv::Mat& img) {

  cv::Mat img_;
  cv::cvtColor(img, img_, cv::COLOR_BGR2RGB);
  
  bot_core::image_t msg;
  assert(img_.type() == CV_8UC1 || img_.type() == CV_8UC3);
  std::cerr << "img_: " << img_.size() << " " << img_.channels() << std::endl;
  std::cerr << "img_: row_stride: " << img_.cols * img_.channels() << std::endl;
  msg.utime = bot_timestamp_now();
  msg.width = img_.cols;
  msg.height = img_.rows;
  msg.row_stride = img_.cols * img_.channels();
  msg.pixelformat = (img_.channels() == 3) ?
      bot_core::image_t::PIXEL_FORMAT_RGB : bot_core::image_t::PIXEL_FORMAT_GRAY;

  msg.data.resize(msg.row_stride * img_.rows);
  std::copy(img_.data, img_.data + msg.row_stride * img_.rows, msg.data.begin());
  msg.size = msg.data.size();

  msg.nmetadata = 0;
  
  vis.lcm.publish(channel.c_str(), &msg);

}

int64_t publish_camera_frame(double utime) {
        
  //----------------------------------
  // Publish camera frame of reference for drawing
  //----------------------------------
  vs::obj_collection_t objs_msg; 
  objs_msg.id = vis.get_id("KINECT_POSE"); 
  objs_msg.name = "KINECT_POSE"; 
  objs_msg.type = VS_OBJ_COLLECTION_T_AXIS3D; 
  objs_msg.reset = true; 

  BotTrans sensor_frame;
  bot_frames_get_trans_with_utime (vis.b_frames, "KINECT", "local", utime, &sensor_frame);
  double rpy[3]; bot_quat_to_roll_pitch_yaw(sensor_frame.rot_quat, rpy);

  objs_msg.objs.resize(1); 
  objs_msg.objs[0].id = 1; 
  objs_msg.objs[0].x = sensor_frame.trans_vec[0], 
      objs_msg.objs[0].y = sensor_frame.trans_vec[1], objs_msg.objs[0].z = sensor_frame.trans_vec[2]; 
  objs_msg.objs[0].roll = rpy[0], objs_msg.objs[0].pitch = rpy[1], objs_msg.objs[0].yaw = rpy[2]; 
  objs_msg.nobjs = objs_msg.objs.size(); 

  vis.lcm.publish("OBJ_COLLECTION", &objs_msg);
  return objs_msg.id;
}

// bot_core_pose_t transform_to_sensor_frame(double utime, 
//                                           bot_core_pose_t tag_pose) { 
//     BotTrans sensor_frame;
//     bot_frames_get_trans_with_utime (vis.b_frames, "KINECT", "local", utime, &sensor_frame);

//     cv::Mat_<double> Ts_w(4,4);
//     bot_trans_get_mat_4x4(&sensor_frame, (double*)Ts_w.data);

//     cv::Mat_<double> Tobs_s(4,4); 
//     bot_core_pose_to_mat(tag_pose, (double*)Tobs_s.data);

//     cv::Mat_<double> Tobs_w = Ts_w * Tobs_s; 

//     bot_core_pose_t tag_pose_tf; 
//     mat_to_bot_core_pose((double*)Tobs_w.data, tag_pose_tf);

//     return tag_pose_tf;
// }
    
}
