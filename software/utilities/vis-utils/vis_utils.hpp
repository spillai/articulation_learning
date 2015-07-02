#ifndef VIS_UTILS_HPP__
#define VIS_UTILS_HPP__

// Visualization utils

// lcm
#include <lcm/lcm-cpp.hpp>
#include <lcmtypes/bot2_param.h>

// libbot/lcm includes
#include <bot_core/bot_core.h>
#include <lcmtypes/bot_core.hpp>
#include <bot_frames/bot_frames.h>
#include <bot_param/param_client.h>
#include <bot_param/param_util.h>
#include <bot_lcmgl_client/lcmgl.h>

// opencv includes
#include <opencv2/opencv.hpp>

// visualization msgs
#include <lcmtypes/visualization.h>
#include <lcmtypes/visualization.hpp>

// pcl-utils
#include <pcl-utils/pcl_utils.hpp>

// pcl includes
#include <pcl/io/pcd_io.h>

namespace vis_utils { 
void 
populate_viz_cloud(vs::point3d_list_t& viz_list, 
                   const cv::Mat_<cv::Vec3f>& cloud, const cv::Mat& img, 
                   int64_t object_id = bot_timestamp_now());

void 
populate_viz_cloud(vs::point3d_list_t& viz_list, 
                   const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud, 
                   int64_t object_id = bot_timestamp_now());


void publish_cloud(const std::string&, const cv::Mat_<cv::Vec3f>& cloud, const cv::Mat& img);
void publish_cloud(const std::string& , const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& );

void publish_image_t(const std::string& channel, const cv::Mat& img);

int64_t publish_camera_frame(double utime=bot_timestamp_now());
std::vector<double>
botframes_get_pose(const std::string& from, const std::string& to); // ,
                    // double utime=bot_timestamp_now());
struct VisualizationWrapper { 
  lcm::LCM lcm;
        
  BotParam   *b_server;
  BotFrames *b_frames;

  std::map<std::string, int64_t> id_map;
        
  VisualizationWrapper() { 
    // Bot Param/frames init
    b_server = bot_param_new_from_server(lcm.getUnderlyingLCM(), 1);
    b_frames = bot_frames_get_global (lcm.getUnderlyingLCM(), b_server);
  }
  ~VisualizationWrapper() {}

  bool find_id(const std::string& channel) {
    return (id_map.find(channel) != id_map.end());
  }
  
  int64_t get_id(const std::string& channel) {
    int64_t idx = -1;
    if (id_map.find(channel) == id_map.end()) {
      idx = id_map.size() + 123456;
      id_map[channel] = idx;
    } else 
      idx = id_map[channel];
    assert(idx >= 0);
    return idx;
  }
      
};
}

#endif // VIS_UTILS_HPP
