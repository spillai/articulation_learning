#ifndef LCM_READER_UTIL_HPP_
#define LCM_READER_UTIL_HPP_

// lcm
#include <lcm/lcm-cpp.hpp>
#include <lcmtypes/bot2_param.h>

// libbot cpp
#include <lcmtypes/bot_core.hpp>

// libbot includes
#include <bot_core/bot_core.h>
#include <bot_frames/bot_frames.h>
#include <bot_param/param_client.h>
#include <bot_param/param_util.h>
#include <bot_lcmgl_client/lcmgl.h>
#include <ConciseArgs>

// kinect frame
#include <lcmtypes/kinect.hpp>
#include <kinect/kinect-utils.h>

// opencv includes
#include <opencv2/opencv.hpp>

// boost includes
#include <boost/thread/thread.hpp>
#include <boost/thread/mutex.hpp>
#include <queue>

typedef void (*kinect_frame_msg_t_handler_cpp) (const lcm::ReceiveBuffer* rbuf, 
                                const std::string& chan,
                                const kinect::frame_msg_t *msg);

struct LCMLogReaderOptions { 
  std::string fn;    // Filename
  std::string ch;    // Specific channel subscribe
  float fps, scale;
  int start_frame, end_frame;

  lcm::LCM* lcm;
  kinect_frame_msg_t_handler_cpp handler;
    
  void* user_data;    // user data

  LCMLogReaderOptions () { 
    fn = "";
    ch = "";

    // default starts and ends as expected
    start_frame = 0, end_frame = -1; 
    scale = 1.f;
    // User data
    user_data = NULL;
  }    
}; 

typedef std::pair<float, kinect::frame_msg_t> kinect_frame;
class LCMLogReader { 

 private: 
  void internal_init();
  
 public: 
  lcm::LCM lcm;
  LCMLogReaderOptions options;

  lcm::LogFile* log; 
  const lcm::LogEvent* event;
  int frame_num;
  int64_t first_event;
  
  double usleep_interval;

  std::vector<int64_t> event_utimes; // event utimes
  std::map<int64_t, int64_t> utime_map; // sensor to log utime map

  // float scale;
  cv::Mat img, depth_img, cloud; 

  LCMLogReader();
  LCMLogReader(const std::string& _fn, float scale = 1.f, bool sequential_read=true);
  ~LCMLogReader();
  int64_t find_closest_utime(int64_t utime);
  void init_index();
  void init (const LCMLogReaderOptions& _options, bool sequential_read = true);
  bool good() { return (log->good() && options.ch != ""); }
  void reset();

  int getNumFrames();
  bool getNextFrame();
  kinect::frame_msg_t getNextKinectFrame();
  kinect::frame_msg_t getKinectFrame(double index); // avoid double (python boost issue)
  kinect::frame_msg_t getKinectFrameWithTimestamp(double sensor_utime); // avoid double (python boost issue)
};

#endif // LCM_READER_UTIL_HPP


// // pcl includes
// #include <pcl/io/pcd_io.h>
// #include <pcl/PointIndices.h>
// #include <pcl/point_types.h>
// #include <pcl/filters/extract_indices.h>
// #include <pcl/filters/random_sample.h>
// #include <pcl/kdtree/kdtree.h>
// #include <pcl/kdtree/kdtree_flann.h>
// #include <pcl/segmentation/extract_clusters.h>
// #include <pcl/features/normal_3d.h>
// #include <pcl/features/integral_image_normal.h>
// #include <pcl/surface/mls.h>

// pcl includes for kinect
// #include <pointcloud_tools/pointcloud_lcm.hpp>
// #include <pointcloud_tools/pointcloud_vis.hpp>

// // // default handler receiving images
// static void _on_kinect_image_frame (const lcm_recv_buf_t *rbuf, const char *channel,
//                                        const kinect_frame_msg_t *msg, 
//                                        void *user_data ) {
//     double tic; 

//     // Check msg type
//     // state_t *state = (state_t*) user_data;
//     if (msg->depth.depth_data_format != KINECT_DEPTH_MSG_T_DEPTH_MM) { 
//         std::cerr << "Warning: kinect data format is not KINECT_DEPTH_MSG_T_DEPTH_MM" << std::endl 
//                   << "Program may not function properly" << std::endl;
//     }
//     assert (msg->depth.depth_data_format == KINECT_DEPTH_MSG_T_DEPTH_MM);

//     //----------------------------------
//     // Unpack Point cloud
//     //----------------------------------
//     double t1 = bot_timestamp_now();
//     cv::Mat3b img(msg->image.height, msg->image.width);
//     cv::Mat_<float> depth_img = cv::Mat_<float>(img.size());

//     // opencv_utils::unpack_kinect_frame(msg, img);

//     // state->pc_lcm->unpack_kinect_frame(msg, img.data, depth_img.data, cloud);
//     // cv::resize(img, rgb_img, cv::Size(int(img.cols / SCALE), int(img.rows / SCALE)));
//     // printf("%s===> UNPACK TIME: %4.2f ms%s\n", esc_yellow, (bot_timestamp_now() - t1) * 1e-3, esc_def); 
//     // std::cerr << "cloud: " << cloud->points.size() << std::endl;
//     // std::cerr << "rgb: " << rgb_img.size() << std::endl;

//     //--------------------------------------------
//     // Convert Color
//     //--------------------------------------------
//     // cvtColor(rgb_img, rgb_img, CV_RGB2BGR);

//     return;
// }

