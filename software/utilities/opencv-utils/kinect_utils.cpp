#include "kinect_utils.hpp"

namespace opencv_utils { 

void unpack_rgb(const kinect::frame_msg_t* msg, cv::Mat& img) { 

  /// 1.1 RGB:
  img = cv::Mat(msg->image.height, msg->image.width, CV_8UC3);
  if(msg->image.image_data_format == kinect::image_msg_t::VIDEO_RGB) {
    assert(msg->image.image_data.size() == msg->image.height * msg->image.width * 3);
    std::copy(msg->image.image_data.begin(), 
              msg->image.image_data.end(), img.data);
  } else if(msg->image.image_data_format == kinect::image_msg_t::VIDEO_RGB_JPEG) {
    jpeg_decompress_8u_rgb (&msg->image.image_data[0], msg->image.image_data_nbytes,
                            img.data, msg->image.width, msg->image.height, msg->image.width* 3);
  }

  // RGB2BGR (opencv fmt)
  cvtColor(img, img, cv::COLOR_RGB2BGR);
  return;
}

void unpack_depth(const kinect::frame_msg_t* msg, cv::Mat_<uint16_t>& img) { 

  if (msg->depth.depth_data_format != kinect::depth_msg_t::DEPTH_MM) { 
    std::cerr << "Warning: kinect data format is not KINECT_DEPTH_MSG_T_DEPTH_MM" << std::endl 
              << "Program may not function properly" << std::endl;
  }
  assert (msg->depth.depth_data_format == kinect::depth_msg_t::DEPTH_MM);

  /////////////////////////////////////////////////////////////////////
  /// 1.2. DEPTH:
  img = cv::Mat_<uint16_t>(msg->depth.height, msg->depth.width);

  // 1.2.1 De-compress if necessary:
  if(msg->depth.compression != kinect::depth_msg_t::COMPRESSION_NONE) {
    // //std:: cout << "compression \n ";
    // if(msg->depth.uncompressed_size > uncompress_buffer_size) {
    //     uncompress_buffer_size = msg->depth.uncompressed_size;
    //     uncompress_buffer = (uint8_t*) realloc(uncompress_buffer, uncompress_buffer_size);
    // }
    unsigned long dlen = msg->depth.uncompressed_size;
    int status = uncompress( (uint8_t*) img.data, &dlen, 
                             &(msg->depth.depth_data[0]), msg->depth.depth_data_nbytes);
    if(status != Z_OK) {
      return;
    }
  }else{
    int npixels = msg->depth.width * msg->depth.height;
    memcpy(img.data, &(msg->depth.depth_data[0]), npixels * sizeof(uint16_t));
  }

  return;
}

void scale_image(cv::Mat& img, int interp, const float scale) {
  // Downsample (if necessary)
  if (scale != 1.f)
    cv::resize(img, img, cv::Size(int(img.cols * scale), int(img.rows * scale)), 
               0, 0, interp);
}

void unpack_rgb_scaled(const kinect::frame_msg_t* msg, cv::Mat& img, const float scale) { 

  // RGB unpack 
  unpack_rgb(msg, img);

  // Scale Image
  scale_image(img, cv::INTER_CUBIC, scale);
  return;
}

void unpack_depth_scaled(const kinect::frame_msg_t* msg, cv::Mat_<uint16_t>& depth,
                         const float scale) { 

  // Depth unpack 
  cv::Mat_<uint16_t> _depth;
  unpack_depth(msg, _depth);
    
  // Skip pixels instead of resizing (if necessary)
  scale_image(_depth, cv::INTER_NEAREST, scale);
  depth = _depth;
  
  // if (scale != 1.f) { 
  //   int dec = (int) (1.f / scale);
  //   depth = cv::Mat_<uint16_t>(cv::Size(int(_depth.cols * scale), int(_depth.rows * scale)));
            
  //   for(int _y=0,y=0; _y<_depth.rows; _y=_y+dec,y++)
  //     for(int _x=0,x=0; _x<_depth.cols; _x=_x+dec,x++)
  //       depth(y,x) = _depth(_y,_x);        
  // } else 
  //   depth = _depth;
    
  return;
}

void unpack_cloud_scaled(const cv::Mat_<uint16_t>& depth, cv::Mat_<cv::Vec3f>& cloud, float scale) { 

  // Allocate (nan init)
  cloud = cv::Mat3f(cv::Size(int(depth.cols * scale), 
                             int(depth.rows * scale)), cv::Vec3f(NAN, NAN, NAN));
    
  // Skip pixels instead of resizing (if necessary)
  int dec = (int) (1.f / scale);
        
  bool flip_coords = false;
  for(int _y=0,y=0; _y<depth.rows; _y=_y+dec,y++)
    for(int _x=0,x=0; _x<depth.cols; _x=_x+dec,x++) { 
      cv::Vec3f& vec = cloud.at<cv::Vec3f>(y,x);
      double disparity_d = (double) depth.at<uint16_t>(_y,_x) * 1e-3; 
      double constant = 1.f / 576.09757860; // 1.0f/kcal->intrinsics_rgb.fx;

      // cloud pt is NaN                    
      if (disparity_d == 0) 
        continue; 

      if (flip_coords) { 
        vec[1] =  - (((double) _x)- 319.50)*disparity_d*constant; //y right+ (check)
        vec[2] = - (((double) _y)- 239.50)*disparity_d*constant;  //z up+
        vec[0] = disparity_d;  //x forward+
      } else {
        vec[0] =  (((double) _x)- 319.50)*disparity_d*constant; //y right+ (check)
        vec[1] =  (((double) _y)- 239.50)*disparity_d*constant;  //x up+
        vec[2] = disparity_d;  //x forward+
      }
    }
    
  return;
}

cv::Mat
estimate_normals(const cv::Mat& cloud, cv::Mat& pts) {
  
}

std::vector<cv::Point3f>
estimate_normals(const cv::Mat3f& cloud, std::vector<cv::Point2f>& pts) {

}

cv::Point3f
get_xyz(const cv::Mat_<uint16_t>& depth, cv::Point2f& pt) { 

  bool flip_coords = false;

  float _y = pt.y, _x = pt.x;
  double disparity_d = (double) depth.at<uint16_t>(_y, _x) * 1e-3; 
  double constant = 1.f / 576.09757860; // 1.0f/kcal->intrinsics_rgb.fx;

  // cloud pt is NaN                    
  if (disparity_d == 0) 
    return cv::Point3f(NAN, NAN, NAN); 

  cv::Point3f vec;
  if (flip_coords) { 
    vec.y =  - (((double) _x)- 319.50)*disparity_d*constant; //y right+ (check)
    vec.z = - (((double) _y)- 239.50)*disparity_d*constant;  //z up+
    vec.x = disparity_d;  //x forward+
  } else {
    vec.x =  (((double) _x)- 319.50)*disparity_d*constant; //y right+ (check)
    vec.y =  (((double) _y)- 239.50)*disparity_d*constant;  //x up+
    vec.z = disparity_d;  //x forward+
  }
  return vec;
}


std::vector<cv::Point3f>
get_xyz(const cv::Mat_<uint16_t>& depth, std::vector<cv::Point2f>& pts) { 

  std::vector<cv::Point3f> pts3d(pts.size(),cv::Point3f(NAN, NAN, NAN));

  bool flip_coords = false;
  for (int j=0; j<pts3d.size(); j++) {
    cv::Point3f& vec = pts3d[j];
    float _y = pts[j].y, _x = pts[j].x;
    double disparity_d = (double) depth.at<uint16_t>(_y, _x) * 1e-3; 
    double constant = 1.f / 576.09757860; // 1.0f/kcal->intrinsics_rgb.fx;

    // cloud pt is NaN                    
    if (disparity_d == 0) 
      continue; 

    if (flip_coords) { 
      vec.y =  - (((double) _x)- 319.50)*disparity_d*constant; //y right+ (check)
      vec.z = - (((double) _y)- 239.50)*disparity_d*constant;  //z up+
      vec.x = disparity_d;  //x forward+
    } else {
      vec.x =  (((double) _x)- 319.50)*disparity_d*constant; //y right+ (check)
      vec.y =  (((double) _y)- 239.50)*disparity_d*constant;  //x up+
      vec.z = disparity_d;  //x forward+
    }
  }

  return pts3d;
}


void unpack_kinect_frame(const kinect::frame_msg_t* msg, cv::Mat& img, float scale) { 

  // unpack_kinect_frame first (scaled RGB)
  unpack_rgb_scaled(msg, img, scale);

  return;
}

void unpack_kinect_frame_with_depth(const kinect::frame_msg_t* msg, cv::Mat& img, 
                                    cv::Mat_<uint16_t>& depth, float scale) { 

  // unpack_kinect_frame first (scaled RGB)
  unpack_rgb_scaled(msg, img, scale);

  // Depth unpack
  unpack_depth_scaled(msg, depth, scale);

  return;
}

void unpack_kinect_frame_with_cloud(const kinect::frame_msg_t* msg, cv::Mat& img, 
                                    cv::Mat_<cv::Vec3f>& cloud, float scale) { 

  // unpack_kinect_frame first (scaled RGB)
  unpack_rgb_scaled(msg, img, scale);

  // Depth unpack
  cv::Mat_<uint16_t> depth;
  unpack_depth_scaled(msg, depth, scale);

  // Cloud scaled
  unpack_cloud_scaled(depth, cloud, scale);

  return;
}

cv::Mat3b color_depth_map(const cv::Mat& depth, float scale, bool min_max) { 

  // Convert to 8-bit
  cv::Mat1b depth8; 
  convertScaleAbs(depth, depth8, scale);
        
  if (min_max) 
    cv::normalize(depth8, depth8, 0, 255, cv::NORM_MINMAX, CV_8UC1);

  // Convert to colormap
  cv::Mat3b cdepth;
  applyColorMap(depth8, cdepth, cv::COLORMAP_JET);
        
  return cdepth;
}

}
