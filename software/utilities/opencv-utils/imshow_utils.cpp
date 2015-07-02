#include <stdio.h>
#include "imshow_utils.hpp"
#include <sys/time.h>

using namespace std;

namespace opencv_utils { 

#define VIEWER_NAME "OPENCV VIEWER"
#define WIDTH 640*3
#define HEIGHT 480*3


#define FOCAL_LENGTH 600
#define CUBE_SIZE 10

// Singleton object
static OpenCVImageViewer viewer;

void imshow(const std::string& name, const cv::Mat& img) {
  assert(&viewer != 0);
  viewer.imshow(name, img);
  // cv::imshow(name, img);
}

void write(const std::string& fn) {
  assert(&viewer != 0);
  viewer.write(fn);
}

OpenCVImageViewer::OpenCVImageViewer() {
  inited = false;
  opengl_mat = cv::Mat_<float>::eye(4,4);
  // std::cerr << "OPENCV VIEWER CTOR " << this << std::endl;
}

OpenCVImageViewer::~OpenCVImageViewer() { 
}

static void renderCube(float size)
{
  glBegin(GL_QUADS);
  // Front Face
  glNormal3f( 0.0f, 0.0f, 1.0f);
  glVertex3f( 0.0f,  0.0f,  0.0f);
  glVertex3f( size,  0.0f,  0.0f);
  glVertex3f( size,  size,  0.0f);
  glVertex3f( 0.0f,  size,  0.0f);
  // Back Face
  glNormal3f( 0.0f, 0.0f,-1.0f);
  glVertex3f( 0.0f,  0.0f, size);
  glVertex3f( 0.0f,  size, size);
  glVertex3f( size,  size, size);
  glVertex3f( size,  0.0f, size);
  // Top Face
  glNormal3f( 0.0f, 1.0f, 0.0f);
  glVertex3f( 0.0f,  size,  0.0f);
  glVertex3f( size,  size,  0.0f);
  glVertex3f( size,  size, size);
  glVertex3f( 0.0f,  size, size);
  // Bottom Face
  glNormal3f( 0.0f,-1.0f, 0.0f);
  glVertex3f( 0.0f,  0.0f,  0.0f);
  glVertex3f( 0.0f,  0.0f, size);
  glVertex3f( size,  0.0f, size);
  glVertex3f( size,  0.0f,  0.0f);
  // Right face
  glNormal3f( 1.0f, 0.0f, 0.0f);
  glVertex3f( size,  0.0f, 0.0f);
  glVertex3f( size,  0.0f, size);
  glVertex3f( size,  size, size);
  glVertex3f( size,  size, 0.0f);
  // Left Face
  glNormal3f(-1.0f, 0.0f, 0.0f);
  glVertex3f( 0.0f,  0.0f, 0.0f);
  glVertex3f( 0.0f,  size, 0.0f);
  glVertex3f( 0.0f,  size, size);
  glVertex3f( 0.0f,  0.0f, size);
  glEnd();
}

// static void on_opengl(void* user_data) {
//   //Draw the object with the estimated pose
//   glLoadIdentity();
//   glScalef( 1.0f, 1.0f, -1.0f);
 
//   glMultMatrixf( (float*) user_data );
        
//   glEnable( GL_LIGHTING );
//   glEnable( GL_LIGHT0 );
//   glEnable( GL_BLEND );
 
//   glBlendFunc(GL_SRC_ALPHA, GL_ONE);
        
//   renderCube( CUBE_SIZE );
        
//   glDisable(GL_BLEND);
//   glDisable( GL_LIGHTING );

//   std::cerr << "call back on_opengl" << std::endl;
// }

void OpenCVImageViewer::init() { 
  if (inited) return;
  inited = true;

  // One time
  // setMouseCallback(VIEWER_NAME, on_mouse, &mouse);
  // setWindowProperty(VIEWER_NAME, CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);

  // Setup window
  cv::namedWindow(VIEWER_NAME, CV_GUI_EXPANDED); // CV_WINDOW_AUTOSIZE);
  cv::resizeWindow(VIEWER_NAME, 1280, 960);

  // // Setup opengl callback CV_WINDOW_OPENGL |
  // cv::namedWindow("3D-VIEWER", cv::WINDOW_OPENGL | CV_GUI_EXPANDED);
  // cv::resizeWindow("3D-VIEWER", 1280, 960);
  // cv::setOpenGlDrawCallback("3D-VIEWER", on_opengl, (void*) opengl_mat.data);

}

static void button_callback(int b, void* data) {
  OpenCVImageViewer* viewer = (OpenCVImageViewer*)data;
  // std::cerr << viewer->image_map.size() << std::endl;
  // std::cerr << "button callback" << std::endl;
}

static bool image_info_sort_asc(const image_info* lhs, const image_info* rhs) { 
  return lhs->area > rhs->area;
}

static std::ostream& operator<<(std::ostream& os, cv::Rect& r) { 
  os << "[" << r.x << "," << r.y << "," << r.width << "," << r.height << "]";
  return os;
}

void OpenCVImageViewer::draw_image_to_frame(cv::Mat& frame, image_info& im_info) { 
  // Set the full ROI to zeros
  cv::Mat roi(frame, im_info.roi);
  roi = cv::Scalar::all(0);

  // Preserve aspect ratio
  int im_tw, im_th;
  double s1 = im_h * 1.f / im_info.img.rows;
  double s2 = im_w * 1.f / im_info.img.cols;
  double s = std::min(s1, s2);
  im_tw = int(s*im_info.img.cols), im_th = int(s*im_info.img.rows);

  cv::Rect roi_rect(im_info.roi.x+im_info.roi.width/2-im_tw/2, 
                    im_info.roi.y+im_info.roi.height/2-im_th/2, 
                    im_tw, im_th);
  // roi_rect &= im_info.roi;
  roi = cv::Mat(frame, roi_rect);

  cv::resize(im_info.img, roi, cv::Size(im_tw, im_th), 0, 0, cv::INTER_AREA);    

  cv::rectangle(frame, cv::Point(im_info.roi.x, im_info.roi.y+im_info.roi.height-15), 
                cv::Point(im_info.roi.x+im_info.roi.width, im_info.roi.y+im_info.roi.height), cv::Scalar(20,20,20), CV_FILLED, CV_AA);
  cv::putText(frame, cv::format("%s", im_info.name.c_str()),
              cv::Point2f(im_info.roi.x+5,im_info.roi.y+im_info.roi.height-5), 0, 0.30, 
              cv::Scalar(255,255,255),1, CV_AA);
  cv::rectangle(frame, cv::Point(im_info.roi.x, im_info.roi.y), 
                cv::Point(im_info.roi.x+im_info.roi.width, im_info.roi.y+im_info.roi.height), cv::Scalar(55,55, 55), 1, CV_AA);

}

void OpenCVImageViewer::configure_grid() { 
  if (!image_map.size()) return;
        
  // One-time init if image_map
  init(); 

  int dim1 = std::ceil(std::sqrt(image_map.size()));
  int dim2 = std::ceil(image_map.size() * 1.f / dim1);
  tile_width = std::max(dim1,dim2);
  tile_height = std::min(dim1,dim2);

  frame = cv::Mat3b(tile_height * 480, tile_width * 640, cv::Vec3b(0,0,0));

  im_w = frame.cols / tile_width;
  im_h = frame.rows / tile_height;

  int j = 0;

  for (std::map<std::string, image_info>::iterator it=image_map.begin(); it!=image_map.end(); it++) { 
    const std::string& name = it->first;
    image_info& im_info = it->second;

    int x = j % tile_width;
    int y = j / tile_width;

    im_info.roi = cv::Rect(x*im_w, y*im_h, im_w, im_h);
    im_info.placed = true;

    draw_image_to_frame(frame, im_info);

    j++;
  }
    
  return;
}

void OpenCVImageViewer::imshow(const std::string& name, const cv::Mat& img) { 
  // std::cerr << "opencv_utils::imshow" << std::endl;
  if (image_map.find(name) == image_map.end()) { 
    image_map[name] = image_info(name.c_str(), img.clone());
    // cv::createButton(name,button_callback,this,CV_CHECKBOX,1);
    // std::cerr << "Adding: " << name << std::endl;

    // Configure grid
    configure_grid();

  } else { 
    image_info& im_info = image_map[name];
    im_info.set_image(img);
        
    draw_image_to_frame(frame, im_info);
  }

  if (image_map.find(name) != image_map.end() && !frame.empty()) { 
    cv::imshow(VIEWER_NAME, frame);
    // cv::imshow("3D-VIEWER", frame);
  }
}

void OpenCVImageViewer::write(const std::string& fn) { 
  //----------------------------------
  // Create video writer
  //----------------------------------
  if (fn != "") {
    if (!writer.isOpened()) { 
      writer.open(fn, cv::VideoWriter::fourcc('M','P','4','2'),
                  30, frame.size(), 1);
      std::cerr << "===> Video writer created" << std::endl;
    }
  }

  writer << frame;
  std::cerr << "writing" << std::endl;
  return;
}

cv::Mat drawCorrespondences(const cv::Mat& img1, const vector<cv::Point2f>& features1, 
                            const cv::Mat& img2, const vector<cv::Point2f>& features2) {
  cv::Mat part, img_corr(cv::Size(img1.cols + img2.cols, MAX(img1.rows, img2.rows)), CV_8UC3);
  img_corr = cv::Scalar::all(0);
  part = img_corr(cv::Rect(0, 0, img1.cols, img1.rows));
  img1.copyTo(part); // cvtColor(img1, part, COLOR_GRAY2RGB);
  part = img_corr(cv::Rect(img1.cols, 0, img2.cols, img2.rows));
  img1.copyTo(part); // cvtColor(img1, part, COLOR_GRAY2RGB);

  for (size_t i = 0; i < features1.size(); i++) {
    cv::circle(img_corr, features1[i], 3, CV_RGB(255, 0, 0));
  }

  for (size_t i = 0; i < features2.size(); i++) {
    cv::Point pt(cvRound(features2[i].x + img1.cols), cvRound(features2[i].y));
    cv::circle(img_corr, pt, 3, cv::Scalar(0, 0, 255));
    // line(img_corr, features1[desc_idx[i].trainIdx].pt, pt, Scalar(0, 255, 0));
  }

  return img_corr;
}
}
