#ifndef OPENCV_IMGPROC_UTILS_H__
#define OPENCV_IMGPROC_UTILS_H__

#include <math.h>
#include <stdio.h>
#include <limits>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc_c.h>

static std::ostream& operator<<(std::ostream& out, const cv::Vec3f& m){
    out << m[0] << "," << m[1] << "," << m[2];
    return out;
}

static std::ostream& operator<<(std::ostream& out, const cv::KeyPoint& kp) { 
    out << "(" << kp.pt.x << "," << kp.pt.y << "): response: " << kp.response 
        << " angle:" << kp.angle << " size: " << kp.size 
        << " octave: " << kp.octave << " class_id: " << kp.class_id;
}

namespace opencv_utils { 
void simple_sobel(cv::Mat& dst, cv::Mat& img);
void convert_to_argb(const cv::Mat& img_mat, unsigned int* img);
void convert_to_mat(unsigned int* img, cv::Mat& img_mat);
std::vector<std::vector<cv::Point> >
find_enclosing_contour(std::vector<cv::Point>& pts, const cv::Size& size,
                       const int& erode_iterations = 0);

class MSER
{
public:
    //! the full constructor
    MSER( int _delta=5, int _min_area=60, int _max_area=14400,
          double _max_variation=0.25, double _min_diversity=.2,
          int _max_evolution=200, double _area_threshold=1.01,
          double _min_margin=0.003, int _edge_blur_size=5 );
  
    //! the operator that extracts the MSERs from the image or the specific part of it
  void operator()( const cv::Mat& image, std::vector<std::vector<cv::Point> >& msers,
                   std::vector<cv::Vec4i>& hierarchy, 
                   cv::Mat& der, const cv::Mat& mask=cv::Mat()) const;
  // AlgorithmInfo* info() const;

  void processImage(const cv::Mat& image, const cv::Mat& mask=cv::Mat());

  std::vector<std::vector<cv::Point> > getContours();
  std::vector<cv::Vec4i> getHierarchy();
 
protected:
  void detectImpl( const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& der, const cv::Mat& mask=cv::Mat() ) const;

    int delta;
    int minArea;
    int maxArea;
    double maxVariation;
    double minDiversity;
    int maxEvolution;
    double areaThreshold;
    double minMargin;
    int edgeBlurSize;
  std::vector<std::vector<cv::Point> > dstcontours_;
  std::vector<cv::Vec4i> hierarchy_;  
};

typedef MSER MserFeatureDetector;
}

#endif // OPENCV_IMGPROC_UTILS_H
