#include <stdio.h>
#include "plot_utils.hpp"
#include "imshow_utils.hpp"
#include <sys/time.h>

namespace opencv_utils { 

cv::Scalar heatmap_bgr( float val ) { 
  float b = val > 0.333f ? 1 : val;
  val -= 0.333f;
  float g = val > 0.333f ? 1 : val;
  val -= 0.333f;
  float r = val > 0.333f ? 1 : val;
  return cv::Scalar(b * 255, g * 255, r * 255);
}

cv::Vec3b heatmap_bgrvec( float val ) { 
  float b = val > 0.333f ? 1 : val;
  val -= 0.333f;
  float g = val > 0.333f ? 1 : val;
  val -= 0.333f;
  float r = val > 0.333f ? 1 : val;
  return cv::Vec3b(b * 255, g * 255, r * 255);
}


void fillColors( std::vector<cv::Vec3b>& colors ) {
  if (!colors.size()) return;
  for( size_t ci = 0; ci < colors.size()-1; ci++ ) { 
    colors[ci] = hsv_to_bgrvec(cv::Vec3b(int(ci * 180.f / (colors.size()-1)), 255, 255));
  }
        
  if (colors.size())
    colors.end()[-1] = hsv_to_bgrvec(cv::Vec3b(180, 255, 255));
}

void fillColors( std::vector<cv::Scalar>& colors, bool normalize, bool hsv ) {
  if (!colors.size()) return;

  if (hsv) { 
    for( size_t ci = 0; ci < colors.size()-1; ci++ )
      colors[ci] = cv::Scalar(int(ci * 180.f / (colors.size()-1)), 255, 255);
    if (colors.size())
      colors.end()[-1] = cv::Scalar(255, 255, 255, 1);
  } else {
    cv::RNG rng = cv::theRNG();
    for( size_t ci = 0; ci < colors.size(); ci++ ) 
      colors[ci] = cv::Scalar( rng(256), rng(256), rng(256), 1 );
    
  }
}

// void fillColors( std::vector<cv::Scalar>& colors ) {
// }

cv::Mat plot_1D_hist(std::vector<float>& hist) { 

  int width = 600;
  double minVal=0,maxVal=0;
  cv::minMaxLoc(hist, &minVal, &maxVal, 0, 0);

  cv::Mat histImgLAB = cv::Mat::zeros(100, width, CV_8UC3);
  for( int i = 0; i < hist.size(); i++ ) {
    float scale = width*1.f/hist.size();
    float val = (cv::saturate_cast<int>(hist[i]*histImgLAB.rows-minVal) * 1.f) / (maxVal-minVal);
    cv::rectangle( histImgLAB, cv::Point(i*scale,histImgLAB.rows),
                   cv::Point2f((i+1)*scale,histImgLAB.rows - val),
                   cv::Scalar(0,240,0),-1, CV_AA );
    putText(histImgLAB, cv::format("%3.2f",minVal),
            cv::Point(20,100-20), 0, 0.4, cv::Scalar(100,100,100), 1);
    putText(histImgLAB, cv::format("%3.2f",maxVal),
            cv::Point(width-40,100-20), 0, 0.4, cv::Scalar(100,100,100), 1);
    putText(histImgLAB, cv::format("%3.2f",maxVal),
            cv::Point(width-40,20), 0, 0.4, cv::Scalar(100,100,100), 1);

    
  }
  return histImgLAB;
}

void putText(const std::string& text, cv::Mat& img) { 
  int n_lines = std::count(text.begin(), text.end(), '\n');
  cv::putText(img, cv::format("%s", text.c_str()),
              cv::Point2f(5,img.rows - 5*(n_lines+1)), 0, 0.3, 
              cv::Scalar(255,255,255),1);
        
}

static cv::Vec3b computeColor(float fx, float fy) {
  static bool first = true;

  // relative lengths of color transitions:
  // these are chosen based on perceptual similarity
  // (e.g. one can distinguish more shades between red and yellow
  //  than between yellow and green)
  const int RY = 15;
  const int YG = 6;
  const int GC = 4;
  const int CB = 11;
  const int BM = 13;
  const int MR = 6;
  const int NCOLS = RY + YG + GC + CB + BM + MR;
  static cv::Vec3i colorWheel[NCOLS];

  if (first)
  {
    int k = 0;

    for (int i = 0; i < RY; ++i, ++k)
      colorWheel[k] = cv::Vec3i(255, 255 * i / RY, 0);

    for (int i = 0; i < YG; ++i, ++k)
      colorWheel[k] = cv::Vec3i(255 - 255 * i / YG, 255, 0);

    for (int i = 0; i < GC; ++i, ++k)
      colorWheel[k] = cv::Vec3i(0, 255, 255 * i / GC);

    for (int i = 0; i < CB; ++i, ++k)
      colorWheel[k] = cv::Vec3i(0, 255 - 255 * i / CB, 255);

    for (int i = 0; i < BM; ++i, ++k)
      colorWheel[k] = cv::Vec3i(255 * i / BM, 0, 255);

    for (int i = 0; i < MR; ++i, ++k)
      colorWheel[k] = cv::Vec3i(255, 0, 255 - 255 * i / MR);

    first = false;
  }

  const float rad = sqrt(fx * fx + fy * fy);
  const float a = atan2(-fy, -fx) / (float)CV_PI;

  const float fk = (a + 1.0f) / 2.0f * (NCOLS - 1);
  const int k0 = static_cast<int>(fk);
  const int k1 = (k0 + 1) % NCOLS;
  const float f = fk - k0;

  cv::Vec3b pix;

  for (int b = 0; b < 3; b++)
  {
    const float col0 = colorWheel[k0][b] / 255.f;
    const float col1 = colorWheel[k1][b] / 255.f;

    float col = (1 - f) * col0 + f * col1;

    if (rad <= 1)
      col = 1 - rad * (1 - col); // increase saturation with radius
    else
      col *= .75; // out of range

    pix[2 - b] = static_cast<uchar>(255.f * col);
  }

  return pix;
}

inline bool isFlowCorrect(cv::Point2f u) {
  return !cvIsNaN(u.x) && !cvIsNaN(u.y) && fabs(u.x) < 1e9 && fabs(u.y) < 1e9;
}

void drawOpticalFlow(const cv::Mat_<cv::Point2f>& flow, cv::Mat& dst, float maxmotion) {
  dst.create(flow.size(), CV_8UC3);
  dst.setTo(cv::Scalar::all(0));

  // determine motion rang:e
  float maxrad = maxmotion;

  if (maxmotion <= 0)
  {
    maxrad = 1;
    for (int y = 0; y < flow.rows; ++y)
    {
      for (int x = 0; x < flow.cols; ++x)
      {
        cv::Point2f u = flow(y, x);

        if (!isFlowCorrect(u))
          continue;

        maxrad = std::max(double(maxrad), sqrt(u.x * u.x + u.y * u.y));
      }
    }
  }

  for (int y = 0; y < flow.rows; ++y)
  {
    for (int x = 0; x < flow.cols; ++x)
    {
      cv::Point2f u = flow(y, x);

      if (isFlowCorrect(u))
        dst.at<cv::Vec3b>(y, x) = computeColor(u.x / maxrad, u.y / maxrad);
    }
  }
}

void drawFlow(const cv::Mat_<cv::Vec2f>& flow, cv::Mat& dst, float maxmotion) {
  dst.create(flow.size(), CV_8UC3);
  dst.setTo(cv::Scalar::all(0));

  // determine motion rang:e
  float maxrad = maxmotion;

  if (maxmotion <= 0)
  {
    maxrad = 1;
    for (int y = 0; y < flow.rows; ++y)
    {
      for (int x = 0; x < flow.cols; ++x)
      {
        cv::Vec2f u = flow(y, x);

        if (!isFlowCorrect(u))
          continue;

        maxrad = std::max(double(maxrad), sqrt(u[0] * u[0] + u[1] * u[1]));
      }
    }
  }

  for (int y = 0; y < flow.rows; ++y)
  {
    for (int x = 0; x < flow.cols; ++x)
    {
      cv::Vec2f u = flow(y, x);

      if (isFlowCorrect(u))
        dst.at<cv::Vec3b>(y, x) = computeColor(u[0] / maxrad, u[1] / maxrad);
    }
  }
}

typedef std::map<int, FixedLengthQueue<float, 100> > data_queue;
// std::map<std::string, cv::Mat3b> plot_channels;
// std::map<std::string, data_queue> plot_data;
data_queue plot_data;

std::vector<cv::Scalar> colors(0);

cv::Mat plot_1D(const std::string& ch, std::vector<std::pair<int, float> >& data) {
  // if (plot_data.find(ch) == plot_data.end()) { 
  // //   plot_channels[ch] = cv::Mat3b::zeros(500, 800);
  //   plot_data[ch] = data_queue();
  //   fillColors(colors);
  // }

  if (!colors.size()) {
    colors.resize(10);
    fillColors(colors);
  }

  // // Insert data
  // for (int j=0; j<data.size(); j++)
  //   plot_data[ch][data[j].first].push(data[j].second);

  // Insert
  for (int j=0; j<data.size(); j++)
    plot_data[data[j].first].push(data[j].second);  
  
  // Plot data
  cv::Mat3b im = cv::Mat3b::zeros(100, 800);
  for (data_queue::const_iterator it = plot_data.begin();
       it != plot_data.end(); it++) {
    const std::deque<float>& d = it->second;
    for (int j=0; j<d.size()-1; j++) 
      cv::line(im, cv::Point2f(j*8, d[j] * 100.f), cv::Point2f((j+1)*8, d[j+1] * 100.f),
               colors[it->first % 10], 1, CV_AA);
  }
  return im;
}

}

