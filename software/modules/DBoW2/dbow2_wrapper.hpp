// DBoW2 implementation wrapper
// Refer to 
// http://webdiis.unizar.es/~dorian/index.php?p=32

// // DBoW2
// #include "DBoW2.h" // defines Surf64Vocabulary and Surf64Database

// #include "DUtils.h"
// #include "DUtilsCV.h" // defines macros CVXX
// #include "DVision.h"

// DBoW2 include
#include <fs_perception_wrappers/DBoW2/DBoW2.h>
#include <fs_perception_wrappers/DBoW2/DUtils.h>
#include <fs_perception_wrappers/DBoW2/DUtilsCV.h>
#include <fs_perception_wrappers/DBoW2/DVision.h>

// frame_utils include
#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/features2d.hpp>

#include <pcl-utils/frame_utils.hpp>
#include <perception_opencv_utils/opencv_utils.hpp>

// feature_types
#include <features3d/feature_types.hpp>

using namespace DBoW2;
using namespace DUtils;

namespace fs { namespace vision { 
class DBoW2 {
 public:

  DBoW2();
  ~DBoW2();

  void addFrame(const opencv_utils::Frame& frame);
  void addImage(const cv::Mat& img);
  void addDescription(const cv::Mat& desc);

  std::vector<Result> queryFrame(const opencv_utils::Frame& frame, int results);
  std::vector<Result> queryImage(const cv::Mat& img, int results);
  std::vector<Result> queryDescription(const cv::Mat& desc, int results);
  
  void reset();
  void save(const std::string& filename);
  void load(const std::string& filename);
  void build(const int k=9, const int L=4, const bool use_direct_index=true);
  void buildVOC(const int k=9, const int L=4);
  void buildDB(const bool use_direct_index=true);
  
 private:
  void changeStructure(const std::vector<float> &plain,
                            std::vector<std::vector<float> > &out,
                       int L);
  
  std::vector<std::vector<std::vector<float> > > features;

  cv::Ptr<cv::FeatureDetector> detector;
  cv::Ptr<cv::DescriptorExtractor> extractor;

  // cv::SURF surf;

  Surf64Database db; // (voc, false, 0); // false = do not use direct index
};

} // namespace vision
} // namespace fs
