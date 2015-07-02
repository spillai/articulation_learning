#include "dbow2_wrapper.hpp"

namespace fs { namespace vision { 

DBoW2::DBoW2() {
  // extended surf gives 128-dimensional vectors
  cv::initModule_nonfree();
  // const bool EXTENDED_SURF = true;
  // surf = cv::SURF(200, 4, 2, EXTENDED_SURF);
  // SurfFeatureDetector detector(5000);
  
  // extractor = cv::Ptr<cv::DescriptorExtractor>(
  //     new cv::SurfDescriptorExtractor(200, 4, 2, false, true)); // false, true
  detector = cv::FeatureDetector::create("SIFT");
  extractor = cv::DescriptorExtractor::create("SIFT");
 
}

DBoW2::~DBoW2() {
}

void DBoW2::reset() {
  features.clear();
  // features.reserve(NIMAGES);
}

void DBoW2::addFrame(const opencv_utils::Frame& frame) {
  addImage(frame.getGrayRef());
}

void DBoW2::addImage(const cv::Mat& img) {
  // Step 1: Detect the keypoints using SURF Detector
  std::vector<cv::KeyPoint> keypoints;
  detector->detect(img, keypoints);

  // Step 2: Calculate descriptors (feature vectors)
  cv::Mat desc;
  extractor->compute(img, keypoints, desc);

  // Step 3: Add description
  addDescription(desc);
}

void DBoW2::addDescription(const cv::Mat& desc) {
  // Change structure of desc, and add to features
  std::vector<std::vector<float> > descriptors(desc.rows, std::vector<float>(desc.cols));
  for (int j=0; j<desc.rows; j++)
    desc.row(j).copyTo(descriptors[j]);
  features.push_back(descriptors);
}


std::vector<Result>
DBoW2::queryFrame(const opencv_utils::Frame& frame, int results) {
  return queryImage(frame.getGrayRef(), results);
}

std::vector<Result>
DBoW2::queryImage(const cv::Mat& img, int results) {
  // Step 1: Detect the keypoints using SURF Detector
  std::vector<cv::KeyPoint> keypoints;
  detector->detect(img, keypoints);

  // Step 2: Calculate descriptors (feature vectors)
  cv::Mat desc;
  extractor->compute(img, keypoints, desc);

  // Step 3: Query extracted description
  return queryDescription(desc, results);
}


std::vector<Result>
DBoW2::queryDescription(const cv::Mat& desc, int results) {
  // Change structure, and query
  std::vector<std::vector<float> > descriptors(desc.rows, std::vector<float>(desc.cols));
  for (int j=0; j<desc.rows; j++)
    desc.row(j).copyTo(descriptors[j]);

  QueryResults ret;
  db.query(descriptors, ret, results);

  std::vector<Result> res(ret.begin(), ret.end());
  // ret[0] is always the same image in this case, because we added it to the 
  // database. ret[1] is the second best match.
  cout << "Searching for Image " << ret << endl;
  
  return res;
}

void DBoW2::save(const std::string& db_filename) {
  cout << "Saving database..." << endl;
  db.save(db_filename);
  cout << "... done!" << db << endl;
}


void DBoW2::load(const std::string& db_filename) {
  cout << "Loading database..." << endl;
  db = Surf64Database(db_filename);
  cout << "... done!" << db << endl;
}

void DBoW2::build(const int k, const int L, const bool use_direct_index) {
  // branching factor and depth levels 
  const WeightingType weight = TF_IDF;
  const ScoringType score = L1_NORM;

  Surf64Vocabulary voc(k, L, weight, score);

  cout << "Creating a small " << k << "^" << L << " vocabulary..." << endl;
  voc.create(features);
  cout << "... done!" << endl;
  cout << "Vocabulary information: " << endl
       << voc << endl << endl;

  cout << "Creating a small database..." << endl;
  db = Surf64Database (voc, use_direct_index, 0); // false = do not use direct index
  cout << "... done!" << endl;
  cout << "Database information: " << endl << db << endl;

  // (so ignore the last param)
  // The direct index is useful if we want to retrieve the features that 
  // belong to some vocabulary node.
  // db creates a copy of the vocabulary, we may get rid of "voc" now

    // add images to the database
  for(int i = 0; i < features.size(); i++) {
    db.add(features[i]);
  }
  cout << "... done!" << endl;
  cout << "Database information: " << endl << db << endl;
 
}

void DBoW2::buildVOC(const int k, const int L) {
  // branching factor and depth levels 
  const WeightingType weight = TF_IDF;
  const ScoringType score = L1_NORM;

  Surf64Vocabulary voc(k, L, weight, score);

  cout << "Creating a small " << k << "^" << L << " vocabulary..." << endl;
  voc.create(features);
  cout << "... done!" << endl;

  cout << "Vocabulary information: " << endl
       << voc << endl << endl;

  // // lets do something with this vocabulary
  // cout << "Matching images against themselves (0 low, 1 high): " << endl;
  // BowVector v1, v2;
  // for(int i = 0; i < features.size(); i++)
  // {
  //   voc.transform(features[i], v1);
  //   for(int j = 0; j < features.size(); j++)
  //   {
  //     voc.transform(features[j], v2);
      
  //     double score = voc.score(v1, v2);
  //     cout << "Image " << i << " vs Image " << j << ": " << score << endl;
  //   }
  // }

  // save the vocabulary to disk
  cout << endl << "Saving vocabulary..." << endl;
  voc.save("small_voc.yml.gz");
  cout << "Done" << endl;

}

void DBoW2::buildDB(bool use_direct_index) {
  cout << "Creating a small database..." << endl;

  // load the vocabulary from disk
  Surf64Vocabulary voc("small_voc.yml.gz");
  
  db = Surf64Database (voc, use_direct_index, 0); // false = do not use direct index
  // (so ignore the last param)
  // The direct index is useful if we want to retrieve the features that 
  // belong to some vocabulary node.
  // db creates a copy of the vocabulary, we may get rid of "voc" now

  // add images to the database
  for(int i = 0; i < features.size(); i++) {
    db.add(features[i]);
  }
  cout << "... done!" << endl;
  cout << "Database information: " << endl << db << endl;

  // // and query the database
  // cout << "Querying the database: " << endl;

  // QueryResults ret;
  // for(int i = 0; i < features.size(); i++)
  // {
  //   db.query(features[i], ret, 4);

  //   // ret[0] is always the same image in this case, because we added it to the 
  //   // database. ret[1] is the second best match.

  //   cout << "Searching for Image " << i << ". " << ret << endl;
  // }

  // cout << endl;

  // we can save the database. The created file includes the vocabulary
  // and the entries added
  cout << "Saving database..." << endl;
  db.save("small_db.yml.gz");
  cout << "... done!" << endl;
  
  // // once saved, we can load it again  
  // cout << "Retrieving database once again..." << endl;
  // Surf64Database db2("small_db.yml.gz");
  // cout << "... done! This is: " << endl << db2 << endl;
}


void DBoW2::changeStructure(const std::vector<float> &plain,
                            std::vector<std::vector<float> > &out,
                            int L) {
  out.resize(plain.size() / L);

  unsigned int j = 0;
  for(unsigned int i = 0; i < plain.size(); i += L, ++j)
  {
    out[j].resize(L);
    std::copy(plain.begin() + i, plain.begin() + i + L, out[j].begin());
  }
}


} // namespace vision
} // namespace fs
