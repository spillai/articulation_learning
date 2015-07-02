// Author: Sudeep Pillai (spillai@csail.mit.edu) Mar 16, 2013

// Track Manager include
#include "track_manager.hpp"

// opencv includes
#include <opencv2/opencv.hpp>

// opencv-utils includes
// #include <perception_opencv_utils/opencv_utils.hpp>

namespace fsvision {

//============================================
// Track class
//============================================
template <typename T>
Track<T>::Track () {
  predictions_ = 0; 
}

template <typename T>
Track<T>::~Track () {
}

template <typename T>
int Track<T>::size() { 
  return track_.size();
}

template <typename T>
void Track<T>::add(const T& feature) {
  predictions_ = (feature.get_status() == Feature::PREDICTED) ? predictions_ + 1 : 0;
  track_.push_back(feature);
  // assert(!feature.desc_.empty());
}

template <typename T>
void Track<T>::prune_by_size(const int size) { 
  if (track_.size() < size) return;
  track_.erase(track_.begin(), track_.end() - size);
}

template <typename T>
void Track<T>::prune_by_timespan_ms(const int timespan_ms) { 
  int64_t latest_utime = track_.end()[-1].get_utime();
  int64_t first_utime = track_.begin()[0].get_utime();
  if (latest_utime - first_utime > timespan_ms * 1e3) { 
    int idx = track_.size()-1; 
    for (; idx>=0; idx--) 
      if (latest_utime - track_[idx].get_utime() > timespan_ms * 1e3)
        break;
    track_.erase(track_.begin(), track_.begin() + idx);
  }
}

template <typename T>
void Track<T>::set_status(const int& status) { 
}

// Instantiate the templates for the class
template class Track<Feature2D>;
template class Track<Feature3D>;


//============================================
// TrackManager class
//============================================
template <typename T>
TrackManager<T>::TrackManager () {
  profiler_.setName("TM");
  profiler_.debug("CTOR");
  track_id_ = 1;

  previous_add_time_ = 0;
  previous_add_count_ = 0;
  track_add_rate_ = 0;

  num_feats_ = 1500; 
  allowed_predictions_ = 0;
  allowed_skips_ = 0; 
}

template <typename T>
TrackManager<T>::~TrackManager () {
}

template <typename T>
void TrackManager<T>::setParams (const int num_feats, const int allowed_predictions, const int allowed_skips) {
  num_feats_ = num_feats;
  allowed_predictions_ = allowed_predictions;
  allowed_skips_ = allowed_skips;
  
}

// void TrackManager::reset() { 
//     tracklets.clear();
//     track_confidence.clear();
//     past_utimes.clear();
//     reset_trackletID();
// }




//--------------------------------------------
// Ready helper
//--------------------------------------------
template <typename T>
bool TrackManager<T>::ready(const int num_feats) { 
  profiler_.enter("ready()");
  if (tracklet_map_.size() < num_feats)
    return true;
  else 
    return false;
  profiler_.leave("ready()");
}

//--------------------------------------------
// Only process frames that are newer
// Ignore frames that are old
// Track Clean up before update
//--------------------------------------------
template <typename T>
void TrackManager<T>::cleanup(const int64_t& utime ) { 
  // profiler_.enter("cleanup(utime)");

  std::stringstream ss;
  ss << " BEFORE: Tracklets Size: " << tracklet_map_.size() ;
  
  //----------------------------------
  // - clean up tracks that are old
  // - prune tracks that are longer than track_size_limit_
  //----------------------------------
  for (typename std::map<int64_t, Track<T> >::iterator it = tracklet_map_.begin(); it != tracklet_map_.end(); ) { 
    Track<T> & tracklet = it->second;
    assert(tracklet.track_.size());
    int64_t latest_utime = tracklet.track_.end()[-1].get_utime(); 

    // // 1. Clean up first
    // // If update frame is ahead of last track update by track_max_delay_ms seconds
    // if (utime - latest_utime >= track_max_delay_ms_ * 1e3) {
    //   it = tracklet_map_.erase(it);
    //   continue;
    // }

    // // 2. Clean up old predictions
    // // If update frame is ahead of last track update by track_max_delay_ms seconds
    // if (tracklet.predictions_ > allowed_predictions_) {
    //   it = tracklet_map_.erase(it);
    //   continue;
    // }

    
    // 3. Prune tracks that are longer than track_size_limit_
    if (tracklet.track_.size()>track_size_limit_) { 
      // std::cerr << "prune before track size: " << track.size() << std::endl;
      tracklet.prune_by_size(track_size_limit_);
      // std::cerr << "prune after track size: " << track.size() << std::endl;
    }

    // Done with pruning and cleanup
    ++it;
  }
  ss << " AFTER: Tracklets Size: " << tracklet_map_.size() << std::endl;
  // profiler_.leave("cleanup(utime)");
  // profiler_.debug("cleanup(utime)"+ss.str());
  return; 
}



//--------------------------------------------
// Prepare stable feats from existing tracklets
// TODO: 
// Add functionality for adding keypoints only if within some window
//  - use past utimes
//--------------------------------------------
template <typename T> 
void TrackManager<T>::getStableFeatures(std::vector<T>& tpts) { 
  // profiler_.enter("getStableFeatures(tpts)");

  int st = std::max(0, int(past_utimes.size() - allowed_skips_ - 1));
  int64_t min_utime = past_utimes[st]; 

  tpts.clear(); 
  for (typename std::map<int64_t, Track<T> >::iterator it =
           tracklet_map_.begin(); it != tracklet_map_.end(); it++) { 
    const Track<T> & tracklet = it->second;
    const std::vector<T>& track = tracklet.track_;
    assert(track.size());
    assert(it->first == track.end()[-1].get_id());
    const T& feat = track.end()[-1];
    assert(past_utimes.size());

    int64_t last_utime = past_utimes.back();
    // Check if within prediction limit
    if (feat.get_utime() < min_utime) continue; //  && tracklet.predictions_ > allowed_predictions_) continue;

    
    // if (feat.get_utime() != last_utime) continue;

    // Populate track
    tpts.push_back(feat); 

  }

  std::stringstream ss;
  ss << "Stable Features: " << tpts.size() << std::endl;
  // profiler_.leave("getStableFeatures(tpts)");
  // profiler_.debug("getStableFeatures(tpts) " + ss.str());
  return;
}


//--------------------------------------------
// Prepare stable feats from existing tracklets
// TODO: 
// Add functionality for adding keypoints only if within some window
//  - use past utimes
//--------------------------------------------
template <typename T> 
void TrackManager<T>::getAllFeatures(std::vector<T>& tpts, int min_track_size) { 
  // profiler_.enter("getAllFeatures(tpts)");

  tpts.clear(); 
  for (typename std::map<int64_t, Track<T> >::iterator it = tracklet_map_.begin(); it != tracklet_map_.end(); it++) { 
    const Track<T> & tracklet = it->second;
    const std::vector<T>& track = tracklet.track_;
    assert(track.size());
    // assert(it->first == track.end()[-1].get_id());
    // const T& feat = track.end()[-1];
            
    assert(past_utimes.size());
   
    // Populate track
    if (track.size() >= min_track_size)
      tpts.insert(tpts.end(), track.begin(), track.end());

  }

  std::stringstream ss;
  ss << "Stable Features: " << tpts.size() << std::endl;
  // profiler_.leave("getAllFeatures(tpts)");
  // profiler_.debug("getAllFeatures(tpts) "+ss.str());
  return;
}


template <typename T> 
void TrackManager<T>::addFeatures(const std::vector<T>& fpts) {
  // profiler_.enter("addFeatures(tpts)");
  std::stringstream ss;
  ss << "BEFORE: Tracklets Size: " << tracklet_map_.size() << std::endl;
 
  //--------------------------------------------
  // Check size and get utime now
  //--------------------------------------------
  std::vector<T> tpts = fpts;
  if (!tpts.size()) return;
  int64_t now = tpts.back().get_utime();

  //--------------------------------------------
  // Add features
  //--------------------------------------------
  for (int j=0; j<tpts.size(); j++) { 
    if (tpts[j].get_id() < 0) { 
      int64_t id = nextTrackID();
      tpts[j].set_id(id);
      tracklet_map_[id].add(tpts[j]);
    }

    int64_t id = tpts[j].get_id();
    tracklet_map_[id].add(tpts[j]);
  }

  //--------------------------------------------
  // Provide utime info
  //--------------------------------------------
  past_utimes.push_back(now);
  // (FIX)
  if (past_utimes.size() > 100) past_utimes.pop_front();

  // //--------------------------------------------
  // // Update utime history
  // //--------------------------------------------
  // {
  //   utimes_history.clear();
  //   int skip = 0; 
  //   for (std::deque<int64_t>::reverse_iterator rit = past_utimes.rbegin();
  //        rit != past_utimes.rend(); rit++)
  //     utimes_history[*rit] = skip++;
  // }
  
  // ss << "AFTER: Tracklets Size: " << tracklet_map_.size() << std::endl;
  // profiler_.leave("addFeatures(tpts)");
  // profiler_.debug("addFeatures(tpts) "+ss.str());
}

template <typename T> 
void TrackManager<T>::plot(const int64_t& utime, const cv::Mat& img) { 
  profiler_.enter("plot(utime, img)");
  if (img.empty()) return;

  cv::Mat display = img.clone();
  cv::Mat dalpha = 0.5 * display.clone();
  // cv::imshow("Tracklets", display);    
  // int64_t max_track_length = 1, min_track_length = std::numeric_limits<int>::max();
  // for (std::map<int64_t, int64_t>::iterator it = track_confidence.begin(); 
  //      it != track_confidence.end(); it++) { 
  //     const int64_t feat_id = it->first; 
  //     max_track_length = std::max(it->second, max_track_length);
  //     min_track_length = std::min(it->second, min_track_length);
  // }
  // float conf_norm = 255.f / (max_track_length - min_track_length);

  cv::Vec3b h(0,200,200);
  float r = 5; // spvision::GPFT_ADD_POINT_DISTANCE_THRESHOLD;
  for (typename std::map<int64_t, Track<T> >::iterator it = tracklet_map_.begin();
       it != tracklet_map_.end(); it++) { 
    const int64_t& fid = it->first; 
    const Track<T> & tracklet = it->second;
    const std::vector<T>& track = tracklet.track_;

    if (track.size() < 10) continue;

    // Plot with red, else green?
    bool current_track = bool(track.end()[-1].get_utime() == utime);
    if (!current_track) continue;
    cv::Scalar track_color = current_track ? CV_RGB(0,255,0) : CV_RGB(255,0,0);
    // std::cerr << "FeatID: " << feat_id << " " 
    // << track.end()[-1].point2D << std::endl;

    cv::Point2f pt = track.end()[-1].get_point();
    // float theta = track.end()[-1].angle * CV_PI / 180;

    // Draw trace of tracklet
    for (int k=track.size()-2, j=0; k>=0; k--, j++) { 
      int color = int(k * 255.f / (track.size() - 1));
      cv::line(dalpha, track[k].get_point(), track[k+1].get_point(), 
               current_track ? CV_RGB(0,color,0) : CV_RGB(color,0,0), 
               1, CV_AA, 0);
    }

    // double conf = (track_confidence[it->first] - min_track_length) * conf_norm;
    // h = cv::Vec3b(conf,conf,conf);
    // circle(dalpha, sc * pt, r, cv::Scalar(h[0],h[1],h[2]), 1,CV_AA);

    // Draw the keypoint circumference
    float hue = (1.f - 1.f / track.size()) * 180;
    cv::Scalar h(opencv_utils::hsv_to_bgrvec(cv::Vec3b(hue, 255, 255)));

    circle(dalpha, pt, r, h, 1,CV_AA);


    // // Draw the keypoint angle
    // cv::line(dalpha, sc * pt, sc * (pt + cv::Point2f(r*cos(theta), r*sin(theta))), 
    //          cv::Scalar(h[0],h[1],h[2]), 2, CV_AA, 0);

    // Draw the center 
    circle(dalpha, pt, 1, cv::Scalar(200,200,200), 1,CV_AA);
                
    // Draw the ID
    // putText(dalpha,cv::format("%i",fid), pt, 0, 0.3, track_color,1,CV_AA);
  }
  addWeighted(display, 0.2, dalpha, 0.5, 0, display);

  cv::imshow("Tracklets", display);
  profiler_.leave("plot(utime, img)");
  return;
           
}


// //--------------------------------------------
// // Now the rest unprocessed current points are new
// // Add new features
// //--------------------------------------------
// void TrackManager::addSuperFeatures(int64_t utime, std::list<SuperFeature>& c_sfpts_list, 
//                                        float ADD_3D_POINT_THRESHOLD) { 
//     int count_added = 0, count_updated = 0;

//     // Set all to invalid unless new observations are added
//     for (TrackMapIt it = tracklets.begin(); it != tracklets.end(); it++) { 
//         Track& track = it->second;
//         track.valid = false;
//     }
//     // Add new observations
//     for (std::list<SuperFeature>::iterator it = c_sfpts_list.begin(); 
//          it != c_sfpts_list.end(); it++) { 
//         if (tracklets[it->id].size()) { 
//             if (cv::norm(it->point3D - tracklets[it->id].back().point3D) > ADD_3D_POINT_THRESHOLD)
//                 continue;
//         }
//         it->id = (it->id >= 0) ? it->id : nextTrackletID(); 
//         (it->id >= 0) ? count_updated++ : count_added++;                    
//         tracklets[it->id].push_back(*it);
//         tracklets[it->id].valid = true;
//         updateConfidence(it->id);
//     }

//     printf("Total: %i, Added: %i, Updated: %i points\n", 
//            c_sfpts_list.size(), count_added, count_updated); 
//     printf("Tracklets : %i\n",tracklets.size());

//     past_utimes.push_front(utime);
//     return;
// }

// void TrackManager::plot(int64_t utime, const cv::Mat& img) { 
//     if (img.empty()) return;

//     double sc = 1.f;
//     cv::Mat display = img.clone();
//     cv::Mat dalpha = 0.5 * display;
    
//     {   int64_t max_track_length = 1, min_track_length = std::numeric_limits<int>::max();
//         for (std::map<int64_t, int64_t>::iterator it = track_confidence.begin(); 
//              it != track_confidence.end(); it++) { 
//             const int64_t feat_id = it->first; 
//             max_track_length = std::max(it->second, max_track_length);
//             min_track_length = std::min(it->second, min_track_length);
//         }
//         float conf_norm = 255.f / (max_track_length - min_track_length);

//         cv::Vec3b h(0,200,200);
//         float r = 5; // spvision::GPFT_ADD_POINT_DISTANCE_THRESHOLD;
//         for (TrackMapIt it = tracklets.begin(); it != tracklets.end(); it++) { 
//             const int64_t feat_id = it->first; 
//             const Track& track = it->second; 
            
//             if (!track.valid) continue;
//             if (track.end()[-1].utime != utime) continue;

//             // std::cerr << "FeatID: " << feat_id << " " << track.end()[-1].point2D << std::endl;
//             cv::Point2f pt = track.end()[-1].point2D;
//             float theta = track.end()[-1].angle * CV_PI / 180;

//             // Draw trace of tracklet
//             for (int k=track.size()-2, j=0; k>=0; k--, j++)
//                 cv::line(dalpha, sc * track[k].point2D, sc * track[k+1].point2D, 
//                          CV_RGB(0,0,200), 1, CV_AA, 0);

//             double conf = (track_confidence[it->first] - min_track_length) * conf_norm;
//             h = cv::Vec3b(conf,conf,conf);

//             // Draw the keypoint circumference
//             circle(dalpha, sc * pt, r, cv::Scalar(h[0],h[1],h[2]), 1,CV_AA);

//             // // Draw the keypoint angle
//             // cv::line(dalpha, sc * pt, sc * (pt + cv::Point2f(r*cos(theta), r*sin(theta))), 
//             //          cv::Scalar(h[0],h[1],h[2]), 2, CV_AA, 0);

//             // Draw the center 
//             circle(dalpha, sc * pt, 1, cv::Scalar(200,200,200), 1,CV_AA);
                
//             // Draw the ID
//             putText(dalpha,cv::format("%i",feat_id), sc * pt, 0, 0.3, cv::Scalar(0,200,0),1,CV_AA);
//         }
//         addWeighted(display, 0.2, dalpha, 0.5, 0, display);

//     }
//     opencv_utils::imshow("Tracklets", display);
//     return;
           
// }

// Instantiate the templates for the class
template class TrackManager<Feature2D>;
template class TrackManager<Feature3D>;
}
