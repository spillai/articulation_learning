// Author: Sudeep Pillai (spillai@csail.mit.edu) Mar 16, 2013
// 

#ifndef TRACK_MANAGER_HPP_
#define TRACK_MANAGER_HPP_

// Feature includes
#include "feature_types.hpp"
#include <deque>

// include profiler
#include <fs-utils/profiler.hpp>

namespace fsvision { 

//============================================
// Track class
//============================================
template <typename T>
class Track { 
  // enum TrackStatus { 
  //   GOOD=0, WAITING, BAD, INVALID,
  //   OUT_OF_BOUNDS, UNKNOWN
  // };

 public: 
  float avg_velocity;
  //  length_;
  int predictions_;
  std::vector<T> track_;
  

  Track ();
  ~Track ();
  int size();
  void add(const T& feature);
  void prune_by_size(const int size); 
  void prune_by_timespan_ms(const int timespan_ms); 
  void set_status(const int& status); 

};


//============================================
// TrackManager class
//============================================

// T=Feature2D, Feature3D

#define TRACK_SIZE_LIMIT 20
#define TRACK_MAX_DELAY_MS 50000 // discount this for now
template <typename T>
class TrackManager { 
 public: 

  // For IDing purposes
  int64_t track_id_;
  // ::vector<int> track_lengths_;
  
  // Constants
  static const int track_size_limit_ = TRACK_SIZE_LIMIT;
  static const int track_max_delay_ms_ = TRACK_MAX_DELAY_MS;
  
  // Tracklet map
  std::map<int64_t, Track<T> > tracklet_map_;
  std::deque<int64_t> past_utimes;
  std::map<int64_t, int> utimes_history;
  // std::map<int64_t, int64_t> track_confidence; 

  // Profiler
  Profiler profiler_;

  // Start Time (estimate rate of track addition)
  double previous_add_time_;
  double previous_add_count_;
  double track_add_rate_;
  int allowed_predictions_;
  int allowed_skips_;
  int num_feats_;
  
 public: 
  TrackManager ();
  ~TrackManager (); 
  void setParams (const int num_feats, const int allowed_predictions, const int allowed_skips);
  
  // Track IDing 
  int64_t nextTrackID() { return ++track_id_; } 
  int64_t trackID() { return track_id_;  }
  bool resetTrackID() { track_id_ = 1; return true; }

  // void reset(); 
  // bool cleanup(int64_t utime, float MAX_TRACK_TIMESTAMP_DELAY);

  bool ready(const int num_feats); 
  void cleanup(const int64_t& utime );
  void addFeatures(const std::vector<T>& fpts);
  void getStableFeatures(std::vector<T>& tpts);
  void getAllFeatures(std::vector<T>& tpts, int min_track_size=TRACK_SIZE_LIMIT);


  // void addSuperFeatures(int64_t utime, std::list<SuperFeature>& c_sfpts_list, 
  //                       float ADD_3D_POINT_THRESHOLD = 1.f);

  void plot(const int64_t& utime, const cv::Mat& img);

  // inline void updateConfidence(int64_t id) { 
  //     if (track_confidence.find(id) == track_confidence.end())
  //         track_confidence[id] = 1;
  //     else 
  //         track_confidence[id]++;
  // }
};


// typedef std::map<int64_t, Track2D> Track2DMap; 
// typedef std::map<int64_t, Track2D>::iterator Track2DMapIt; 


}
#endif
