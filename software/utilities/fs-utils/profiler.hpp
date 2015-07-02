// 
// Author: Sudeep Pillai (spillai@csail.mit.edu)
// Updates: Aug 05, 2013
// 

#ifndef PROFILER_HPP_
#define PROFILER_HPP_

// Standard includes
#include <unistd.h>
#include <stdio.h>
#include <iostream>
// #include <fstream>
#include <sstream>
#include <algorithm>
#include <map>
#include <set> 

namespace fsvision { 

struct profile_info {
  int64_t start_; 
  int64_t end_;
  std::string key_;
        
  profile_info() 
      : start_(0), end_(0) {}
  profile_info(const std::string& key, const int64_t& start) 
      : key_(key), start_(start), end_(0) {}
};

struct profile_agg_info {
  int64_t start_, duration_; 
  std::string key_;
  int count_; 
  profile_agg_info() 
      : duration_(0), count_(0) {}
  profile_agg_info(const std::string& key, const int64_t start, const int64_t duration) 
      : key_(key), start_(start), duration_(duration), count_(1) {}
  void add(const std::string& iname, const int64_t duration) {
    duration_ += duration;
    count_++;

    // double avg = avg_duration() * 1e-3;
    // int mod = std::max(1, int(20 / avg));
    // if (count_ % mod == 0) { // greater than 100 ms
    //   std::cerr << "[" << iname << "] " << key_ 
    //             << " ===> " << avg << " ms " << std::endl;
    // }
  }
  double avg_duration() {
    return duration_ * 1.f / count_;
  }

};


static bool profile_sort(const profile_agg_info& lhs, 
                         const profile_agg_info& rhs) {
  return lhs.start_ < rhs.start_;
}

class Profiler { 
  std::map<std::string, profile_info> profile_map_;
  std::map<std::string, profile_agg_info> profile_agg_map_;
        
 private: 
  std::string iname_;
  int depth_;
  bool print_;

  // std::ofstream fstream_;
  
 public: 
  Profiler();
  Profiler(const std::string& iname, bool colors=true, bool print=true);
  ~Profiler();

  void setName(const std::string& iname, bool colors=true, bool print=true);
  void enter(const std::string& key);
  void leave(const std::string& key);
  void debug(const std::string& str);
  void write(const std::string& str);
  void print_all();
};
}
#endif
