#ifndef FS_PROFILER_HPP_
#define FS_PROFILER_HPP_

// 
// Author: Sudeep Pillai (spillai@csail.mit.edu)
// Updates: Aug 05, 2013
// 

// profiler include
#include "profiler.hpp"

// libbot/lcm includes
#include <bot_core/bot_core.h>

namespace fsvision {

static const char *_esc_colors[] = {"\033[0;31m", "\033[0;32m", "\033[0;33m", "\033[0;34m",
                                    "\033[0;35m", "\033[0;36m", "\033[0;37m", "\033[1;33m",
                                    "\033\[0m"};
static const std::vector<std::string> esc_colors(_esc_colors, std::end(_esc_colors));

static std::map<std::string, int> color_manager;

static std::string esc_def() {
  return esc_colors.end()[-1];
}

static std::string get_esc_color(const std::string& str) {
  int idx = -1;
  if (color_manager.find(str) == color_manager.end()) {
    idx = color_manager.size();
    color_manager.insert(std::make_pair(str, idx));
  } else 
    idx = color_manager[str];
  assert(idx>=0);
  return esc_colors[idx%(esc_colors.size()-1)];  
}

Profiler::Profiler() { 
}

Profiler::Profiler(const std::string& iname, bool colors, bool print) { 
  setName(iname, colors, print);
}

Profiler::~Profiler() { 
  // Could show all tracked/profiled
  // print_all();
  // fstream_.close();
}

void 
Profiler::setName(const std::string& iname, bool colors, bool print) {
  if (colors)
    iname_ = get_esc_color(iname) + iname + esc_def(); 
  else
    iname_ = iname;
  print_ = print;
  // fstream_.open(iname+".profile.txt");
}
    
void
Profiler::enter(const std::string& key) { 
  int64_t now = bot_timestamp_now();
  profile_map_[key] = profile_info(key, now);
  ++depth_;
}

void
Profiler::leave(const std::string& key) { 
  int64_t now = bot_timestamp_now();
  assert(profile_map_.find(key) != profile_map_.end());
  profile_map_[key].end_ = now;
  int64_t diff = profile_map_[key].end_ - profile_map_[key].start_;

  // Aggregates
  if (profile_agg_map_.find(key) != profile_agg_map_.end()) {
    profile_agg_map_[key].add(iname_, diff);
  } else {
    profile_agg_map_[key] = profile_agg_info(key, profile_map_[key].start_, diff);
  }
  
  // std::string indents = "> ";
  // // for (int j=0; j<depth_; j++) indents = "--" + indents;
  if (print_)
    std::cerr << "[" << iname_ << "] "  << key 
              << " ===> " << (diff) * 1e-3 
              << " ms " << std::endl;

  --depth_;
}

void 
Profiler::debug(const std::string& str) { 

  std::string indents = "> ";
  std::cerr << "[" << iname_ << "] " << indents << " DEBUG " 
            << "\n" << str << std::endl;
  
}

void 
Profiler::write(const std::string& str) { 
  // fstream_ << str ;
}

// static bool
// profile_sort(const profile_agg_info& lhs, const profile_agg_info& rhs) {
//   return lhs.duration_ < rhs.duration_;
// }

void 
Profiler::print_all() { 
  // Assemble profiles
  std::vector<profile_agg_info> profiles;
  for (std::map<std::string, profile_agg_info>::iterator it = profile_agg_map_.begin(); 
       it != profile_agg_map_.end(); it++) { 
    profiles.push_back(it->second);
  }
        
  // Sort by start times
  std::sort(profiles.begin(), profiles.end(), profile_sort);

  // Print 
  for (int j=0; j<profiles.size(); j++) { 
    std::cerr << "[" << iname_ << "] :: " << profiles[j].key_ 
              << " \n===> " << (profiles[j].avg_duration() ) * 1e-3 
              << " ms " << std::endl;
  }
}
}
#endif // #ifndef FS_PROFILER_HPP_
