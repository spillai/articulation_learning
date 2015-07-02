// 
// Author: Sudeep Pillai (spillai@csail.mit.edu) 
// Updates: Aug 13, 2013
// 

#ifndef POSE_UTILS_HPP_
#define POSE_UTILS_HPP_

// libbot
#include <bot_core/bot_core.h>

// libbot/lcm includes
#include <bot_core/bot_core.h>

// libbot cpp
#include <lcmtypes/bot_core.hpp>

// opencv includes
#include <opencv2/opencv.hpp>

// std includes
#include <ostream>
#include <iostream>

namespace pose_utils {

// Orientation orientation w,x,y,z
class pose_t : public bot_core::pose_t {
 public:
  pose_t ();
  pose_t (const bot_core::pose_t& p);
  pose_t (const BotTrans& bt);
  pose_t (const int64_t utime,
          const cv::Vec3f& _p1, const cv::Vec3f& _v1, const cv::Vec3f& _v2);

  const double* get_quat() const;
  
  const double* get_pos() const;
  
  bot_core::pose_t& get_pose ();

  // t: 3x1 (double [3])
  void set_translation(const double* t);

  // t: 3x1 (double [3])
  void set_translation(const double x, const double y, const double z);
  
  // Rotation matrix: 3x3 (double [9])
  void set_rotation(const double* R);

  // rpy: 3x1 (double [3])
  // rotation(const double r, const double p, const double y);
  
  pose_t inv();
  int64_t id;
};

// p = p1 * p2
pose_t pose_t_premul(const pose_t& p1,const pose_t& p2);  
pose_t operator* (const pose_t& p1, const pose_t& p2); 
std::ostream& operator << (std::ostream& os, const pose_t& p);

// pose_t
// pose_t::operator* (const pose_t& m) {


// static cv::Mat_<double> bot_core_pose_to_mat2(const bot_core::pose_t& msg) { 

//     cv::Mat_<double> T = cv::Mat_<double>::eye(4,4);

//     Rect Rrect(0,0,3,3), trect(3,0,1,3);
//     cv::Mat_<double> R = cv::Mat_<double>(T, Rrect);
//     cv::Mat_<double> t = cv::Mat_<double>(T, trect);
    
//     bot_quat_to_matrix(msg.orientation, (double*)R.data);
//     t(0,0) = msg.pos[0], t(1,0) = msg.pos[1], t(2,0) = msg.pos[2];

//     std::cerr << "T: " << T << std::endl;

//     return T;
// }

// static void bot_core_pose_to_mat(const bot_core::pose_t& msg, double* T) { 

//     double R[9];
//     bot_quat_to_matrix(msg.orientation, R);

//     T[3] = msg.pos[0], T[7] = msg.pos[1], T[11] = msg.pos[2];

//     T[0] = R[0], T[1] = R[1], T[2] = R[2];
//     T[4] = R[3], T[5] = R[4], T[6] = R[5];
//     T[8] = R[6], T[9] = R[7], T[10] = R[8];
//     T[12] = 0, T[13] = 0, T[14] = 0, T[15] = 1;
//     return;
// }

// static 
// void mat_to_bot_core_pose(double* T, bot_core::pose_t& msg) { 

//     double R[9];
//     R[0] = T[0], R[1] = T[1], R[2] = T[2];
//     R[3] = T[4], R[4] = T[5], R[5] = T[6];
//     R[6] = T[8], R[7] = T[9], R[8] = T[10];

//     bot_matrix_to_quat(R, msg.orientation);
//     msg.pos[0] = T[3] , msg.pos[1] = T[7], msg.pos[2] = T[11];
//     return;
// }

  
// }

}

#endif
