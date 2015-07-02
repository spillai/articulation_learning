#include "pose_utils.hpp"

namespace pose_utils { 

static inline bool is_nan(const cv::Vec3f& v) {
  return (v != v);
}

// Orientation orientation w,x,y,z
pose_t::pose_t() { 
  memset(pos, 0, sizeof(double) * 3);
  memset(orientation, 0, sizeof(double) * 4);
  orientation[0] = 1;
}
  
pose_t::pose_t (const bot_core::pose_t& p) {
  utime = p.utime;
  memcpy(pos, p.pos, sizeof(double) * 3);
  memcpy(vel, p.vel, sizeof(double) * 3);
  memcpy(orientation, p.orientation, sizeof(double) * 4);

  memcpy(rotation_rate, p.rotation_rate, sizeof(double) * 3);
  memcpy(accel, p.accel, sizeof(double) * 3);
}

pose_t::pose_t (const BotTrans& bt) {
  memcpy(pos, bt.trans_vec, sizeof(double) * 3);
  memcpy(orientation, bt.rot_quat, sizeof(double) * 4);
}

pose_t::pose_t(const int64_t _utime,
               const cv::Vec3f& _p1, const cv::Vec3f& _v1, const cv::Vec3f& _v2) {
  utime = _utime;
  if (is_nan(_p1)) {
    memset(pos, NAN, sizeof(double) * 3);
    memset(orientation, NAN, sizeof(double) * 4);
    return;
  }
  set_translation(_p1[0],_p1[1],_p1[2]);
  
  if (is_nan(_v1) || is_nan(_v2)) {
    memset(orientation, NAN, sizeof(double) * 4);
    return;
  }
  
  cv::Vec3f v1(_v1), v2(_v2);
  v1 *= 1.0 / cv::norm(v1);
  v2 *= 1.0 / cv::norm(v2);
  
  cv::Vec3f v3 = v1.cross(v2);
  v3 *= 1.0 / cv::norm(v3);

  v2 = v3.cross(v1);
  v2 *= 1.0 / cv::norm(v2);

  double T1[9];
  T1[0] = v1[0], T1[1] = v2[0], T1[2] = v3[0];
  T1[3] = v1[1], T1[4] = v2[1], T1[5] = v3[1];
  T1[6] = v1[2], T1[7] = v2[2], T1[8] = v3[2];
  set_rotation(T1);
}

const double* 
pose_t::get_pos() const {
  return pos;
}

bot_core::pose_t& pose_t::get_pose () {
  return *this;
}

// t: 3x1 (double [3])
void
pose_t::set_translation(const double* t) {
  pos[0] = t[0], pos[1] = t[1], pos[2] = t[2]; 
}

// t: 3x1 (double [3])
void
pose_t::set_translation(const double x, const double y, const double z) {
  pos[0] = x, pos[1] = y, pos[2] = z; 
}

  
// Rotation matrix: 3x3 (double [9])
void
pose_t::set_rotation(const double* R) {
  bot_matrix_to_quat(R, orientation);
}
  
pose_t
pose_t::inv() {
  pose_t p = *this;
  
  p.pos[0] = -p.pos[0];
  p.pos[1] = -p.pos[1];
  p.pos[2] = -p.pos[2];
  bot_quat_rotate_rev(p.orientation, p.pos);
  p.orientation[1] = -p.orientation[1];
  p.orientation[2] = -p.orientation[2];
  p.orientation[3] = -p.orientation[3];

  return p;
}

// p = p1 * p2
pose_t pose_t_premul(const pose_t& p1, const pose_t& p2) {
  pose_t p;
  bot_quat_mult(p.orientation, p1.orientation, p2.orientation);
  bot_quat_rotate_and_translate(p1.orientation, p1.pos, p2.pos, p.pos);

#if 0
  cv::Mat_<double> T(4,4), T1(4,4), T2(4,4);
  bot_quat_pos_to_matrix(p1.orientation, p1.pos, (double*)T1.data);
  bot_quat_pos_to_matrix(p2.orientation, p2.pos, (double*)T2.data);
  bot_quat_pos_to_matrix(p.orientation, p.pos, (double*)T.data);

  std::cerr << "T1: " << T1 << std::endl;
  std::cerr << "T2: " << T2 << std::endl;
  std::cerr << "T=T1*T2: " << T1*T2 << std::endl;
  std::cerr << "T(pose1,pose2): " << T << std::endl;
#endif

  return p;
}
 
pose_t
operator* (const pose_t& p1, const pose_t& p2) {
  return pose_t_premul(p1,p2);
}

std::ostream& operator << (std::ostream& os, const pose_t& p) {
  os << "pose_t [xyz:" << p.pos[0] << "," << p.pos[1] << "," << p.pos[2] << "] " << std::endl;
  double rpy[3]; bot_quat_to_roll_pitch_yaw (p.orientation, rpy);
  os << "[quat:"
            << p.orientation[0] << "," << p.orientation[1] << ","
            << p.orientation[2] << "," << p.orientation[3] << "] "
            << "[rpy:"
            << rpy[0] << "," << rpy[1] << "," << rpy[2] << "] " << std::endl;
  return os;
}

}
