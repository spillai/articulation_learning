// Author: Sudeep Pillai (spillai@csail.mit.edu) 
// Updates: Oct 1, 2013

#include "quaternion.hpp"

namespace fs { namespace math { 

Quaternion::Quaternion() {
  orientation = std::vector<double>(4, 0); 
  orientation[0] = 1;
}

Quaternion::~Quaternion() {
}

Quaternion::Quaternion(const Quaternion& q) {
  orientation = q.orientation; 
}

Quaternion::Quaternion(const double* val) {
  const int nvals = sizeof(val)/sizeof(double);
  switch(nvals) {
    case 3:
      bot_roll_pitch_yaw_to_quat(val, &orientation[0]);
      break;
    case 4:
      memcpy(&orientation[0], val, sizeof(double) * 4);
      break;
    case 9:
      bot_matrix_to_quat(val, &orientation[0]);
      break;
    default:
      assert(0);
      break;
  }
}

Quaternion::Quaternion(const double axis[3], const double angle) {
  // bot_axis_angle_to_quat(axis, angle, orientation);
  assert(0);
}

Quaternion
Quaternion::inverse() {
  double q[4];
  q[0] = orientation[0]; 
  q[1] = -orientation[1];
  q[2] = -orientation[2];
  q[3] = -orientation[3];
  return Quaternion(q);
}

Quaternion operator* (const Quaternion& q1, const Quaternion& q2) {
  Quaternion q;
  bot_quat_mult(&q.orientation[0], &q1.orientation[0], &q2.orientation[0]);
  return q;
}

std::ostream& operator << (std::ostream& os, const Quaternion& p) {
  os << "Quaternion: ";
      // << p.tvec[0] << "," << p.tvec[1] << "," << p.tvec[2] << "] " << std::endl;
  double rpy[3];
  bot_quat_to_roll_pitch_yaw (&p.orientation[0], rpy);
  os << "[quat:"
     << p.orientation[0] << "," << p.orientation[1] << ","
     << p.orientation[2] << "," << p.orientation[3] << "] "
     << "[rpy:"
     << rpy[0] << "," << rpy[1] << "," << rpy[2] << "] " << std::endl;
  return os;
}

} // namespace math
} // namespace fs
