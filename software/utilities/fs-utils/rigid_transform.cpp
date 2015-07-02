// Author: Sudeep Pillai (spillai@csail.mit.edu) 
// Updates: Oct 1, 2013

#include "rigid_transform.hpp"

namespace fs { namespace math {

RigidTransform::RigidTransform() {
  tvec = std::vector<double>(3, 0);
}

RigidTransform::~RigidTransform() {

}

RigidTransform::RigidTransform(const RigidTransform& rt) {
  tvec = rt.tvec;
  quat = rt.quat;
}

RigidTransform::RigidTransform(const Quaternion& q, const double t[3]) {
  tvec = std::vector<double>(t, t+3);
  quat = q;
}

RigidTransform::RigidTransform(const double T[16]) {
  double R[9] = { T[0], T[1], T[2],
                  T[4], T[5], T[6],
                  T[8], T[9], T[10] };
  double t[3] = { T[3], T[7], T[11] };
  
  tvec = std::vector<double>(t, t+3);
  quat = Quaternion(R);
}

RigidTransform
RigidTransform::inverse() const {
  RigidTransform p = *this;
  p.tvec[0] = -p.tvec[0];
  p.tvec[1] = -p.tvec[1];
  p.tvec[2] = -p.tvec[2];
  bot_quat_rotate_rev(&p.quat.orientation[0], &p.tvec[0]);
  p.quat.inverse();
}

RigidTransform
operator*(const RigidTransform& rt1, const RigidTransform& rt2)  {
  RigidTransform rt;
  rt.quat = rt1.quat * rt2.quat; 
  bot_quat_rotate_and_translate(&rt1.quat.orientation[0], &rt1.tvec[0], &rt2.tvec[0], &rt.tvec[0]);
  return rt;
}

std::ostream& operator << (std::ostream& os, const RigidTransform& p) {
  os << "RigidTransform: ";
  os << "[tvec: ]" << p.tvec[0] << "," << p.tvec[1] << "," << p.tvec[2] << std::endl;
  return os;
}

} // namespace math
} // namespace fs
