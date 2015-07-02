// Author: Sudeep Pillai (spillai@csail.mit.edu) 
// Updates: Oct 1, 2013

#ifndef FS_MATH_RIGID_TRANSFORM_HPP_
#define FS_MATH_RIGID_TRANSFORM_HPP_

#include "quaternion.hpp"

namespace fs { namespace math { 
class RigidTransform {

 public:
  RigidTransform();
  ~RigidTransform();
  RigidTransform(const RigidTransform& q);
  RigidTransform(const Quaternion& q, const double t[3]);
  RigidTransform(const double T[16]);
  RigidTransform inverse() const; 

 public:
  std::vector<double> tvec;
  Quaternion quat;
};
RigidTransform operator*(const RigidTransform& q1, const RigidTransform& q2) ;
std::ostream& operator << (std::ostream& os, const RigidTransform& p);

} // namespace math
} // namespace fs

#endif // FS_MATH_RIGID_TRANSFORM_HPP_
