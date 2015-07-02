// Author: Sudeep Pillai (spillai@csail.mit.edu) 
// Updates: Oct 1, 2013

#ifndef FS_MATH_QUATERNION_HPP_
#define FS_MATH_QUATERNION_HPP_

// libbot/lcm includes
#include <bot_core/bot_core.h>

// libbot cpp
#include <lcmtypes/bot_core.hpp>

// std includes
#include <ostream>
#include <iostream>

namespace fs { namespace math {

class Quaternion {
public:
  Quaternion();
  ~Quaternion();
  Quaternion(const Quaternion& q);
  Quaternion(const double* val);
  Quaternion(const double axis[3], const double angle);
  // Quaternion& operator=(const Quaternion&);
  Quaternion inverse(); 

 public:
  std::vector<double> orientation;
};
  
Quaternion operator*(const Quaternion& q1, const Quaternion& q2) ;
std::ostream& operator << (std::ostream& os, const Quaternion& p);

} // namespace math
} // namespace fs

#endif //#ifndef FS_MATH_QUATERNION_HPP_
