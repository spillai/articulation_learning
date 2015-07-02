#ifndef _EIGEN_NUMPY_H_
#define _EIGEN_NUMPY_H


#include <Python.h>
#include <numpy/ndarrayobject.h>

#include <Eigen/Eigen>
// #include <boost/numpy.hpp>
#include <glog/logging.h>
// #include <numpy/arrayobject.h>

namespace fs { namespace eigen {
void export_converters();
} // namespace eigen
} // namespace fs
#endif


