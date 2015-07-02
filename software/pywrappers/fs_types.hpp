#ifndef FS_TYPES_HPP_
#define FS_TYPES_HPP_

#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

#include "utils/template.h"
#include "utils/container.h"
#include "utils/opencv_numpy_conversion.hpp"
#include "utils/eigen_numpy_conversion.hpp"
// #include "utils/pair.h"

// opencv includes
#include <opencv2/opencv.hpp>

namespace fs { namespace python {

bool init_and_export_converters();

} // namespace python
} // namespace fs

#endif // FS_TYPES_HPP_
