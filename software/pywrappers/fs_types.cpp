#include "fs_types.hpp"

namespace fs { namespace python {

static void py_init() {
  Py_Initialize();
  import_array();
}

static bool export_type_conversions_once = false;
bool init_and_export_converters() {

  if (export_type_conversions_once)
    return false;
  
  // std::cerr << "PYTHON TYPE CONVERTERS exported" << std::endl;
  export_type_conversions_once = true;

  // Py_Init and array import
  py_init();
  fs::opencv::export_converters();
  // fs::eigen::export_converters();

  // vectors
  expose_template_type<int>();
  expose_template_type<float>();
  expose_template_type<double>();

  expose_template_type< std::vector<int> >();
  expose_template_type< std::vector<float> >();
  expose_template_type< std::vector<double> >();

  expose_template_type<std::map<int, std::vector<int> > >();
  expose_template_type<std::map<int, std::vector<float> > >();

  expose_template_type<std::map<std::string, float> >();
  expose_template_type<std::map<std::string, double> >();

  return true;
}



BOOST_PYTHON_MODULE(fs_types)
{
  // Main types export
  fs::python::init_and_export_converters();
  py::scope scope = py::scope();

  // cv::Point2f
  py::class_<cv::Point2f>("Point2f")
      .def_readwrite("x", &cv::Point2f::x)
      .def_readwrite("y", &cv::Point2f::y)
      ;
  
  // cv::KeyPoint
  py::class_<cv::KeyPoint>("KeyPoint")
      .def_readwrite("pt", &cv::KeyPoint::pt)
      .def_readwrite("size", &cv::KeyPoint::size)
      .def_readwrite("angle", &cv::KeyPoint::angle)
      .def_readwrite("response", &cv::KeyPoint::response)
      .def_readwrite("octave", &cv::KeyPoint::octave)
      .def_readwrite("class_id", &cv::KeyPoint::class_id)
      ;
  
}

} // namespace python
} // namespace fs

