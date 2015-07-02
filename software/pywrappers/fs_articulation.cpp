// Wrapper for Articulation module
// lcm
#include <lcm/lcm-cpp.hpp>
// #include <lcmtypes/bot2_param.h>

#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

// fs_types
#include "fs_types.hpp"

#include <articulation/factory.h>
#include <articulation/utils.hpp>
#include <articulation/structs.h>
#include <articulation/ArticulatedObject.hpp>

using namespace articulation_models;
// using namespace articulation;

namespace py = boost::python;

namespace fs { namespace vision { 
class ArticulationLearner {
 public:
  ArticulationLearner() {}
  ArticulationLearner(const std::string& msg_buf, const std::string& filters="rigid rotational prismatic") {

    // Setting filters
    params.factory.setFilter(filters);
    
    // // Init server
    // InitParams();
    
    articulation::track_list_msg_t msg;
    msg.decode((void*) msg_buf.c_str(), 0, msg_buf.size());

    // Test validity of decode
    lcm.publish("ARTICULATION_OBJECT_TRACKS", &msg);
    
    // Object msg is updated
    articulation::articulated_object_msg_t object_msg;
    object_msg.parts.resize(msg.num_tracks);
    for (int j=0; j<msg.num_tracks; j++) {
      std::cerr << "TRACK : " << msg.tracks[j].id << " => " << j << std::endl;

      // copy tracks
      object_msg.parts[j] = msg.tracks[j];
    }

    // Setting object model
    object = ArticulatedObject(params); 
    object.SetObjectModel(object_msg); 
  }

 
  std::string project(const std::string& msg_buf) {
    articulation::track_list_msg_t msg;
    msg.decode((void*) msg_buf.c_str(), 0, msg_buf.size());
  }

  std::string articulated_object_str(articulation::articulated_object_msg_t& obj) {
    // Fix issue with sizes before publishing
    obj.num_parts = obj.parts.size();
    obj.num_params = obj.params.size();
    obj.num_models = obj.models.size();
    for (int j=0; j<obj.num_models; j++) {
      obj.models[j].track.num_channels = obj.models[j].track.channels.size();
      for (int k=0; k<obj.models[j].track.channels.size(); k++)
        obj.models[j].track.channels[k].num_values = obj.models[j].track.channels[k].values.size();
      obj.models[j].track.num_poses = obj.models[j].track.pose.size();
      obj.models[j].track.num_poses_projected = obj.models[j].track.pose_projected.size();
      obj.models[j].track.num_poses_resampled = obj.models[j].track.pose_resampled.size();
    }
    obj.num_markers = obj.markers.size();

    // Encode
    int sz = obj.getEncodedSize();
      char *buf = (char*) malloc (sz);
      obj.encode(buf, 0, sz);
      
      std::string ret;
      ret.assign(buf, sz);
      free(buf);
      return ret;
  }
  
  std::string fit(bool optimize, bool return_tree) {
    // Fit the object model
    object.FitModels(optimize);

    articulation::articulated_object_msg_t fully_connected_object = object.GetObjectModel();
    if (!return_tree) {
      return articulated_object_str(fully_connected_object);
    }

    std::cerr << "\n============== fit_models DONE ===========" << std::endl;
    std::cerr << "\n============== get_spanning_tree ===========" << std::endl; 
    // structureSelectSpanningTree
    object.ComputeSpanningTree();
    std::cerr << "\n============== get_spanning_tree DONE ===========" << std::endl; 
    // structureSelectSpanningTree
    // std::cerr << "\n============== get_fast_graph ===========" << std::endl; 
    // structureSelectFastGraph
    // object.getFastGraph();
    std::cerr << "\n============== get_fast_graph DONE ===========" << std::endl; 
    // structureSelectFastGraph
    std::cerr << "\n============== get_graph ===========" << std::endl; 
    // structureSelectGraph
    object.getGraph();
    std::cerr << "\n============== get_graph DONE ===========" << std::endl; 

    //----------------------------------
    // Publish articulated object
    //----------------------------------
    articulation::articulated_object_msg_t fitted_object = object.GetObjectModel();
    // lcm.publish("ARTICULATED_OBJECT", &fitted_object);

    // Return the fitted object
    return articulated_object_str(fitted_object);
  }

  // void InitParams() {
  //   // // Not using config file for filter models
  //   // b_server = bot_param_new_from_server(lcm.getUnderlyingLCM(), 1);
  //   // params.LoadParams(b_server);
  // }
  
 private:
  ArticulatedObject object; // (params); 
  KinematicParams params;
  
  lcm::LCM lcm;
};

} // namespace vision
} // namespace fs

namespace fs { namespace python { 

BOOST_PYTHON_MODULE(fs_articulation)
{
  // Py Init and Main types export
  fs::python::init_and_export_converters();

  // ArticulationLearner
  py::class_<fs::vision::ArticulationLearner>("ArticulationLearner")
      .def(py::init< std::string, py::optional<std::string> >
           (py::args("msg", "filters")))
      .def("fit", &fs::vision::ArticulationLearner::fit, 
           (py::arg("optimize")=true, py::arg("return_tree")=true))
      ;
}

} // namespace python
} // namespace fs
