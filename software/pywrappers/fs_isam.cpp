#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

// frame utils
#include <pcl-utils/frame_utils.hpp>
#include <perception_opencv_utils/opencv_utils.hpp>

// features3d
#include <features3d/feature_types.hpp>

// fs_types
#include "fs_types.hpp"

// isam
#include <isam/Noise.h>
#include <isam/Slam.h>
// #include <isam/slam2d.h>

#include <isam/Point3d.h>
#include <isam/Rot3d.h>
#include <isam/Pose3d.h>
#include <isam/slam3d.h>
#include <isam/util.h>

// #include <gtsam/geometry/Point3.h>
// #include <gtsam/geometry/Point3.h>

// Eigen
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <opencv2/core/eigen.hpp>

using namespace isam;

namespace py = boost::python;

namespace fs { namespace vision { 

// Wrapper to the isam::Slam class, with opencv support, and simple batch
// optimization routine 
class Slam3D {
 private:
  isam::Slam slam;
  std::map<int, isam::Pose3d_Node*> nodes;
  std::vector<isam::Pose3d_Factor*> unary_factors;
  std::vector<isam::Pose3d_Pose3d_Factor*> edge_factors;
  // isam::Covariances covs;

 public:
  Slam3D() {
    // for more complex cases use Powell's dog leg and
    // optionally a robust estimator (pseudo Huber) to deal with occasional outliers

    isam::Properties prop = slam.properties();
    prop.method = DOG_LEG;
    slam.set_properties(prop);

  }

  Slam3D(const Slam3D& slam3d) {
    // slam = slam3d;
  }

  ~Slam3D() {

  }

  // from Matrix
  isam::Pose3d toPose3d(const cv::Mat_<double>& T) {
    Eigen::MatrixXd T_;
    cv::cv2eigen(T, T_);
    return isam::Pose3d(T_);
  }

  
  // w,x,y,z
  isam::Pose3d toPose3d(const cv::Mat_<double>& quat, const cv::Mat_<double>& tvec) {
    isam::Pose3d p
        (isam::Point3d(tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2)), 
         isam::Rot3d(Eigen::Quaterniond(quat.at<double>(0),quat.at<double>(1),
                                        quat.at<double>(2),quat.at<double>(3))));
    return p;
  }

  cv::Mat_<double> fromPose3d(const isam::Pose3d& pose) {
    cv::Mat_<double> T;
    cv::eigen2cv(pose.wTo(), T);
    return T;
  }

  
  void add_node(int idx) {
    std::cerr << "iSAM SLAM3D: adding node: " << idx << std::endl;

    assert(nodes.find(idx) != nodes.end());
    nodes[idx] = new isam::Pose3d_Node();
    slam.add_node(nodes[idx]);
    
  }

  bool remove_node(int idx) {
    std::cerr << "iSAM SLAM3D: removing node: " << idx << std::endl;
    std::map<int, isam::Pose3d_Node*>::iterator it = nodes.find(idx);
    if (it != nodes.end()) {
      slam.remove_node(nodes[idx]);
      nodes.erase(it);
      return true;
    }
    return false;
  }
  
  void add_node_with_factor(int idx,
                            const cv::Mat_<double>& prior, 
                            // const cv::Mat_<double>& prior_quat,
                            // const cv::Mat_<double>& prior_tvec,
                            const cv::Mat_<double>& noise) {
    // std::cerr << "adding unary factor : " << idx << std::endl;
    // std::cerr << "unary factor prior : " << prior << std::endl;
    // std::cerr << "unary factor noise : " << noise << std::endl << std::endl;
    std::cerr << "iSAM SLAM3D: adding node with factor: " << idx << std::endl;
    // w,x,y,z

    
    nodes[idx] = new isam::Pose3d_Node();
    
    Eigen::MatrixXd _noise;
    cv::cv2eigen(noise, _noise);

    // std::cerr << "unary factor noise: " << _noise << std::endl;
    unary_factors.push_back(new isam::Pose3d_Factor(nodes[idx],
                                                    toPose3d(prior),
                                                    isam::Covariance(_noise)));

    slam.add_node(nodes[idx]);
    slam.add_factor(unary_factors.end()[-1]);
  }
  
  bool add_edge_factor(int idxi, int idxj,
                       const cv::Mat_<double>& odo, 
                       // const cv::Mat_<double>& odo_quat, const cv::Mat_<double>& odo_tvec,
                       const cv::Mat_<double>& noise) {
    std::cerr << "iSAM SLAM3D: adding edge factor between : " << idxi << "," << idxj << std::endl;
    // std::cerr << "edge factor odo : " << odo << std::endl;
    // std::cerr << "edge factor noise : " << noise << std::endl << std::endl;

    assert((nodes.find(idxi) != nodes.end()) && (nodes.find(idxj) != nodes.end()));

    Eigen::MatrixXd _noise;
    cv::cv2eigen(noise, _noise);

    edge_factors.push_back(new isam::Pose3d_Pose3d_Factor(nodes[idxi], nodes[idxj],
                                                          toPose3d(odo),
                                                          isam::Covariance(_noise)));
    slam.add_factor(edge_factors.end()[-1]);
    return true;
  }
  
  void batch_optimization() {
    // Optimize
    std::cerr << "iSAM SLAM3D: BATCH OPTIMIZATION: " << std::endl;
    slam.batch_optimization();
  }

  std::vector< cv::Mat_<double> > get_nodes() {
    // std::vector<isam::Pose3d> _nodes(nodes.size());
    std::vector< cv::Mat_<double> > _nodes(nodes.size());

    int idx = 0; 
    for (std::map<int, isam::Pose3d_Node*>::iterator iter = nodes.begin();
         iter != nodes.end(); iter++) {
      _nodes[idx++] = fromPose3d(iter->second->value());
    }
    return _nodes;    
  }

  // std::vector<cv::Mat> get_marginals() {
  //   int idx = 0;
  //   std::list<isam::Pose3d_Node*> _nodes(nodes.size());
  //   for (std::map<int, isam::Pose3d_Node*>::iterator iter = nodes.begin();
  //        iter != nodes.end(); iter++) {
  //     _nodes[idx++] = iter->second;
  //   }

  //   std::vector<cv::Mat> covs(nodes.size());
  //   std::list<Eigen::MatrixXd> _covs = slam.covariances().marginal(_nodes);
    
  //   idx = 0;
  //   for (std::list<Eigen::MatrixXd>::iterator iter = _covs.begin();
  //        iter != _covs.end(); iter++) {
  //     cv::eigen2cv(_covs[idx], covs[idx++]);
  //   }
  //   return covs;
  // }
  
  // note that as the nodes have no IDs to print, it's not easy to know
  // which node is which; the user really has to keep track of that, or
  // the user could extend the Pose3d_Node class to for example contain
  // a string ID.
  void print_all() {
    std::list<isam::Node*> ids = slam.get_nodes();
    for (list<isam::Node*>::iterator iter = ids.begin(); iter != ids.end(); iter++) {
      isam::Pose3d_Node* pose = dynamic_cast<isam::Pose3d_Node*>(*iter);
      cout << pose->value() << endl;
    }
  }


  // note that as the nodes have no IDs to print, it's not easy to know
  // which node is which; the user really has to keep track of that, or
  // the user could extend the Pose3d_Node class to for example contain
  // a string ID.
  void print_all2() {
    for (std::map<int, isam::Pose3d_Node*>::iterator iter = nodes.begin();
         iter != nodes.end(); iter++) {
      isam::Pose3d_Node* pose = iter->second;
      cout << iter->first << ": " << pose->value() << endl;
    }
  }

  
  void print_stats() { 
    slam.print_stats();
  }

  void print_graph() { 
    slam.print_graph();
  }
};
} // namespace vision
} // namespace fs

namespace fs { namespace python { 

BOOST_PYTHON_MODULE(fs_isam)
{
  // Main types export
  fs::python::init_and_export_converters();

  // Slam3D
  py::class_<fs::vision::Slam3D>("Slam3D")
      .def("add_node", &fs::vision::Slam3D::add_node)
      .def("remove_node", &fs::vision::Slam3D::remove_node)
      
      .def("add_node_with_factor", &fs::vision::Slam3D::add_node_with_factor,
           py::args("i", "T", "noise"))
      
      .def("add_edge_factor", &fs::vision::Slam3D::add_edge_factor,
           py::args("i", "j", "T", "noise"))

      .def("batch_optimization", &fs::vision::Slam3D::batch_optimization)
      .def("get_nodes", &fs::vision::Slam3D::get_nodes)
      // .def("get_marginals", &fs::vision::Slam3D::get_marginals)
      .def("print_all", &fs::vision::Slam3D::print_all)
      .def("print_all2", &fs::vision::Slam3D::print_all2)
      .def("print_stats", &fs::vision::Slam3D::print_stats)
      .def("print_graph", &fs::vision::Slam3D::print_graph)
      ;
  
  // expose_template_type<std::vector<isam::Pose3d> >(); // for get_nodes
  expose_template_type<std::vector<cv::Mat_<double> > >(); // for get_nodes, get_marginals
  
}

} // namespace python
} // namespace fs
