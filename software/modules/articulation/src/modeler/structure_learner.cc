// Articulation Structure learner
// 1. Run Detector on log file
// articulation-detector -l ~/data/2013-07-31-drawer-apriltags/top_drawer_motion10fps -e 100
// 2. Run structure learner with '-k 20' (run every 20 frames)
// articulation-learner -k 20
// 3. Run viewer
// articulation-renderer / articulation-viewer


// lcm
#include <lcm/lcm-cpp.hpp>
#include <lcmtypes/bot2_param.h>

// libbot/lcm includes
#include <bot_core/bot_core.h>
#include <bot_frames/bot_frames.h>
#include <bot_param/param_client.h>
#include <bot_param/param_util.h>
#include <bot_lcmgl_client/lcmgl.h>
#include <GL/gl.h>

// opencv-utils includes
#include <perception_opencv_utils/opencv_utils.hpp>

// lcm log player wrapper
#include <lcm-utils/lcm-reader-util.hpp>

// lcm messages
#include <lcmtypes/articulation.hpp> 

// optargs
#include <ConciseArgs>

// visualization msgs
// #include <lcmtypes/visualization.h>
#include <lcmtypes/visualization.hpp>

#include <boost/foreach.hpp>

#include <articulation/factory.h>
#include <articulation/utils.hpp>
#include <articulation/structs.h>
#include <articulation/ArticulatedObject.hpp>

using namespace std;

using namespace articulation_models;
using namespace articulation;

// State 
struct state_t {
  lcm::LCM lcm;

  
  BotParam   *b_server;
  BotFrames *b_frames;

  // Factory
  MultiModelFactory factory;

  // Model
  GenericModelVector models_valid;

  // Track ID map
  std::map<int, int> id2idx_map, idx2id_map;
    
  // Articulation objects, tracks
  KinematicParams params;
  
  // Frames
  int nframe;

  // Viz related
  double kinect_to_local_m_opengl[16];
  bot_lcmgl_t *lcmgl_features;
  
  state_t() : 
      b_server(NULL),
      b_frames(NULL), 
      nframe(0) {

    // Check lcm
    assert(lcm.good());

    //----------------------------------
    // Bot Param/frames init
    //----------------------------------
    b_server = bot_param_new_from_server(lcm.getUnderlyingLCM(), 1);
    b_frames = bot_frames_get_global (lcm.getUnderlyingLCM(), b_server);
    lcmgl_features = bot_lcmgl_init (lcm.getUnderlyingLCM(), "Articulation Learner");

    double kinect_to_local_m[16];
    bot_frames_get_trans_mat_4x4(b_frames,"KINECT",
                                 bot_frames_get_root_name(b_frames),
                                 kinect_to_local_m);

    // opengl expects column-major matrices
    bot_matrix_transpose_4x4d(kinect_to_local_m, kinect_to_local_m_opengl);

    
  }
    
  void on_pose_tracks (const lcm::ReceiveBuffer* rbuf, const std::string& chan,
                       const articulation::track_list_msg_t *msg);
  void viz_articulation (ArticulatedObject& object);

  void draw_link(const std::string& str, const articulation::pose_msg_t& from_pose,
                 const articulation::pose_msg_t& to_pose,
                 const articulation::pose_msg_t& relative =
                 transformToPose(btTransform::getIdentity()), 
                 double size=20,
                 double r=0, double g=0.8, double b=0, double alpha=0.8);

  void draw_link(const std::string& str,
                 const btTransform& from_pose,
                 const btTransform& to_pose,
                 const btTransform& relative =
                 btTransform::getIdentity(),
                 double size=20,
                 double r=0, double g=0.8, double b=0, double alpha=0.8);

  
  void draw_point(const std::string& str, const articulation::pose_msg_t& pose,
                 const articulation::pose_msg_t& relative =
                 transformToPose(btTransform::getIdentity()), 
                  double size=20,
                  double r=0, double g=0.8, double b=0, double alpha=0.8);

  void draw_point(const std::string& str,
                 const btTransform& pose,
                 const btTransform& relative =
                 btTransform::getIdentity(),
                 double size=20,
                 double r=0, double g=0.8, double b=0, double alpha=0.8);

  
  ~state_t() { 
  }

};
state_t state;


struct StructureLearnerOptions { 
    int vRUN_EVERY_K_FRAMES;

    StructureLearnerOptions () : 
        vRUN_EVERY_K_FRAMES(20) {}
};
StructureLearnerOptions options;

// void state_t::on_pose_tracks (const lcm::ReceiveBuffer* rbuf, const std::string& chan,
//                               const articulation::track_msg_t *msg) {

void state_t::on_pose_tracks (const lcm::ReceiveBuffer* rbuf, const std::string& chan,
                              const articulation::track_list_msg_t *msg) {

  
    std::cerr << "."; 
    //----------------------------------
    // Collect Tracks
    //----------------------------------

    // // Set tracks for track-id
    // if (object_msg.parts.size() <= msg->id)
    //     object_msg.parts.resize(msg->id+1);
    // object_msg.parts[msg->id] = *msg;
    // std::cerr << "TRACK : " << msg->id << std::endl;


    if (msg->num_tracks < 2) { 
        std::cerr << "Insufficient track information, track size: " 
                  << msg->num_tracks << std::endl;
        return;
    }
       
    articulation::articulated_object_msg_t object_msg;
    object_msg.parts.resize(msg->num_tracks);
    for (int j=0; j<msg->num_tracks; j++) {
      std::cerr << "TRACK : " << msg->tracks[j].id << " => " << j << std::endl;
      object_msg.parts[j] = msg->tracks[j];
    }
    
    
    //----------------------------------
    // Run every few frames
    //----------------------------------
    nframe++;
    if (nframe % options.vRUN_EVERY_K_FRAMES != 0)
        return;
    
    //----------------------------------
    // Fit model to collected tracks
    //----------------------------------
    std::cerr << "\n============== fit_models ===========" << std::endl;
    ArticulatedObject object(params); 
    object.SetObjectModel(object_msg); 
    object.FitModels();

    articulation::articulated_object_msg_t fully_connected_object = object.GetObjectModel();
    std::cerr << "\n============== fit_models DONE ===========" << std::endl;
    std::cerr << "\n============== get_spanning_tree ===========" << std::endl; 
    // structureSelectSpanningTree
    object.ComputeSpanningTree();
    std::cerr << "\n============== get_spanning_tree DONE ===========" << std::endl; 
    // structureSelectSpanningTree
    std::cerr << "\n============== get_fast_graph ===========" << std::endl; 
    // structureSelectFastGraph
    object.getFastGraph();
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

    // Fix issue with sizes before publishing
    fitted_object.num_parts = fitted_object.parts.size();
    fitted_object.num_params = fitted_object.params.size();
    fitted_object.num_models = fitted_object.models.size();
    fitted_object.num_markers = fitted_object.markers.size();
    lcm.publish("ARTICULATED_OBJECT", &fitted_object);

    //----------------------------------
    // Debug info
    //----------------------------------
    std::cerr << "**** ARTICULATED OBJECT PARAMS ****" << std::endl;
    for (int j=0; j<fitted_object.params.size(); j++) { 
        std::cerr << fitted_object.params[j].name << ": " << fitted_object.params[j].value << std::endl;
    }

    std::cerr << "**** MODEL PARAMS ****" << std::endl;
    for (int j=0; j<fitted_object.models.size(); j++) {
        articulation::model_msg_t& model = fitted_object.models[j];
        
        std::cerr << model.id << "**" << model.name << "**" << 
            model.id / fitted_object.parts.size() << "->" << 
            model.id % fitted_object.parts.size() << " " << std::endl;

        for (int k=0; k<model.params.size(); k++) { 
            std::cerr << model.params[k].name << ": " << model.params[k].value << std::endl;
        }
        std::cerr << std::endl;
    }
    std::cerr << "-------------- DONE -----------" << std::endl;


    //----------------------------------
    // Viz
    //----------------------------------
    viz_articulation(object);
}

void state_t::draw_link(const std::string& str,
                        const btTransform& from_pose,
                        const btTransform& to_pose,
                        const btTransform& relative,
                        double size,
                        double r, double g, double b, double alpha) { 
  draw_link(str, transformToPose(from_pose), transformToPose(to_pose),
            transformToPose(relative), size, r, g, b, alpha);
}

void state_t::draw_link(const std::string& str,
                        const articulation::pose_msg_t& from_pose,
                        const articulation::pose_msg_t& to_pose,
                        const articulation::pose_msg_t& relative,
                        double size,
                        double r, double g, double b, double alpha) {

  // gl draw with queue
  bot_lcmgl_t* lcmgl = lcmgl_features;

  // Rotation to viewer
  lcmglPushMatrix();
  lcmglMultMatrixd(kinect_to_local_m_opengl);

  // Blending
  lcmglEnable(GL_BLEND);
  
  // Draw link bar
  lcmglEnable(GL_DEPTH_TEST);
  lcmglColor4f(r, g, b, alpha);
  lcmglLineWidth(size);

  lcmglBegin(GL_LINES);
  lcmglVertex3f(from_pose.pos[0],
                from_pose.pos[1],
                from_pose.pos[2]);
  lcmglVertex3f(to_pose.pos[0],
                to_pose.pos[1],
                to_pose.pos[2]);
  lcmglEnd();

  // Disable blending 
  lcmglDisable(GL_BLEND);
  lcmglDisable(GL_DEPTH_TEST);

  if (str.length()) { 
    // Draw model name (in between)
    const double pos_local[3] = { 0.5 *from_pose.pos[0] + 0.5 * to_pose.pos[0], 
                                  0.5 *from_pose.pos[1] + 0.5 * to_pose.pos[1], 
                                  0.5 *from_pose.pos[2] + 0.5 * to_pose.pos[2] };
    lcmglColor4f (1, 1, 1, 1);
    bot_lcmgl_text (lcmgl, pos_local, str.c_str());
  }
  
  // Pop sensor frame
  lcmglPopMatrix();

  draw_point("", to_pose, relative, 10, r, g, b, alpha);
  
}


void state_t::draw_point(const std::string& str,
                         const btTransform& pose, 
                        const btTransform& relative,
                        double size,
                         double r, double g, double b, double alpha) { 
  draw_point(str, transformToPose(pose), 
             transformToPose(relative), size, r, g, b);
}

void state_t::draw_point(const std::string& str,
                        const articulation::pose_msg_t& pose,
                        const articulation::pose_msg_t& relative,
                        double size,
                        double r, double g, double b, double alpha) {

  // gl draw with queue
  bot_lcmgl_t* lcmgl = lcmgl_features;

  // Rotation to viewer
  lcmglPushMatrix();
  lcmglMultMatrixd(kinect_to_local_m_opengl);

  // Blending
  lcmglEnable(GL_BLEND);
  
  // Draw link bar
  lcmglEnable(GL_DEPTH_TEST);
  lcmglPointSize(size);
  lcmglColor4f(r, g, b, alpha);
  lcmglBegin(GL_POINTS);
  lcmglVertex3f(pose.pos[0], pose.pos[1], pose.pos[2]);
  lcmglEnd();

  
  // Disable blending 
  lcmglDisable(GL_BLEND);
  lcmglDisable(GL_DEPTH_TEST);
  
  // Draw model name (in between)
  const double pos_local[3] = { pose.pos[0], 
                                pose.pos[1], 
                                pose.pos[2] };
  bot_lcmgl_push_attrib (lcmgl, GL_CURRENT_BIT | GL_ENABLE_BIT);
  lcmglColor4f (1, 1, 1, 1);
  bot_lcmgl_text (lcmgl, pos_local, str.c_str());

  // Draw axis
  lcmglTranslatef(pose.pos[0], pose.pos[1], pose.pos[2]);

  double rpy[3];
  bot_quat_to_roll_pitch_yaw(pose.orientation, rpy);
  
  lcmglRotatef(bot_to_degrees(rpy[2]),  0., 0., 1.);
  lcmglRotatef(bot_to_degrees(rpy[1]),0., 1., 0.);
  lcmglRotatef(bot_to_degrees(rpy[0]), 1., 0., 0.);

  float sz = 0.05;
  lcmglLineWidth(3.f);
  lcmglBegin(GL_LINES);
    lcmglColor3f(1.0,0.0,0.0); lcmglVertex3f(0.0,0.0,0.0); lcmglVertex3f(sz*1.0,0.0,0.0);
    lcmglColor3f(0.0,1.0,0.0); lcmglVertex3f(0.0,0.0,0.0); lcmglVertex3f(0.0,sz*1.0,0.0);
    lcmglColor3f(0.0,0.0,1.0); lcmglVertex3f(0.0,0.0,0.0); lcmglVertex3f(0.0,0.0,sz*1.0);
  lcmglEnd();
  
  bot_lcmgl_pop_attrib (lcmgl);

  // Pop sensor frame
  lcmglPopMatrix();
  
}



std::ostream& operator << (std::ostream& os, const articulation::pose_msg_t& p) {
  os << "pose_t [xyz:" << p.pos[0] << "," << p.pos[1] << "," << p.pos[2] << "] " << std::endl;
  os << "[quat:"
            << p.orientation[0] << "," << p.orientation[1] << ","
            << p.orientation[2] << "," << p.orientation[3] << "] "
            << "[rpy:"
            << p.orientation[0] << "," << p.orientation[1] << ","
            << p.orientation[2] << "," << p.orientation[3] << "] " << std::endl;
  return os;
}

std::ostream& operator << (std::ostream& os, const btTransform& p) {
  std::cerr << transformToPose(p);
}

btVector3 getOrtho(btVector3 a,btVector3 b) {
  btVector3 bb = b - a.dot(b)*a;
  bb.normalize();
  a.normalize();
  return a.cross(bb);
}


void state_t::viz_articulation(ArticulatedObject& object) {

  for(KinematicGraph::iterator i= object.currentGraph.begin();
      i!=object.currentGraph.end(); i++) {
    int from = i->first.first;
    int to = i->first.second;
    articulation_models::GenericModelPtr &model = i->second;
    const pose_msg_t& pose_from = object.object_msg.parts[from].pose.back();
    const pose_msg_t& pose_to = object.object_msg.parts[to].pose.back();

    std::stringstream ssfrom; ssfrom << from;
    std::stringstream ssto; ssto << to;
    draw_point(ssfrom.str(), pose_from,
                transformToPose(btTransform::getIdentity()), 10, 0.8, 0, 0);
    draw_point(ssto.str(), pose_to,
               transformToPose(btTransform::getIdentity()), 10, 0.8, 0, 0);
    
    std::stringstream modelss; 
    if(model->getModelName()=="rigid") {
      modelss << "Rigid"
        << " (" << from << "->" << to << ")";

      draw_link(modelss.str(), pose_from, pose_to);
      // draw_point("rigid_origin_from", pose_from,
      //            transformToPose(btTransform::getIdentity()), 10, 0.8, 0, 0);
    } else if(model->getModelName()=="prismatic") {
      modelss << "Prismatic"
        << " (" << from << "->" << to << ")";

      pose_msg_t origin = model->predictPose(model->predictConfiguration( transformToPose(btTransform::getIdentity() )));
      btTransform pose_orthogonal = poseToTransform(pose_from)*poseToTransform(origin);

      btTransform pose_max = poseToTransform(pose_from)*
          poseToTransform(model->predictPose(model->getMinConfigurationObserved()));
      btTransform pose_min = poseToTransform(pose_from)*
          poseToTransform(model->predictPose(model->getMaxConfigurationObserved()));


      // std::cerr << "minobs: " << model->getMinConfigurationObserved() << std::endl;
      // std::cerr << "maxobs: " << model->getMinConfigurationObserved() << std::endl;
      // std::cerr << "pose_min: " << pose_min << " max: " << pose_max << std::endl;
            
      // // draw tics
      // for(double q = model->getMinConfigurationObserved()[0];
      //     q<model->getMaxConfigurationObserved()[0];
      //     q += 0.05) {
      //   V_Configuration qq(1);
      //   qq(0) = q;
      //   btTransform pose_q = poseToTransform(pose_from)*
      //       poseToTransform(model->predictPose(qq));

      //   btVector3 dir = poseToTransform(pose_from) *
      //       boost::static_pointer_cast<PrismaticModel>(model)->prismatic_dir
      //       - poseToTransform(pose_from) * btVector3(0,0,0);
      //   dir.normalize();

      //   draw_link("",
      //             btTransform(btQuaternion(0,0,0,1),pose_q.getOrigin()), 
      //             btTransform(btQuaternion(0,0,0,1),pose_q.getOrigin()
      //                         + getOrtho(dir,btVector3(0,0,1))*0.01),
      //             btTransform(btQuaternion(0,0,0,1),pose_q.getOrigin()), 
      //             5,0.3,0.0,0.3);
      //   draw_link("",
      //             btTransform(btQuaternion(0,0,0,1),pose_q.getOrigin()), 
      //             btTransform(btQuaternion(0,0,0,1),pose_q.getOrigin()
      //                         + getOrtho(dir,btVector3(0,0,1))*-0.01),
      //             btTransform(btQuaternion(0,0,0,1),pose_q.getOrigin()), 
      //             5,0.3,0.0,0.3);
      // }
            
      // Link
      draw_link("", pose_from, pose_to,
                transformToPose(btTransform::getIdentity()),
                20., 0, 0.3, 0, 0.3);

      // Extent of motion
      pose_msg_t dpose = pose_to;
      dpose.pos[0] = pose_from.pos[0] - pose_to.pos[0];
      dpose.pos[1] = pose_from.pos[1] - pose_to.pos[1];
      dpose.pos[2] = pose_from.pos[2] - pose_to.pos[2];

      draw_link(modelss.str(), transformToPose(pose_min),
                transformToPose(pose_max),
                dpose,60,0,0.6,0);
    } else if(model->getModelName()=="rotational") {
      modelss << "Rotational"
        << " (" << from << "->" << to << ")";
      // modelss << "Rot. radius: " << setprecision(3) << model->getParam("rot_radius");

      pose_msg_t rot_center = 
          transformToPose(poseToTransform(pose_from)*
                          btTransform(boost::static_pointer_cast<RotationalModel>(model)->rot_axis,
                                      boost::static_pointer_cast<RotationalModel>(model)->rot_center)
                          );
      // draw_link(modelss.str(),pose_from,rot_center,
      //           transformToPose(btTransform::getIdentity()),
      //           40,0.4,0.4,0.0,0.3); 
      draw_link(modelss.str(),rot_center,pose_to,
                transformToPose(btTransform::getIdentity()),
                20,0,0,0.5);
      // draw_link("",rot_center,pose_to,
      //           transformToPose(btTransform::getIdentity()),
      //           20,0.3,0.,0.3); // ,0.666,1.0,1.0);

      #if 0
      double q0 = model->predictConfiguration( transformToPose(btTransform::getIdentity()) )[0];
      double q1 = model->predictConfiguration( model->model.track.pose.back() )[0];
      if(q0>q1) {
        double q= q1;
        q1 = q0;
        q0 = q;
      }
      if(q1-q0 > M_PI) {
        // go the other way around
        double q= q1;
        q1 = q0+2*M_PI;
        q0 = q;
      }


      V_Configuration Qmin = model->getMinConfigurationObserved();
      V_Configuration Qmax = model->getMaxConfigurationObserved();

      V_Configuration Q(1);
      btTransform t1,t2,t3,t4;
      double STEPSIZE = M_PI/16;
      // double radius = 0.05;
      double radius = model->getParam("rot_radius");
      for(double q=0;q<2*M_PI+STEPSIZE;q+=STEPSIZE) {
        bool in_range = q>=Qmin[0] && q<=Qmax[0];
        if(Qmin[0]<0) in_range = q>=(Qmin[0]+2*M_PI) || q<=Qmax[0];
        if (!in_range) continue;
        Q[0] = q;
        t1 = 	poseToTransform(rot_center) *
            btTransform(btQuaternion(btVector3(0,0,1),-q),btVector3(0,0,0)) *
            btTransform(btQuaternion(0,0,0,1),btVector3(radius,0,0.0));
        t2 = 	poseToTransform(rot_center) *
            btTransform(btQuaternion(btVector3(0,0,1),-(q+STEPSIZE)),btVector3(0,0,0)) *
            btTransform(btQuaternion(0,0,0,1),btVector3(radius,0,0.0));
        t3 = 	poseToTransform(rot_center) *
            btTransform(btQuaternion(btVector3(0,0,1),-(q+STEPSIZE)),btVector3(0,0,0)) *
            btTransform(btQuaternion(0,0,0,1),btVector3(radius*1.1,0,0.0));
        t4 = 	poseToTransform(rot_center) *
            btTransform(btQuaternion(btVector3(0,0,1),-(q+STEPSIZE)),btVector3(0,0,0)) *
            btTransform(btQuaternion(0,0,0,1),btVector3(radius*0.9,0,0.0));
        draw_link("",
                  transformToPose(t1*btTransform(btQuaternion(0,0,0,1),btVector3(-0.015,0,0))),
                  transformToPose(t2*btTransform(btQuaternion(0,0,0,1),btVector3(-0.015,0,0))));
      }
#endif
      // rotation axis:
      draw_link("",
                rot_center,
                transformToPose(poseToTransform(rot_center) *
                                btTransform(btQuaternion(0,0,0,1),btVector3(0,0,-0.10))),
                rot_center);
      draw_link("",
                rot_center,
                transformToPose(poseToTransform(rot_center) *
                                btTransform(btQuaternion(0,0,0,1),btVector3(0,0,+0.10))),
                rot_center);            


    }

  }

  bot_lcmgl_switch_buffer(lcmgl_features);  
}

int main(int argc, char** argv) 
{
    //----------------------------------
    // Opt args
    //----------------------------------
    ConciseArgs opt(argc, (char**)argv);
    opt.add(options.vRUN_EVERY_K_FRAMES, "k", "k-frames","Run every K frames");
    opt.parse();

    //----------------------------------
    // args output
    //----------------------------------
    std::cerr << "===========  Structure Learner ============" << std::endl;
    std::cerr << "MODES 1: articulation-structure-learner\n";
    std::cerr << "=============================================\n";

    std::cerr << "=> Note: Hit 'space' to proceed to next frame" << std::endl;
    std::cerr << "=> Note: Hit 'p' to proceed to previous frame" << std::endl;
    std::cerr << "=> Note: Hit 'n' to proceed to previous frame" << std::endl;
    std::cerr << "===============================================" << std::endl;

    //----------------------------------
    // Subscribe, and start main loop
    //----------------------------------
    state.lcm.subscribe("ARTICULATION_OBJECT_TRACKS", &state_t::on_pose_tracks, &state);

    // Not using config file for filter models
    state.params.LoadParams(state.b_server);

    // // Set PRIOR Params for Articulated Object
    // setParam(state.object_msg.params, "sigma_position", structure_params.sigma_position,
    //          articulation::model_param_msg_t::PRIOR);
    // setParam(state.object_msg.params, "sigma_orientation", structure_params.sigma_orientation,
    //          articulation::model_param_msg_t::PRIOR);
    // setParam(state.object_msg.params, "reduce_dofs", structure_params.reduce_dofs,
    //          articulation::model_param_msg_t::PRIOR);

    while (state.lcm.handle() == 0);

    return 0;

}




// int main(int argc, char** argv) {
// 	ros::init(argc, argv, "structure_learner_server");
// 	nh = new ros::NodeHandle();
// 	nh_local = new ros::NodeHandle("~");

// 	params.LoadParams(*nh_local,false);

// 	model_pub = nh->advertise<articulation_msgs::ModelMsg> ("model", 0);
// 	track_pub = nh->advertise<articulation_msgs::TrackMsg> ("track", 0);
// 	marker_pub = nh->advertise<visualization_msgs::MarkerArray> ("structure_array", 0);
// 	ros::Publisher marker2_pub = nh->advertise<visualization_msgs::Marker> ("structure", 0);

// 	ros::ServiceServer fitService = nh->advertiseService("fit_models",
// 			structureFitModels);
// 	ros::ServiceServer selectServiceSpanningTree = nh->advertiseService("get_spanning_tree",
// 			structureSelectSpanningTree);
// 	ros::ServiceServer selectServiceFastGraph = nh->advertiseService("get_fast_graph",
// 			structureSelectFastGraph);
// 	ros::ServiceServer selectServiceGraph = nh->advertiseService("get_graph",
// 			structureSelectGraph);
// 	ros::ServiceServer selectServiceGraphAll = nh->advertiseService("get_graph_all",
// 			structureSelectGraphAll);
// 	ros::ServiceServer selectVisualizeGraph = nh->advertiseService("visualize_graph",
// 			visualizeGraph);

// 	ROS_INFO("Ready to fit articulation models and structure to articulated objects.");
// 	ros::spin();
// }
