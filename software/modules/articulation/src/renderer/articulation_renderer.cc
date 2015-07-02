// Articulation renderer

// lcm
#include <lcm/lcm-cpp.hpp>

// libbot/lcm includes
#include <bot_core/bot_core.h>
#include <bot_frames/bot_frames.h>
#include <bot_param/param_client.h>
#include <bot_param/param_util.h>
#include <bot_lcmgl_client/lcmgl.h>

// lcm messages
#include <lcmtypes/articulation.hpp> 

#include "LinearMath/btTransform.h"

#include "articulation_renderer.h"

// #include <articulation/factory.h>
#include <articulation/structs.h>
#include <articulation/utils.hpp>
#include <articulation/ArticulatedObject.hpp>
#include <articulation/structs.h>

// std includes
#include <iostream>
#include <iomanip>
#include <map>
#include <deque>
#include <sstream>

using namespace articulation;
using namespace articulation_models;

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


// std::deque<bot_core_rigid_transform_t*> tf_msgs;
struct ArticulationRenderer {
  BotRenderer renderer;

  BotGtkParamWidget *pw;
  BotViewer   *viewer;
  BotFrames *frames;
  BotParam   *server;
  
  lcm::LCM lcm;


  std::string articulation_frame;

  std::map<int, articulation::track_msg_t> track_msg_map;

  ArticulatedObject aobject;
  KinematicParams params;

  void on_track_msg (const lcm::ReceiveBuffer* rbuf, 
                     const std::string& chan,
                     const articulation::track_list_msg_t *msg);
    
  void on_articulated_object (const lcm::ReceiveBuffer* rbuf, 
                              const std::string& chan,
                              const articulation::articulated_object_msg_t *msg);


  void draw_link(const std::string& str, const articulation::pose_msg_t& from_pose,
                 const articulation::pose_msg_t& to_pose,
                 const articulation::pose_msg_t& relative =
                 transformToPose(btTransform::getIdentity()), 
                 double size=20, double r=0.4, double g=0.4, double b=0);

  void draw_link(const std::string& str,
                 const btTransform& from_pose,
                 const btTransform& to_pose,
                 const btTransform& relative =
                 btTransform::getIdentity(),
                 double size=20,
                 double r=0.4, double g=0.4, double b=0);
  void draw_tracks();
  void draw_model();
};

typedef std::map<std::string, double> MapValue;

MapValue build_param_map(const articulation::model_msg_t& msg) { 
    MapValue mv;
    for (int j=0; j<msg.num_params; j++) 
        mv[msg.params[j].name] = msg.params[j].value;
    return mv;
}

void ArticulationRenderer::on_track_msg (const lcm::ReceiveBuffer* rbuf, 
                                         const std::string& chan,
                                         const articulation::track_list_msg_t *msg) { 
  track_msg_map.clear();
  for (int j=0; j<msg->num_tracks; j++) {
    track_msg_map[msg->tracks[j].id] = msg->tracks[j];
  }
  bot_viewer_request_redraw(viewer);
}


void ArticulationRenderer::on_articulated_object (const lcm::ReceiveBuffer* rbuf, 
                                                  const std::string& chan,
                                                  const articulation::articulated_object_msg_t *msg) { 
    // read kin params
  // aobject = ArticulatedObject(params);
  // aobject.SetObjectModel(*msg);
    bot_viewer_request_redraw(viewer);
}



// static void 
// on_object_track_list (const lcm_recv_buf_t *rbuf, const char *channel,
// 		   const articulation_track_list_msg_t *track_list_msg, void *user_data )
// {
//     ArticulationRenderer *self = (ArticulationRenderer*) user_data;

//     if (self->track_list_msg)
//         articulation_track_list_msg_t_destroy(self->track_list_msg);
//     articulation_track_list_msg_t* msg = articulation_track_list_msg_t_copy(track_list_msg);    
//     bot_viewer_request_redraw(self->viewer);
// }


static void on_param_widget_changed (BotGtkParamWidget *pw, const char *name, void *user) {
    ArticulationRenderer *state = (ArticulationRenderer*) user;
    if (!&state->renderer)
    	return;
    bot_viewer_request_redraw(state->viewer);
}

static inline void
_matrix_vector_multiply_3x4_4d (const double m[12], const double v[4],
        double result[3])
{
    result[0] = m[0]*v[0] + m[1]*v[1] + m[2] *v[2] + m[3] *v[3];
    result[1] = m[4]*v[0] + m[5]*v[1] + m[6] *v[2] + m[7] *v[3];
    result[2] = m[8]*v[0] + m[9]*v[1] + m[10]*v[2] + m[11]*v[3];
}

static inline void
_matrix_transpose_4x4d (const double m[16], double result[16])
{
    result[0] = m[0];
    result[1] = m[4];
    result[2] = m[8];
    result[3] = m[12];
    result[4] = m[1];
    result[5] = m[5];
    result[6] = m[9];
    result[7] = m[13];
    result[8] = m[2];
    result[9] = m[6];
    result[10] = m[10];
    result[11] = m[14];
    result[12] = m[3];
    result[13] = m[7];
    result[14] = m[11];
    result[15] = m[15];
}

void draw_axis(float size, float opacity) { 
    glPointSize(2.f);
    glBegin(GL_POINTS);
    glColor4f(0.2, 0.2, 0.2, opacity);
    glVertex3f(0,0,0);
    glEnd();

    //x-axis
    glBegin(GL_LINES);
    glColor4f(1, 0, 0, 0.8 * opacity);
    glVertex3f(size, 0, 0);
    glVertex3f(0, 0, 0);
    glEnd();

    //y-axis
    glBegin(GL_LINES);
    glColor4f(0, 1, 0, 0.8 * opacity);
    glVertex3f(0, size, 0);
    glVertex3f(0, 0, 0);
    glEnd();

    //z-axis
    glBegin(GL_LINES);
    glColor4f(0, 0, 1, 0.8 * opacity);
    glVertex3f(0, 0, size);
    glVertex3f(0, 0, 0);
    glEnd();
}

btVector3 getOrtho(btVector3 a,btVector3 b) {
  btVector3 bb = b - a.dot(b)*a;
  bb.normalize();
  a.normalize();
  return a.cross(bb);
}

void ArticulationRenderer::draw_link(const std::string& str,
                                     const btTransform& from_pose,
                                     const btTransform& to_pose,
                                     const btTransform& relative,
                                     double size,
                                     double r, double g, double b) { 
  draw_link(str, transformToPose(from_pose), transformToPose(to_pose),
            transformToPose(relative), size, r, g, b);
}

void ArticulationRenderer::draw_link(const std::string& str,
                                     const articulation::pose_msg_t& from_pose,
                                     const articulation::pose_msg_t& to_pose,
                                     const articulation::pose_msg_t& relative,
                                     double size,
                                     double r, double g, double b) { 

    // Draw link bar
    glLineWidth(size);
    glColor4f(r, g, b, 0.6);
    glBegin(GL_LINES);
    glVertex3f(from_pose.pos[0], from_pose.pos[1], from_pose.pos[2]);
    glVertex3f(to_pose.pos[0], to_pose.pos[1], to_pose.pos[2]);
    glEnd();

    // Draw model name (in between)
    const double pos_local[3] = { 0.5 *from_pose.pos[0] + 0.5 * to_pose.pos[0], 
                                  0.5 *from_pose.pos[1] + 0.5 * to_pose.pos[1], 
                                  0.5 *from_pose.pos[2] + 0.5 * to_pose.pos[2] };
    glPushAttrib (GL_CURRENT_BIT | GL_ENABLE_BIT);
    glColor4f (.1, .1, .1, 1);
    bot_gl_draw_text (pos_local, GLUT_BITMAP_HELVETICA_12, str.c_str(),
                      BOT_GL_DRAW_TEXT_JUSTIFY_LEFT);
    // ,BOT_GL_DRAW_TEXT_DROP_SHADOW);
    glPopAttrib ();

}

void ArticulationRenderer::draw_tracks() { 

  ArticulationRenderer *state = (ArticulationRenderer*) this;
  
    float size = .02f; 

    for (std::map<int, articulation::track_msg_t>::iterator it = state->track_msg_map.begin(); 
         it != state->track_msg_map.end(); it++) { 

        const std::vector<articulation::pose_msg_t>& poses =  it->second.pose; 
        
        for (int k=0; k<poses.size(); k++) { 
            const articulation::pose_msg_t& pose_msg = poses[k];
            
            double feat_to_sensor_m[16];
            double feat_to_sensor_m_opengl[16];
            bot_quat_pos_to_matrix(pose_msg.orientation, pose_msg.pos, feat_to_sensor_m);
            bot_matrix_transpose_4x4d(feat_to_sensor_m, feat_to_sensor_m_opengl);

            if (k == poses.size() - 1) { 
                glLineWidth(4);
                size = 0.05f; 
            } else {
                glLineWidth(2);
                size = 0.02f;
            }

            float opacity = 0.3 + (k+1) * 1.f / poses.size();

            glPushMatrix(); // feat_to_sensor
            glMultMatrixd(feat_to_sensor_m_opengl);

            // Draw Axis
            draw_axis(size, opacity);

            // Draw Text
            if (k == poses.size() -1) { 
                // Draw Text
                char buf[256];
                const double pos_local[3] = {0,0,.10};
                sprintf (buf, "ID: %i", it->first);
                glPushAttrib (GL_CURRENT_BIT | GL_ENABLE_BIT);
                glColor4f (.1, .1, .1, 1);
                bot_gl_draw_text (pos_local, GLUT_BITMAP_HELVETICA_12, buf,
                                  BOT_GL_DRAW_TEXT_ANCHOR_TOP);

                glPopAttrib ();
            }

            glPopMatrix(); // feat_to_sensor

        }
    }
}

void ArticulationRenderer::draw_model() {
    ArticulationRenderer *state = (ArticulationRenderer*) this;
    
    // // Draw DOFs
    // const int num_parts = state->articulated_obj_msg.num_parts;
    // for (int j=0; j<state->articulated_obj_msg.num_models; j++) {

    //   const articulation::model_msg_t& model = state->articulated_obj_msg.models[j]; 
    //   MapValue model_map = build_param_map(model);
      
    //   int from = model.id / num_parts;
    //   int to = model.id % num_parts;

    //   std::stringstream modelss; 
    //   modelss << model.name << " (" << from << "->" << to << ")";

    //   const pose_msg_t& pose_from = state->articulated_obj_msg.parts[from].pose.back();
    //   const pose_msg_t& pose_to = state->articulated_obj_msg.parts[to].pose.back();

    //   if (model.name == "rigid") {
    //       draw_link(modelss.str(), pose_from, pose_to);
    //   } else if(model.getModelName()=="prismatic") {
    //     model_map["q_min[0]"]
    //         // pose_msg_t origin = model->predictPose(model->predictConfiguration( transformToPose(btTransform::getIdentity() )));
    //         // btTransform pose_orthogonal = poseToTransform(pose_from)*poseToTransform(origin);

    //         // btTransform pose_max = poseToTransform(pose_from)*
    //         //     poseToTransform(model->predictPose(model->getMinConfigurationObserved()));
    //         // btTransform pose_min = poseToTransform(pose_from)*
    //         //     poseToTransform(model->predictPose(model->getMaxConfigurationObserved()));
    //   }      
    // }


    // for(KinematicGraph::iterator i= state->aobject.currentGraph.begin();
    //     i!=state->aobject.currentGraph.end(); i++) {
    //     int from = i->first.first;
    //     int to = i->first.second;
    //     articulation_models::GenericModelPtr &model = i->second;
    //     const pose_msg_t& pose_from = state->aobject.object_msg.parts[from].pose.back();
    //     const pose_msg_t& pose_to = state->aobject.object_msg.parts[to].pose.back();

    //     std::stringstream modelss; 
    //     modelss << model->getModelName()
    //             << " (" << from << "->" << to << ")";

    //     if(model->getModelName()=="rigid") {
    //       // draw_link(modelss.str(), pose_from, pose_to);
    //     } else if(model->getModelName()=="prismatic") {
    //       pose_msg_t origin = model->predictPose(model->predictConfiguration( transformToPose(btTransform::getIdentity() )));
    //       btTransform pose_orthogonal = poseToTransform(pose_from)*poseToTransform(origin);

    //       btTransform pose_max = poseToTransform(pose_from)*
    //             poseToTransform(model->predictPose(model->getMinConfigurationObserved()));
    //         btTransform pose_min = poseToTransform(pose_from)*
    //             poseToTransform(model->predictPose(model->getMaxConfigurationObserved()));


    //         std::cerr << "minobs: " << model->getMinConfigurationObserved() << std::endl;
    //         std::cerr << "maxobs: " << model->getMinConfigurationObserved() << std::endl;
    //         std::cerr << "pose_min: " << pose_min << " max: " << pose_max << std::endl;
            
    //         // // draw tics
    //         // for(double q = model->getMinConfigurationObserved()[0];
    //         //     q<model->getMaxConfigurationObserved()[0];
    //         //     q += 0.05) {
    //         //   V_Configuration qq(1);
    //         //   qq(0) = q;
    //         //   btTransform pose_q = poseToTransform(pose_from)*
    //         //       poseToTransform(model->predictPose(qq));

    //         //   btVector3 dir = poseToTransform(pose_from) *
    //         //       boost::static_pointer_cast<PrismaticModel>(model)->prismatic_dir
    //         //       - poseToTransform(pose_from) * btVector3(0,0,0);
    //         //   dir.normalize();

    //         //   draw_link("decoration.poses",
    //         //             btTransform(btQuaternion(0,0,0,1),pose_q.getOrigin()), 
    //         //             btTransform(btQuaternion(0,0,0,1),pose_q.getOrigin()
    //         //                         + getOrtho(dir,btVector3(0,0,1))*0.05),
    //         //             btTransform(btQuaternion(0,0,0,1),pose_q.getOrigin()), 
    //         //             5,0.3,0.0,0.3);
              
    //         //   //   AddMarkerLine("decoration.poses",
    //         //   //                 pose_q.getOrigin(),
    //         //   //                 pose_q.getOrigin() + getOrtho(dir,tf::Vector3(0,0,1))*0.005,
    //         //   //                 0.0025,0.0,0.0,0.3,pose_q.getOrigin());
    //         //   //   AddMarkerLine("decoration.poses",
    //         //   //                 pose_q.getOrigin() ,
    //         //   //                 pose_q.getOrigin() + getOrtho(dir,tf::Vector3(0,0,1))*(-0.005),
    //         //   //                 0.0025,0.0,0.0,0.3,pose_q.getOrigin());
    //         // }
            
    //         // Link
    //         draw_link(modelss.str(), pose_from, pose_to);

    //         // Extent of motion
    //         draw_link("pris", transformToPose(pose_min),
    //                   transformToPose(pose_max),
    //                   transformToPose(btTransform::getIdentity()),30,0.3,0,0.3);
    //     }
    // }
      #if 0
        else if(model->getModelName()=="rotational") {
            pose_msg_t rot_center = 
                transformToPose(poseToTransform(pose_from)*
                                btTransform(boost::static_pointer_cast<RotationalModel>(model)->rot_axis,
                                            boost::static_pointer_cast<RotationalModel>(model)->rot_center)
                                );
            draw_link(modelss.str(),pose_from,rot_center); // ,0.01,0.666,1.0,1.0);
            draw_link(modelss.str(),rot_center,pose_to); // ,0.01,0.666,1.0,1.0);
            draw_link("",rot_center,pose_to,
                      transformToPose(btTransform::getIdentity()),20,0.3,0.,0.3); // ,0.666,1.0,1.0);

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
            double radius = model->getParam("rot_radius");
            for(double q=0;q<2*M_PI+STEPSIZE;q+=STEPSIZE) {
                bool in_range = q>=Qmin[0] && q<=Qmax[0];
                if(Qmin[0]<0) in_range = q>=(Qmin[0]+2*M_PI) || q<=Qmax[0];
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

            // rotation axis:
            draw_link("rotation_axis",
                      rot_center,
                      transformToPose(poseToTransform(rot_center) *
                                      btTransform(btQuaternion(0,0,0,1),btVector3(0,0,-0.10))),
                                      rot_center);
            draw_link("rotation_axis",
                      rot_center,
                      transformToPose(poseToTransform(rot_center) *
                                      btTransform(btQuaternion(0,0,0,1),btVector3(0,0,+0.10))),
                                      rot_center);            


        }

        // MapValue map = build_param_map(state->articulated_obj_msg.models[j]);
        // if (map.find("rot_center.x") != map.end() && 
        //     map.find("rot_center.y") != map.end() && 
        //     map.find("rot_center.z") != map.end() && 
        //     map.find("rot_orientation.x") != map.end() && 
        //     map.find("rot_orientation.y") != map.end() && 
        //     map.find("rot_orientation.z") != map.end() && 
        //     map.find("rot_orientation.w") != map.end() && 
        //     map.find("rot_orientation.w") != map.end()) { 

        //     double pos[3];
        //     pos[0] = map["rot_center.x"];
        //     pos[1] = map["rot_center.y"];
        //     pos[2] = map["rot_center.z"];
                
        //     double orientation[4];
        //     orientation[0] = map["rot_orientation.w"];
        //     orientation[1] = map["rot_orientation.x"];
        //     orientation[2] = map["rot_orientation.y"];
        //     orientation[3] = map["rot_orientation.z"];

        //     double rot_m[16];
        //     double rot_m_opengl[16];
        //     bot_quat_pos_to_matrix(orientation, pos, rot_m);
        //     bot_matrix_transpose_4x4d(rot_m, rot_m_opengl);

        //     glPushMatrix(); 
        //     glMultMatrixd(rot_m_opengl);

        //     size = .2f; 
        //     glPointSize(12.f);
        //     glBegin(GL_POINTS);
        //     glColor4f(0.2, 0.2, 0.2, 1);
        //     glVertex3f(0,0,0);
        //     glEnd();

        //     glLineWidth(24.f);

        //     //z-axis
        //     glBegin(GL_LINES);
        //     glColor4f(0, .6, .6, 0.5 );
        //     glVertex3f(0, 0, size);
        //     glVertex3f(0, 0, 0.01);
        //     glEnd();
                
        //     glPopMatrix();
                    
        // }

        // if (map.find("prismatic_dir.x") != map.end() && 
        //     map.find("prismatic_dir.y") != map.end() && 
        //     map.find("prismatic_dir.z") != map.end() && 
        //     map.find("rigid_position.x") != map.end() && 
        //     map.find("rigid_position.y") != map.end() && 
        //     map.find("rigid_position.z") != map.end()) { 

        //     double pos[3];
        //     pos[0] = map["rigid_position.x"];
        //     pos[1] = map["rigid_position.y"];
        //     pos[2] = map["rigid_position.z"];

        //     double dpos[3];
        //     dpos[0] = map["prismatic_dir.x"];
        //     dpos[1] = map["prismatic_dir.y"];
        //     dpos[2] = map["prismatic_dir.z"];

        //     glBegin(GL_LINES);
        //     glColor4f(0, .6, .6, .5);
        //     glVertex3f(pos[0],pos[1],pos[2]);
        //     glVertex3f(pos[0]+dpos[0]*size,pos[1]+dpos[1]*size,pos[2]+dpos[2]*size);
        //     glEnd();
                
        // }
            

        // if (map.find("rot_radius") != map.end() && (state->articulated_obj_msg.models[j].name == "rotational") ) 
        //     modelss << " radius:" << map["rot_radius"] << " m";
        // if (map.find("q_min[0]") != map.end() && map.find("q_max[0]") != map.end() && 
        //     (state->articulated_obj_msg.models[j].name ==  "prismatic") ) { 
        //     modelss << " len:" << map["q_max[0]"] - map["q_min[0]"] << " m";
        // }
            

        // const articulation::track_msg_t& from_track_msg = state->track_msg_map[from_id];
        // const articulation::track_msg_t& to_track_msg = state->track_msg_map[to_id];

        // if (!from_track_msg.num_poses || !to_track_msg.num_poses) 
        //     continue;

        // const articulation::pose_msg_t& from_pose = from_track_msg.pose[from_track_msg.num_poses-1];
        // const articulation::pose_msg_t& to_pose = to_track_msg.pose[to_track_msg.num_poses-1];

        // // Draw the link and the text
        // draw_link(from_pose, to_pose);
    }
#endif


}

static void _draw(BotViewer *viewer, BotRenderer *renderer)
{

    ArticulationRenderer *state = (ArticulationRenderer*) renderer->user;
    
    // Return if nothing to draw
    if(!state->track_msg_map.size()) return;

    // ---------------------- 
    // Setup transformation to sensor
    // ---------------------- 
    glPushMatrix();
    if (state->frames==NULL || 
        !bot_frames_have_trans(state->frames, 
                               state->articulation_frame.c_str(), 
                               bot_frames_get_root_name(state->frames))){
      // rotate so that X is forward and Z is up
      glRotatef(-90, 1, 0, 0);
    }
    else{
      //project to current frame
      double sensor_to_local_m[16];
      bot_frames_get_trans_mat_4x4(state->frames, state->articulation_frame.c_str(), 
                                   bot_frames_get_root_name(state->frames),
                                   sensor_to_local_m);

      // opengl expects column-major matrices
      double sensor_to_local_m_opengl[16];
      bot_matrix_transpose_4x4d(sensor_to_local_m, sensor_to_local_m_opengl);
      glMultMatrixd(sensor_to_local_m_opengl);
    }
    

    // ---------------------- 
    // Depth test
    // ---------------------- 
    glEnable(GL_DEPTH_TEST);

    // ---------------------- 
    // Blending
    // ---------------------- 
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);


    // ---------------------- 
    // Draw Tracks
    // ---------------------- 
    state->draw_tracks();

    // ---------------------- 
    // Draw model params
    // ----------------------
    state->draw_model();

    // if (self->show_projected_poses) { 
       
    //     // Draw Projected Poses
    //     for (int j=0; j<self->obj_msg->num_parts; j++) { 

    //         articulation_object_pose_track_msg_t& track_msg = self->obj_msg->parts[j];
                
    //         for (int k=0; k<track_msg.num_poses_projected; k++) { 
    //             articulation_object_pose_msg_t& pose_msg = track_msg.pose_projected[k];
                    
    //             double feat_to_sensor_m[16];
    //             double feat_to_sensor_m_opengl[16];

    //             bot_quat_pos_to_matrix(pose_msg.orientation, pose_msg.pos, feat_to_sensor_m);
    //             bot_matrix_transpose_4x4d(feat_to_sensor_m, feat_to_sensor_m_opengl);

    //             float opacity = (k+1) * 1.f / track_msg.num_poses_projected;

    //             glPushMatrix(); // feat_to_sensor
    //             glMultMatrixd(feat_to_sensor_m_opengl);
                    
    //             glPointSize(6.f);
    //             glBegin(GL_POINTS);
    //             glColor4f(0.2, 0.2, 0.2, opacity);
    //             glVertex3f(0,0,0);
    //             glEnd();

    //             //x-axis
    //             glBegin(GL_LINES);
    //             glColor4f(1, 0, 0, 0.5 * opacity);
    //             glVertex3f(size, 0, 0);
    //             glVertex3f(0.01, 0, 0);
    //             glEnd();

    //             //y-axis
    //             glBegin(GL_LINES);
    //             glColor4f(0, 1, 0, 0.5 * opacity);
    //             glVertex3f(0, size, 0);
    //             glVertex3f(0, 0.01, 0);
    //             glEnd();

    //             //z-axis
    //             glBegin(GL_LINES);
    //             glColor4f(0, 0, 1, 0.5 * opacity);
    //             glVertex3f(0, 0, size);
    //             glVertex3f(0, 0, 0.01);
    //             glEnd();

    //             glPopMatrix(); // feat_to_sensor
    //         }
    //     }
    // }

    // if (self->show_visual_cues) { 





    // // ---------------------- 
    // // Draw model params
    // // ---------------------- 
    // char str[1024];
    // for (int j=0; j<self->obj_msg->num_params; j++) 
    //     sprintf(str, "%s: %f\n",self->obj_msg->params[j].name,self->obj_msg->params[j].value);;

    // GLdouble model_matrix[16];
    // GLdouble proj_matrix[16];
    // GLint viewport[4];

    // glGetDoublev (GL_MODELVIEW_MATRIX, model_matrix);
    // glGetDoublev (GL_PROJECTION_MATRIX, proj_matrix);
    // glGetIntegerv (GL_VIEWPORT, viewport);

    // // Render the current robot status
    // glMatrixMode(GL_PROJECTION);
    // glPushMatrix();
    // glLoadIdentity();
    // gluOrtho2D(0, viewport[2], 0, viewport[3]);

    // glMatrixMode(GL_MODELVIEW);
    // glPushMatrix();
    // glLoadIdentity();

    // double state_xyz[] = {50, 90, 100};
    // bot_gl_draw_text(state_xyz, NULL, str,
    //                  BOT_GL_DRAW_TEXT_JUSTIFY_CENTER |
    //                  BOT_GL_DRAW_TEXT_ANCHOR_VCENTER |
    //                  BOT_GL_DRAW_TEXT_ANCHOR_HCENTER |
    //                  BOT_GL_DRAW_TEXT_DROP_SHADOW);


    // glMatrixMode(GL_PROJECTION);
    // glPopMatrix();
    // glMatrixMode(GL_MODELVIEW);
    // glPopMatrix();


    // Disable blending 
    glDisable(GL_BLEND);

    // Disable depth test
    glDisable(GL_DEPTH_TEST);

    // Pop sensor_to_local transformation
    glPopMatrix(); 

  return;
}

static void _free(BotRenderer *renderer)
{
    ArticulationRenderer *state = (ArticulationRenderer*) renderer;
    free(state);
}


void get_params(ArticulationRenderer* state) { 

    state->params.sigma_position = 
        bot_param_get_double_or_fail(state->server,
                                     "articulation_structure_learner.sigma_position");
    state->params.sigma_orientation = 
        bot_param_get_double_or_fail(state->server,
                                     "articulation_structure_learner.sigma_orientation");
    state->params.sigmax_position = 
        bot_param_get_double_or_fail(state->server,
                                     "articulation_structure_learner.sigmax_position");
    state->params.sigmax_orientation = 
        bot_param_get_double_or_fail(state->server,
                                     "articulation_structure_learner.sigmax_orientation");
    state->params.eval_every = 
        bot_param_get_int_or_fail(state->server,
                                     "articulation_structure_learner.eval_every");
    state->params.eval_every_power = 
        bot_param_get_double_or_fail(state->server,
                                     "articulation_structure_learner.eval_every_power");
    state->params.supress_similar = 
        bot_param_get_boolean_or_fail(state->server,
                                     "articulation_structure_learner.supress_similar");
    state->params.reuse_model = 
        bot_param_get_boolean_or_fail(state->server,
                                     "articulation_structure_learner.reuse_model");

    char* restricted_graphs_str = 
        bot_param_get_str_or_fail(state->server,
                                  "articulation_structure_learner.restricted_graphs");
    state->params.restricted_graphs = std::string(restricted_graphs_str);

    state->params.restrict_graphs = 
        bot_param_get_boolean_or_fail(state->server,
                                     "articulation_structure_learner.restrict_graphs");
    state->params.reduce_dofs = 
        bot_param_get_boolean_or_fail(state->server,
                                     "articulation_structure_learner.reduce_dofs");
    state->params.search_all_models = 
        bot_param_get_boolean_or_fail(state->server,
                                     "articulation_structure_learner.search_all_models");

    char* full_eval_str = 
        bot_param_get_str_or_fail(state->server,
                                  "articulation_structure_learner.full_eval");
    state->params.full_eval = std::string(full_eval_str);

    return;
}

void 
articulation_add_renderer_to_viewer(BotViewer* viewer, int priority, lcm_t* lcm, BotFrames * frames, const char * articulation_frame)
{
    ArticulationRenderer* state = new ArticulationRenderer();

    state->frames = frames;
    if (state->frames!=NULL)
        state->articulation_frame = std::string(articulation_frame);
    state->server = bot_param_new_from_server(lcm, 1);
    state->lcm = lcm::LCM(lcm);
    
    BotRenderer *renderer = &state->renderer;

    state->viewer = viewer;
    state->pw = BOT_GTK_PARAM_WIDGET(bot_gtk_param_widget_new());

    renderer->draw = _draw;
    renderer->destroy = _free;
    renderer->name = (char*)"Articulation Renderer";
    renderer->widget = GTK_WIDGET(state->pw);
    renderer->enabled = 1;
    renderer->user = state;

    // Load params
    state->params.LoadParams(state->server);
    // state->obj_msg = NULL;

    // bot_gtk_param_widget_add_booleans(state->pw, 
    //                                   BOT_GTK_PARAM_WIDGET_CHECKBOX, 
    //                                   PARAM_NAME_CLOUD_SHOW, 0, NULL);

    g_signal_connect (G_OBJECT (state->pw), "changed",
                      G_CALLBACK (on_param_widget_changed), state);

    state->lcm.subscribe("ARTICULATION_OBJECT_TRACKS", &ArticulationRenderer::on_track_msg, state);
    state->lcm.subscribe("ARTICULATED_OBJECT", &ArticulationRenderer::on_articulated_object, state);

    // // articulation_track_list_msg_t_subscribe(state->lcm, "POSE_TRACK", on_object_track_list, state);
    // // articulation_track_list_msg_t_subscribe(state->lcm, "ARTICULATION_TRACK_LIST", on_object_track_list, state);


    bot_viewer_add_renderer(viewer, renderer, priority);
}
