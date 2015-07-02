/*
 * utils.hpp
 *
 *  Created on: Oct 21, 2009
 *      Author: sturm
 */

#ifndef ARTICULATION_MODELS_UTILS_HPP_
#define ARTICULATION_MODELS_UTILS_HPP_

#include "LinearMath/btTransform.h"
#include <lcmtypes/articulation.hpp>
/* #include <lcmtypes/articulation_object_pose_msg_t.h> */
/* #include <lcmtypes/articulation_object_pose_track_msg_t.h> */
/* #include <lcmtypes/articulation_articulated_object_msg_t.h> */
#include <Eigen/Core>
#include <vector>
#include <map>
#include <sys/time.h>
#include <boost/shared_ptr.hpp>

typedef boost::shared_ptr<articulation::pose_msg_t> pose_msg_t_ptr;
typedef boost::shared_ptr<articulation::pose_msg_t const> pose_msg_t_constptr;
typedef boost::shared_ptr<articulation::track_msg_t> track_msg_t_ptr;
typedef boost::shared_ptr<articulation::track_msg_t const> track_msg_t_constptr;


namespace articulation_models {

#define DEBUG 1

#ifndef SQR
#define SQR(a) ((a)*(a))
#endif

#ifndef MIN
#define MIN(a,b) ((a<=b)?(a):(b))
#endif

#ifndef MAX
#define MAX(a,b) ((a>=b)?(a):(b))
#endif

#define PRINT_TRANSFORM(tf) \
	"[ "<<tf.getOrigin().x() <<"; "<<tf.getOrigin().y() <<"; "<<tf.getOrigin().z() <<"]" << \
	"( "<<tf.getRotation().x() <<"; "<<tf.getRotation().y() <<"; "<<tf.getRotation().z()<<"; "<<tf.getRotation().w() <<") "

typedef Eigen::MatrixXd M_CartesianJacobian;
typedef Eigen::VectorXd V_Configuration;

// ************************************************************** // 
// Generic utility functions
// ************************************************************** // 

static int64_t
timestamp_us (void)
{
    struct timeval tv;
    gettimeofday (&tv, NULL);
    return (int64_t) tv.tv_sec * 1000000 + tv.tv_usec;
}

// ************************************************************** // 
// Structures for model learning
// ************************************************************** // 
    
// struct Quaternion { 
//     double x, y, z, w;
// Quaternion() : w(1), x(0), y(0), z(0) { }
// };

// struct Point { 
//     double x, y, z;
// Point() : x(0), y(0), z(0) { }
//     Point(double* pos) { 
//         x = pos[0], 
//             y = pos[1], 
//             z = pos[2];
//     }
// };

// struct Pose { 
//     Quaternion orientation;
//     Point pos;
//     Pose() {};
//     /* Pose(const articulation_object_pose_msg_t& msg) {  */
//     /*     orientation.w = msg.orientation[0]; */
//     /*     orientation.x = msg.orientation[1]; */
//     /*     orientation.y = msg.orientation[2]; */
//     /*     orientation.z = msg.orientation[3]; */

//     /*     pos.x = msg.pos[0]; */
//     /*     pos.y = msg.pos[1]; */
//     /*     pos.z = msg.pos[2]; */
//     /* } */
//     Pose(const double* _pos, const double* _orientation) { 
//         orientation.w = _orientation[0];
//         orientation.x = _orientation[1];
//         orientation.y = _orientation[2];
//         orientation.z = _orientation[3];

//         pos.x = _pos[0];
//         pos.y = _pos[1];
//         pos.z = _pos[2];
//     }
// };
// typedef articulation::pose_msg_t Pose;
//     typedef boost::shared_ptr<Pose > PosePtr;
// typedef boost::shared_ptr<Pose const> PoseConstPtr;

// struct ObjectPose { 
//     int64_t utime;
//     int64_t id;
//     Pose pose;
     
// ObjectPose() : id(-1), utime(0) {}
//     ObjectPose(const articulation::pose_msg_t& msg) { 
//         utime = msg.utime;
//         id = msg.id;
//         pose = Pose(msg.pos, msg.orientation);
//     }
//     ObjectPose(const articulation::pose_msg_t* msg) { 
//         utime = msg->utime;
//         id = msg->id;
//         pose = Pose(msg->pos, msg->orientation);
//     }
// };
    // typedef articulation::pose_msg_t ObjectPose;
    // typedef boost::shared_ptr<ObjectPose > ObjectPosePtr;
    // typedef boost::shared_ptr<ObjectPose const> ObjectPoseConstPtr;

// struct ModelParam { 
//     std::string name;
//     float value;
//     int type;

//     enum { PRIOR = 0, 
//            PARAM = 1, 
//            EVAL = 2
//     };

//     ModelParam() {}
//     ModelParam(const articulation::model_param_msg_t* msg) { 
//         name = std::string(msg->name);
//         value = msg->value;
//         type = msg->type;
//     }
// };
// typedef boost::shared_ptr<ModelParam > ModelParamPtr;
// typedef boost::shared_ptr<ModelParam const> ModelParamConstPtr;

// struct ChannelFloat32 { 
//     std::string name;
//     std::vector<float> values;
// };

// struct ObjectPoseTrack { 
//     int64_t id;
//     std::vector<ObjectPose> pose;
//     std::vector<int32_t> pose_flags;
//     std::vector<ObjectPose> pose_projected;
//     std::vector<ObjectPose> pose_resampled;
//     std::vector<ChannelFloat32> channels;

//     enum { POSE_VISIBLE = 1, 
//            POSE_END_OF_SEGMENT = 2
//     };
     
//     ObjectPoseTrack() {}
//     ObjectPoseTrack(const articulation::track_msg_t* msg) { 
//         id = msg->id;

//         if (msg->num_poses) { 
//             assert(msg->num_poses == msg->pose.size());
//             pose = std::vector<ObjectPose>(msg->num_poses);
//             for (int j=0; j<msg->num_poses; j++) 
//                 pose[j] = ObjectPose(msg->pose[j]);

//             pose_flags = std::vector<int32_t>(msg->num_poses);
//             for (int j=0; j<msg->num_poses; j++) 
//                 pose_flags[j] = msg->pose_flags[j];
//         }

//         if (msg->num_poses_projected) { 
//             pose_projected = std::vector<ObjectPose>(msg->num_poses_projected);
//             for (int j=0; j<msg->num_poses_projected; j++) 
//                 pose_projected[j] = ObjectPose(msg->pose_projected[j]);
//         }

//         if (msg->num_poses_resampled) { 
//             pose_resampled = std::vector<ObjectPose>(msg->num_poses_resampled);
//             for (int j=0; j<msg->num_poses_resampled; j++) 
//                 pose_resampled[j] = ObjectPose(msg->pose_resampled[j]);
//         }

         
//     }
// };
// typedef boost::shared_ptr<ObjectPoseTrack > ObjectPoseTrackPtr;
// typedef boost::shared_ptr<ObjectPoseTrack const> ObjectPoseTrackConstPtr;

// struct ArticulationModel { 
//     int id;
//     std::string name;
//     std::vector<ModelParam> params;
//     ObjectPoseTrack track;

// ArticulationModel() : id(-1) {}
//     ArticulationModel(const articulation::model_msg_t* msg) { 
//         id = msg->id;
//         name = std::string(msg->name);

//         if (msg->num_params) { 
//             params = std::vector<ModelParam>(msg->num_params);
//             for (int j=0; j<msg->num_params; j++) 
//                 params[j] = ModelParam(&msg->params[j]);
//         }

//         // track = ObjectPoseTrack(&msg->track);        
//     }
// };
// typedef boost::shared_ptr<ArticulationModel > ArticulationModelPtr;
// typedef boost::shared_ptr<ArticulationModel const> ArticulationModelConstPtr;

// struct ArticulatedObject { 
//     std::vector<ObjectPoseTrack> parts;
//     std::vector<ModelParam> params;
//     std::vector<ArticulationModel> models;
//     std::vector<ObjectPose> markers;

//     ArticulatedObject() {}
//     ArticulatedObject(const articulation::articulated_object_msg_t* msg) { 
//         if (msg->num_parts) { 
//             parts = std::vector<ObjectPoseTrack>(msg->num_parts);
//             for (int j=0; j<msg->num_parts; j++) 
//                 parts[j] = ObjectPoseTrack(&msg->parts[j]);
//         }

//         if (msg->num_params) { 
//             params = std::vector<ModelParam>(msg->num_params);
//             for (int j=0; j<msg->num_params; j++) 
//                 params[j] = ModelParam(&msg->params[j]);
//         }

//         if (msg->num_models) { 
//             models = std::vector<ArticulationModel>(msg->num_models);
//             for (int j=0; j<msg->num_models; j++) 
//                 models[j] = ArticulationModel(&msg->models[j]);
//         }

//         if (msg->num_markers) { 
//             markers = std::vector<ObjectPose>(msg->num_markers);
//             for (int j=0; j<msg->num_markers; j++) 
//                 markers[j] = ObjectPose(&msg->markers[j]);
//         }
//     }
// };
// typedef boost::shared_ptr<ArticulatedObject > ArticulatedObjectPtr;
// typedef boost::shared_ptr<ArticulatedObject const> ArticulatedObjectConstPtr;

// inline btQuaternion orientationToQuaternion(double orientation[]) {
// 	return btQuaternion(orientation[1],orientation[2],
//                             orientation[3],orientation[0]);
// }

inline btVector3 positionToVector(double position[]) {
	return btVector3(position[0],position[1],position[2]);
}

 inline double l2_dist(double* p1, double* p2) { 
     return sqrt((p1[0]-p2[0])*(p1[0]-p2[0]) + (p1[1]-p2[1])*(p1[1]-p2[1]) + (p1[2]-p2[2])*(p1[2]-p2[2]));
 }
 // inline double l2_dist(Point& p1, Point& p2) { 
 //     return sqrt((p1.x-p2.x)*(p1.x-p2.x) + (p1.y-p2.y)*(p1.y-p2.y) + (p1.z-p2.z)*(p1.z-p2.z));
 // }

    inline btTransform poseToTransform(articulation::pose_msg_t pose) {
        // Careful btTransform(x,y,z,w)
    return(btTransform(btQuaternion(pose.orientation[1], pose.orientation[2], 
                                    pose.orientation[3], pose.orientation[0]), 
                       btVector3(pose.pos[0], pose.pos[1], pose.pos[2])));
}

    inline void quaternionToOrientation(btQuaternion quat, double* q) {
     q[1] = quat.x();
     q[2] = quat.y();
     q[3] = quat.z();
     q[0] = quat.w();
     return;
 }

    inline void vectorToPosition(btVector3 point, double* p) {
     p[0] = point.x();
     p[1] = point.y();
     p[2] = point.z();
     return;
}

 inline articulation::pose_msg_t transformToPose(btTransform transform) {
     articulation::pose_msg_t p;
     p.id = 0;
     p.utime = 0;
     quaternionToOrientation( transform.getRotation(), p.orientation);
     vectorToPosition( transform.getOrigin(), p.pos );
     return p;
}

    int openChannel(articulation::track_msg_t& track, std::string name, bool autocreate);

 // articulation_msgs::TrackMsg flipTrack(articulation_msgs::TrackMsg input, int corner=0);

 Eigen::VectorXd pointToEigen(double p[]);
    void eigenToPoint(Eigen::VectorXd v, double* );
 void setParamIfNotDefined(std::vector<articulation::model_param_msg_t> &vec,
                           std::string name, double value, uint8_t type=articulation::model_param_msg_t::PRIOR);
void setParam(std::vector<articulation::model_param_msg_t> &vec,
		std::string name, double value, uint8_t type=articulation::model_param_msg_t::PRIOR);
 // void setParamIfNotDefined(std::vector<articulation::model_param_msg_t> &vec,
 //                           std::string name, double value, 
 //                           uint8_t type=articulation::model_param_msg_t::PRIOR);
 // void setParam(std::vector<articulation::model_param_msg_t> &vec,
 //               std::string name, double value, 
 //               uint8_t type=articulation::model_param_msg_t::PRIOR);
    double getParam(std::vector<articulation::model_param_msg_t> &vec,
		std::string name);
bool hasParam(std::vector<articulation::model_param_msg_t> &vec,
		std::string name);
Eigen::VectorXd vectorToEigen(V_Configuration q);
Eigen::MatrixXd matrixToEigen(M_CartesianJacobian J);


bool check_values(const btVector3 &vec);
bool check_values(const btQuaternion &vec);
bool check_values(double v);
bool check_values(float v);
double getBIC(double loglh, size_t k, size_t n);
btMatrix3x3 RPY_to_MAT(double roll, double pitch, double yaw);
void MAT_to_RPY(const btMatrix3x3& mat, double &roll, double &pitch, double &yaw);
}

#endif /* ARTICULATION_MODELS_UTILS_HPP_ */
