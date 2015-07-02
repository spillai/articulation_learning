/*
 * rigid_model.cpp
 *
 *  Created on: Oct 21, 2009
 *      Author: sturm
 */

#include <articulation/rigid_model.h>
#include <articulation/utils.hpp>


using namespace std;

// using namespace articulation_msgs;

namespace articulation_models {

void readParamsFromModel();
void writeParamsToModel();

RigidModel::RigidModel():GenericModel() {
	rigid_position = btVector3(0,0,0);
	rigid_orientation = btQuaternion(0,0,0,1);
	rigid_width = 1;
	rigid_height = 1;

	complexity = 6;
}

void RigidModel::readParamsFromModel() {
	GenericModel::readParamsFromModel();
	getParam("rigid_position",rigid_position);
	getParam("rigid_orientation",rigid_orientation);
	getParam("rigid_width",rigid_width);
	getParam("rigid_height",rigid_height);
}

void RigidModel::writeParamsToModel() {
	GenericModel::writeParamsToModel();
	setParam("rigid_position",rigid_position,articulation::model_param_msg_t::PARAM);
	setParam("rigid_orientation",rigid_orientation,articulation::model_param_msg_t::PARAM);
	setParam("rigid_width",rigid_width,articulation::model_param_msg_t::PARAM);
	setParam("rigid_height",rigid_height,articulation::model_param_msg_t::PARAM);
}

// articulation_object_pose_msg_t predictPose(V_Configuration q) { 
//     articulation_models::Pose p = predictPose(q);
//     articulation_object_pose_msg_t msg;
//     msg.pos[0] = p.pos.x;
//     msg.pos[1] = p.pos.y;
//     msg.pos[2] = p.pos.z;

//     msg.orientation[0] = p.orientation.w;
//     msg.orientation[1] = p.orientation.x;
//     msg.orientation[2] = p.orientation.y;
//     msg.orientation[3] = p.orientation.z;

//     msg.utime = 0;
//     msg.id = -1;
//     return msg;
// }

articulation::pose_msg_t RigidModel::predictPose(V_Configuration q) {
    articulation::pose_msg_t objpose = transformToPose( btTransform(rigid_orientation,rigid_position) );
    objpose.id = -1;
    objpose.utime = 0;
    return objpose;
}

void RigidModel::projectConfigurationToChannels() {
    int ch_w = openChannel("width",false);
    int ch_h = openChannel("height",false);

    size_t n = model.track.pose.size();
    for(size_t i=0;i<n;i++) {
        if(ch_w>=0) model.track.channels[ch_w].values[i] = rigid_width;
        if(ch_h>=0) model.track.channels[ch_h].values[i] = rigid_height;
    }
}

bool RigidModel::guessParameters() {
    if(model.track.pose.size() == 0)
        return false;

    size_t i = rand() % getSamples();
    
    btTransform pose = poseToTransform(model.track.pose[i]);

    rigid_position = pose.getOrigin();
    rigid_orientation = pose.getRotation();
    return true;
}

void RigidModel::updateParameters(std::vector<double> delta) {
	rigid_position = rigid_position + btVector3(delta[0],delta[1],delta[2]);
	btQuaternion q;
	q.setEuler(delta[3],delta[4],delta[5]);
	rigid_orientation = rigid_orientation * q;
}

}
