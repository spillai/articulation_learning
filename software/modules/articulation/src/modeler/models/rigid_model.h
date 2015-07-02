/*
 * rigid_model.h
 *
 *  Created on: Oct 21, 2009
 *      Author: sturm
 */

#ifndef RIGID_MODEL_H_
#define RIGID_MODEL_H_

#include "generic_model.h"

namespace articulation_models {

class RigidModel: public GenericModel {
public:
	btVector3 rigid_position;
	btQuaternion rigid_orientation;
	double rigid_width,rigid_height;

	RigidModel();

	// -- params
	void readParamsFromModel();
	void writeParamsToModel();

	size_t getDOFs() { return 0; }

        // articulation_object_pose_msg_t predictPose(V_Configuration q);
        articulation::pose_msg_t predictPose(V_Configuration q);
	void projectConfigurationToChannels();

	bool guessParameters();
	void updateParameters(std::vector<double> delta);
};

}

#endif /* RIGID_MODEL_H_ */
