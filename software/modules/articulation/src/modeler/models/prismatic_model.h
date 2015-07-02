/*
 * prismatic_model.h
 *
 *  Created on: Oct 21, 2009
 *      Author: sturm
 */

#ifndef PRISMATIC_MODEL_H_
#define PRISMATIC_MODEL_H_

#include "rigid_model.h"

namespace articulation_models {

class PrismaticModel: public RigidModel {
public:
	btVector3 prismatic_dir;

	PrismaticModel();

	// -- params
	void readParamsFromModel();
	void writeParamsToModel();

	size_t getDOFs() { return 1; }

	V_Configuration predictConfiguration(articulation::pose_msg_t obj_pose);
        articulation::pose_msg_t predictPose(V_Configuration q);
	M_CartesianJacobian predictHessian(V_Configuration q,double delta=1e-6);

	bool guessParameters();
	void updateParameters(std::vector<double> delta);
	bool normalizeParameters();
};

}

#endif /* PRISMATIC_MODEL_H_ */
