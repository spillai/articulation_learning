/*
 * pca_gp_model.h
 *
 *  Created on: Feb 10, 2010
 *      Author: sturm
 */

#ifndef PCA_GP_MODEL_H_
#define PCA_GP_MODEL_H_

#include "rigid_model.h"
#include "prismatic_model.h"
#include "gaussian_process/SingleGP.h"

namespace articulation_models {

class PCAGPModel: public GenericModel {
public:
	btVector3 rigid_position;
	btVector3 prismatic_dir;
	size_t training_samples;

	PCAGPModel();
	virtual ~PCAGPModel();

	std::vector<gaussian_process::SingleGP*> gp;
	double downsample;
	bool initialized;

	btTransform pose(size_t index);
	void storeData(bool inliersOnly);

	// -- params
	void readParamsFromModel();
	void writeParamsToModel();

	size_t getDOFs() { return 1; }

	bool fitModel(bool optimize=true);
	void buildGPs();
        V_Configuration predictConfiguration(articulation::pose_msg_t objpose);
        articulation::pose_msg_t predictPose(V_Configuration q);

	void checkInitialized();
	void projectPoseToConfiguration();
	void projectConfigurationToPose();
	void projectConfigurationToJacobian();
	bool evaluateModel();
	bool normalizeParameters();

};

}

#endif /* PCA_GP_MODEL_H_ */
