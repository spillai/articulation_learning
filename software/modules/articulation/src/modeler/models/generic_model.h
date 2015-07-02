/*
 * generic_model.h
 *
 *  Created on: Oct 19, 2009
 *      Author: sturm
 */

#ifndef GENERIC_MODEL_H_
#define GENERIC_MODEL_H_

/* #include "articulation_msgs/ModelMsg.h" */
/* #include "articulation_msgs/TrackMsg.h" */
/* #include "articulation_msgs/ParamMsg.h" */
#include <lcmtypes/articulation.hpp> 
/* #include <lcmtypes/articulation_model_msg_t.h> */
/* #include <lcmtypes/articulation_model_param_msg_t.h> */
/* #include <lcmtypes/articulation_object_pose_msg_t.h> */
/* #include <lcmtypes/articulation_object_pose_track_msg_t.h> */
#include <boost/shared_ptr.hpp>


#include <articulation/utils.hpp>

namespace articulation_models {
class GenericModel {
public:
	// global params
	double sigma_position;
	double sigma_orientation;
	double supress_similar;
	double outlier_ratio;
	double sac_iterations;
	double optimizer_iterations;

	// cached variables
	double complexity;
	double avg_error_position;
	double avg_error_orientation;
	double loglikelihood;
	double bic;
	double prior_outlier_ratio;
	Eigen::MatrixXd jacobian;
	Eigen::MatrixXd hessian;

	double last_error_jacobian;

	double evaluated;

	int channelOutlier;
	int channelLogLikelihood;
	int channelInlierLogLikelihood;
	std::vector<int> channelConfiguration;

        articulation::model_msg_t model;

public:
	GenericModel();
        virtual void setModel(const articulation::model_msg_t& model);
	virtual articulation::model_msg_t getModel();
	virtual void setTrack(const articulation::track_msg_t& track);
	virtual articulation::track_msg_t getTrack();
	void setId(int id);
	int getId();
	// -- model information
	virtual std::string getModelName();
	virtual size_t getDOFs();
	virtual size_t getSamples();
	// -- params
	virtual void readParamsFromModel();
	virtual void writeParamsToModel();
	virtual void prepareChannels();
	//
	bool hasParam(std::string name);
	double getParam(std::string name);
	void getParam(std::string name,double& data);
	void getParam(std::string name,btVector3 &vec);
	void getParam(std::string name,btQuaternion &quat);
	void getParam(std::string name,btTransform &t);
	void getParam(std::string name,Eigen::VectorXd &vec);
	void getParam(std::string name,Eigen::MatrixXd &mat);
	void setParam(std::string name,double value,int type);
	void setParam(std::string name,const btVector3 &vec,int type);
	void setParam(std::string name,const btQuaternion &quat,int type);
	void setParam(std::string name,const btTransform &t,int type);
	void setParam(std::string name,const Eigen::VectorXd &vec,int type);
	void setParam(std::string name,const Eigen::MatrixXd &mat,int type);
	// -- track data
	virtual int openChannel(std::string name,bool autocreate=true);
	// -- train model
	virtual bool fitModel(bool optimize=true);
	virtual bool fitMinMaxConfigurations();
	// -- evaluate model
	virtual bool evaluateModel();
	virtual double evalLatestJacobian();
	virtual double getPositionError();
	virtual double getOrientationError();
	virtual double getBIC();
	// -- cartesian space
        virtual articulation::pose_msg_t predictPose(V_Configuration q);
	// virtual articulation_models::Pose predictPose(V_Configuration q);
	virtual M_CartesianJacobian predictJacobian(V_Configuration q,double delta = 1e-6);
	virtual M_CartesianJacobian predictHessian(V_Configuration q,double delta = 1e-6);
	// -- configuration space convenience functions
	virtual void setConfiguration(size_t index,V_Configuration q);
	virtual void setJacobian(size_t index,M_CartesianJacobian J);
	virtual V_Configuration getConfiguration(size_t index);
	// -- configuration space
	virtual V_Configuration predictConfiguration(articulation::pose_msg_t obj_pose);
	// virtual V_Configuration predictConfiguration(articulation::pose_msg_t pose);
	virtual V_Configuration getMinConfigurationObserved();
	virtual V_Configuration getMaxConfigurationObserved();
	// -- projections of track data
	virtual void projectPoseToConfiguration();
	virtual void projectConfigurationToPose();
	virtual void projectConfigurationToJacobian();
	virtual void sampleConfigurationSpace(double resolution);
	virtual void keepLatestPoseOnly();
	// -- likelihood
	virtual double getInlierLogLikelihood( size_t index );
	virtual double getOutlierLogLikelihood();
	virtual double getLogLikelihoodForPoseIndex(size_t index);
	virtual double getLogLikelihood(bool estimate_outlier_ratio);
	virtual bool guessParameters();
	virtual bool sampleConsensus();
	virtual bool optimizeParameters();
	virtual bool normalizeParameters();
	std::vector<articulation::model_param_msg_t> params_initial;
	virtual void updateParameters(std::vector<double> delta);
};

typedef boost::shared_ptr<GenericModel > GenericModelPtr;
typedef boost::shared_ptr<GenericModel const> GenericModelConstPtr;

}

#endif /* GENERIC_MODEL_H_ */
