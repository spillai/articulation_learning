/*
 * factory.h
 *
 *  Created on: Oct 22, 2009
 *      Author: sturm
 */

#ifndef FACTORY_H_
#define FACTORY_H_

#include <boost/shared_ptr.hpp>
#include <typeinfo>

#include "generic_model.h"
#include <lcmtypes/articulation.hpp>
/* #include <lcmtypes/articulation_model_msg_t.h> */
/* #include <lcmtypes/articulation_object_pose_track_msg_t.h> */

namespace articulation_models {

typedef boost::shared_ptr< GenericModel > GenericModelPtr;
typedef std::vector< GenericModelPtr > GenericModelVector;

class GenericModelFactory {
public:
    virtual GenericModelPtr createModel(const articulation::track_msg_t& trackMsg) = 0;
    virtual GenericModelPtr createModel(const articulation::model_msg_t& modelMsg) = 0;
    virtual std::string getLongName() = 0;
    virtual std::string getClassName() = 0;
};

template<class T>
class SingleModelFactory: public GenericModelFactory {
public:
	std::string longname;
	std::string classname;
	SingleModelFactory(std::string longname): longname(longname),classname(typeid(T).name()) {}
	GenericModelPtr createModel(const articulation::track_msg_t& trackMsg) {
		GenericModelPtr model(new T());
		model->setTrack( trackMsg );
		return model;
	}
	/* GenericModelPtr createModel(const articulation_object_pose_track_msg_t& trackMsg) { */
	/* 	GenericModelPtr model(new T()); */
	/* 	model->setTrack( trackMsg ); */
	/* 	return model; */
	/* } */
	GenericModelPtr createModel(const articulation::model_msg_t& modelMsg) {
		GenericModelPtr model(new T());
		model->setModel( modelMsg );
		return model;
	}
	/* GenericModelPtr createModel(const articulation_msgs::ModelMsg& modelMsg) { */
	/* 	GenericModelPtr model(new T()); */
	/* 	model->setModel( modelMsg ); */
	/* 	return model; */
	/* } */
	std::string getLongName() {
		return longname;
	}
	std::string getClassName() {
		return classname;
	}
};

class MultiModelFactory {
public:
	static MultiModelFactory instance;
	std::vector<GenericModelFactory*> all_factories;
	std::vector<GenericModelFactory*> factories;

	MultiModelFactory();
	GenericModelVector createModels(const articulation::track_msg_t& trackMsg);
	// GenericModelVector createModels(const articulation_msgs::TrackMsg& trackMsg);
	GenericModelVector createModels(const articulation::model_msg_t& modelMsg);
	// GenericModelVector createModels(const articulation_msgs::ModelMsg& modelMsg);
	GenericModelPtr restoreModel(const articulation::model_msg_t& modelMsg);
	// GenericModelPtr restoreModel(const articulation_msgs::ModelMsg& modelMsg);
	int getModelIndex(std::string name);
	int getFactoryCount();
	void listModelFactories();
	static std::string getLongName(GenericModel* model);
	void setFilter(std::string filter);
};

}
#endif /* FACTORY_H_ */
