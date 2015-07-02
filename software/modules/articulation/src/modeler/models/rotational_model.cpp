/*
 * rotational_model.cpp
 *
 *  Created on: Oct 22, 2009
 *      Author: sturm
 */

#include <articulation/rotational_model.h>
#include <articulation/utils.hpp>

using namespace std;

#include <iomanip>
#define VEC(a) setprecision(5)<<fixed<<a.x()<<" "<<a.y()<<" "<<a.z()<<" "<<a.w()<<" l="<<a.length()
#define VEC2(a) "t=["<<VEC(a.getOrigin())<<"] r=[]"<<VEC(a.getRotation())<<"]"
#define PRINT(a) cout << #a <<"=" << VEC(a)<<endl;
#define PRINT2(a) cout << #a <<"=" << VEC2(a)<<endl;

namespace articulation_models {


RotationalModel::RotationalModel() {
	rot_center = btVector3 (0,0,0);
	rot_axis = btQuaternion (0,0,0,1);
	rot_radius = 1;
	rot_orientation = btQuaternion (0,0,0,1);
	complexity = 3+2+1+3;	// rotation center + rotation axis + radius + orientation
	rot_mode = 0;// use position (=0) or orientation (=1) for configuration estimation
}

// -- params
void RotationalModel::readParamsFromModel() {
	GenericModel::readParamsFromModel();
	getParam("rot_center",rot_center);
	getParam("rot_axis",rot_axis);
	getParam("rot_radius",rot_radius);
	getParam("rot_orientation",rot_orientation);
	getParam("rot_mode",rot_mode);
}

void RotationalModel::writeParamsToModel() {
	GenericModel::writeParamsToModel();
	setParam("rot_center",rot_center,articulation::model_param_msg_t::PARAM);
	setParam("rot_axis",rot_axis,articulation::model_param_msg_t::PARAM);
	setParam("rot_radius",rot_radius,articulation::model_param_msg_t::PARAM);
	setParam("rot_orientation",rot_orientation,articulation::model_param_msg_t::PARAM);
	setParam("rot_mode",rot_mode,articulation::model_param_msg_t::PARAM);
}

V_Configuration RotationalModel::predictConfiguration(articulation::pose_msg_t objpose) { 
	V_Configuration q(1);

	if(rot_mode==1) {
		// estimate configuration from pose orientation
		btMatrix3x3 m_pose( poseToTransform(objpose).getBasis() );
		btMatrix3x3 m_axis(rot_axis);
		btMatrix3x3 m_orient(rot_orientation);

		btMatrix3x3 rot_z = m_axis.inverse() * m_pose * m_orient.inverse();
		btQuaternion rot_z_quat;
		rot_z.getRotation(rot_z_quat);

		q(0) = rot_z_quat.getAngle();
	} else {

		btTransform center(rot_axis,rot_center);

		btTransform rel = center.inverseTimes( poseToTransform(objpose) );

		q(0) = -atan2(rel.getOrigin().y(), rel.getOrigin().x());
	}

	return q;
}

articulation::pose_msg_t RotationalModel::predictPose(V_Configuration q) {
    articulation::pose_msg_t objpose;
    objpose.id = -1;
    objpose.utime = 0;

    btTransform center(rot_axis,rot_center);
    btTransform rotZ(btQuaternion(btVector3(0, 0, 1), -q[0]), btVector3(0, 0, 0));
    btTransform r(btQuaternion(0,0,0,1), btVector3(rot_radius, 0, 0));
    btTransform offset(rot_orientation, btVector3(0, 0, 0));

    objpose = transformToPose( center * rotZ * r * offset );

    return objpose;
}

bool RotationalModel::guessParameters() {
	if(model.track.pose.size() < 3)
		return false;

	size_t i,j,k;
	do{
		i = rand() % getSamples();
		j = rand() % getSamples();
		k = rand() % getSamples();
	} while (i==j || j==k || i==k);

	btTransform pose1 = poseToTransform(model.track.pose[i]);
	btTransform pose2 = poseToTransform(model.track.pose[j]);
	btTransform pose3 = poseToTransform(model.track.pose[k]);


//	if( pose1.getOrigin().distance(pose2.getOrigin())/sigma_position <
//		pose1.getRotation().angle(pose2.getRotation())/sigma_orientation
	if(rand() % 2 == 0)
	{
		rot_mode = 1;
//		cout <<"using rot computation"<<endl;

		// angle between pose 1 and pose 2
		double angle_12 = pose1.inverseTimes(pose2).getRotation().getAngle();

		// distance between pose 1 and pose 2
		double dist_12 = pose1.getOrigin().distance( pose2.getOrigin() );
		// cout << "Distance_12: " << dist_12 << std::endl;
                
		// rot axis between pose 1 and pose 2
		btVector3 rot_12 = pose1.getBasis() * pose1.inverseTimes(pose2).getRotation().getAxis() * -1;
		// PRINT(rot_12);

		rot_center = btVector3(0.5,0.5,0.5);
		rot_axis = btQuaternion(0,0,0,1);
		rot_orientation = btQuaternion(0,0,0,1);
		rot_radius = 0.0;

		rot_radius = (dist_12 * 0.5) / sin( angle_12 * 0.5 );
		// cout << "Rotation radius_12: " << rot_radius << std::endl;
                
//		PRINT(pose1.getOrigin());
//		PRINT(pose2.getOrigin());
		btVector3 c1 = (pose1.getOrigin() + pose2.getOrigin())/2;
//		PRINT(c1);
		btVector3 v1 = (pose2.getOrigin() - pose1.getOrigin());
//		PRINT(v1);
		v1.normalize();
//		PRINT(v1);
		v1 = v1 - rot_12.dot(v1)*rot_12;
//		PRINT(v1);
		v1.normalize();
//		PRINT(v1);
//		PRINT(rot_12);
		rot_center = c1 + v1.cross(rot_12) * rot_radius * cos(angle_12/2);
//		PRINT(rot_center);

		btVector3 d(1,0,0);
		d = pose1.getOrigin() - rot_center;
		d.normalize();
//		PRINT(d);

		btVector3 rot_z = rot_12;
		btVector3 rot_x = d - rot_z.dot(d)*rot_z;
		rot_x.normalize();
		btVector3 rot_y = rot_z.cross(rot_x);
//		rot_x = btVector3(-1,0,0);
//		rot_y = btVector3(0,-1,0);
//		rot_z = btVector3(0,0,-1);
		btMatrix3x3(
				rot_x.x(),rot_y.x(),rot_z.x(),
				rot_x.y(),rot_y.y(),rot_z.y(),
				rot_x.z(),rot_y.z(),rot_z.z()).getRotation(rot_axis);
//		PRINT(rot_x);
//		PRINT(rot_y);
//		PRINT(rot_z);
//		PRINT(rot_axis);
//		rot_axis=btQuaternion(0,0,0,1);

		rot_orientation = rot_axis.inverse() * pose1.getRotation();

		// eval ---------------------
		btTransform t_rotaxis(rot_axis,btVector3(0,0,0));
		btTransform t_radius(btQuaternion(0,0,0,1),btVector3(rot_radius,0,0));
		btTransform t_orient(rot_orientation,btVector3(0,0,0));
		btTransform t_rotcenter(btQuaternion(0,0,0,1),rot_center);

		// show diff to ppose1 and ppose2
		btTransform diff;
		btTransform pp1 =
				t_rotcenter *
				t_rotaxis *
				btTransform(btQuaternion(btVector3(0,0,1),0.00),btVector3(0,0,0)) *
				t_radius *
				t_orient;
		diff = pose1.inverseTimes(pp1);
//		cout <<"pp1: angle=" <<0.00<<" poserr="<<diff.getOrigin().length()<<" orienterr="<<diff.getRotation().getAngle()<<endl;

		btTransform pp2 =
				t_rotcenter *
				t_rotaxis *
				btTransform(btQuaternion(btVector3(0,0,1),angle_12),btVector3(0,0,0)) *
				t_radius *
				t_orient;
		diff = pose2.inverseTimes(pp2);
//		cout <<"pp2: angle=" <<angle_12<<" poserr="<<diff.getOrigin().length()<<" orienterr="<<diff.getRotation().getAngle()<<endl;

		for(size_t a=0;a<getSamples();a++) {
			V_Configuration q =predictConfiguration(model.track.pose[a]);
			btTransform p2 = poseToTransform(predictPose(q));
			diff = poseToTransform(model.track.pose[a]).inverseTimes(p2);
//			cout <<"angle=" <<q[0]<<" poserr="<<diff.getOrigin().length()<<" orienterr="<<diff.getRotation().getAngle()<<endl;
//			btTransform pp =
//					t_rotcenter *
//					t_rotaxis *
//					btTransform(btQuaternion(btVector3(0,0,1),-q[0]),btVector3(0,0,0)) *
//					t_radius *
//					t_orient;
//			diff = poseToTransform(model.track.pose[a]).inverseTimes(pp);
//			cout <<"vs angle=" <<q[0]<<" poserr="<<diff.getOrigin().length()<<" orienterr="<<diff.getRotation().getAngle()<<endl;
		}
	} else {
		rot_mode = 0;
//		cout<<"using plane computation"<<endl;
		// first, find the plane
		btVector3 plane_pos = pose1.getOrigin();
		btVector3 plane_v = pose2.getOrigin() - pose1.getOrigin();
		btVector3 plane_w = pose3.getOrigin() - pose1.getOrigin();
	//	PRINT(plane_pos);
	//	PRINT(plane_v);
	//	PRINT(plane_w);
		plane_v.normalize();
		plane_w.normalize();

		btVector3 plane_x = plane_v;
		btVector3 plane_y = plane_w - (plane_w.dot(plane_v))*plane_v;
		plane_x.normalize();
		plane_y.normalize();
		btVector3 plane_z = plane_x.cross(plane_y);
	//	PRINT(plane_x);
	//	PRINT(plane_y);
	//	PRINT(plane_z);


		btMatrix3x3 plane_rot(
				plane_x.x(),plane_y.x(),plane_z.x(),
				plane_x.y(),plane_y.y(),plane_z.y(),
				plane_x.z(),plane_y.z(),plane_z.z()
				);

		btTransform plane(plane_rot,plane_pos);

		btTransform onplane_pose1 = plane.inverseTimes(pose1);
		btTransform onplane_pose2 = plane.inverseTimes(pose2);
		btTransform onplane_pose3 = plane.inverseTimes(pose3);
	//	cout <<"onplane_pose1"<<VEC2(onplane_pose1)<<endl;
	//	cout <<"onplane_pose2"<<VEC2(onplane_pose2)<<endl;
	//	cout <<"onplane_pose3"<<VEC2(onplane_pose3)<<endl;

		//http://local.wasp.uwa.edu.au/~pbourke/geometry/lineline2d/
		btVector3 p1 = (onplane_pose1.getOrigin() + onplane_pose2.getOrigin())/2;
		btVector3 p21 = (onplane_pose2.getOrigin() - onplane_pose1.getOrigin()).rotate(btVector3(0,0,1),M_PI/2);;

		btVector3 p3 = (onplane_pose1.getOrigin() + onplane_pose3.getOrigin())/2;
		btVector3 p43 = (onplane_pose3.getOrigin() - onplane_pose1.getOrigin()).rotate(btVector3(0,0,1),M_PI/2);;

		btVector3 p13 = p1 - p3;

		double u = 	( p43.x()*p13.y() - p43.y()*p13.x() ) /
					( p43.y()*p21.x() - p43.x()*p21.y() );
		btVector3 onplane_center = p1 + u*p21;

		btTransform rotcenter(plane_rot,plane * onplane_center);
		rot_center = rotcenter.getOrigin();
		rot_axis = rotcenter.getRotation();

		rot_radius = rotcenter.inverseTimes(pose1).getOrigin().length();
		rot_orientation = btQuaternion(0,0,0,1);
		V_Configuration q = predictConfiguration( model.track.pose[i]);
	//	cout <<"q="<<q[0]<<endl;
		btTransform pred1 = poseToTransform( predictPose(q) );
	//	PRINT2(pose1);
	//	PRINT2(pred1);
		rot_orientation = pred1.inverseTimes(pose1).getRotation();
	}

//	cout<<"rot_radius="<<rot_radius<<endl;
//	PRINT(rot_center);
//	PRINT(rot_axis);
//	PRINT(rot_orientation);

	if(!check_values(rot_center)) return false;
	if(!check_values(rot_axis)) return false;
	if(!check_values(rot_radius)) return false;
	if(!check_values(rot_orientation)) return false;

	return true;
}

void RotationalModel::updateParameters(std::vector<double> delta) {
	rot_center = rot_center + btVector3(delta[0],delta[1],delta[2]);
	btQuaternion q;
	// q.setRPY(delta[3],delta[4],0.00);
        q.setEuler(delta[3], delta[4], 0.00);
	rot_axis = rot_axis * q;

	rot_radius = rot_radius + delta[5];

	btQuaternion q2;
        q2.setEuler(delta[6], delta[7], delta[8]);
	rot_orientation = rot_orientation * q2;
}

bool RotationalModel::normalizeParameters() {
//	if(model.track.pose.size()>2) {
//		{
//			V_Configuration q = predictConfiguration(model.track.pose.front());
//			rot_axis = rot_axis * btQuaternion(btVector3(0, 0, 1), -q[0]);
//		}
//		if(predictConfiguration(model.track.pose.back())[0]<0)
//			rot_axis = rot_axis * btQuaternion(btVector3(1,0,0),M_PI);
//
//		rot_orientation = rot_axis.inverse() * poseToTransform(model.track.pose.front()).getRotation();
//	}
	return true;
}
}
