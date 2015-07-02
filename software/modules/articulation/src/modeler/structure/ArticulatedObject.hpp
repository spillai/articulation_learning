/*
 * ArticulatedObject.h
 *
 *  Created on: Sep 14, 2010
 *      Author: sturm
 */

#ifndef ARTICULATEDOBJECT_HPP_
#define ARTICULATEDOBJECT_HPP_

#include <lcmtypes/articulation.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>

#include <map>
#include <set>
#include <iostream>

#include <articulation/factory.h>
#include <articulation/utils.hpp>
#include <articulation/structs.h>


class ArticulatedObject: public KinematicParams, public KinematicData {
public:
  // articulation_msgs::ArticulatedObjectMsg object_msg;
  articulation::articulated_object_msg_t object_msg;
  KinematicGraph currentGraph;
  std::map< std::string, KinematicGraph > graphMap;
  std::map<int, std::map<uint64_t, articulation::pose_msg_t> > pose_map;
  
    ArticulatedObject();
    ArticulatedObject(const KinematicParams &other);
    void SetObjectModel(const articulation::articulated_object_msg_t &msg);
    articulation::articulated_object_msg_t& GetObjectModel();
    void FitModels(bool optimize=true);
    KinematicGraph getSpanningTree();
    void ComputeSpanningTree();
    void getFastGraph();
    void getGraph();
    void enumerateGraphs();
    void saveEval();
};

#endif /* ARTICULATEDOBJECT_HPP_ */
