/*
 * ArticulatedObject.cpp
 *
 *  Created on: Sep 14, 2010
 *      Author: sturm
 */

#include "ArticulatedObject.hpp"
// #include "articulation_structure/PoseStampedIdMsg.h"

using namespace articulation_models;
// using namespace articulation_structure;
// using namespace geometry_msgs;
using namespace std;
// using namespace boost;

ArticulatedObject::ArticulatedObject() {

}

ArticulatedObject::ArticulatedObject(const KinematicParams &other):
	KinematicParams(other) {
}

static uint64_t pose_key(const std::pair<uint64_t, articulation::pose_msg_t>& pair) {
  return pair.first;
}

void ArticulatedObject::SetObjectModel(const articulation::articulated_object_msg_t &msg) {
    std::cerr << "SetObjectModel: " << msg.parts.size() << " parts " << std::endl;

    // Copy object_msg
    object_msg = msg;

    // ADDED: spillai { 
    // Find intersection sets of utime
    std::vector<std::set<int> > id_sets(object_msg.parts.size());
    for (int j=0; j<object_msg.parts.size(); j++) {
        std::cerr << "Part: " << j << ": " << object_msg.parts[j].pose.size() << " poses " << std::endl;            
        for (int k=0; k<object_msg.parts[j].pose.size(); k++) 
            id_sets[j].insert(object_msg.parts[j].pose[k].utime);
    }

    // Find intersection sets for each 
    std::set<int> intersection_set;
    for (std::set<int>::iterator it = id_sets[0].begin(); it != id_sets[0].end(); it++) { 
        int id1 = *it;

        bool found = true;
        for (int j=1; j<id_sets.size(); j++) { 
            found = found && (id_sets[j].find(id1) != id_sets[j].end());
            if (!found) break;
        }

        if (found) intersection_set.insert(id1);
    }

    // Return if we can't find any overlapping observations
    if (!intersection_set.size())
        return;

    // align poses correctly
    for (int j=0; j<object_msg.parts.size(); j++) { 
        std::vector<articulation::pose_msg_t>& poses = object_msg.parts[j].pose;
        for (std::vector<articulation::pose_msg_t>::iterator it = poses.begin();
             it != poses.end(); ) { 
          if (intersection_set.find(it->utime) == intersection_set.end())
            it = poses.erase(it);
          else 
            it++;                    
        }
    }
    // }

    // check whether all parts have the same number of obs
    if(object_msg.parts.size()>0) {
        for(size_t part=1;part < object_msg.parts.size();part++) {
            assert(object_msg.parts[0].pose.size() == object_msg.parts[part].pose.size());
        }
        printf("ObjectArticulation::SetObjectModel with %d parts and %d obs each\n",(int)object_msg.parts.size(),(int)object_msg.parts[0].pose.size());
    }

    // // Sync
    // for (int j=0; j<object_msg.parts.size(); j++) {
    //   std::cerr << "Part: " << j << ": "
    //             << object_msg.parts[j].pose.size() << " poses " << std::endl;            
    //   for (int k=0; k<object_msg.parts[j].pose.size(); k++)  {
    //     // utime_map[object_msg.parts[j].id].insert(object_msg.parts[j].pose[k].utime);
    //     pose_map[j][object_msg.parts[j].pose[k].utime] = object_msg.parts[j].pose[k];
    //   }
    // }
    
    // copy global params to KinematicParams
    //	LoadParams(*nh_local,false);

    // copy data to KinematicParams
    if(hasParam(object_msg.params,"sigma_position")) {
        this->sigma_position = getParam(object_msg.params,"sigma_position");
        this->sigmax_position = getParam(object_msg.params,"sigma_position");
    }
    if(hasParam(object_msg.params,"sigma_orientation")) {
        this->sigma_orientation = getParam(object_msg.params,"sigma_orientation");
        this->sigmax_orientation = getParam(object_msg.params,"sigma_orientation");
    }
    if(hasParam(object_msg.params,"sigmax_position"))
        this->sigmax_position = getParam(object_msg.params,"sigmax_position");
    if(hasParam(object_msg.params,"sigmax_orientation"))
        this->sigmax_orientation = getParam(object_msg.params,"sigmax_orientation");

    if(hasParam(object_msg.params,"reduce_dofs"))
        this->reduce_dofs = getParam(object_msg.params,"reduce_dofs")!=0;

    cout << "(param) sigma_position=" << sigma_position << endl;
    cout << "(param) sigma_orientation=" << sigma_orientation << endl;
    cout << "(param) sigmax_position=" << sigmax_position << endl;
    cout << "(param) sigmax_orientation=" << sigmax_orientation << endl;
    cout << "(param) full_eval=" << full_eval<< endl;
    cout << "(param) eval_every=" << eval_every<< endl;

#if 0

    typedef std::map <uint64_t, articulation::pose_msg_t> map_utime_pose;
    for (std::map<int, map_utime_pose>::iterator it1 = pose_map.begin();
         it1 != pose_map.end(); it1++) {

      map_utime_pose& utime1_map = it1->second;
      std::vector<uint64_t> keys1;
      std::transform(utime1_map.begin(), utime1_map.end(),
                     keys1.begin(), pose_key); 

      for (std::map<int, map_utime_pose> ::iterator it2 = pose_map.begin();
           it2 != pose_map.end(); it2++) {

        if (it1->first >= it2->first)
          continue;
        map_utime_pose& utime2_map = it2->second;
        std::vector<uint64_t> keys2;
        std::transform(utime2_map.begin(), utime2_map.end(),
                       keys2.begin(), pose_key); 
        
        std::vector<uint64_t> intersection;
        std::set_intersection(keys1.begin(), keys1.end(),
                              keys2.begin(), keys2.end(),
                              intersection.begin());

        for (int j=0; j<intersection.size(); j++) {
          uint64_t stamp = intersection[j];
          articulation::pose_msg_t& pose = utime1_map[stamp];
          articulation::pose_msg_t& pose2 = utime2_map[stamp];
          addPose(pose, pose2, it1->first, it2->first,*this);         

          articulation::pose_msg_t pose_id; 
          pose_id = pose;
          pose_id.id = it1->first;
          pose_id.utime = double(stamp);
          
          boost::shared_ptr<articulation::pose_msg_t const> pose_id_ptr = 
              boost::make_shared<articulation::pose_msg_t>(pose_id);
          stampedMarker[stamp][it1->first] = pose_id_ptr;
          markerStamped[it1->first][stamp] = pose_id_ptr;

        }
      }
    }
    
# else
    // copy data to KinematicData
    for(size_t part=0;part < object_msg.parts.size();part++) {
        if(object_msg.parts[part].pose.size())
            latestTimestamp[part] = object_msg.parts[part].pose.size()-1 ;

        for(size_t obs=0;obs<object_msg.parts[part].pose.size();obs++) {
            // prepare data type for storage

            // REMOVED: spillai (stamp is propagated within pose_msg_t)
            // PoseStamped pose;
            // pose.header.stamp = ros::Time( obs );
            articulation::pose_msg_t pose;
            pose = object_msg.parts[part].pose[obs];

            // MODIFIED: spillai (stamp propagate)
            double stamp = pose.utime;

            for(size_t part2=0;part2 < object_msg.parts.size();part2++) {
                // PoseStamped pose2;
                articulation::pose_msg_t pose2;
                pose2 = object_msg.parts[part2].pose[obs];
                // pose2.header.stamp = ros::Time( obs );

                if (part >= part2)
                    continue;

                addPose(pose, pose2, part,part2,*this);
            }

            // REMOVED: spillai
            // PoseStampedIdMsg pose_id;
            // pose_id.pose = pose;
            // pose_id.id = part;
            // PoseStampedIdMsgConstPtr pose_id_ptr = boost::make_shared<PoseStampedIdMsg>(pose_id);

            articulation::pose_msg_t pose_id; 
            pose_id = pose;
            pose_id.id = part;
            pose_id.utime = stamp;

            boost::shared_ptr<articulation::pose_msg_t const> pose_id_ptr = 
                boost::make_shared<articulation::pose_msg_t>(pose_id);
            stampedMarker[stamp][part] = pose_id_ptr;
            markerStamped[part][stamp] = pose_id_ptr;
        }
    }
#endif
    
    // restore models (if any)
    for(vector<articulation::model_msg_t>::iterator m=object_msg.models.begin();
        m!=object_msg.models.end(); m++) {
        int from = m->id / object_msg.parts.size();
        int to = m->id % object_msg.parts.size();
        //		cout << "restoring model from "<<from<<" to "<<to << endl;
        models[from][to] = factory.restoreModel(*m);
        currentGraph.push_back(
                               std::pair< std::pair<int,int>,articulation_models::GenericModelPtr > 
                               ( std::pair<int,int>(from,to), models[from][to]) 
                               );
    }
}

articulation::articulated_object_msg_t& 
ArticulatedObject::GetObjectModel() {
  // copy projected poses to object_msg.parts
  for(size_t part=0;part<object_msg.parts.size();part++) {
    object_msg.parts[part].pose_projected.resize(object_msg.parts[part].pose.size());
    for(size_t obs=0;obs<object_msg.parts[part].pose.size();obs++) {
      if(stampedMarkerProjected[obs][part]) {
        object_msg.parts[part].pose_projected[obs] = * (stampedMarkerProjected[obs][part]);
      }
    }
  }

  // copy currentGraph to object_msg
  object_msg.models.clear();
  for(KinematicGraph::iterator m=currentGraph.begin();m!=currentGraph.end();m++) {
    m->second->model.id = m->first.first * object_msg.parts.size() + m->first.second;
    //		cout <<"writing id="<<m->second->model.id<<" for edge "<<m->first.first<<" to "<<m->first.second<<", for parts="<<object_msg.parts.size()<<endl;
    m->second->model.track.id = m->first.first * object_msg.parts.size() + m->first.second;
    // m->second->model.track.header = object_msg.header;

    // Getting model with projected poses
    // Added spillai
    m->second->projectConfigurationToPose();
    m->second->sampleConfigurationSpace(0.005);

    std::cerr << "GETTING OBJECT MODEL WITH PROJECTION" << std::endl;
    //
    object_msg.models.push_back(m->second->getModel());
    cout << "saving model from "<<m->first.first<<" to "<<m->first.second << endl;
  }
  return object_msg;
}

void ArticulatedObject::FitModels(bool optimize) {
    //#pragma omp parallel for schedule(dynamic,1)
    double t1 = timestamp_us();
    for(size_t part=0;part < object_msg.parts.size();part++) {
        for(size_t part2=0;part2 < object_msg.parts.size();part2++) {
            if (part >= part2)
                continue;
            std::cerr << "ObjectArticulation::FitModels  updateModel("
                      <<part<<","<<part2<<")"<<std::endl;
            if (this)
              updateModel(part,part2,*this, optimize);
        }
    }
    currentGraph.clear();
    for(std::map< int, std::map<int,std::vector<articulation_models::GenericModelPtr> > >::iterator i=models_all.begin();
        i!=models_all.end(); i++) {
        for(std::map<int,std::vector<articulation_models::GenericModelPtr> >::iterator j=i->second.begin();
            j!=i->second.end(); j++) {
            for(std::vector<articulation_models::GenericModelPtr> ::iterator k = j->second.begin();
                k!=j->second.end(); k++) {
                currentGraph.push_back(
                                       std::pair< std::pair<int,int>,articulation_models::GenericModelPtr>(std::pair<int,int>(i->first,j->first),*k));
            }
        }
    }
    setParam(object_msg.params,
             "runtime_fitting",
                 (timestamp_us() - t1) * 1e-3,
             articulation::model_param_msg_t::EVAL);
    setParam(object_msg.params,
             "object_parts",
             object_msg.parts.size(),
             articulation::model_param_msg_t::EVAL);
    setParam(object_msg.params,
             "object_samples",
             intersectionOfStamps().size(),
             articulation::model_param_msg_t::EVAL);
}

class EdgeModel {
public:
    EdgeModel(int from,int to,GenericModelPtr model)
	:from(from),to(to),model(model)
    {
    }
    int from;
    int to;
    GenericModelPtr model;
    bool operator<(const EdgeModel &other) const { return model->bic < other.model->bic; }
};

KinematicGraph ArticulatedObject::getSpanningTree() {
    vector< EdgeModel > edges;
    for(std::map< int, std::map<int,articulation_models::GenericModelPtr> >::iterator i=models.begin();i!=models.end();i++)
        for(std::map<int,articulation_models::GenericModelPtr>::iterator j=i->second.begin();j!=i->second.end();j++)
            if(j->second)
                edges.push_back( EdgeModel(i->first,j->first,j->second) );

    vector< EdgeModel > edges_accepted;
    vector< int > vertices;
    vertices.push_back(0);
    while(true) {
        // now only keep edges that don't begin and end on seen vertices
        vector< EdgeModel > edge_candidates;
        for(vector< EdgeModel >::iterator i=edges.begin();i!=edges.end();i++) {
            bool from_seen = false;
            bool to_seen = false;
            for(vector< int >::iterator j=vertices.begin();j!=vertices.end();j++) {
                if(i->from == *j) from_seen = true;
                if(i->to == *j) to_seen = true;
            }
            if((from_seen && !to_seen) || (!from_seen && to_seen)) {
                edge_candidates.push_back(*i);
            }
        }
        if(edge_candidates.size()==0)
            break;

        // sort edges
        sort(edge_candidates.begin(),edge_candidates.end());

        // add best edge
        //		cout <<"best remaining edge:"<<edge_candidates.front().from<<" to "<<edge_candidates.front().to<<" bic="<<edge_candidates[0].model->bic<<endl;
        edges_accepted.push_back(edge_candidates.front());
        vertices.push_back(edge_candidates.front().from);
        vertices.push_back(edge_candidates.front().to);
    }

    KinematicGraph E_new;
    for(vector< EdgeModel >::iterator i=edges_accepted.begin();i!=edges_accepted.end();i++) {
        E_new.push_back(
                        std::pair< std::pair<int,int>,articulation_models::GenericModelPtr> (
                                                                                             std::pair<int,int>(i->from,i->to),
                                                                                             i->model
                                                                                             ) );
    }
    E_new.DOF = E_new.getNominalDOFs();
    E_new.evaluate(intersectionOfStamps(),*this,*this);


    return(E_new);
}


void ArticulatedObject::ComputeSpanningTree() {
    double t1 = timestamp_us();

    currentGraph = getSpanningTree();
    setParam(object_msg.params,
             "bic[\""+currentGraph.getTreeName(true,false,true)+"\"]",
             currentGraph.BIC,
             articulation::model_param_msg_t::EVAL);
    setParam(object_msg.params,
             "bic_spanningtree",
             currentGraph.BIC,
             articulation::model_param_msg_t::EVAL);
    setParam(object_msg.params,
             "evals_spanningtree",
             1,
             articulation::model_param_msg_t::EVAL);
    setParam(object_msg.params,
             "runtime_spanningtree",
             (timestamp_us() - t1) * 1e-3,
             articulation::model_param_msg_t::EVAL);
    setParam(object_msg.params,"spanningtree.dof",currentGraph.DOF,articulation::model_param_msg_t::EVAL);
    setParam(object_msg.params,"spanningtree.dof.nominal",currentGraph.getNominalDOFs(),articulation::model_param_msg_t::EVAL);

    saveEval();
}

void ArticulatedObject::getFastGraph() {
    double t1 = timestamp_us();

    enumerateGraphs();
    std::vector<double> stamps= intersectionOfStamps();

    // initialize with spanning tree
    KinematicGraph tree = getSpanningTree();
    tree.evaluate(stamps,*this,*this);

    cout <<"spanning tree: "<<tree.getTreeName(true,false,true)<<endl;
    // find it graph list (ordering might be different, but topological distance should be zero)
    string current;
    for(std::map< std::string, KinematicGraph >::iterator j=graphMap.begin();j!=graphMap.end();j++) {
        //		cout <<"  "<<j->second.getTreeName(true,false,true)<< " "<< tree.distanceTo(j->second)<<endl;
        if(tree.distanceTo(j->second)==0) {
            current = j->second.getTreeName(true,false,true);
            break;
        }
    }
    cout <<"starting from current graph: "<<current<<endl;
    cout <<"finding neighbors, total graphs = "<<graphMap.size()<<endl;

    if(graphMap.find(current)==graphMap.end()) {
        cout <<"current graph is not in graphMap???"<<endl;
        return;
    }
    graphMap[current].evaluate(stamps,*this,*this);

    int evals = 0;
    cout <<"  starting   "<<graphMap[current].BIC<<"  "<<current<<" pos="<<graphMap[current].avg_pos_err<<" orient="<<graphMap[current].avg_orient_err<< endl;
    string previous;

    vector< string > v;
    for(std::map< std::string, KinematicGraph >::iterator j=graphMap.begin();j!=graphMap.end();j++)
        v.push_back(j->first);

    while(current!=previous) {
        previous = current;

        setParam(object_msg.params,
                 "fastgraph_eval[\""+current+"\"]",
                 graphMap[current].BIC,
                 articulation::model_param_msg_t::EVAL);

        //#pragma omp parallel for schedule(dynamic,1)
        for(size_t i=0;i<v.size();i++) {
            if(graphMap[current].distanceTo(graphMap[ v[i] ])==1) {
                graphMap[ v[i] ].evaluate(stamps,*this,*this);
                //#pragma omp critical
                setParam(object_msg.params,
                         "bic[\""+graphMap[ v[i] ].getTreeName(true,false,true)+"\"]",
                         graphMap[ v[i] ].BIC,
                         articulation::model_param_msg_t::EVAL);
                evals ++;
                cout <<"  evaluating "<< graphMap[ v[i] ].BIC<<
                    " ("<<graphMap[ v[i] ].getTreeName(true,true,true)<<
                    " pos="<<graphMap[ v[i] ].avg_pos_err<<
                    " orient="<<graphMap[ v[i] ].avg_orient_err<<
                    endl;
                if(graphMap[ v[i] ].BIC < graphMap[current].BIC) {
                    current = graphMap[ v[i] ].getTreeName(true,false,true);
                    //					break;
                }
            } else {
                //					cout <<"  not evaluating " <<graphMap[ v[i] ]..getTreeName(true,true,true)<<" dist="<< graphMap[current].distanceTo(graphMap[ v[i] ].)<< endl;
            }
        }
        cout <<graphMap[current].BIC<<"  "<<current<<endl;
    }

    cout <<"final:  "<<graphMap[current].BIC<<"  "<<current<<" pos="<<graphMap[current].avg_pos_err<<" orient="<<graphMap[current].avg_orient_err<< " stamps="<<stamps.size()<<endl;
    cout <<" evals: "<<evals;
    cout << endl;

    currentGraph = graphMap[current];
    currentGraph.evaluate(stamps,*this,*this);
    saveEval();

    setParam(object_msg.params,
             "evals_fastgraph",
             evals,
             articulation::model_param_msg_t::EVAL);
    setParam(object_msg.params,
             "runtime_fastgraph",
             (timestamp_us() - t1) * 1e-3,
             articulation::model_param_msg_t::EVAL);
    setParam(object_msg.params,
             "bic_fastgraph",
             currentGraph.BIC,
             articulation::model_param_msg_t::EVAL);
    setParam(object_msg.params,"fastgraph.dof",currentGraph.DOF,articulation::model_param_msg_t::EVAL);
    setParam(object_msg.params,"fastgraph.dof.nominal",currentGraph.getNominalDOFs(),articulation::model_param_msg_t::EVAL);

}

void ArticulatedObject::enumerateGraphs() {
  //	cout <<"enumerating"<<endl;
  graphMap.clear();
  map<int,int> marker_to_vertex;
  map<int,int> vertex_to_marker;
  int n=0;
  //	cout <<"latestTimestamp.size()=="<<latestTimestamp.size()<<endl;
  for(map<int,double>::iterator it = latestTimestamp.begin();
      it != latestTimestamp.end(); it++ ) {
    //		cout <<"it->first="<<it->first <<" n="<<n<<endl;
    marker_to_vertex[it->first] = n;
    vertex_to_marker[n] = it->first;
    n++;
  }

  int vertices = object_msg.parts.size();
  int edges = vertices*vertices;

  //	cout <<"vertices="<<vertices<<endl;
  //	cout <<"edges="<<edges<<endl;
  for(long graph=1; graph < (1<<(edges-1)); graph++) {
    // iterate over all possible graphs
    //		cout <<"graph="<<graph<<endl;
    KinematicGraph tree;
    for(int e=0;e<edges;e++) {
      if((graph & (1<<e))>0) {
        // edge is present in this particular graph
        int e_from = vertex_to_marker[e / vertices];
        int e_to = vertex_to_marker[e % vertices];
        if(!models[e_from][e_to]) {
          //					cout <<"from="<<e_from<<" to="<<e_to<<endl;
          tree.clear();
          break;
        }
        tree.push_back(pair< pair<int,int>,GenericModelPtr>(pair<int,int>(e_from,e_to),models[e_from][e_to]));
        //					tree.push_back(pair< pair<int,int>,GenericModelPtr>(pair<int,int>(e_to,e_from),models[e_to][e_from]));
      }
    }
    std::vector<bool> connected(vertex_to_marker.size(),false);
    connected[0] = true;
    bool connected_changed=true;
    while(connected_changed) {
      connected_changed = false;
      for(KinematicGraph::iterator i =tree.begin();i!=tree.end();i++) {
        if( (connected[i->first.first] && !connected[i->first.second]) ||
            (!connected[i->first.first] && connected[i->first.second]) ) {
          connected[i->first.first]  = connected[i->first.second] = true;
          connected_changed = true;
        }
      }
    }
    bool allConnected = true;
    for(size_t i=0;i<connected.size();i++) {
      if(!connected[i]) allConnected=false;
    }

    tree.DOF = tree.getNominalDOFs();// compute sum
    if((int)tree.size()<vertices-1) {
      //			cout <<"not enough links in graph"<<endl;
      continue;
    }
    if( restrict_graphs ) {
      if(
             (restricted_graphs.find(" "+tree.getTreeName(false,false,false)+" ")==restricted_graphs.npos) &&
             (restricted_graphs.find(" "+tree.getTreeName(true,false,false)+" ")==restricted_graphs.npos) &&
             (restricted_graphs.find(" "+tree.getTreeName(true,true,false)+" ")==restricted_graphs.npos)  &&
             (restricted_graphs.find(" "+tree.getTreeName(true,true,true)+" ")==restricted_graphs.npos) ) {
        //				cout <<"graph is not in restricted set"<<endl;
        continue;
      }
    }

    //		if(!allConnected) {
    //			cout <<" not fully connected "<<tree.getTreeName(true,true,true)<<endl;
    //		}
    if(allConnected) {
      //			cout <<" generating graphs for "<<tree.getTreeName(true,true,true)<<endl;
      // generate tree with all possible link models
      // find the number of possible combinations
      KinematicGraph basetree = tree;
      tree.clear();
      int combination_total = 1;
      for(KinematicGraphType::iterator it=basetree.begin();it!=basetree.end();it++) {
        combination_total *= models_all[it->first.first][it->first.second].size();
      }
      //			cout <<"combination_total="<<combination_total<<endl;
      for(int combination=0;combination<combination_total;combination++) {
        if(search_all_models) {
          tree.clear();
          int c_tmp = combination;
          for(KinematicGraphType::iterator it=basetree.begin();it!=basetree.end();it++) {
            int idx = c_tmp % models_all[it->first.first][it->first.second].size();
            tree.push_back( pair< pair<int,int>,GenericModelPtr> ( pair<int,int>(it->first.first,it->first.second), models_all[it->first.first][it->first.second][idx]));
            c_tmp /= models_all[it->first.first][it->first.second].size();
          }
        } else {
          tree = basetree;
        }

        int DOFs = tree.getNominalDOFs();// compute sum
        tree.DOF = DOFs;
        //				cout <<"DOFs="<<DOFs<<endl;
        for(int reducedDOFs=(reduce_dofs?(DOFs==0?0:1):DOFs);reducedDOFs<=DOFs;reducedDOFs++) {
          KinematicGraph copyOfTree(tree,true);
          copyOfTree.DOF = reducedDOFs;
          graphMap[copyOfTree.getTreeName(true,false,true)] = KinematicGraph(copyOfTree,true);
        }
        if(!search_all_models) break;
      }
    }
  }
}

void ArticulatedObject::getGraph() {
    double t1 = timestamp_us();

    enumerateGraphs();
    std::vector<double> stamps= intersectionOfStamps();

    string current;
    int evals = 0;

    vector< string > v;
    for(std::map< std::string, KinematicGraph >::iterator j=graphMap.begin();j!=graphMap.end();j++)
        v.push_back(j->first);

    //#pragma omp parallel for schedule(dynamic,1)
    for(size_t i=0;i<v.size();i++) {
        graphMap[ v[i] ].evaluate(stamps,*this,*this);

        //#pragma omp critical
        setParam(object_msg.params,
                 "bic[\""+graphMap[ v[i] ].getTreeName(true,false,true)+"\"]",
                 graphMap[ v[i] ].BIC,
                 articulation::model_param_msg_t::EVAL);
        evals ++;
        cout <<"  evaluating "<< graphMap[ v[i] ].BIC<<" ("<<graphMap[ v[i] ].getTreeName(true,true,true)<<" pos="<<graphMap[ v[i] ].avg_pos_err<<" orient="<<graphMap[ v[i] ].avg_orient_err;
        if(current == "" || (graphMap[ v[i] ].BIC < graphMap[current].BIC)) {
            current = graphMap[ v[i] ].getTreeName(true,false,true);
            cout <<"*";
        }
        cout << endl;
    }

    cout <<"final:  "<<graphMap[current].BIC<<"  "<<current<<" pos="<<graphMap[current].avg_pos_err<<" orient="<<graphMap[current].avg_orient_err<< " stamps="<<stamps.size()<<endl;
    cout <<" evals: "<<evals;
    cout << endl;

    currentGraph = graphMap[current];
    currentGraph.evaluate(stamps,*this,*this);

    setParam(object_msg.params,
             "evals_graph",
             evals,
             articulation::model_param_msg_t::EVAL);
    setParam(object_msg.params,
             "runtime_graph",
             (timestamp_us() - t1) * 1e-3,
             articulation::model_param_msg_t::EVAL);
    setParam(object_msg.params,
             "bic_graph",
             currentGraph.BIC,
             articulation::model_param_msg_t::EVAL);
    setParam(object_msg.params,"graph.dof",currentGraph.DOF,articulation::model_param_msg_t::EVAL);
    setParam(object_msg.params,"graph.dof.nominal",currentGraph.getNominalDOFs(),articulation::model_param_msg_t::EVAL);

    saveEval();
}

void ArticulatedObject::saveEval() {
    // copy evaluation to object_msg
    setParam(object_msg.params,"bic",currentGraph.BIC,articulation::model_param_msg_t::EVAL);
    setParam(object_msg.params,"dof",currentGraph.DOF,articulation::model_param_msg_t::EVAL);
    setParam(object_msg.params,"dof.nominal",currentGraph.getNominalDOFs(),articulation::model_param_msg_t::EVAL);
    setParam(object_msg.params,"loglikelihood",currentGraph.loglh,articulation::model_param_msg_t::EVAL);
    setParam(object_msg.params,"complexity",currentGraph.k,articulation::model_param_msg_t::EVAL);
    setParam(object_msg.params,"avg_error_position",currentGraph.avg_pos_err,articulation::model_param_msg_t::EVAL);
    setParam(object_msg.params,"avg_error_orientation",currentGraph.avg_orient_err,articulation::model_param_msg_t::EVAL);
}
