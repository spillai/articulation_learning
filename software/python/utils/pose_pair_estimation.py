#!/usr/bin/python 
# TODO: Compute relative pose mean instead of absolute mean pose

from __future__ import division

import time
import numpy as np
import pandas as pd
import networkx as nx
import random
import itertools
import cv2

import transformations as tf
import rigid_transform as rtf

import draw_utils

from collections import defaultdict

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from fs_pcl_utils import CorrespondenceRejectionSAC
from fs_isam import Pose3d, Slam3D

# def tf_compress(T): 
#     return rtf.RigidTransform.from_homogenous_matrix(T)

# def tf_uncompress(v): 
#     return v.to_homogeneous_matrix()


def pose_pair_construct(p1,n1,p2,n2): 
    """ Construct pose-pair from (pt1,normal1), and (pt2,normal2) """
    v1 = p2-p1; v1 /= np.linalg.norm(v1)
    v2 = -v1

    g = np.array([0., -1., 0.])
    # R1, R2 = rtf.tf_construct(n1,v1), rtf.tf_construct(n2,v2)
    R1, R2 = rtf.tf_construct(n1,g), rtf.tf_construct(n2,g)
    T1, T2 = rtf.tf_compose(R1, p1), rtf.tf_compose(R2, p2)    

    return rtf.RigidTransform.from_homogenous_matrix(T1), \
        rtf.RigidTransform.from_homogenous_matrix(T2)

def mean_pose(poses): 
    """HACK: Compute the mean pose of set of tfs"""
    print [pose for pose in poses]
    return poses[0]
    # tvecs = [pose.tvec.reshape(-1) for pose in poses]
    # mu_pos = np.mean(np.vstack(tvecs), axis=0)

    # rpys = [np.fmod(np.array(pose.quat.to_roll_pitch_yaw()),
    #                 2*np.pi) for pose in poses]
    # mu_rpy = np.mean(np.vstack(rpys), axis=0);
    # mu_quat = rtf.Quaternion.from_roll_pitch_yaw(mu_rpy[0], mu_rpy[1], mu_rpy[2])

    # # print mu_pos, np.vstack(tvecs)
    
    # return rtf.RigidTransform(mu_quat, mu_pos)

class PosePairEstimation: 
    def __init__(self, data, evaluate_top=10): 

        self.MAX_CLUSTER_SIZE = 20;

        # label, utime - > pose
        self.pose_map = defaultdict(
            lambda: defaultdict()
        )

        # Build cluster index map
        self.identify_cluster_inds(data, evaluate_top=evaluate_top)

        # Estimate pose given data
        self.per_label_estimation(data)

    def get_pose_list(self): 
        """
        Retrieve the pose list of trajectories, and include sensor pose
        Label: label + 1
        """
        pose_list = []
        utimes_set = set()
        for label,v in self.pose_map.iteritems(): 
            assert(label >= 0)

            # Sort by utime
            vals = [(utime,pose) for utime,pose in v.iteritems()]
            vals.sort(key=lambda x: x[0])

            # Add to pose list
            pose_list.append((label, vals))

            # Provide sensor ref. for each utime
            for utime, pose in v.iteritems(): 
                utimes_set.add(utime)

        ref_label = -1;
        pose_list.append((ref_label, [(utime,rtf.RigidTransform.identity()) 
                                      for utime in sorted(utimes_set)]))

        # Sort by label
        pose_list.sort(key=lambda x: x[0])                    
        return pose_list

    def viz_data(self): 
        viz_poses, viz_tvecs1, viz_tvecs2 = [], [], []
        for label,v in self.pose_map.iteritems(): 

            viz_tvecsl = []
            for ut,pose in v.iteritems(): 
                viz_poses.append(pose)
                viz_tvecsl.append(pose.tvec.reshape((-1,3)))
            viz_tvecsl = np.vstack(viz_tvecsl)

            viz_tvecs1.append(viz_tvecsl[:-1])
            viz_tvecs2.append(viz_tvecsl[1:])

        draw_utils.publish_pose_list2('POSE_OPT', viz_poses, sensor_tf=True)       
        draw_utils.publish_line_segments('POSE_OPT_TRAJ', 
                                         np.vstack(viz_tvecs1),
                                         np.vstack(viz_tvecs2), c='g')       

    def get_max_movement_trajectories(self, data, label_inds, k=1): 
        """
        Pick trajectories that have the largest displacement, 
        or least signal to noise ratio
        """
        if len(label_inds) < k: 
            print 'Insuffient trajectories for max_movement'
            return []

        top_inds = []
        for ind in label_inds: 
            ut_inds, = np.where(data.idx[ind,:] != -1)
            X = data.xyz[ind,ut_inds].reshape((-1,3))
            Xmin, Xmax = np.min(X, axis=0), np.max(X, axis=0);
            top_inds.append((ind, np.linalg.norm(np.fabs(Xmin-Xmax)) ))

        # Reverse true: for longest traj, false: for shortest traj
        top_inds.sort(key=lambda x: x[1], reverse=True)
        return np.array([ ind for ind,score in top_inds[:k] ])

    def per_label_estimation(self, data): 
        # Eval cluster 
        for label,label_inds in self.cluster_ind_map.iteritems(): 
	    print 'Estimating pose for label %i: %s' % (label,label_inds)

            # Build overlapping utimes
            utimes_inds = []
            for ind in label_inds: 
                ut_inds, = np.where(data.idx[ind,:] != -1);
                utimes_inds.extend(ut_inds)
            # Find utime inds that occur at least twice
            utimes_inds, utimes_inv = np.unique(utimes_inds, return_inverse=True)
            utimes_inds_count = np.bincount(utimes_inv)
            utimes_inds = np.sort(utimes_inds[np.where(utimes_inds_count>=2)])

            if len(utimes_inds) < 2: 
                print 'Insufficient constraints for pose estimation'
                continue

            # Get top k trajectories for articulation estimation
            top_label_inds = self.get_max_movement_trajectories(data, label_inds, k=4)
            print 'Top movement inds: ', top_label_inds

            # Build pose pair manager
            ppm = PoseManager(data=data)

            # For each utime, add observation
            for ut_idx in utimes_inds: 
                
                # Look at each feature, and add if available
                linds, = np.where(data.idx[top_label_inds,ut_idx] != -1)
                linds = np.sort(top_label_inds[linds]) # sort to hash pair correctly

                # indices should be atleast 2, since we filtered it above
                # assert(len(linds)>=2)

                # For every permutation (pairs)
                # Add pose-pairs as observations
                for ind1,ind2 in itertools.combinations(linds,r=2):
                    ppm.add_observation(ut_idx, 
                                        ind1, # data.feature_ids[ind1], 
                                        data.xyz[ind1,ut_idx], data.normal[ind1,ut_idx],
                                        ind2, # data.feature_ids[ind2], 
                                        data.xyz[ind2,ut_idx], data.normal[ind1,ut_idx]
                    )


            # # Add to final pose map
            # for n,d in ppm.pp_graph.nodes_iter(data=True): 
            #     self.pose_map[label][d['data'].utime]  = d['data'].pp_ij_w

            # # Viz graph nodes
            # viz_poses = [d['data'].pp_ij_w for n,d in ppm.pp_graph.nodes_iter(data=True)]
            # ppm.debug_pose(viz_poses, ch='POSEPAIR_CTR')

            # # break
            # continue;

            # Once observations are added
            # Initialize pose
            # And establish connections
            st = time.time()
            ppm.establish_connections()
            print 'Total time: ', time.time() - st
            
            # Optimize for pose
            ppm.optimize()

            # Pose map
            print 'Propagate poses'
            for ut_idx,pose in ppm.final_pose_map.iteritems(): 
                self.pose_map[label][ut_idx] = pose

            print data.feature_utimes[ut_idx]



    def identify_cluster_inds(self, data, evaluate_top): 
        # Build cluster index map
        self.cluster_ind_map = {}

        # Count number of instances of labels
        # Avoid -1 label
        unique_labels, unique_inv = np.unique(data.labels[np.where(data.labels != -1)], 
                                              return_inverse=True)
        num_labels = len(unique_labels)
        label_count = np.bincount(unique_inv)
        label_votes = zip(unique_labels, label_count)
        print 'Total: %i labels' % num_labels

        # Pick the top labeled trajectories
        label_votes.sort(key=lambda x: x[1], reverse=True)
        labels = [lv[0] for lv in label_votes[:evaluate_top]]

        print 'Pose Pair Estimation of top %i labels' % len(labels)

        # Find exemplars for each cluster 
	for label in labels: 
	    if label < 0: continue; # don't do anything for unlabeled cases

            # Indices that have the label == label
	    inds, = np.where(data.labels == label)
	    assert(len(inds) != 0)

            # Arbitrarily pick random set of label_inds (if many available)
            label_inds = inds; # random.sample(inds, min(self.MAX_CLUSTER_SIZE, len(inds)));

            # For querying relative poses
	    self.cluster_ind_map[label] = np.array(label_inds, dtype=np.int32)
        

class PosePairNode: 
    def __init__(self, utime, i, j, pp_ij, pp_ji): 
        if i >= j: 
            raise AssertionError('Pose Pair Order not preserved')

        self.i = i
        self.j = j
        self.utime = utime

        self.pp_ij_w = pp_ij
        self.pp_ji_w = pp_ji

        # print 'PPN: ', 'i', self.i, 'j', self.j, 'utime', self.utime, 'pp_ij', self.pp_ij_w, 'pp_ji', self.pp_ji_w

class PoseManager: 
    def __init__(self, data): 
        # data from tracker
        self.data = data
        
        # random.seed(1)
        self.next_idx = 0

        # Graph matrix for idx map
        self.pp_graph_utime_map = defaultdict(
            lambda: defaultdict()
        )
        self.pp_graph_pair_map = defaultdict(
            lambda: defaultdict()
        )
        
        # Pose-Pair graph
        self.pp_graph = nx.Graph()
        self.pose_graph = nx.Graph()
        self.relpose_graph = nx.Graph()

        # Final Pose map of track i @ t=t 
        self.final_pose_map = defaultdict(
            lambda: rtf.RigidTransform.identity()
        ) # (id-pair : Pose-pair)

        self.pose_pub = defaultdict(list)
        self.pose_links1_pub, self.pose_links2_pub = [], []

    def debug_pose(self, pose, ch='POSE_DEBUG', reset=False): 
        if reset: self.pose_pub[ch] = []
        self.pose_pub[ch].append(pose)
        draw_utils.publish_pose_list2(ch, self.pose_pub[ch], sensor_tf=True)

    def debug_link(self, tvec1, tvec2, reset=False): 
        if reset: self.pose_links1_pub, self.pose_links2_pub = [], []
        self.pose_links1_pub.append(tvec1.reshape((-1,3)))
        self.pose_links2_pub.append(tvec2.reshape((-1,3)))
        draw_utils.publish_line_segments('POSE_DEBUG_EDGES', 
                                         np.vstack(self.pose_links1_pub), 
                                         np.vstack(self.pose_links2_pub))       
        draw_utils.publish_point_cloud('POSE_DEBUG_VERTICES', np.vstack(self.pose_links1_pub))

    def debug_pose_link(self, pose1, pose2, reset=False): 
        if reset: self.pose_links1_pub, self.pose_links2_pub = [], []
        self.pose_links1_pub.append(pose1.tvec.reshape((-1,3)))
        self.pose_links2_pub.append(pose2.tvec.reshape((-1,3)))
        draw_utils.publish_line_segments('POSE_DEBUG_EDGES', 
                                         np.vstack(self.pose_links1_pub), 
                                         np.vstack(self.pose_links2_pub))       

    def next_index(self): 
        self.next_idx += 1
        return self.next_idx

    def initialize(self, ut_idx): 
        # Look at previously added nodes, both inter/intra
        nodes = np.array(
            list(set([node for node in 
                      itertools.chain.from_iterable(self.pp_graph_utime_map[ut_idx].keys())]
                 ))
        )
        
        X0 = self.data.xyz[nodes, ut_idx]
        N0 = self.data.normal[nodes, ut_idx]
        p0, n0 = np.mean(X0, axis=0), np.mean(N0, axis=0)
        n0 /= np.linalg.norm(n0)

        R = rtf.tf_construct(n0, np.array([0., 1., 0.]))

        # Flip coordinates for pcl
        xold, zold = R[:,0].copy(), R[:,2].copy()
        R[:,0], R[:,2] = zold, -xold

        pose0 = rtf.RigidTransform.from_homogenous_matrix(rtf.tf_compose(R, p0))

        # Debug mean pose
        self.debug_pose(pose0, ch='INIT_POSE')
        print 'publishing INIT_POSE'

        # Add mean pose to the node information
        self.relpose_graph.add_node(ut_idx, data=pose0)

        # self.final_pose_map[ut_idx] = mu_pose

    def add_incomplete_observation(self, ut_idx, fidi, pi, ni): 
        # print 'Adding OBS: ', ut_idx, fidi, fidj

        # Construct Pose-Pair
        pp_ij_w, pp_ji_w = pose_pair_construct(pi, ni, pi + 1, ni)
        # if fidi == 619: self.debug_pose(pp_ij_w)

        # Add obs. to graph
        nidx = self.pp_graph.number_of_nodes()

        self.pp_graph_utime_map[ut_idx][(fidi,fidj)] = nidx
        self.pp_graph_pair_map[(fidi,fidj)][ut_idx] = nidx

        # Add pose-pair observation
        self.pp_graph.add_node(nidx, 
                               data=PosePairNode(utime=ut_idx, i=fidi, j=fidj, 
                                                 pp_ij=pp_ij_w, pp_ji=pp_ji_w))

    def add_observation(self, ut_idx, fidi, pi, ni, fidj, pj, nj): 
        # print 'Adding OBS: ', ut_idx, fidi, fidj

        # Construct Pose-Pair
        pp_ij_w, pp_ji_w = pose_pair_construct(pi, ni, pj, nj)
        # if fidi == 619: self.debug_pose(pp_ij_w)

        # Add obs. to graph
        nidx = self.pp_graph.number_of_nodes()

        self.pp_graph_utime_map[ut_idx][(fidi,fidj)] = nidx
        self.pp_graph_pair_map[(fidi,fidj)][ut_idx] = nidx

        # Add pose-pair observation
        self.pp_graph.add_node(nidx, 
                               data=PosePairNode(utime=ut_idx, i=fidi, j=fidj, 
                                                 pp_ij=pp_ij_w, pp_ji=pp_ji_w))

    def nnplus1(self, n): 
        return n * (n+1) / 2

    def shortest_path(self, source, target): 
        for t in target: 
            path = nx.shortest_path(self.pose_graph, 
                                    source=source, 
                                    target=t, weight='weight')
            if len(path): return path
        return None
        

    def establish_connections(self):
        # Build utime graph
        utimes_inds = sorted(self.pp_graph_utime_map.keys())

        st = time.time()
        # Add nodes
        for ut_ind in utimes_inds: 
            self.pose_graph.add_node(ut_ind)
        print '==========> Add nodes', time.time() - st

        # Add edges: this can be improved
        for ut_ind1, ut_ind2 in itertools.combinations(utimes_inds, r=2): 
            self.pose_graph.add_edge(ut_ind1, ut_ind2, 
                                     weight=self.nnplus1(abs(ut_ind2-ut_ind1)+1))

        print '==========> Add nodes and edges', time.time() - st

        # # Initialize and Establish connections
        # targets = [utimes_inds[0]]
        missing = []
        # for idx, ut_ind in enumerate(utimes_inds): 
        #     if idx == 0: 
        #         self.initialize(ut_ind)
        #     else: 
        #         # Find shortest path between current node, 
        #         # and its previous nodes (targets)
        #         path = self.shortest_path(source=ut_ind, target=targets)

        #         # Update set of targets so that we can recursively 
        #         # determine closest node to attach to 
        #         targets = path

        #         if path is None: 
        #             missing.append(ut_ind)
        #             continue

        #         # Add an edge as suggested by the path
        #         for n1, n2 in zip(path[:-1], path[1:]): 
        #             self.add_edge(n2, n1)
                
        #         # self.add_edge(ut_idx1, ut_idx2)
        #     # print 'EDGE: ', ut_idx1, ut_idx2

        print '==========> Done adding nodes and edges', time.time() - st
        print '==========> Initializing utimes', time.time() - st
        # Add global links
        utimes_inds_skip = utimes_inds[::4]
        self.initialize(utimes_inds[0])

        added = []
        for ut_ind1, ut_ind2 in zip(utimes_inds_skip[:-1], utimes_inds_skip[1:]): 
            if self.add_edge(ut_ind1, ut_ind2): 
                added.append((ut_ind1, ut_ind2))

        # for ut_ind in utimes_inds_skip[3::3]:  # every 5, after first 5 inds
        #     if self.add_edge(utimes_inds[0], ut_ind): 
        #         added.append((utimes_inds[0], ut_ind))


        print 'Added: %i, Missing: %i' % (len(added), len(missing))
        # print 'CC: ', nx.connected_components(self.relpose_graph)
        print '==========> Done adding additional edges', time.time() - st


    # Find least squares solution between two sets of points
    def add_edge(self, ut_idx1, ut_idx2):
        # Look at previously added nodes, both inter/intra
        intranodes1 = set([node for node in 
                           itertools.chain.from_iterable(self.pp_graph_utime_map[ut_idx1].keys())])
        intranodes2 = set([node for node in 
                           itertools.chain.from_iterable(self.pp_graph_utime_map[ut_idx2].keys())])
        common_nodes = np.array(list(intranodes1.intersection(intranodes2)))

        print 'COMMON_NODES', common_nodes, ut_idx1, ut_idx2
        if len(common_nodes) < 3: 
               print 'Insufficient constraints skipping %i, %i ' % (ut_idx1, ut_idx2)
               return False;

        # Find the relative motion, first remove offsets
        X1 = self.data.xyz[common_nodes, ut_idx1]
        X2 = self.data.xyz[common_nodes, ut_idx2]

        # Debug the links
        self.debug_link(X1.copy(), X2.copy())

        # Remove offsets
        muX = np.mean(X1, axis=0)
        X1, X2 = X1 - muX, X2 - muX

        # Pose estimation via Correspondence Rejection SAC
        T12 = rtf.RigidTransform.from_homogenous_matrix(
            CorrespondenceRejectionSAC(source=X1, target=X2, 
                                       inlier_threshold=0.05, max_iterations=10)
        )


        # Give relative pose information
        self.relpose_graph.add_edge(ut_idx1, ut_idx2, data=T12)
        return True

    def optimize(self): 
        # SLAM relative pose formulation
        # print 'Edges: ', self.pose_graph.edges()
        print '===================================='
        print 'SLAM OPT'
        print 'Optimizing over %i nodes' % self.relpose_graph.number_of_nodes()
        print '===================================='

        slam = Slam3D()
        obs_axes, plot_axes = 'rxyz', 'sxyz'
        print 'Adding priors: '
        # Add priors
        noise = np.diag([0.05, 0.05, 0.05, 0.1, 0.1, 0.1])
        first_node = None
        for ut,d in self.relpose_graph.nodes(data=True): 
            if 'data' in d: 
                roll, pitch, yaw, x, y, z = d['data'].to_roll_pitch_yaw_x_y_z(axes=obs_axes)
                slam.add_node_with_factor(ut, 
                                          Pose3d(x=x, y=y, z=z, 
                                                 yaw=yaw, pitch=pitch, roll=roll), noise);
                # print 'PRIOR: ', ut, d['data']
                first_node = ut
            else: 
                slam.add_node(ut)
                # print 'adding node, ', ut
            # print 'EDGES: ', self.pose_graph[ut]
        assert(first_node is not None)
        
        # Add edge factors
        print 'Adding edges: '
        T = nx.minimum_spanning_tree(self.relpose_graph)
        added = set([])
        for ut1,ut2 in nx.dfs_edges(T, source=first_node): 
            d = self.relpose_graph[ut1][ut2]
            if 'data' not in d: continue
            roll, pitch, yaw, x, y, z = d['data'].to_roll_pitch_yaw_x_y_z(axes=obs_axes)
            # roll, pitch, yaw = q.to_roll_pitch_yaw()
            # roll, pitch, yaw, x, y, z = 0, 0, 0, 0.01, 0.01, 0.01
            slam.add_edge_factor(ut1, ut2, 
                                 Pose3d(x=x, y=y, z=z, 
                                        yaw=yaw, pitch=pitch, roll=roll), noise)

            # print 'EDGE: ', ut1, ut2, d['data']
            if len(added) and not (ut1 in added or ut2 in added): 
                print 'FAILED!'
                break
            added.add(ut1), added.add(ut2)

        # Relative pose optimization
        print 'Optimizing: '
        t1 = time.time()
        slam.batch_optimization();
        print 'Graph Optimization took', time.time() - t1, 's'
        print '===================================='

        # Update pose
        for ut, node in zip(self.relpose_graph.nodes(), slam.get_nodes()): 
            q = rtf.Quaternion.from_roll_pitch_yaw(node.roll, node.pitch, node.yaw, 
                                                   axes=plot_axes)
            self.final_pose_map[ut] = rtf.RigidTransform(q, [node.x, node.y, node.z])
            #     node.roll, node.pitch, node.yaw, 
            #     node.x, node.y, node.z
            # )



    # # Find least squares solution between two sets of pose pairs
    # def add_edge(self, ut_idx1, ut_idx2):

    #     # Look at previously added nodes, both inter/intra
    #     intranodes1 = set(self.pp_graph_utime_map[ut_idx1].keys())
    #     intranodes2 = set(self.pp_graph_utime_map[ut_idx2].keys())
    #     common_nodes = intranodes1.intersection(intranodes2)
    #     # print 'COMMON_NODES', len(common_nodes), intranodes1, intranodes2
    #     # assert(len(common_nodes>0)
    #     RANSAC_SAMPLE = 10

    #     pp_t12 = []
    #     pp2s = []
    #     for pair in random.sample(common_nodes, min(RANSAC_SAMPLE, len(common_nodes))): 
    #         idx1 = self.pp_graph_pair_map[pair][ut_idx1]
    #         idx2 = self.pp_graph_pair_map[pair][ut_idx2]

    #         pp1 = self.pp_graph.node[idx1]['data'].pp_ij_w
    #         pp2 = self.pp_graph.node[idx2]['data'].pp_ij_w
    #         # self.debug_pose_link(pp1, pp2)

    #         # pp1.quat = rtf.Quaternion.identity()
    #         # pp2.quat = rtf.Quaternion.identity()

    #         # print 'a:', ut_idx1, pp1
    #         # print 'b:', ut_idx2, pp2
            
    #         # Find relative pose
    #         T12 = pp1.inverse() * pp2
    #         # print 'T12:', T12

    #         pp_t12.append(T12)
    #         pp2s.append(pp2)

    #     # print pp_t12
    #     # print '1 and 2: ', intranodes1, intranodes2
    #     # print 'COMMON: ', common_nodes

    #     if not len(pp_t12): return False

    #     # # print ut_idx1, ut_idx2, len(pp_t12)
    #     # for dpose, pose in zip(pp_t12, pp2s): 
    #     #     self.debug_pose(rtf.RigidTransform(dpose.quat, pose.tvec))
            
    #     # Give relative pose information
    #     mu_pp_t12 = mean_pose(pp_t12)
    #     self.relpose_graph.add_edge(ut_idx1, ut_idx2, data= mu_pp_t12)

    #     return True
    #     # print 'MU: ', mu_pp_t12

