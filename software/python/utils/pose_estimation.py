#!/usr/bin/python 
# TODO: Compute relative pose mean instead of absolute mean pose

from __future__ import division

import random, time, itertools, operator, logging
import numpy as np
import cv2
import networkx as nx

# from pysfm.bundle import Camera, Bundle, Track
# from pysfm.bundle_adjuster import BundleAdjuster

from itertools import combinations
from collections import defaultdict
from rigid_transform import Quaternion, RigidTransform, \
    tf_compose, tf_construct, normalize_vec, tf_construct_3pt

import utils.draw_utils as draw_utils
import utils.plot_utils as plot_utils

from utils.geom_utils import scale_points, contours_from_endpoints

np.set_printoptions(precision=2, suppress=True, threshold='nan', linewidth=160)

from fs_pcl_utils import CorrespondenceRejectionSAC, TransformationEstimationSVD
from fs_isam import Slam3D

def monotone_increasing(lst):
    pairs = zip(lst, lst[1:])
    return all(itertools.starmap(operator.le, pairs))

def monotone_decreasing(lst):
    pairs = zip(lst, lst[1:])
    return all(itertools.starmap(operator.ge, pairs))

def monotone(lst): 
    return monotone_decreasing(lst) or monotone_increasing(lst)

def pose_pair_construct(p1,n1,p2,n2): 
    """ Construct pose-pair from (pt1,normal1), and (pt2,normal2) """
    v1 = p2-p1; v1 /= np.linalg.norm(v1)
    R1 = tf_construct(n1,v1)
    return RigidTransform.from_Rt(R1, p1)

class PoseEstimation:
    def __init__(self, data, cluster_inds, params): 

        self.log = logging.getLogger(self.__class__.__name__)

        self.data = data

        # Keep frame pose
        self.params = params

        # Evaluate only the top k moving trajectories in a cluster
        self.eval_top_k = 10

        # label, utime - > pose
        self.initial_pose_map = defaultdict(
            lambda: defaultdict()
        )
        self.final_pose_map = defaultdict(
            lambda: defaultdict()
        )

        # label -> ref_pose
        self.reference_pose_map = dict()

        # Save cluster inds
        self.cluster_inds = cluster_inds

        # Estimate pose given data
        self.per_label_estimation(data)

    def get_initial_poselist_map(self): 
        return self.get_poselist_map(self.initial_pose_map)

    def get_initial_pose_list(self): 
        return self.get_pose_list(self.initial_pose_map)

    def get_final_pose_list(self): 
        return self.get_pose_list(self.final_pose_map)

    def get_poselist_map(self, pose_map, ref_label=-1): 
        """
        Retrieve the pose list of trajectories, and include sensor pose
        """
        utimes_set = set()
        poselist_map = {}

        for label,v in pose_map.iteritems(): 
            assert(label >= 0)

            # Sort by actual utime
            vals = [(self.data.feature_utimes[utime],pose) for utime,pose in v.iteritems()]
            vals.sort(key=lambda x: x[0])

            # Add to pose list
            poselist_map[label] = vals

            # Provide sensor ref. for each utime
            for utime, pose in v.iteritems(): 
                utimes_set.add(self.data.feature_utimes[utime])

            self.log.debug('LABEL %i: POSES: %i' % (label, len(vals)))

        # Create a RIGID reference frame for all utimes
        # ref_label = np.max(map(lambda (l,v): l, pose_map.iteritems())) + 1
        self.log.debug('REF_LABEL: %i' % ref_label)
        poselist_map[ref_label] = [(utime,RigidTransform.identity()) 
                                      for utime in sorted(utimes_set)]
        return poselist_map

    def get_pose_list(self, pose_map, ref_label=-1): 
        """
        Retrieve the pose list of trajectories, and include sensor pose
        """
        pose_list = []
        utimes_set = set()
        for label,v in pose_map.iteritems(): 
            assert(label >= 0)

            # Sort by actual utime
            vals = [(self.data.feature_utimes[utime],pose) for utime,pose in v.iteritems()]
            vals.sort(key=lambda x: x[0])

            # Add to pose list
            pose_list.append((label, vals))

            # Provide sensor ref. for each utime
            for utime, pose in v.iteritems(): 
                utimes_set.add(self.data.feature_utimes[utime])

            self.log.debug('LABEL %i: POSES: %i' % (label, len(vals)))

        # Create a RIGID reference frame for all utimes
        # ref_label = np.max(map(lambda (l,v): l, pose_map.iteritems())) + 1
        self.log.debug('REF_LABEL: %i' % ref_label)
        pose_list.append((ref_label, [(utime,RigidTransform.identity()) 
                                      for utime in sorted(utimes_set)]))

        # Sort by label
        pose_list.sort(key=lambda x: x[0])                    
        return pose_list

    def viz_poses_data(self, pose_map, ch):
        viz_poses, viz_tvecs1, viz_tvecs2 = [], [], []
        for label,v in pose_map.iteritems(): 

            viz_tvecsl = []
            for ut,pose in v.iteritems(): 
                viz_poses.append(pose)
                viz_tvecsl.append(pose.tvec.reshape((-1,3)))
            viz_tvecsl = np.vstack(viz_tvecsl)

            viz_tvecs1.append(viz_tvecsl[:-1])
            viz_tvecs2.append(viz_tvecsl[1:])

        self.log.debug('NUM_POSES: %i' % len(viz_poses))

        draw_utils.publish_pose_list2(ch, viz_poses, sensor_tf='KINECT')       
        if len(viz_tvecs1) and len(viz_tvecs2): 
            draw_utils.publish_line_segments(ch + '_TRAJ', 
                                             np.vstack(viz_tvecs1),
                                             np.vstack(viz_tvecs2), c='g', sensor_tf='KINECT')  

     
    def viz_data(self): 
        self.viz_poses_data(self.initial_pose_map, 'INITIAL_POSE_OPT')
        self.viz_poses_data(self.final_pose_map, 'FINAL_POSE_OPT')

    def get_max_movement_trajectories(self, data, label_inds, k=1): 
        """
        Pick trajectories that have the largest displacement, 
        or least signal to noise ratio
        """

        # For each of the indices, find the overall motion of the trajecory
        top_inds = []
        for ind in label_inds: 
            ut_inds, = np.where(data.idx[ind,:] != -1)

            # Evaluate range of motion 
            X = data.xyz[ind,ut_inds].reshape((-1,3))
            Xmin, Xmax = np.min(X, axis=0), np.max(X, axis=0);

            # Store ind, and corr. range
            top_inds.append((ind, np.linalg.norm(np.fabs(Xmin-Xmax)) ))

        # Sort inds by descending range
        # Reverse true: for longest traj, false: for shortest traj
        top_inds.sort(key=lambda (ind,d): d, reverse=True)

        # Pick inds with top ranges
        return np.array([ ind for ind,score in top_inds[:k] ])

    def per_label_estimation(self, data): 

        # Eval clusters in descending order of size
        for label,label_inds in self.cluster_inds.iteritems(): 
	    self.log.debug('Estimating pose for label %i: %i' % (label,len(label_inds)))

            # Get top k trajectories for articulation estimation
            top_label_inds = self.get_max_movement_trajectories(data, label_inds, k=self.eval_top_k)
            self.log.debug('Top movement inds: %i ' % len(top_label_inds))

            # Check sufficent label inds available (ensuring check from earlier)
            assert(len(top_label_inds) >= 3); 

            # Build overlapping utimes
            # FIX THIS!!
            utimes_inds = []
            for ind in top_label_inds: 
                ut_inds, = np.where(data.idx[ind,:] != -1);
                utimes_inds.extend(ut_inds)

            # Find utime inds that occur at least twice
            utimes_inds, utimes_inv = np.unique(utimes_inds, return_inverse=True)
            utimes_inds_count = np.bincount(utimes_inv)
            assert(len(utimes_inds) == len(utimes_inds_count))

            # Sort utimes so that monotonically increasing
            utimes_inds = np.sort(map(lambda (ind,c): ind, 
                                      filter(lambda (ind,c): c >= 3, 
                                             zip(utimes_inds, utimes_inds_count))))

            # Check sufficient utimes available
            if len(utimes_inds) < 2: 
                # print 'WARNING: LABEL %i: Insufficient utime constraints!' % label
                continue

            # Build pose pair manager
            ppm = PoseManager(use_normals=self.params.with_normals, 
                              inlier_threshold=self.params.inlier_threshold, 
                              est_method=self.params.method)

            # Add observations for top_label_inds, and utimes_inds
            ppm.add_observations(data=data, 
                                 feature_inds=top_label_inds, utimes_inds=utimes_inds)
            
            # Initialize nodes, and correspondences
            # Store initial_pose_map once nodes are established
            ppm.initialize()
            self.reference_pose_map[label] = ppm.reference_pose()

            # BA
            # ppm.ba_optimize(data, top_label_inds)
            
            # Propagate Pose map
            self.log.debug('Add poses from posemanager to label mapped pose_map')
            for ut_idx,pose in ppm.initial_pose_map.iteritems(): 
                self.initial_pose_map[label][ut_idx] = pose
                
            # Establish connections between utimes
            st = time.time()
            ppm.establish_connections(connect_every_k=5)
            self.log.debug('Total time: %4.3f s' % (time.time() - st))

            # Optimize for pose
            ppm.optimize()

            # Propagate Pose map
            self.log.debug('Add poses from posemanager to label mapped pose_map')
            for ut_idx,pose in ppm.final_pose_map.iteritems(): 
                self.final_pose_map[label][ut_idx] = pose

        
class PoseManager: 
    def __init__(self, use_normals=True, inlier_threshold=0.04, est_method='ransac'): 

        self.log = logging.getLogger(self.__class__.__name__)

        self.use_normals = use_normals
        self.inlier_threshold = inlier_threshold
        self.est_method = est_method

        # Keep init utime, and feat index
        self.init_pose = None
        self.init_ut_idx = None
        self.init_feat_idx = None

        # Build ( utime->inds ) map
        self.utime_map = dict()

        # Pose graph
        self.relpose_graph = nx.DiGraph()

        # Init, and Final Pose map of track ( utime->pose )
        self.final_pose_map = dict()
        self.initial_pose_map = dict()

        self.pose_pub = defaultdict(list)
        self.pose_links1_pub, self.pose_links2_pub = [], []
        self.points_pub = defaultdict(list)

    def add_observations(self, data, feature_inds, utimes_inds): 
        # Store data for further lookups
        # Feature inds are in decreasing order of max-movement
        self.data = data
        self.feature_inds = feature_inds
        self.utimes_inds = utimes_inds

        # For each utime, add observation
        for ut_idx in self.utimes_inds: 
                
            # Look at each feature, and add if available
            # Note: linds are in decreasing order of max-movement
            linds, = np.where(data.idx[feature_inds,ut_idx] != -1)
            if len(linds) < 2: continue;
            
            # should be in sorted order of max_movement
            linds = feature_inds[linds] 

            # indices should be atleast 2, since we filtered it above
            assert(len(linds)>=2)

            # Add observation
            self.utime_map[ut_idx] = linds

    # Build utime graph
    def establish_connections(self, connect_every_k):
        st = time.time()

        # Add links between every k utimes
        connect_utimes_inds = self.utimes_inds[::connect_every_k]

        # Add consecutive utimes (skipped utimes)
        added, total = 0, 0

        # for ut_idx0, ut_idx1, ut_idx2 in \
        #     zip(np.roll(connect_utimes_inds, shift=0)[:-2],
        #         np.roll(connect_utimes_inds, shift=-1)[:-2],
        #         np.roll(connect_utimes_inds, shift=-2)[:-2]):
        #     if self.add_edge_factor(ut_idx0, ut_idx1, ut_idx2): 
        #         added += 1
        #     total += 1

        for ut_idx0, ut_idx1 in zip(connect_utimes_inds[:-1], connect_utimes_inds[1:]):
            # a. Check if added: ensures that iSAM does not complain 
            if ut_idx0 not in set(self.relpose_graph.nodes()): 
                continue

            # b. Add edge constraint
            if self.add_edge_factor2(ut_idx0, ut_idx1): 
                added += 1
            total += 1

        self.log.debug('EDGES Added: %i out of %i total' % (added, total))
        self.log.debug('==========> Done adding edges %4.3f' % (time.time() - st))

    def reference_pose(self): 
        return self.init_pose        

    def initialize(self): 
        assert(monotone_increasing(self.utimes_inds));

        # 1a. Pick max movement feature index for initial utime
        # TODO: double check validity of init_ut_idx
        self.init_ut_idx = self.utimes_inds[0]
        self.init_feat_idx = self.utime_map[self.init_ut_idx][0]
        # self.init_feat_idx = self.feature_inds[0]
        self.log.debug('Initializing label at ut_idx: %i' % self.init_ut_idx)

        # 1b. Initialize pose for feature @ initial utime, and max-movement
        self.init_pose = self.setup_initial_pose(self.init_feat_idx, self.init_ut_idx)
        if self.init_pose is None or self.init_ut_idx is None: 
            raise RuntimeError('Failed to initialize cluster! %s %s' % 
                               (self.init_pose, self.init_ut_idx) )

        # 1c. Add node factor for the first node
        added, total = 1, 1
        self.relpose_graph.add_node(self.init_ut_idx, data=self.init_pose)
        # print 'Adding INIT node ', self.relpose_graph.node[self.init_ut_idx]

        # 2. Compute node factors from neighbors (1,2), (2,3), ...
        ut_idx0 = self.init_ut_idx
        for ut_idx1 in self.utimes_inds[1:]: 
        # for ut_idx0, ut_idx1 in zip(self.utimes_inds[:-1], self.utimes_inds[1:]): 

            # 2a. Check if added: ensures that iSAM does not complain 
            if ut_idx0 not in set(self.relpose_graph.nodes()): 
                continue

            # 2b. Use pose @ t-1, to estimate pose @ t
            # Pass init_feat_idx as fail-safe if T(t-1->t) estimation fails
            pose0 = self.relpose_graph.node[ut_idx0]['data']
            pose1 = self.pose_estimation_from_correspondence(pose0, 
                                                             # self.init_feat_idx, 
                                                             ut_idx0, ut_idx1)

            # 2c. Ensure valid pose, and add to pose graph
            if pose1 is not None: 
                # print 'Adding ', pose1, ' between nodes: ', ut_idx0, 'and', ut_idx1
                self.relpose_graph.add_node(ut_idx1, data=pose1)
                added += 1
            total += 1 

        self.log.debug('NODES Added: %i out of %i total' % (added, total))

        # 3. Store initial poses
        for ut_idx,d in self.relpose_graph.nodes(data=True): 
            self.initial_pose_map[ut_idx] = d['data']

        # # Setup initial TF
        # for ut_idx in self.utimes_inds: 
        #     self.init_pose = self.setup_initial_pose(self.init_feat_idx, ut_idx)
        #     if self.init_pose is not None: 
        #         print 'Initializing label at ut_idx: ', ut_idx

        #         # Initialize utime index
        #         self.init_ut_idx = ut_idx
        #         break

        # if self.init_pose is None or self.init_ut_idx is None: 
        #     print 'Warning: Failed to initialize label!'
        #     assert(False)

        # Compute node factor (prior), from first node

        # for idx, ut_idx in enumerate(self.utimes_inds): 
        #     # if idx % 2 != 0: continue
        #     print 'Adding node factor from corr', self.init_feat_idx, self.init_ut_idx, ut_idx
        #     pose = self.add_node_factor_from_correspondence(self.init_pose, 
        #                                                     self.init_feat_idx, 
        #                                                     self.init_ut_idx, ut_idx)
        #     if pose is not None: 
        #         self.relpose_graph.add_node(ut_idx, data=pose)
        #         added += 1

        #     # if self.add_node_factor(self.init_feat_idx, ut_idx): 
        #     # if added >= 1: break

        #     total += 1
        # # print 'INIT: ', self.init_feat_idx

    def setup_initial_pose(self, feat_idx, ut_idx): 
        # Initialize with pos. X from feat_idx, and mean surface normal N0
        p0 = self.data.xyz[feat_idx, ut_idx]
        if np.linalg.norm(p0) < 0.1: 
            self.log.debug('WARNING: Failed to add node factor due to zero depth!!')
            return None

        # Compute surface normal
        n0 = self.data.normal[feat_idx, ut_idx]
        if np.isnan(p0).any() or np.isnan(n0).any() or np.isnan(np.linalg.norm(n0)): 
            self.log.debug('WARNING: Failed to add node factor due to nans!!')
            return None

        # Construct initial pose
        # Note: Flip coordinates for pcl
        R = tf_construct(normalize_vec(n0), np.array([0., -1., 0.]))
        return RigidTransform.from_Rt(R, p0)

    # def add_incomplete_node(self, p0, n0): 
    #     # Optionally refine pose if surface normal available
    #     if np.isnan(n0).any() or np.isnan(np.linalg.norm(n0)) or np.linalg.norm(n0) < 0.1 or \
    #        np.isnan(p0).any() or np.linalg.norm(p0) < 0.1: 
    #         return None

    #     Ri = tf_construct(normalize_vec(n0), np.array([0., 1., 0.]))        
    #     # zold = Ri[:,2].copy()
    #     # R = tf_construct(normalize_vec(n0), normalize_vec(np.cross(zold, n0)))
    #     return RigidTransform.from_Rt(Ri, p0) # or pose.tvec

    def correspondence_estimation_with_normals(self, Xs, Ns, Xt, Nt): 
        rel_poses = []
        inds = np.arange(len(Xs))

        for i1, i2 in combinations(inds, 2): 
            if np.linalg.norm(Xs[i1] - Xs[i2]) < 0.05: continue
            # print 'NORM: ', np.linalg.norm(Ns[i1]), np.linalg.norm(Ns[i2])
            try: 
                p0 = pose_pair_construct(Xs[i1]-Xs[i1], Ns[i1], Xs[i2]-Xs[i1], Ns[i2]) 
                p1 = pose_pair_construct(Xt[i1]-Xs[i1], Nt[i1], Xt[i2]-Xs[i1], Nt[i2]) 
                p01 = p1.oplus(p0.inverse()) 
                rel_poses.extend([p01])

            except: 
                continue

        if not len(rel_poses): None
        rel_poses_sorted = sorted(rel_poses, key=lambda p01: np.mean(np.linalg.norm(Xt-p01*Xs, axis=1)))
        for p01 in rel_poses_sorted: 
            Xt_ = p01 * Xs
            p10 = p01.inverse()

            self.log.debug('Err: ----> %s (Xerr:%f, Nerr:%f)' % \
                (p01, 
                 np.mean(np.linalg.norm(p01 * Xs - Xt, axis=1)), 
                 np.mean(np.sum(np.multiply(p01.rotate_vec(Ns), Nt), axis=1))
             ))

        self.log.debug('-----------------------------')
        draw_utils.publish_pose_list2('rel_poses', rel_poses, sensor_tf='KINECT')       
        return rel_poses_sorted[0]

    def pose_estimation_from_correspondence(self, posew0, ut_idx0, ut_idx1): 
        # Find overlapping nodes between the 2 utimes
        inds1 = set(self.utime_map[ut_idx0])
        inds2 = set(self.utime_map[ut_idx1])
        common_nodes = np.array(list(inds1.intersection(inds2)))

        # Check if there's at least 3 correspondences
        if len(common_nodes) < 3: 
               self.log.debug('WARNING: Insufficient constraints skipping %i, %i!' % \
                  (ut_idx0, ut_idx1))
               return None # self.add_incomplete_node(p0, n0)
        # self.log.debug('COMMON_NODES', common_nodes, ut_idx0, ut_idx1

        # Find the valid relative motion, first remove offsets
        # w0 convention: [0] wrt [w]orld/sensor
        Xw0 = self.data.xyz[common_nodes, ut_idx0]
        Nw0 = self.data.normal[common_nodes, ut_idx0]
        Xw1 = self.data.xyz[common_nodes, ut_idx1]
        Nw1 = self.data.normal[common_nodes, ut_idx1]

        # Remove invalid inds
        Xw01inds = ~np.isnan(Xw0).any(axis=1) & ~np.isnan(Xw1).any(axis=1)
        Xw0, Xw1 = Xw0[Xw01inds], Xw1[Xw01inds]
        Nw0, Nw1 = Nw0[Xw01inds], Nw1[Xw01inds]

        # draw_utils.publish_line_segments('CORR', Xw0, Xw1, sensor_tf='KINECT')
        # time.sleep(1)

        if len(Xw1) < 3: 
            self.log.debug('WARNING: Insufficient VALID constraints skipping %i, %i!' % \
                (ut_idx0, ut_idx1))
            return None # self.add_incomplete_node(p0, n0)

        # Tf pts to initial ref. frame
        pose0w = posew0.inverse()
        X00, X01 = pose0w * Xw0, pose0w * Xw1

        # N00, N01 = pose0w.rotate_vec(Nw0), pose0w.rotate_vec(Nw1)
        # pose01_ = self.correspondence_estimation_with_normals(X00, N00, X01, N01)

        # Remove offsets
        muX00 = np.mean(np.vstack([X00]), axis=0)
        X00, X01 = X00 - muX00, X01 - muX00

        # Pose estimation via Correspondence Rejection SAC
        # T01: T1wrt0
        if self.est_method == 'ransac': 
            T01, inliers = CorrespondenceRejectionSAC(source=X00, target=X01, 
                                                      source_dense=np.array([]),
                                                      target_dense=np.array([]),
                                                      inlier_threshold=self.inlier_threshold, 
                                                      max_iterations=50)
        elif self.est_method == 'svd': 
            T01 = TransformationEstimationSVD(source=X00, target=X01, 
                                              source_dense=np.array([]),
                                              target_dense=np.array([]))
        else: 
            raise RuntimeError('Unknown pose estimation method: %s' % self.est_method)
    
        try: 
            # Initial tf, final tf
            pose01 = RigidTransform.from_homogenous_matrix(T01)
            draw_utils.publish_pose_list2('rel_poses_sac', [pose01], sensor_tf='KINECT')       

            posew1 = posew0.oplus(pose01)
            # print 'pw0:', posew0, '\np01:', pose01, '\npw1:', posew1
            # print 'p01:', pose01

            pose10 = pose01.inverse()
            # print 'PoseX ----> %s (Xerr:%f, Nerr:%f)' % \
            #     (pose01, 
            #      np.mean(np.linalg.norm(pose01 * X00 - X01, axis=1)), 
            #      np.mean(np.sum(np.multiply(pose01.rotate_vec(N00), N01), axis=1))

            # )

            # print 'PoseN ----> %s (Xerr:%f, Nerr:%f)' % \
            #     (pose01_, 
            #      np.mean(np.linalg.norm(pose01_ * X00 - X01, axis=1)), 
            #      np.mean(np.sum(np.multiply(pose01_.rotate_vec(N00), N01), axis=1))

            # )
            # print '=============================================='

        except: 
            self.log.debug('WARNING: Insufficient samples for RANSAC skipping %i, %i!' % \
                (ut_idx0, ut_idx1))
            return None; # self.add_incomplete_node(p0, n0)

        # Don't refine
        return posew1

        # # Optionally refine pose if surface normal available
        # if np.isnan(n0).any() or np.isnan(np.linalg.norm(n0)) or np.linalg.norm(n0) < 0.1 or \
        #    np.isnan(p0).any() or np.linalg.norm(p0) < 0.1: 
        #     return posew1

        # # Correct pose with surface normal
        # poseT = pose.to_homogeneous_matrix()
        # zold = poseT[:3,2].copy()
        # R = tf_construct(normalize_vec(n0), normalize_vec(np.cross(zold, n0)))
        # pose = RigidTransform.from_Rt(R, p0) # or pose.tvec
        
        # # Debug pose
        # # self.debug_pose(pose, channel='DEBUG_T01_relative', sensor_tf='KINECT')
        # # self.debug_pose(init_pose, channel='DEBUG_T01_init', sensor_tf='KINECT')

        # # if np.any(rt12.tvec > 0.0): 
        # #     print 'Init Err: ', np.mean(np.sqrt(np.sum(np.square(X1 - X2), axis=1)))
        # #     print 'INLIERS ', inliers
        # #     print 'Test: ', np.hstack([rt12 * X1, X2])
        # #     print 'RT: ', rt12
        # #     print 'Final Err: ', np.mean(np.sqrt(np.sum(np.square(rt12 * X1 - X2), axis=1)))
        # #     print '*********************************'
        # return pose

    # def add_node_factor(self, feat_idx, ut_idx): 
    #     # Initialize with pos. X from feat_idx, and mean surface normal N0
    #     p0 = self.data.xyz[feat_idx, ut_idx]
    #     if np.linalg.norm(p0) < 0.1: 
    #         print 'WARNING: Failed to add node factor due to zero depth!!'
    #         return False

    #     # Compute mean surface normal
    #     # n0 = self.data.normal[self.feature_inds, ut_idx]
    #     # n0 = np.mean(n0[~np.isnan(n0).any(axis=1)], axis=0)
    #     n0 = self.data.normal[feat_idx, ut_idx]
    #     if np.isnan(p0).any() or np.isnan(n0).any() or np.isnan(np.linalg.norm(n0)): 
    #         print 'WARNING: Failed to add node factor due to nans!!'
    #         return False

    #     # Construct initial pose
    #     # Note: Flip coordinates for pcl
    #     R = tf_construct(n0 * 1.0 / np.linalg.norm(n0), np.array([0., 1., 0.]))
    #     xold, zold = R[:,0].copy(), R[:,2].copy()
    #     R[:,0], R[:,2] = -zold, xold

    #     # Rigid transform with pos. p0, 
    #     # and ref. frame (z pointing opp. to surface normal)
    #     pose = RigidTransform.from_Rt(R, p0)

    #     # Debug mean pose
    #     # self.debug_pose(pose, ch='INIT_POSE')
    #     # print 'publishing INIT_POSE'

    #     # # Set the initial node factor
    #     # if ut_idx == self.init_ut_idx: 
    #     #     self.pose_init = pose

    #     # Add mean pose to the node information
    #     self.relpose_graph.add_node(ut_idx, data=pose)
    #     return True

    def add_edge_factor2(self, ut_idx0, ut_idx1): 
        if ut_idx0 in self.initial_pose_map: 

        # ut_idx0 in self.relpose_graph.nodes() and \
        #    ut_idx1 in self.relpose_graph.nodes() and \
        #    'data' in self.relpose_graph.node[ut_idx0] and \
        #    'data' in self.relpose_graph.node[ut_idx1]: 
            p0 = self.initial_pose_map[ut_idx0]
            p1 = self.pose_estimation_from_correspondence(p0, ut_idx0, ut_idx1)
            # print 'Adding edge p1: ', p1, 'between ', ut_idx0, ' ', ut_idx1, p0
            
        else: 
            return False

        if p1 is None: 
            return False

        # Give relative pose information
        self.relpose_graph.add_edge(ut_idx0, ut_idx1, data=p0.inverse().oplus(p1))
        return True

    # Find least squares solution between two sets of points
    def add_edge_factor(self, ut_idx0, ut_idx1, ut_idx2):
        
        # # Find overlapping nodes between the 2 utimes
        # inds1 = set(self.utime_map[ut_idx1])
        # inds2 = set(self.utime_map[ut_idx2])
        # common_nodes = np.array(list(inds1.intersection(inds2)))

        # # Check if there's at least 3 correspondences
        # if len(common_nodes) < 3: 
        #        # print 'WARNING: Insufficient constraints skipping %i, %i!' % \
        #        #    (ut_idx1, ut_idx2)
        #        return False;
        # # print 'COMMON_NODES', common_nodes, ut_idx1, ut_idx2

        # # Find the relative motion, first remove offsets
        # # self.frame_pose = draw_utils.get_frame('KINECT')
        # X1 = self.data.xyz[common_nodes, ut_idx1]
        # X2 = self.data.xyz[common_nodes, ut_idx2]


        # # Debug the links
        # # self.debug_link(X1.copy(), X2.copy())

        # # Remove offsets
        # muX = np.mean(np.vstack([X1]), axis=0)
        # X1, X2 = X1 - muX, X2 - muX

        # # Pose estimation via Correspondence Rejection SAC
        # # T12 = T1wrt2 = {T_1}^2
        # T12, inliers = CorrespondenceRejectionSAC(source=X1, target=X2, 
        #                                           inlier_threshold=0.01, max_iterations=100)

        if ut_idx0 in self.relpose_graph.nodes() and \
           ut_idx1 in self.relpose_graph.nodes() and \
           'data' in self.relpose_graph.node[ut_idx0] and \
           'data' in self.relpose_graph.node[ut_idx1]: 
            rt0, rt1 = self.relpose_graph.node[ut_idx0]['data'], \
                       self.relpose_graph.node[ut_idx1]['data']
            # T12 = (rt0.inverse() * rt1).to_homogeneous_matrix()

            # T12 = np.eye(4)
            # T12[:3,3] = rt1.tvec - rt0.tvec
            # rt12 = RigidTransform.from_homogenous_matrix(T12)
            rt12 = (rt0.inverse()).oplus(rt1)
        else: 
            return False
        # if np.any(rt12.tvec > 0.0): 
        #     print 'Init Err: ', np.mean(np.sqrt(np.sum(np.square(X1 - X2), axis=1)))
        #     print 'INLIERS ', inliers
        #     print 'Test: ', np.hstack([rt12 * X1, X2])
        #     print 'RT: ', rt12
        #     print 'Final Err: ', np.mean(np.sqrt(np.sum(np.square(rt12 * X1 - X2), axis=1)))
        #     print '*********************************'

        # Give relative pose information
        self.relpose_graph.add_edge(ut_idx1, ut_idx2, data=rt12)
        # print 'Adding edge factor between %i and %i: %s' % (ut_idx1, ut_idx2, rt12)
        return True


    def debug_points(self, points, colors, channel='POINTS_DEBUG', reset=False): 
        if reset: self.points_pub[channel] = []
        self.points_pub[channel].append(np.hstack([points, colors]))
        XC = np.vstack(self.points_pub[channel])
        draw_utils.publish_point_cloud(channel, XC[:,:3], c=XC[:,3:], sensor_tf='KINECT')

    def debug_pose(self, pose, channel='POSE_DEBUG', reset=False, sensor_tf='KINECT'): 
        if reset: self.pose_pub[channel] = []
        self.pose_pub[channel].append(pose)
        draw_utils.publish_pose_list2(channel, self.pose_pub[channel], sensor_tf=sensor_tf)

    def debug_link(self, tvec1, tvec2, reset=False): 
        if reset: self.pose_links1_pub, self.pose_links2_pub = [], []
        self.pose_links1_pub.append(tvec1.reshape((-1,3)))
        self.pose_links2_pub.append(tvec2.reshape((-1,3)))
        draw_utils.publish_line_segments('POSE_DEBUG_EDGES', 
                                         np.vstack(self.pose_links1_pub), 
                                         np.vstack(self.pose_links2_pub), sensor_tf='KINECT')
        draw_utils.publish_point_cloud('POSE_DEBUG_VERTICES', 
                                       np.vstack(self.pose_links2_pub), sensor_tf='KINECT')

    def debug_pose_link(self, pose1, pose2, reset=False): 
        if reset: self.pose_links1_pub, self.pose_links2_pub = [], []
        self.pose_links1_pub.append(pose1.tvec.reshape((-1,3)))
        self.pose_links2_pub.append(pose2.tvec.reshape((-1,3)))
        draw_utils.publish_line_segments('POSE_DEBUG_EDGES', 
                                         np.vstack(self.pose_links1_pub), 
                                         np.vstack(self.pose_links2_pub), sensor_tf='KINECT')       

    # def ba_optimize(self, data, top_label_inds):

    #     # K = np.array([[528.49404721, 0, 319.5],
    #     #               [0, 528.49404721, 239.5],
    #     #               [0, 0, 1]], dtype=np.float64)

    #     K = np.array([[576.09757860, 0, 319.5],
    #                   [0, 576.09757860, 239.5],
    #                   [0, 0, 1]], dtype=np.float64)


    #     # Bundle adjustment
    #     ba = Bundle()
    #     ba.K = K

    #     obj_poses = self.initial_pose_map.values()
    #     cam_poses = map(lambda p: (obj_poses[0].inverse()).oplus(p), obj_poses)
    #     for pose in cam_poses: 
    #         P = pose.to_homogeneous_matrix()[:3]
    #         ba.add_camera(Camera(P[:,:3], P[:,3]))

    #     for tid, ind in enumerate(top_label_inds): 
    #         ut_inds = np.where(data.idx[ind,:] != -1)[0]
    #         track = Track(list(ut_inds.astype(int)), data.xy[ind, ut_inds])
    #         ba.add_track(track)
    #         ba.reconstruction[tid] = data.xyz[ind, ut_inds[1]]

    #     self.log.debug('PREDICT: ', ba.predict(0, 6), data.xy[top_label_inds[6], 0])


    #     for track in ba.tracks:
    #         track.reconstruction = ba.triangulate(track)

    #     reconst = np.vstack([track.reconstruction for track in ba.tracks])
    #     draw_utils.publish_point_cloud('ba_reconstruction', 
    #                                    reconst, sensor_tf='KINECT')

    #     for j,track in enumerate(ba.tracks):
    #         msm = dict()
    #         for i,c in enumerate(ba.cameras): 
    #             try: 
    #                 msm[i] = track.get_measurement(i)
    #             except: 
    #                 pass
    #         track.measurements = msm
    #         # track.measurements = { i : track.get_measurement(i) for i,c in enumerate(ba.cameras) }


    #     param_mask = np.ones(ba.num_params()-6, bool)
    #     param_mask[:6] = False
    #     param_mask[9] = False

    #     b_adj = BundleAdjuster(ba)
    #     b_adj.optimize(param_mask, max_steps=20)
    #     # for cam in ba.cameras: 
    #     #     print cam.R, cam.t
    #     cam_opt_poses = [RigidTransform.from_Rt(cam.R, cam.t) for cam in b_adj.bundle.cameras]
    #     draw_utils.publish_pose_list2('cam_poses', cam_poses, sensor_tf='KINECT')       
    #     draw_utils.publish_pose_list2('cam_opt_poses', cam_opt_poses, sensor_tf='KINECT')       


    # SLAM relative pose formulation
    def optimize(self): 
        self.log.debug('====================================')
        self.log.debug('POSE GRAPH OPTIMIZATION')
        self.log.debug('Optimizing over %i nodes, %i edges' % \
                       (self.relpose_graph.number_of_nodes(), self.relpose_graph.number_of_edges()))
        self.log.debug('====================================')

        # Init Slam3d isam class
        slam = Slam3D()

        # Inconsistent coordinate frames
        obs_axes, plot_axes = 'szyx', 'sxyz'

        # Checks, and inits
        self.log.debug('====================================')
        assert(self.init_ut_idx is not None)
        self.log.debug('First Node utime: %i' % self.init_ut_idx)

        # Add node factors (priors)
        self.log.debug('====================================')
        self.log.debug('Adding priors: ')
        init_nodes = []
        for ut_idx,d in self.relpose_graph.nodes(data=True): 
            if 'data' in d: 

                # Node cov. 
                if ut_idx == self.init_ut_idx: 
                    noise = np.diag([0.005, 0.005, 0.005, 0.005, 0.005, 0.005])
                else: 
                    noise = np.diag([0.1, 0.1, 0.1, 0.2, 0.2, 0.2])

                # Add SLAM node with ut_idx identifying it
                slam.add_node_with_factor(i=ut_idx, 
                                          T=d['data'].to_homogeneous_matrix(), 
                                          noise=noise);

                init_nodes.append(ut_idx)
            else: 
                slam.add_node(ut_idx)
        self.log.debug('INIT NODES: %s' % init_nodes)
        
        # Add edge factors
        self.log.debug('====================================')
        self.log.debug('Adding edges: ')
        # self.log.debug('Init NODES: ', init_nodes

        # Edge cov. 
        noise = np.diag([0.1, 0.1, 0.1, 0.01, 0.01, 0.01])

        # Perform a DFS, so that edge factors are not added arbitrarily
        # iSAM chokes when adding factors that are not connected
        edges = set([e for e in self.relpose_graph.edges()])
    
        # Start from each of the inited nodes
        for init_node in init_nodes: 
            dfs_edges = nx.dfs_edges(self.relpose_graph, source=init_node)
            
            # Remove edges from the original set of edges
            edges_to_visit = []
            for edge in dfs_edges: 
                if edge in edges: 
                    edges.remove(edge)
                    # self.log.debug('Removing', edge, edges
                    edges_to_visit.append(edge)

            if len(edges_to_visit): 
                self.log.debug('EDGES: %s ' % edges_to_visit)

            # Recursively 
            for ut_idx1, ut_idx2 in edges_to_visit:

                # Check if available
                if ut_idx1 not in self.relpose_graph.nodes() or \
                   ut_idx2 not in self.relpose_graph.nodes(): 
                    assert('Node not available! Failed to add relpose edge!')

                # Get the data
                d = self.relpose_graph[ut_idx1][ut_idx2]

                # Add SLAM edge factor connecting ut_idx1 and ut_idx2
                slam.add_edge_factor(i=ut_idx1, j=ut_idx2, 
                                     T=d['data'].to_homogeneous_matrix(), 
                                     noise=noise);

        # Pose Graph Optimization
        self.log.debug('====================================')
        self.log.debug('Optimizing: ')
        t1 = time.time()
        slam.batch_optimization();
        self.log.debug('Optimization took %4.3f s' % (time.time() - t1))
        self.log.debug('====================================')

        # Update pose
        self.log.debug('====================================')
        self.log.debug('Pose update:')
        for ut_idx, node in zip(self.relpose_graph.nodes(), slam.get_nodes()): 
            self.final_pose_map[ut_idx] = RigidTransform.from_homogenous_matrix(node)


if __name__ == "__main__": 
    from rigid_transform import make_random_transform

    print 'Test CorrespondenceRejectionSAC'
    np.set_printoptions(precision=3, suppress=True)

    all_inliers = []
    for _ in range(100): 
        # Make a random transform
        rt = make_random_transform(30)

        # Randomly sample points
        Xs = np.random.normal(loc=(np.random.random(),np.random.random(),np.random.random()), 
                              scale=(np.random.random(),np.random.random(),np.random.random()),
                              size=(20,3))

        # Rigidly transform source points to target
        # Xsh = np.hstack([Xs, np.ones((len(Xs),1))])
        Xt = rt * Xs

        # Add noise
        W = np.random.normal(loc=(0,0,0), scale=(np.random.random() * 0.01,
                                                 np.random.random() * 0.01,
                                                 np.random.random() * 0.01), size=(20,3))
        # print 'Noise: ', W
        Xt += W

        # Corr. Rejection SAC
        Tsac, inliers = CorrespondenceRejectionSAC(source=Xs[:,:3], target=Xt[:,:3], 
                                          inlier_threshold=0.005, max_iterations=100)

        T = rt.to_homogeneous_matrix()
        all_inliers.append(inliers)

        # print 'T: ', T
        # print 'Tsac: ', Tsac
        # print T - Tsac
        assert(np.all(np.fabs(T - Tsac) < 1e-1))
    print 'All Inliers: ', all_inliers
    print 'OK'

