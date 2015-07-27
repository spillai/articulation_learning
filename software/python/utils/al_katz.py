#!/usr/bin/env python
import cv2
import numpy as np
import draw_utils

import itertools, logging
from collections import defaultdict

from utils.rigid_transform import RigidTransform
from utils.db_utils import DictDB, AttrDict
from utils.data_containers import Feature3DData, Pose3DData

from utils.geom_utils import scale_points, contours_from_endpoints
from utils.trackers.offline_klt import OfflineStandardKLT
from utils.trajectory_analysis import TrajectoryClustering
from utils.pose_estimation import PoseEstimation
from utils.articulation_analysis import ArticulationAnalysis

from utils.camera_utils import KinectCamera
from utils.correspondence_estimation import BetweenImagePoseEstimator, remove_nans
from utils.bow_utils import ObjectLocalizationBOW
from utils.frame_utils import frame_desc
np.set_printoptions(precision=2, suppress=True, threshold='nan', linewidth=160)

class KatzALfD:  
    def __init__(self, params, debug_dir=None): 

        self.log = logging.getLogger(self.__class__.__name__)

        self.features_map_ = dict()
        self.params_ = params
        self.debug_dir = debug_dir

    # =============================================================================
    # Training 
    # =============================================================================
    def setup_features(self, features): 
        # Smoothing and pruning of feature trajectories
        # features.prune_by_length(
        #     min_track_length=self.params_.min_track_length
        # )


        # Remove utimes/inds that are discontinuous
        features.prune_discontinuous(            
            min_feature_continuity_distance=self.params_.feature_selection.min_feature_continuity_distance,
            min_normal_continuity_angle=self.params_.feature_selection.min_normal_continuity_angle
        )

        # Pick mostly top moving features
        features.pick_top_by_displacement(k=300)

        return features

    def process_demonstration(self, frames, name): 
        # Extract images, normals, clouds
        for f in frames: 
            # f.fastBilateralFilter(20.0, 0.04)
            f.computeNormals(0.5)

        images = map(lambda x: x.getRGB(), frames)
        Xs = map(lambda x: x.getCloud(), frames)
        Ns = map(lambda x: x.getNormals(), frames)
        
        carr = images[-1][::4,::4].reshape(-1,3) * 1.0 / 255
        draw_utils.publish_point_cloud('CLOUD_KINECT', 
                                       Xs[-1][::4,::4].reshape(-1,3), c=carr[:,::-1].reshape(-1,3),
                                       sensor_tf='KINECT')

        # Extract KLT features
        self.log.info('Feature Extaction ---------------------------------')
        klt = OfflineStandardKLT(images=images, oklt_params=self.params_.tracker)
        klt.features = klt.get_feature_data(Xs, Ns)

        # Process and extract relevant features
        self.log.info('Process Features ----------------------------------')
        features = self.setup_features(klt.features)

        # Viz feature trajectories (after outlier removal)
        features.viz_data(utimes_inds=None)  

        # Trajectory Clustering
        self.log.info('Feature Clustering --------------------------------')
        clusters = self.clustering(features)

        # # # Write to video
        # # features.write_video(frames, 'test.avi')
        # Pose Estimation
        self.log.info('Pose Estimation -----------------------------------')
        poses = self.pose_estimation(features, clusters).get_initial_pose_list()

        # Articulation Analysis
        self.log.info('Articulation Analysis -----------------------------')
        art_est = self.articulation_analysis(poses)

        # Temporary! FIX! TODO
        self.log.info('Saving all variables, and adding to DB -----------')
        # self.model = AttrDict({'features':features, 'clusters':clusters, 'poses': poses, 'art_est':art_est})
        self.model = AttrDict({'poses': poses, 'art_est':art_est})
 
    # =============================================================================
    # Trajectory Clustering, PoseEstimation, Articulation Estimation
    # =============================================================================
    def clustering(self, features): 
        """
        Clustering trajectories based on rigid-point-pair-feature
        Could provide a range, and run for several range chunks
        """
        return TrajectoryEpsilonClustering(features, params=self.params_.clustering, 
                                           visualize=False, save_dir=self.debug_dir)

    def pose_estimation(self, features, clusters): 
        """Pose Estimation from identified clusters"""
        pose_est = PoseEstimation(data=features, 
                                  cluster_inds=clusters.cluster_inds, 
                                  params=self.params_.pose_estimation)
        pose_est.viz_data()
        return pose_est

    def articulation_analysis(self, poses, viz_attrib='pose_projected'): 
        """Analyze articulation give final pose estimates"""
        art_est = ArticulationAnalysis(poses, params=self.params_.articulation, viz_attrib=viz_attrib)
        return art_est

class TrajectoryEpsilonClustering: 
    def __init__(self, data, params, visualize=False, save_dir=None): 

        self.log = logging.getLogger(self.__class__.__name__)

        # Save params
        self.params = params
        self.visualize = visualize
        self.save_dir = save_dir

        # Check valid_feat_inds
        if not len(data.valid_feat_inds): 
            raise RuntimeError('No features available for clustering')
        
        # Cluster index map
        # [label] -> [inds]
        self.cluster_inds = dict()

	# 1. Init labels
	data.labels = np.zeros(data.num_feats, dtype=np.int32);
	data.labels.fill(-1);

        # 2. Single chunk clustering
        # Build similarity matrix for the full time range
        utimes_inds = np.arange(data.num_utimes)
        W_dist = self.build_similarity_matrix(data, utimes_inds)  

        # 3. Build weight matrix with displacements
        W = W_dist

        # 4. Find the clustering between trajectories
        self.log.debug('===> Cluster the trajectories ...')
        valid_feat_labels = self.cluster(data, W )
        data.labels[data.valid_feat_inds] = valid_feat_labels
        self.log.debug('===> Done clustering')

        # 5. Build cluster index map
        # Only map top k clusters (based on size of each cluster)
        self.label_top_clusters(data, top_k=self.params.label_top_k)

        # Visualize clustering
        self.viz_clustering(data, data.valid_feat_inds, utimes_inds)
        self.log.debug('Number of labels: %s' % np.unique(data.labels))

    # Build cluster index map
    # Output: cluster_inds [label] -> [inds corr. to label]
    def label_top_clusters(self, data, top_k): 
        
        # Count number of instances of labels
        # Avoid -1 label
        unique_labels, unique_inv = np.unique(data.labels[np.where(data.labels != -1)], 
                                              return_inverse=True)

        num_labels = len(unique_labels)
        label_count = np.bincount(unique_inv)
        label_votes = zip(unique_labels, label_count)
        self.log.debug('Total: %i labels' % num_labels)

        # Remove labels that have count < 3 
        label_votes = filter(lambda (l,c): c >= 3, label_votes)

        # Sort (label,votes) by votes, and pick the labels with top_k votes
        label_votes.sort(key=lambda (l,c): c, reverse=True)
        labels = map(lambda (l,c): l, label_votes[:top_k])
        self.log.debug('Picking top %i labels: %s' % (len(labels), label_votes[:top_k]))

        # Unlabel rest of the labeled trajectories
        for label in unique_labels: 
            if label in labels: continue
            inds, = np.where(data.labels == label)
            data.labels[inds] = -1

        # Find exemplars for each cluster 
	for label in labels: 
            assert(label >= 0)

            # Indices that are assigned the "label"
	    inds, = np.where(data.labels == label)
            if len(inds) < 3: 
                # Complain about insufficient constraints
                raise RuntimeError('Label %i cluster has only (%i) data constraints!' % (label,len(inds)))

            # Map cluster indices
	    self.cluster_inds[label] = np.array(inds, dtype=np.int32)

    def build_similarity_matrix(self, data, utimes_inds): 
        # Find the overlapping set of features between trajectories
        self.log.debug('===> Extracting overlapping features ...')
        overlap_dict = self.extract_overlapping_features(data, 
                                                         data.valid_feat_inds, 
                                                         utimes_inds)
        self.log.debug('--- Done extracting overlapping features')
        

        # Construct the similarity matrix between trajectories
        # Utime slicing is encapsulated within the overlap_dict
        self.log.debug('===> Compute trajectory similarity ...')
        W_dist = self.compute_trajectory_similarity(data, data.valid_feat_inds, overlap_dict)
        self.log.debug('--- Done computing pose-pairs')

        # Viz similarity in viewer
        # if self.visualize: self.viz_similarity(data, posepair_dict)

        return W_dist;

    # Find overlapping tracks ==============================================
    def extract_overlapping_features(self, data, valid_inds, utimes_inds):
        overlap_dict = {}

        num_inds = len(valid_inds);         
        niter, total_iters = 0, num_inds*(num_inds-1)/2;
        for jind,kind in itertools.combinations(valid_inds, 2):
            # if niter % 200 == 0: self.log.debug('Processed: %i out of %i' % (niter, total_iters))
            niter += 1 

            # find nz utime indices
            tj, = np.where(data.idx[jind,utimes_inds] != -1) # <- space of utimes_inds
            tj = utimes_inds[tj] # <- original space
            tk, = np.where(data.idx[kind,utimes_inds] != -1)
            tk = utimes_inds[tk]

            # set intersection
            intersection = np.intersect1d(tj,tk);
            if intersection.size < self.params.min_track_intersection: continue;
            
            # Build overlap dictionary
            overlap_dict[tuple(sorted((jind,kind)))] = intersection;
        self.log.debug('Processed: %i out of %i' % (niter, total_iters))
        return overlap_dict;

    def compute_relative_motion(self, data, id_pair, overlap_utimes_inds): 
        # Evaluate for id_pair, and overlap_utimes
        jind,kind = id_pair

        # Mean distance change
        distance = data.xyz[jind,overlap_utimes_inds,:] - \
                   data.xyz[kind,overlap_utimes_inds,:]
        distance = distance[~np.isnan(distance).any(axis=1)] 

        l2_distance = np.linalg.norm(distance, axis=1)
        min_l2, max_l2 = np.min(l2_distance), np.max(l2_distance)
        return max_l2-min_l2
        

    # Compute Pose-Pairs for overlapping tracks ============================
    def compute_trajectory_similarity(self, data, valid_inds, overlap_dict): 

        # Valid inds map:  valid_inds2inds => [4,5,9,...] -> [0,1,2,...] = {4:0,5:1,9:2}
        num_valid_feats = len(valid_inds);
        valid_inds2inds = dict( zip( valid_inds, np.arange(num_valid_feats) ) )

        # Build Weight/Affinity matrix
        W_dist = [set([r]) for r in range(num_valid_feats)]

        # Compute Pose Pair Similarity
        # posepair_dict = dict();
        for id_pair,overlap_utimes_inds in overlap_dict.iteritems():

            # eval. relative motion
            motion = self.compute_relative_motion(data, id_pair, overlap_utimes_inds)

            if np.isnan(motion) or motion > self.params.distance_eps: continue
            # posepair_dict[id_pair] = motion;
            # Key stored is valid_feat_id -> valid_feat_id_idx via valid_inds2inds
            r, c = valid_inds2inds[id_pair[0]], valid_inds2inds[id_pair[1]]

            W_dist[r].add(c)
            W_dist[c].add(r)

        return W_dist
        
    # Clustering ===========================================================
    def cluster(self, data, W): 

        # Cluster labels on the valid features
        added = set()
        valid_labels = np.ones(len(W), dtype=np.int32) * -1

        # Sort based on number of neighbors
        W_sorted = sorted(W, key=lambda x: len(x), reverse=True)

        label = 0
        for row in W_sorted: 
            # Find nodes not added yet
            rem = row - added

            # Label remaining nodes
            rem_inds = np.array(list(rem), dtype=np.int32)
            
            valid_labels[rem_inds] = label
            label += 1

            # Add to list
            added = added.union(rem)
            
        # print valid_labels
        return valid_labels
        

    # Draw segmented features ==============================================
    def viz_clustering(self, data, valid_inds, utimes_inds): 
        # Visualize clusters in descending order of size
        viz_pts, viz_normals, viz_colors = [], [], []
        viz_traj1, viz_traj2 = [], []

        # Indices of label in the full dataset (could also look at valid dataset)
        labels = np.unique(data.labels)
        num_labels = len(labels)
        for label in labels:
            # Only among valid_inds
            inds, = np.where(data.labels[valid_inds] == label)
            inds = valid_inds[inds]
            # print 'LABEL: ', label, inds

            # Publish each feature ind
            for idx in inds: 
                ut_inds, = np.where(data.idx[idx,utimes_inds] != -1);
                ut_inds = utimes_inds[ut_inds]

                viz_pts.append(data.xyz[idx,ut_inds,:])

                viz_traj1.append(data.xyz[idx,ut_inds[:-1],:])
                viz_traj2.append(data.xyz[idx,ut_inds[1:],:])

                viz_normals.append(data.xyz[idx,ut_inds,:] + data.normal[idx,ut_inds,:]*0.04)

                # Color by label
                carr = draw_utils.get_color_arr(label, len(ut_inds), palette_size=num_labels,
                                             color_by='label')
                viz_colors.append(carr)

        viz_pts = np.vstack(viz_pts)
        viz_colors = np.vstack(viz_colors)
        viz_normals = np.vstack(viz_normals)
        viz_traj1, viz_traj2 = np.vstack(viz_traj1), np.vstack(viz_traj2)

        draw_utils.publish_point_cloud('CLUSTERED_PTS', viz_pts, 
                                       c=viz_colors, sensor_tf='KINECT')
        draw_utils.publish_line_segments('CLUSTERED_NORMAL', viz_pts, viz_normals, 
                                         c=viz_colors, sensor_tf='KINECT')
        draw_utils.publish_line_segments('CLUSTERED_TRAJ', viz_traj1, viz_traj2, 
                                         c=viz_colors, sensor_tf='KINECT')

    # # Viz feature similarity ===============================================
    # def viz_similarity(self, data, posepair_dict): 
    #     # debug_ids = [8,9,10,11]

    #     viz_pts, viz_normals, viz_colors = [], [], []
    #     for id_pair,ppd in posepair_dict.iteritems():

    #         # if not (id_pair[0] in debug_ids or id_pair[1] in debug_ids): continue
    #         # print id_pair, ppd.distance_match

    #         # Get Pair and 
    #         idx1, idx2 = id_pair
    #         c = plt.cm.jet(ppd.distance_match)
    #         ut_inds1, = np.where(data.idx[idx1,ppd.overlap_utimes_inds] != -1)
    #         ut_inds1 = ppd.overlap_utimes_inds[ut_inds1]
    #         ut_inds2, = np.where(data.idx[idx2,ppd.overlap_utimes_inds] != -1)
    #         ut_inds2 = ppd.overlap_utimes_inds[ut_inds2]

    #         viz_pts.append(data.xyz[idx1,ut_inds1[-1],:])
    #         #viz_pts.append(data.xyz[idx2,ut_inds2[-1],:])
    #         viz_normals.append(data.xyz[idx2,ut_inds2[-1],:])
    #         viz_colors.append(c);

    #     if not len(viz_pts): return

    #     viz_pts = np.vstack(viz_pts)
    #     viz_colors = np.vstack(viz_colors)
    #     viz_normals = np.vstack(viz_normals)

    #     draw_utils.publish_point_cloud('EDGE_DATA', viz_pts, c=viz_colors, sensor_tf='KINECT')
    #     draw_utils.publish_line_segments('EDGE_WEIGHTS', viz_pts, viz_normals, 
    #                                      c=viz_colors, sensor_tf='KINECT')
