#!/usr/bin/python 
# TODO: Compute relative pose mean instead of absolute mean pose

from __future__ import division

import random, time, itertools, operator
import numpy as np
np.set_printoptions(precision=3, suppress=True)
import cv2
import networkx as nx

from collections import defaultdict
from rigid_transform import Quaternion, RigidTransform, tf_compose, tf_construct

import utils.draw_utils as draw_utils
import utils.plot_utils as plot_utils

from utils.trackers.tracker_utils import AprilTagsFeatureTracker
from utils.pose_utils import mean_pose

from fs_pcl_utils import CorrespondenceRejectionSAC
from fs_isam import Slam3D
from fs_utils import RichFeatureMatcher

def remove_nans(pts): 
    return pts[~np.isnan(pts).any(axis=1)]

class BetweenImagePoseEstimator: 
    def __init__(self, frames_map, masks_map, visualize): 

        # Keep data
        self.final_pose_map = dict()
        self.frames_map = frames_map
        self.masks_map = masks_map
        self.frames_idx = dict([ (name, idx) for idx, name in enumerate(frames_map.keys())])
        self.pose_links1_pub, self.pose_links2_pub = [], []

        if not isinstance(frames_map, dict): 
            raise TypeError("Expected dictionary for frames_map")

        if not len(frames_map): 
            raise AssertionError("No frames available")

        # Build graph
        self.g = nx.Graph()

        # Init Slam3d isam class
        slam = Slam3D()

        # Pick an initial node, rooted at identity
        init_node = frames_map.keys()[0];

        # BFS edges 
        for node1, node2 in itertools.combinations(frames_map.keys(), r=2): 
            print 'EDGE: ', node1, node2, node2 in self.frames_map, node1 in self.frames_map

            # Compute features, and correspondences
            g1,g2 = frames_map[node1].getRGB(), frames_map[node2].getRGB()
            g1 = cv2.GaussianBlur(g1, (5, 5), 0)
            g2 = cv2.GaussianBlur(g2, (5, 5), 0)

            # cv2.imshow("mask", masks_map[node1].astype(np.uint8))

            # Compute PyramidFAST + SURF matches
            rfm = RichFeatureMatcher("GFTT", "SIFT", g1, g2, 
                                     np.array([], dtype=np.uint8), np.array([], dtype=np.uint8))
            
            # Get correspondences
            pts1, pts2 = rfm.getMatches()

            # Compute relative pose 
            print node1, node2, self.frames_map[node1], self.frames_map[node2]
            pose12, err = self.get_relative_pose(self.frames_map[node1], pts1, 
                                                 self.frames_map[node2], pts2)
            # pose12, err = self.get_relative_pose2(self.frames_map[node1], pts1, 
            #                                       self.frames_map[node2], pts2)

            if pose12 is None: continue

            # Add relative pose data
            # iSAM expects pose in the inverse form
            self.g.add_edge(node1, node2, data={(node1,node2):pose12, 
                                                (node2,node1):pose12.inverse()}, 
                            weight=err)

            print 'Adding edge factor: ', self.frames_idx[node1], self.frames_idx[node2], \
                pose12
            
            # time.sleep(2)
            # break
            # poses = [RigidTransform.identity(), rt12]
            # texts = ['CAM-1', 'CAM-0']
            # draw_utils.publish_pose_list2('BETWEEN_IMAGE_POSE', poses, texts=texts)

        print '===================================='
        print 'Graph NODES: ', self.g.nodes()
        print 'Graph EDGES: ', self.g.edges()
        print '===================================='

        # Add initial node with a identity factor
        for node in self.frames_map.keys(): 
            node_idx = self.frames_idx[node]
            if node == init_node: 
                noise = np.eye(6) * 0.001
                slam.add_node_with_factor(i=node_idx, 
                                          T=RigidTransform.identity().to_homogeneous_matrix(), 
                                          noise=noise);
                print 'Adding node factor: ', node_idx
            else: 
                slam.add_node(node_idx)
                print 'Adding node: ', node_idx

        # Adding edges
        T = nx.minimum_spanning_tree(self.g, weight='weight')
        bfs_edges = nx.bfs_edges(T, source=init_node)

        # Keep track of added edges
        edges_added = set()
        print 'Adding BFS edges'
        for edge in bfs_edges: 
            node1, node2 = edge
            print 'EDGE: ', node1, node2

            # Get the pose between the nodes
            pose12 = self.g[node1][node2]['data'][(node1,node2)]
            
            print 'EDGE pose : ', pose12 

            # Add relative pose edge factor
            noise = np.diag([0.005, 0.005, 0.005, 0.01, 0.01, 0.01])
            slam.add_edge_factor(i=self.frames_idx[node1], j=self.frames_idx[node2], 
                                 T=pose12.to_homogeneous_matrix(), 
                                 noise=noise);            
            edges_added.add(tuple(sorted(edge)))

        # Pose Graph Optimization
        print '===================================='
        print 'Optimizing: '
        t1 = time.time()
        slam.batch_optimization();
        print 'Optimization took', time.time() - t1, 's'
        print '===================================='

        # Update pose
        # Poses repr. in the local frame
        print '===================================='
        print 'Pose update:'
        for node_name, node_T in zip(self.g.nodes(), slam.get_nodes()): 
            self.final_pose_map[node_name] = RigidTransform.from_homogenous_matrix(node_T) 

        print 'EVALUATED: ', self.final_pose_map.keys()

        # plot_utils.imshow(np.hstack([frame.getRGB().copy() 
        #                              for frame in frames_map.values()]), 
        #                   pattern='bgr')

    def get_viewpoints(self): 
        return self.final_pose_map

    def get_final_pose(self, name): 
        return self.final_pose_map[name]

    def get_relative_cloud_pose(self, name1, name2): 
        return self.g[name1][name2]['data'][(name1,name2)]

    def debug_link(self, tvec1, tvec2, reset=False): 
        if reset: self.pose_links1_pub, self.pose_links2_pub = [], []
        self.pose_links1_pub.append(tvec1.reshape((-1,3)))
        self.pose_links2_pub.append(tvec2.reshape((-1,3)))
        draw_utils.publish_line_segments('MATCH_DEBUG_EDGES', 
                                         np.vstack(self.pose_links1_pub), 
                                         np.vstack(self.pose_links2_pub), sensor_tf='KINECT')
        draw_utils.publish_point_cloud('MATCH_DEBUG_VERTICES', 
                                       np.vstack(self.pose_links2_pub), sensor_tf='KINECT')

    def get_relative_pose(self, frame1, pts1, frame2, pts2): 
        if not len(pts1) or not len(pts2): 
            print 'Warning: No points matched!'
            return None
        assert(len(pts1) == len(pts2))

        cloud1, cloud2 = frame1.getCloud(), frame2.getCloud()
        normals1, normals2 = frame1.getNormals(), frame2.getNormals()

        pts3d1 = np.vstack(map(lambda pt: cloud1[pt[1],pt[0]], np.vstack(pts1).astype(int)))
        pts3d2 = np.vstack(map(lambda pt: cloud2[pt[1],pt[0]], np.vstack(pts2).astype(int)))

        nm3d1 = np.vstack(map(lambda pt: normals1[pt[1],pt[0]], 
                                   np.vstack(pts1).astype(int)))
        nm3d2 = np.vstack(map(lambda pt: normals2[pt[1],pt[0]], 
                                   np.vstack(pts2).astype(int)))

        ilen = len(pts3d1)
        pts3d = np.hstack([pts3d1, pts3d2, nm3d1, nm3d2])
        pts3d = pts3d[~np.isnan(pts3d).any(axis=1)]
        flen = len(pts3d)

        print 'Valid: %i / %i pts' % (flen, ilen)

        # Pose estimation via Correspondence Rejection SAC
        # T12 = T1wrt2 = {T_1}^2
        Xs, Xt, Ns, Nt = pts3d[:,:3], pts3d[:,3:6], \
                         pts3d[:,6:9], pts3d[:,9:12]

        # Debug the links
        muXs = np.mean(Xs, axis=0)

        # Transform cloud to local frame
        sparse_cloud1 = remove_nans(cloud1[::6,::6].reshape((-1,3)))
        sparse_cloud2 = remove_nans(cloud2[::6,::6].reshape((-1,3)))

        T12, inliers = CorrespondenceRejectionSAC(source=Xs, 
                                                  target=Xt, 
                                                  source_dense=np.array([]), # sparse_cloud1, 
                                                  # source_dense=sparse_cloud1, 
                                                  target_dense=np.array([]), # sparse_cloud2, 
                                                  # target_dense=sparse_cloud2,
                                                  inlier_threshold=0.1, 
                                                  max_iterations=300)
        
        try: 
            pose12_ = RigidTransform.from_homogenous_matrix(T12)
        except: 
            return RigidTransform(), 1e9

        print 'Init Err: ', np.mean(np.sqrt(np.sum(np.square(Xs - Xt), axis=1))), Xs[inliers]
        # print 'Test: ', np.hstack([rt12 * Xs, Xt])
        print 'RT: ', pose12_
        err = np.mean(np.sqrt(np.sum(np.square(pose12_ * Xs[inliers] - Xt[inliers]), axis=1)))
        print 'Final Err: ', err
        print 'INLIERS: ', len(inliers), T12

        # draw_utils.publish_point_cloud('0 CLOUD', sparse_cloud1, 
        #                                c='r', sensor_tf='KINECT')
        # draw_utils.publish_point_cloud('1 CLOUD', sparse_cloud2, 
        #                                c='b', sensor_tf='KINECT')
        # draw_utils.publish_point_cloud('2 CLOUD', pose12_ * sparse_cloud1, 
        #                                c='g', sensor_tf='KINECT')

        # Penalize large displacements between cameras
        return pose12_, np.linalg.norm(pose12_.tvec)

    # def get_relative_pose_from_tags(self, frame1, pts1, frame2, pts2): 
    #     self.atag = AprilTagsFeatureTracker(expected_id=0)
    #     self.frame_pose = draw_utils.get_frame('KINECT')
    #     if not len(pts1) or not len(pts2): 
    #         print 'Warning: No points matched!'
    #         return None
    #     assert(len(pts1) == len(pts2))

    #     cloud1, cloud2 = frame1.getCloud(), frame2.getCloud()
    #     normals1, normals2 = frame1.getNormals(), frame2.getNormals()

    #     pts3d1 = np.vstack(map(lambda pt: cloud1[pt[1],pt[0]], np.vstack(pts1).astype(int)))
    #     pts3d2 = np.vstack(map(lambda pt: cloud2[pt[1],pt[0]], np.vstack(pts2).astype(int)))

    #     nm3d1 = np.vstack(map(lambda pt: normals1[pt[1],pt[0]], 
    #                                np.vstack(pts1).astype(int)))
    #     nm3d2 = np.vstack(map(lambda pt: normals2[pt[1],pt[0]], 
    #                                np.vstack(pts2).astype(int)))

    #     ilen = len(pts3d1)
    #     pts3d = np.hstack([pts3d1, pts3d2, nm3d1, nm3d2])
    #     pts3d = pts3d[~np.isnan(pts3d).any(axis=1)]
    #     flen = len(pts3d)

    #     print 'Valid: %i / %i pts' % (flen, ilen)

    #     # Pose estimation via Correspondence Rejection SAC
    #     # T12 = T1wrt2 = {T_1}^2
    #     Xs, Xt, Ns, Nt = pts3d[:,:3], pts3d[:,3:6], \
    #                      pts3d[:,6:9], pts3d[:,9:12]

    #     RTS, RTT = [], []
    #     for idx in range(len(Xs)): 
    #         xs, xt, ns, nt = Xs[idx], Xt[idx], Ns[idx], Nt[idx]
    #         if np.fabs(ns[2]) < 0.85 or np.fabs(nt[2]) < 0.85: continue
            
    #         Rs = tf_construct(ns * 1.0 / np.linalg.norm(ns), np.array([0., 1., 0.]))
    #         Rt = tf_construct(nt * 1.0 / np.linalg.norm(nt), np.array([0., 1., 0.]))
    #         rts, rtt = RigidTransform.from_Rt(Rs, xs), RigidTransform.from_Rt(Rt, xt)

    #         # Transform poses to local frame
    #         RTS.append(self.frame_pose * rts), RTT.append(self.frame_pose * rtt)
    #         # break

    #     # Transform cloud to local frame
    #     sparse_cloud1 = self.frame_pose * remove_nans(cloud1[::6,::6].reshape((-1,3)))
    #     sparse_cloud2 = self.frame_pose * remove_nans(cloud2[::6,::6].reshape((-1,3)))

    #     # Predictions
    #     RTrel = []
    #     for rts, rtt in zip(RTS, RTT): 

    #         # Compute the relative tf
    #         rel = rtt * rts.inverse()

    #         # DEBUG: Display the relative tf, in the local frame
    #         # RTrel.append(rel.inverse() * self.frame_pose)
    #         RTrel.append(rel)

    #         # print 'REL: ', rel, rts * rel , rtt
    #         # draw_utils.publish_point_cloud('2 CLOUD', 
    #         #                                rel * (sparse_cloud1 - rts.tvec) + rts.tvec, 
    #         #                                c='g', sensor_tf='local')

    #     # Compute the tag pose in the local frame
    #     p1 = self.frame_pose * self.atag.get_pose(frame1)
    #     p2 = self.frame_pose * self.atag.get_pose(frame2)
        
    #     # Determine the relative tag pose in local frame
    #     # cloud2 = pose12_ * cloud1
    #     # cloud1 = pose12_.inverse() * cloud2
    #     pose12_ = p2 * p1.inverse()
    #     print 'Pose 12: tag: ', p1, p2

    #     # Penalize large displacements between cameras
    #     return pose12_, np.linalg.norm(pose12_.tvec)
