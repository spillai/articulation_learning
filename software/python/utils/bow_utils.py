#!/usr/bin/python 

from __future__ import division

import random, time, itertools, operator
import numpy as np
np.set_printoptions(precision=3, suppress=True)
import cv2
# import networkx as nx

# from collections import defaultdict
from rigid_transform import Quaternion, RigidTransform, tf_compose, tf_construct

# import utils.draw_utils as draw_utils
# import utils.plot_utils as plot_utils

# from utils.tracker_utils import AprilTagsFeatureTracker
# from utils.pose_utils import mean_pose

from fs_pcl_utils import CorrespondenceRejectionSAC
from fs_utils import DBoW2, RichFeatureMatcher

class SceneLocalizationBOW: 
    def __init__(self): 
        # BOW scene localization/prediction
        self.voc = DBoW2()

        # Data storage for querying
        self.feature_map = dict()

    # Data can be arbitrary: scene_id, feature_id/name
    def add(self, data): 
        self.voc.addDescription(data['desc'])
        self.feature_map[len(self.feature_map)] = data

    def batch_add(self, data_vec): 
        for data in data_vec: 
            self.add(data)

    def build(self): 
        t1 = time.time()
        self.voc.build(k=27,L=3,use_direct_index=False)
        # self.voc.buildVOC(k=27,L=3)
        # self.voc.buildDB(use_direct_index=True)
        print 'Done Building VOC with %i instances, took %f' % (len(self.feature_map), time.time() - t1)

    def query(self, desc, top_k=5): 
        res = self.voc.queryDescription(desc, top_k)
        return [ (r.Score, self.feature_map[r.Id]) for r in res ]

class ObjectLocalizationBOW: 
    def __init__(self): 
        # Initialize scene localization 
        self.scene_localizer = SceneLocalizationBOW()

        # Geometric verification
        self.feature_matcher = RichFeatureMatcher()
        
    # Data can be 2D, 3D pts, object_id etc.
    def add_object(self, data): 
        self.scene_localizer.add(data)

    def build_db(self): 
        self.scene_localizer.build()

    def query_scene(self, data, top_k=5): 
        # 1. Scene identification
        q_desc = data['desc']
        top_objects = self.scene_localizer.query(q_desc, top_k=top_k)
        print 'TOP: %i' % len(top_objects)

        # 2. Geometric verification for each object/scene result against query scene
        verified_object_models = {}
        q_img = data['img']
        q_pts2d, q_pts3d = data['pts2d'].astype(np.int32), data['pts3d']
        for idx, (obj_score,obj) in enumerate(top_objects): 
            o_desc = obj['desc']
            o_img = obj['img']
            o_pts2d, o_pts3d = obj['pts2d'].astype(np.int32), obj['pts3d']

            # 2a. Match features from object result (in DB) and query scene
            m_inds = self.feature_matcher.getMatchInds(q_pts2d, q_desc, o_pts2d, o_desc)
            if m_inds is None: 
                continue
            print '-------------------------------------'
            o_name = obj['name']
            print 'NAME:', o_name
            print 'Score:', obj_score, 'Verified/Total: %i/%i' % (len(m_inds), len(q_pts2d))

            # 2b. Ensure minimum percentage of verified matches
            verified_pc = len(m_inds) * 1.0 / len(q_pts2d)
            if obj_score < 0.2: 
                continue

            # 2c. Estimate the relative pose between object (in DB), and query scene
            valid_inds = np.isfinite(q_pts3d[m_inds[:,0]]).all(axis=1) & \
                         np.isfinite(o_pts3d[m_inds[:,1]]).all(axis=1)
            Xs, Xt = q_pts3d[m_inds[valid_inds,0]], o_pts3d[m_inds[valid_inds,1]]
            print 'Computing pose from %i/%i points' % (len(Xs), len(m_inds))

            Tts, inliers = CorrespondenceRejectionSAC(source=Xt, target=Xs, 
                                                      source_dense=np.array([]), 
                                                      target_dense=np.array([]), 
                                                      inlier_threshold=0.1, max_iterations=300)
            print 'Inliers %i/%i points' % (len(inliers), len(Xs))
            
            # 2d. Ensure minimum percentage of verified inliers
            inliers_pc = len(inliers) * 1.0 / len(Xs)
            if inliers_pc < 0.2: 
                continue

            # 2e. Retain verified objects (object_id, relative pose) if valid transformation
            try: 
                pose_ts = RigidTransform.from_homogenous_matrix(Tts)
            except: 
                return {'rel_pose': RigidTransform(), 'object': None}

            verified_object_models[o_name] = {'rel_pose':pose_ts, 'object':obj}
                
            # 2f. Viz (debug)
            vis = np.hstack([q_img, o_img])
            for q_pt in q_pts2d[m_inds[:,0]]: 
                cv2.circle(vis, tuple(q_pt), 3, (0,255,0), 1, lineType=cv2.LINE_AA)
            for o_pt in o_pts2d[m_inds[:,1]]: 
                cv2.circle(vis, (o_pt[0]+640, o_pt[1]), 3, (0,255,0), 1, lineType=cv2.LINE_AA)

            for q_pt,o_pt in zip(q_pts2d[m_inds[:,0]], o_pts2d[m_inds[:,1]]): 
                cv2.line(vis, tuple(q_pt), (o_pt[0]+640,o_pt[1]), (0,150,0), 1)                

            # cv2.imshow('img-%i' % idx, vis)

            # Temporarily return single object: TODO, FIX!
            return verified_object_models[o_name]

        # cv2.waitKey(10)

        return verified_object_models
