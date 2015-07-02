#!/usr/bin/python 

import cv2
import numpy as np

# import itertools, time, threading, logging, random, os
# from collections import defaultdict, namedtuple, OrderedDict

# from articulation import pose_msg_t, track_msg_t, \
#     track_list_msg_t, articulated_object_msg_t

# from utils.db_utils import AttrDict

import utils.plot_utils as plot_utils
import utils.draw_utils as draw_utils
import utils.imshow_utils as im_utils

from utils.camera_utils import KinectCamera
# from utils.logplayer_utils import LCMLogPlayer
# from utils.correspondence_estimation import BetweenImagePoseEstimator, remove_nans

from rigid_transform import Quaternion, RigidTransform, \
    tf_compose, tf_construct

# from fs_articulation import ArticulationLearner
# from fs_utils import LCMLogReader, publish_image_t
from fs_pcl_utils import change_detection

def kinect_change_detection(rgb, X0, X1): 
    draw_utils.publish_point_cloud('CHANGE_CLOUDS0', X0, c='g')
    draw_utils.publish_point_cloud('CHANGE_CLOUDS1', X1, c='r')
    H,W,D = X1.shape
    X0, X1 = X0.reshape((-1,3)), X1.reshape((-1,3))
    inds1 = change_detection(source=X0, target=X1, resolution=0.04, return_inverse=False)
    inds1 = np.array(inds1)
    
    
    draw_utils.publish_point_cloud('CHANGE_DETECTION', X1[inds1], c='b')

    # Get indices
    ys,xs = inds1/W, inds1%W
    xys = np.dstack([ys, xs]).reshape((-1,2))
    img = rgb

    # Plot indices
    img[ys,xs,:] = 255

    img = cv2.resize(img, (0, 0), fx=4.0, fy=4.0, interpolation=cv2.INTER_NEAREST)
    cv2.imshow('img', img)
    cv2.waitKey(10)
    # inds0 = change_detection(source=clouds1, target=clouds0, 
    #                          resolution=0.15, return_inverse=False)
        
    # def compute_change(self, clouds): 
    #     inliers = []

    #     frame_inds = np.arange(len(clouds))
    #     idx0, rest_inds = frame_inds[0], frame_inds[1:]
        
    #     for idx1 in rest_inds: 
    #         print idx0, idx1
    #         clouds0, clouds1 = clouds[idx0].reshape((-1,3)), clouds[idx1].reshape((-1,3))
    #         draw_utils.publish_point_cloud('CHANGE_CLOUDS0', clouds0, c='g')
    #         draw_utils.publish_point_cloud('CHANGE_CLOUDS1', clouds1, c='r')
    #         inds1 = change_detection(source=clouds0, target=clouds1, 
    #                                  resolution=0.15, return_inverse=False)
    #         inds0 = change_detection(source=clouds1, target=clouds0, 
    #                                  resolution=0.15, return_inverse=False)
    #         print 'INLIERS: ', len(inds1)
            
    #         inliers.append(clouds0[inds0])
    #         inliers.append(clouds1[inds1])

    #         draw_utils.publish_point_cloud('CHANGE_DETECTION', clouds1[inds1], c='b')

    #     draw_utils.publish_point_cloud('ALL_CHANGES', np.vstack(inliers), c='b')
