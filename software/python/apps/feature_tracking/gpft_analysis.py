'''
============================================================
Feature Trajectory Error Analysis using GPFT
============================================================
'''

# Main Idea: Track features over time, use april tags as ground truth Single
# ARtag, and only a single maximal cluster (similar to VO) Compute Homography
# from ground truth, and determine error between actual (x,y) and expected (x,y)

import time
import numpy as np
import tables as tb
import sys, cv2, collections, os.path, time, cPickle
from optparse import OptionParser
from collections import defaultdict, namedtuple, deque
import itertools
import matplotlib as mpl
import matplotlib.pylab as plt

import utils.rigid_transform as rtf
import utils.draw_utils as draw_utils
import utils.pose_utils as pose_utils
import utils.io_utils as io_utils

from utils.io_utils import Feature3DTable, Pose3DTable
from utils.tracker_utils import TrackerInfo, GPFTTracker

from fs_utils import LCMLogReader, MSER3D, GPFT, Feature3D
from fs_fovis_utils import FOVIS
import fs_apriltags
import copy

# # Set up Trackers ==================================================
# class TrackerInfo: 
#     def __init__(self, name, table, description): 
#         self.name = name
#         self.table = table
#         self.description = description

# class GPFTTracker(TrackerInfo): 
#     def __init__(self, name, table, description, allowed_predictions=0):
#         TrackerInfo.__init__(self, name=name, table=table, description=description)
#         self.tracker = GPFT(use_gftt=False, enable_subpixel_refinement=False, 
#                             num_feats=1500, min_add_radius=10, feat_block_size=7, 
#                             feature_match_threshold=20, feature_distance_threshold=15, 
#                             allowed_skips=5, allowed_predictions=allowed_predictions)

#     def get_features(self, frame): 
#         self.tracker.processFrame(frame)
#         return self.tracker.getStableFeatures()


# Trackers map
trackers = {}
trackers['gpft'] = GPFTTracker(name='gpft', table='gpft3d', description=Feature3DTable)
# trackers['gpftv2'] = GPFTTracker(name='gpftv2', table='gpft3d', description=Feature3DTable, allowed_predictions=5)
attrs = {'tags': {'col':(0,0,200), 'radius':4, 'thickness': 2}, 
         'gpft': {'col':(0,200,0), 'radius':1, 'thickness': 1}, 
         'gpftv2': {'col':(0,200,0), 'radius':1, 'thickness': 1}}

if __name__ == "__main__": 
    parses = OptionParser()
    parses.add_option("-l", "--log-file", dest="filename", 
                      help="Input log Filename", 
                      metavar="FILENAME")
    parses.add_option("-s", "--scale", dest="scale", 
                      help="Scale",default=1.0,type="float",
                      metavar="SCALE")
    # parses.add_option("-p", "--subpixel", dest="subpixel", 
    #                   help="Subpixel",default=1.0,type="bool",
    #                   metavar="SUBPIXEL")
    (options, args) = parses.parse_args()
    
    # Set up LCMReader =================================================
    if options.filename is None or not os.path.exists(options.filename):
        print 'Invalid Filename: %s' % options.filename
        sys.exit(0)
    reader = LCMLogReader(filename=options.filename, scale=options.scale, sequential_read=False)
    init_time = time.time();

    # Viz ==============================================================
    ncols = 10
    cols = np.random.randint(0,256,(ncols,3)).astype(int)
    draw_utils.publish_sensor_frame('KINECT_FRAME')
    
    for tracker in trackers.values(): 
        tracker.tracks = defaultdict(lambda: deque(maxlen=20))
        tracker.tracks_log = defaultdict(lambda: deque())

    # Main process loop ================================================
    while True: 
        # Get Frame
        frame = reader.getNextFrame();
        if frame is None: break
        frame.computeNormals(0.5)

        # Plot ground truth as white, rest as colored
        rgb = frame.getRGB()

        # Perform tracking for each tracker
        for tracker in trackers.values(): 
            fpts = tracker.get_features(frame)
            for fpt in fpts: tracker.tracks[fpt.id].append(fpt)
            for fpt in fpts: tracker.tracks_log[fpt.id].append(fpt)

            # # Write to table
            # write_to_table(sftables[tracker.table], fpts)
            
            col = attrs[tracker.name]['col']
            radius = attrs[tracker.name]['radius']
            thickness = attrs[tracker.name]['thickness']

            # Remove unnecessary tracks
            detections = set([fpt.id for fpt in fpts])
            for tid in tracker.tracks.keys(): 
                if tid not in detections: del tracker.tracks[tid]

            for tid, track in tracker.tracks.iteritems(): 
                cv2.circle(rgb, tuple(track[-1].point.reshape(-1).astype(int).tolist()), 
                           radius, tuple((cols[track[-1].id%ncols]).tolist()), 
                           thickness, lineType=cv2.LINE_AA);

                pts = [np.int32(fpt.point.reshape(-1)) for fpt in track]
                cv2.polylines(rgb,[np.array(pts, np.int0)], False, 
                              cols[track[-1].id%ncols].tolist(), lineType=cv2.LINE_AA)

            # Viz normals
            pts3d, normals3d, colors3d = [], [], []
            pts3d.extend([fpt.xyz().reshape(-1) for fpt in fpts])
            normals3d.extend([fpt.normal().reshape(-1)*0.05 for fpt in fpts])
            colors3d.extend([np.array([0.9, 0.1, 0.1, 1.0]) for fpt in fpts])
            draw_utils.publish_point_cloud('FPTS', np.vstack(pts3d), c=np.vstack(colors3d), size=0.02)
            draw_utils.publish_line_segments('FPTS_NORMALS', 
                                             np.vstack(pts3d), 
                                             np.vstack(pts3d) + np.vstack(normals3d), 
                                             c=np.vstack(colors3d), size=0.02)
        
        cv2.imshow('detections', rgb)
        cv2.waitKey(1)



        

        # Only flush after the whole process
        if reader.frame_num % 100 == 0: 
            print 'Detected %i features from %s' % (len(fpts), tracker.name)
            print 'Processed %i frames' % reader.frame_num

    # # Compute percentiles ==============================================
    # track_lengths = np.array(sorted([len(track) for track in tracks_log.values()], reverse=True))
    # cumsum_track_length = np.cumsum(track_lengths)
    # total_track_length = cumsum_track_length[-1]

    # # Only look at 10th to 90th percentile
    # perc = np.array([100, 99, 95, 90, 85, 80, 70, 60, 50])
    # avg_lengths = []
    # for phi,plo in zip(perc[:-1],perc[1:]): 
    #     indslo, = np.where(cumsum_track_length <= (100-plo) * 0.01 * total_track_length)
    #     indshi, = np.where(cumsum_track_length >= (100-phi) * 0.01 * total_track_length)
    #     inds = np.intersect1d(indslo,indshi)
    #     avg_length = np.mean(track_lengths[inds]); 
    #     avg_lengths.append(avg_length)
    #     print '%ith percentile: %i out of %i' % (plo, inds[-1], len(tracks_log.values()))
    #     print '\t Average length: %f' % (avg_length)

    # plt.bar(perc[1:], np.array(avg_lengths))
    # plt.show();

