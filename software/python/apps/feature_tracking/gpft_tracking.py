'''
============================================================
Construct Feature Trajectories using GPFT
============================================================
'''

# Main Idea: Track features over time, use april tags as ground truth
# python gpft_tracking.py -l ~/data/2013-09-03-artags-articulation/lcmlog-2013-09-03.01

import time, argparse
import numpy as np
import sys, cv2, collections, os.path, time
from collections import defaultdict, namedtuple
import itertools

import utils.draw_utils as draw_utils
import utils.io_utils as io_utils

from utils.db_utils import AttrDict, DictDB
from utils.io_utils import Feature3DTable, Pose3DTable, Feature3DWriter, Pose3DWriter
from utils.trackers.tracker_utils import TrackerInfo, AprilTagsFeatureTracker, DenseTrajectoriesTracker
from fs_utils import LCMLogReader, publish_cloud

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(
        usage='python gpft_tracking.py -l <log_file> -s <scale> -i <interval> -a <with_tags>'
    )
    parser.add_argument('--logfile', '-l', 
                        help="Input log Filename")
    parser.add_argument('--interval', '-i', 
                        help="Eval every i frames", default=1, type=int)
    parser.add_argument('--scale', '-s', 
                        help="Scale",default=1.0,type=float,
                        metavar="SCALE")
    parser.add_argument('--simulation', '-m', 
                        help="Simulation mode",default=False, type=bool)
    parser.add_argument('--apriltags', '-a', 
                        help="With AprilTags detection",default=False, type=bool)
    (options, args) = parser.parse_known_args()
    
    # Required options =================================================
    if not options.logfile: 
        parser.error('Filename not given')

    # Trackers map =====================================================
    trackers = []
    if options.apriltags: 
        trackers.append(('apriltags', AprilTagsFeatureTracker()))
    trackers.append(('gpft', DenseTrajectoriesTracker()))

    attrs = {'apriltags': {'col':(0,0,200), 'radius':4, 'thickness': 2}, 
             'gpft': {'col':(0,200,0), 'radius':3, 'thickness': -1}}

    # Set up LCMReader =================================================
    # Only read certain frames
    if options.logfile is None or not os.path.exists(options.logfile):
        print 'Invalid Filename: %s' % options.logfile
        sys.exit(0)
    reader = LCMLogReader(filename=options.logfile, scale=options.scale, sequential_read=False)
    frames = np.arange(0, reader.getNumFrames()-1, options.interval).astype(int)

    init_time = time.time();

    # Open the db  =====================================================
    if not options.simulation: 
        processed_fn = io_utils.get_processed_filename(options.logfile)
        print 'PROCESSED FILENAME', processed_fn
        db = DictDB(filename=processed_fn, mode='w', ext='.h5')

    # Viz ==============================================================
    ncols = 10; cols = np.random.randint(0,256,(ncols,3)).astype(int);
    # draw_utils.publish_sensor_frame('KINECT_FRAME')

    # Main process loop ================================================
    for fno in frames: 
        # Get Frame
        frame = reader.getFrame(fno);
        if frame is None: break

        # Compute normals 
        frame.computeNormals(0.5)
        # publish_cloud("KINECT_FILTERED", frame)

        # Plot ground truth as white, rest as colored
        rgb = frame.getRGB()

        # Perform tracking for each tracker
        mask = np.ones_like(frame.getGray())
        for tname, tracker in trackers: 
            print 'Tracker name: ', tname

            if tname == 'apriltags': 
                fpts, mask = tracker.get_features(frame, return_mask=True)
            else: 
                print 'Tname; ', tname
                # assert (mask is not None)
                fpts = tracker.get_features(frame, mask)
            print 'Detected %i features from %s' % (len(fpts), tname)

            # Write to tracker data
            if not options.simulation: 
                tracker.data.extend(fpts)

            col = attrs[tname]['col']
            radius = attrs[tname]['radius']
            thickness = attrs[tname]['thickness']
            for fpt in fpts: 
                cv2.circle(rgb, tuple(fpt.point.reshape(-1).astype(int).tolist()), 
                           radius, col, thickness, lineType=cv2.LINE_AA);

            # Viz normals
            pts3d, normals3d, colors3d = [], [], []
            pts3d.extend([fpt.xyz().reshape(-1) for fpt in fpts])
            normals3d.extend([fpt.normal().reshape(-1)*0.05 for fpt in fpts])
            colors3d.extend([np.array([0.9, 0.1, 0.1, 1.0]) for fpt in fpts])
            if len(pts3d): 
                draw_utils.publish_point_cloud('FPTS_%s' % (tname), 
                                               np.vstack(pts3d), c=np.vstack(colors3d))
                draw_utils.publish_line_segments('FPTS_NORMALS_%s' % (tname), 
                                                 np.vstack(pts3d), 
                                                 np.vstack(pts3d) + np.vstack(normals3d), 
                                                 c=np.vstack(colors3d))


        cv2.imshow('detections', rgb)
        cv2.waitKey(10)

        # Only flush after the whole process
        if reader.frame_num % 100 == 0: 
            print 'Processed %i frames' % reader.frame_num

    # Write to db
    if not options.simulation: 
        for tname, tracker in trackers:
            db.data[tname] = Feature3DWriter(data=tracker.data)
            print 'TOTAL DATA SAVED: ', len(tracker.data)
    # for gtname, gte in gt_estimators.iteritems():
    #     db.data[gtname] = Pose3DWriter(data=gte.data)
    
    # Close db        
    if not options.simulation: 
        print 'Closing db ...'
        print 'Processed %i frames in %4.1fs' % (reader.frame_num, time.time() - init_time)
        db.flush()
        db.close()
        print 'Done!' 
