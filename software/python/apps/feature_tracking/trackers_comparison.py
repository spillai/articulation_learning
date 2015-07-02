'''
============================================================
Feature Trajectory Trackers Comparison
1. Viz drift for each feature tracker
2. Change frame skips
============================================================
'''
# Run: 
# python trackers_comparison.py -l ~/data/2013-09-03-artags-articulation/lcmlog-2013-09-03.01 -i 5
# python trackers_comparison.py -l ~/data/2013-09-03-artags-articulation/lcmlog-2013-09-03.03 -v 10 -i 10 -o 'chair @10fps'
# -m to simulate (should create the directories)

import time
import numpy as np
import sys, cv2, os.path, time
from optparse import OptionParser
from collections import defaultdict, deque

import utils.draw_utils as draw_utils

from utils.db_utils import AttrDict
from utils.tracker_utils import TrackerInfo, AprilTagsFeatureTracker, \
    DenseTrajectoriesTracker, BirchfieldKLTTracker, LearDenseTrajectoriesTracker

from fs_utils import LCMLogReader 

import matplotlib as mpl
import matplotlib.pylab as plt

from mpltools import style
style.use('ggplot')
mpl.rcParams['backend'] = "Qt4Agg"

if __name__ == "__main__": 
    parser = OptionParser()
    parser.add_option("-l", "--log-file", dest="filename", 
                      help="Input log Filename")
    parser.add_option("-i", "--interval", dest="interval", 
                      help="Eval every i frames", default=1, type="int")
    parser.add_option("-v", "--save-interval", dest="save_interval", 
                      help="Save every i frames", default=5, type="int")
    parser.add_option("-o", "--output-folder", dest="output_folder", 
                      help="Output folder",default="test",type="string")
    parser.add_option("-s", "--scale", dest="scale", 
                      help="Scale",default=1.0,type="float")
    parser.add_option("-m", "--simulate", action="store_true", dest="simulate")
    (options, args) = parser.parse_args()

    # Required options =================================================
    if not options.filename: 
        parser.error('Filename not given')

    log_path, log_file = os.path.split(options.filename)
    graphics_dir='/home/spillai/git/private/fs-private/smthesis/graphics/tmp/'
   
    if not len(options.output_folder): options.output_folder = log_file;
    output_dir = ''.join([graphics_dir, options.output_folder, '/']);
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    # Trackers map =====================================================
    trackers = AttrDict()
    # trackers.apriltags = AprilTagsFeatureTracker()
    trackers.bklt = BirchfieldKLTTracker(); 
    trackers.dtraj = LearDenseTrajectoriesTracker(); 
    trackers.gpft = DenseTrajectoriesTracker() 

    attrs = {'bklt': {'col':(0,200,200), 'radius':3, 'thickness': 1, 'line_thickness': 1}, 
             'gpft': {'col':(0,200,200), 'radius':3, 'thickness': 1, 'line_thickness': 1},
             'dtraj': {'col':(0,200,200), 'radius':3, 'thickness': 1, 'line_thickness': 1},
             'apriltags': {'col':(0,200,200), 'radius':4, 'thickness': 2, 'line_thickness': 2}}
   
    # Set up LCMReader =================================================
    if options.filename is None or not os.path.exists(options.filename):
        print 'Invalid Filename: %s' % options.filename
        sys.exit(0)
    reader = LCMLogReader(filename=options.filename, scale=options.scale, sequential_read=False)
    init_time = time.time();

    # Viz ==============================================================
    # ncols = 20; cols = np.random.randint(0,256,(ncols,3)).astype(int);
    ncols = 1; cols = np.array([[0, 200, 0]]);
    # draw_utils.publish_sensor_frame('KINECT')

    # Setup tracks per tracker
    for tname, tracker in trackers.iteritems(): 
        tracker.tracks_count = defaultdict(lambda: 0)
        tracker.tracks = defaultdict(lambda: deque(maxlen=20))

        # Get config params
        video_fn = graphics_dir+log_file+'-'+tname+'-'+options.output_folder+'-track.avi'
        print 'VIDEO FN: ', video_fn
        tracker.writer = cv2.VideoWriter(video_fn, 
                                         cv2.VideoWriter_fourcc('m','p','4','2'), 
                                         15.0, (640, 480), True)


        # tracker.tracks_log = defaultdict(lambda: deque())

    # Main process loop ================================================
    total_frames = reader.getNumFrames(); 

    tracker_count = defaultdict(list)
    tracker_total_count = defaultdict(list)
    tracker_length = defaultdict(list)

    while True: 
        # Get Frame
        frame = reader.getNextFrame();
        if frame is None: break

        if options.interval != 1 and reader.frame_num % options.interval != 0: 
            continue;
        print reader.frame_num

        # Perform tracking for each tracker
        for tname, tracker in trackers.iteritems(): 

            rgb = np.copy(frame.getRGB());

            fpts = tracker.get_features(frame)
            if fpts is None: continue
            for fpt in fpts: 
                tracker.tracks_count[fpt.id] += 1
                
                if not len(tracker.tracks[fpt.id]):
                    tracker.tracks[fpt.id].append(fpt)
                else: 
                    # if tracker.tracks_count[fpt.id] % max(1, int(10/options.interval)) == 0: 
                    tracker.tracks[fpt.id].append(fpt)
            # for fpt in fpts: tracker.tracks_log[fpt.id].append(fpt)

            if reader.frame_num % 20 == 0: 
                # Tracker count stats
                tracker_count[tname].append(len(fpts)) 
                tracker_total_count[tname].append(len(tracker.tracks_count.values())) 
                vals = tracker.tracks_count.values()
                tracker_length[tname].append(
                    np.mean(sorted(vals, reverse=True)[:max(1, int(len(vals)/4))])
                ) 

            print 'Detected %i features from %s' % (len(fpts), tname)
            print 'Total features: %i' % (len(tracker.tracks))

            col = attrs[tname]['col']
            radius = attrs[tname]['radius']
            thickness = attrs[tname]['thickness']
            line_thickness = attrs[tname]['line_thickness']

            # Remove unnecessary tracks
            detections = set([fpt.id for fpt in fpts])
            for tid in tracker.tracks.keys(): 
                if tid not in detections: del tracker.tracks[tid]

            # Draw tracks and endpoint
            # samples = np.random.choice(tracker.tracks.keys(), 
            #                            size=min(500, len(tracker.tracks.keys())), replace=False)
            samples = sorted([(k,len(v)) for k,v in tracker.tracks.iteritems()], 
                             key=lambda x: x[1], reverse=True)[:2000]

            for sample in samples: 
                tid, _ = sample
                track = tracker.tracks[tid]
                if not len(track): continue
                cv2.circle(rgb, tuple(track[-1].point.reshape(-1).astype(int).tolist()), 
                           radius, col, 
                           thickness, lineType=cv2.LINE_AA);


                pts = [np.int32(fpt.point.reshape(-1)) for fpt in track]
                cv2.polylines(rgb,[np.array(pts, np.int0)], False, 
                              cols[track[-1].id%ncols].tolist(), 
                              thickness=line_thickness, lineType=cv2.LINE_AA)

            if reader.frame_num % options.save_interval == 0: 
                tracker_dir = ''.join([output_dir, tname])
                if not os.path.exists(tracker_dir): os.makedirs(tracker_dir)
                im_fn = ''.join([tracker_dir, '/', log_file, 
                                 '-', str(reader.frame_num),'.png']);

                if not options.simulate:                     
                    # cv2.imwrite(im_fn, rgb);
                    # print 'Writing to %s' % im_fn
                    tracker.writer.write(rgb)
                else: 
                    print 'Simluation mode. Not writing'
                    

            cv2.imshow(tname, rgb)
            cv2.waitKey(1)

        # Only flush after the whole process
        if reader.frame_num % 100 == 0: 
            print 'Processed %i frames' % reader.frame_num

    # print 'Tracker count: ', tracker_count
    # print 'Tracker length: ', tracker_length
    
    f = plt.figure(1)
    ax = f.add_subplot(1, 2, 1)
    xs = np.arange(len(tracker_length['bklt']))  * options.interval
    ax.plot(xs, tracker_length['bklt'], marker='o', markersize=3, linewidth=2, label='KLT')
    ax.plot(xs, tracker_length['dtraj'], marker='o', markersize=3, linewidth=2, label='Dense Traj.')
    ax.plot(xs, tracker_length['gpft'], marker='o', markersize=3, linewidth=2, label='Ours')
    ax.legend(loc='upper left')
    ax.set_xlim([min(xs), max(xs)])
    ax.set_xlabel('Observations')
    ax.set_ylabel('Average Trajectory Length')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), fancybox=True, ncol=3)

    ax = f.add_subplot(1, 2, 2)
    xs = np.arange(len(tracker_count['bklt']))  * options.interval
    ax.plot(xs, tracker_count['bklt'], marker='o', markersize=3, linewidth=2, label='KLT')
    ax.plot(xs, tracker_count['dtraj'], marker='o', markersize=3, linewidth=2, label='Dense Traj.')
    ax.plot(xs, tracker_count['gpft'], marker='o', markersize=3, linewidth=2, label='Ours')

    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), fancybox=True, ncol=3)

    ax.set_xlim([min(xs), max(xs)])
    ax.set_xlabel('Observations')
    ax.set_ylabel('Total Trajectory Count')
    print 'Processed %i frames in %4.1fs' % (reader.frame_num, time.time() - init_time)
    plt.show()
    raw_input()

