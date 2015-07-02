'''
Lucas-Kanade tracker

@author: Sudeep Pillai (Last Edited: 01 May)

Notes: 

a. Forward-Backward OF PyramidLK error 
b. Gating of flows
====================

Lucas-Kanade sparse optical flow demo. Uses goodFeaturesToTrack
for track initialization and back-tracking for match verification
between frames.
'''

# import tables as tb
# import numpy as np
# import sys, cv2, collections, os.path, time
# from common import anorm2, draw_str
# from time import clock
# import lcmutils, dbutils, pcl_utils, normal_estimation

# from utils.io_utils import SuperFeatureTable, KinectData, SuperFeature
# from utils.dbg_utils import set_trace

# import utils.video as video

# from optparse import OptionParser

import numpy as np
import scipy.sparse as sp
np.set_printoptions(precision=2, suppress=True, threshold='nan', linewidth=160)

import cv2, time, os.path
from collections import namedtuple

from utils.db_utils import AttrDict
import utils.draw_utils as draw_utils

class KLTTracker:
    lk_params = dict( maxCorners = 1500, qualityLevel = 0.1, minDistance = 7, blockSize = 7 )
    def __init__(self, params=lk_params):

        self.feature_params = params

        self.track_id = 0;
        self.track_len = 40
        self.detect_interval = 10
        self.max_missed = 2
        self.match_threshold = 80

        self.tracks = collections.defaultdict(collections.deque)
        self.old_tracks = dict()
        self.missed = dict()

        self.frame_idx = 0
    

        
        self.lk_params = dict( winSize  = (5, 5),
                          maxLevel = 3,
                          criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


        self.detector = cv2.FeatureDetector_create("GFTT")
        self.descriptor = cv2.DescriptorExtractor_create("FREAK")
        self.matcher = cv2.DescriptorMatcher_create("BruteForce-Hamming")    
        
    def __get_new_id(self):
        self.track_id += 1
        return self.track_id
    
    def update(self, img, utime=None, X=None, normals=None):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.blur(gray, (9,9))    

        new_pts = []
        if len(self.tracks) > 0:
            img0, img1 = self.prev_gray, gray
            p0 = np.float32([tr[-1].x for tr in self.tracks.values()]).reshape(-1, 1, 2)
            p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **self.lk_params)
            p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **self.lk_params)
            d = abs(p0-p0r).reshape(-1, 2).max(-1)
            good = d < 1

            # Convert to KeyPoints
            kp0 = [cv2.KeyPoint(x,y,1,1,0,-1) for (x,y) in p0.reshape(-1,2)]
            kp1 = [cv2.KeyPoint(x,y,1,1,0,-1) for (x,y) in p1.reshape(-1,2)]    

            # Compute descriptors
            kp0, desc0 = self.descriptor.compute(gray, kp0)
            kp1, desc1 = self.descriptor.compute(gray, kp1)

            # Compute Matches        
            matches = []
            for d0,d1 in zip([d for d in desc0],[d for d in desc1]):
                matches.extend([m.distance for m in self.matcher.match(d0.reshape(1,-1), d1.reshape(1,-1))])

            # Threshold only good matches (match=0: perfect fit)            
            good_match = np.array(matches) < self.match_threshold;
        
            for tid, (x, y), good_flag, match_flag in zip(self.tracks.keys(), p1.reshape(-1, 2), good, good_match):
                if not good_flag or not match_flag:
                    # Update missed tracks list
                    self.missed[tid] = self.missed[tid] + 1    
                    continue
                if x < 0 or y < 0 or x >= img.shape[1] or y >= img.shape[0]: continue

                # Update missed tracks list        
                self.missed[tid] = 0
                self.tracks[tid].append(SuperFeature(utime=utime,feature_id=tid,
                                                    x=(x,y),X=X[y,x,:],normal=normals[y,x,:]))
                new_pts.append(SuperFeature(utime=utime,feature_id=tid,
                                                    x=(x,y),X=X[y,x,:],normal=normals[y,x,:]))    

                # if len(self.tracks[tid]) > self.track_len:
                #     self.tracks[tid].popleft()

            # Check to see if too many missed; if so move to old_tracks
            self.old_tracks = dict()
            #print 'Pre-Removing Features: %i/%i' % (len(self.tracks), len(self.old_tracks))    
            ids = self.tracks.keys();
            for tid in ids:
                if self.missed[tid] > self.max_missed:
                    self.old_tracks[tid] = self.tracks[tid]
                    del self.tracks[tid]
                    del self.missed[tid]
            #print 'Post-Removing Features: %i/%i' % (len(self.tracks), len(self.old_tracks))

        if self.frame_idx % self.detect_interval == 0:
            mask = np.zeros_like(gray)
            mask[:] = 255
            for x, y in [np.int32(tr[-1].x) for tr in self.tracks.values()]:
                cv2.circle(mask, (x, y), 5, 0, -1)
            p = cv2.goodFeaturesToTrack(gray, mask = mask, **self.feature_params)
            if p is not None:
                for x, y in np.float32(p).reshape(-1, 2):
                    if x < 0 or y < 0 or x >= img.shape[1] or y >= img.shape[0]: continue    
                    tid = self.__get_new_id()    
                    self.tracks[tid].append(SuperFeature(utime=utime,feature_id=tid,
                                                    x=(x,y),X=X[y,x,:],normal=normals[y,x,:]))
                    self.missed[tid] = 0    
                    new_pts.append(SuperFeature(utime=utime,feature_id=tid,
                                                x=(x,y),X=X[y,x,:],normal=normals[y,x,:]))    


        self.frame_idx += 1
        self.prev_gray = gray

        return self.tracks, self.old_tracks

def add_to_sftable(sfro, tracks):
    for tr in tracks.values():
        for sf in tr:     
            sfro['utime'] = sf.utime
            sfro['feature_id'] = sf.feature_id
            sfro['x'] = sf.x
            sfro['X'] = sf.X
            sfro['normal'] = sf.normal
            sfro.append()


if __name__ == "__main__": 
    parses = OptionParser()
    parses.add_option("-f", "--filename", dest="filename", 
                      help="Input log Filename", 
                      metavar="FILENAME")
    parses.add_option("-t", "--log-type", dest="logtype", 
                      help="Log type (lcm,hdf5)",default="video",
                      metavar="LOGTYPE")
    parses.add_option("-s", "--simulate", dest="simulate", 
                      help="Simulate", 
                      metavar="S", default=False)
    parses.add_option("-x", "--rewrite", dest="rewrite", 
                      help="Re-write DB", 
                      metavar="X", default=False)
    parses.add_option("-q", "--flush-queue", dest="flush_queue", 
                      help="Flush every N frames", 
                      metavar="Q", default=25)
    parses.add_option("-c", "--channel", dest="channel", 
                      help="Extract Channel", 
                      metavar="C", default='KINECT_FRAME')
    parses.add_option("-n", "--num-frames", dest="frames", 
                      help="Number of Frames", type="int",
                      metavar="N", default=None)
    
    (options, args) = parses.parse_args()

    if options.filename is None or not os.path.exists(options.filename):
        print 'Invalid Filename: %s' % options.filename
        sys.exit(0)

    if not options.simulate: 
        # Open DB
        h5f = tb.openFile('%s.h5' % options.filename, mode='a', title='%s' % options.filename)

        # Create table if it doesn't exist
        if 'superfeatures' in h5f.root:
            h5f.removeNode(h5f.root, name='superfeatures')    

        sftb = h5f.createTable(h5f.root, name='superfeatures', description=SuperFeatureTable, title='SuperFeature Data')
        print 'Creating new Log table: SuperFeature (DOES NOT UPDATE)'

    # else:
    #     sftb = h5f.root.superfeatures
    #     print 'Using existing Log table: SuperFeature'    

    reader = None        
    if options.logtype == "lcm": 
        # Read lcm log file        
        reader = lcmutils.KinectLogReader(filename=options.filename,
                                            channel=options.channel,extract_depth=True, extract_X=True)
        read_next_frame = lambda: reader.get_next_frame_wo_seek()    
    elif options.logtype == "hdf5":
        # Read DB file        
        reader = dbutils.KinectDBReader(db=h5f) #filename=options.filename)
        read_next_frame = lambda: reader.get_next_frame()        
    elif options.logtype == "video":
        # Defaulting to video/synth    
        reader = video.create_capture(video_src)
        read_next_frame = lambda: reader.read()            
    else:
        print 'Unknown Log type: Exiting!'
        sys.exit(0) 
        
    

    # KLT Tracker class
    klt = KLTTracker()

    if not options.simulate: sfro = sftb.row
    sf_tracks, old_sf_tracks = None, None    
    while True:
        if options.logtype == "video":
            ret, frame = cam.read()    
            if not ret: break
        else:
            st = time.time()    
            data = read_next_frame()
            if not data: break    
            frame = cv2.cvtColor(data.rgb, cv2.COLOR_RGB2BGR)            
            print 'Time: %4.3f' % (time.time() - st)    
    
        st = time.time()
        normals = pcl_utils.integral_normal_estimation(np.ascontiguousarray(data.X, dtype=np.float32))
        # normals = normal_estimation.integral_normal_estimation(np.ascontiguousarray(data.X, dtype=np.float32),4)
        print 'Normal estimation Time: %4.3f' % (time.time() - st)        
        sf_tracks, old_sf_tracks = klt.update(frame, utime=data.utime, normals=normals, X=data.X)    

        if not options.simulate: 
            # Append to sf table    
            add_to_sftable(sfro, old_sf_tracks)    

        if klt.frame_idx % 10 == 0:
            print 'Frames: %i' % klt.frame_idx    

            if not options.simulate: sftb.flush()    

        if (options.frames is not None) and (klt.frame_idx >= options.frames):
            print 'Exiting prematurely: requested only %i frames' % options.frames

            if not options.simulate: 
                # Append to sf table    
                add_to_sftable(sfro, sf_tracks)    
                sftb.flush()
            
            break        

        # Display    
        vis = frame.copy()
        tracks = [[sf.x for sf in sf_tr] for sf_tr in sf_tracks.values()]
        cv2.polylines(vis, [np.int32(tr) for tr in tracks], False, (0, 255, 0))
        draw_str(vis, (20, 20), 'track count: %d' % len(tracks))
    
        cv2.imshow('lk_track', vis)

        ch = 0xFF & cv2.waitKey(10)
        if ch == 27:
            break

    if not options.simulate: 
        # Flush and Indexing DB
        print 'Flushing DB ...'    

        # Append to latest tracks to sf table    
        add_to_sftable(sfro, sf_tracks)    
        sftb.flush()

        print 'Indexing DB ...'
        if not sftb.indexed:
            st = time.time()    
            sftb.cols.feature_id.createIndex()
            sftb.cols.utime.createIndex()    
            print 'DB indexed in %4.3f s' % (time.time() - st)
        else:
            sftb.cols.feature_id.reIndex()
            sftb.cols.utime.createIndex()        
            print 'DB already indexed, but reindexing'

        # Build table for utimes, and IDs
        utimes = sorted(list(set(sftb.cols.utime[:])))
        utimes_map = dict(zip(utimes, range(len(utimes))))
        print 'Total utimes: %i' % len(utimes)

        ids = sorted(list(set(sftb.cols.feature_id[:])))
        ids_map = dict(zip(ids, range(len(ids))))
        print 'Total tracks: %i' % len(ids)

        # Replace with new table
        if 'superfeatures_utimes_table' in h5f.root: 
            h5f.removeNode(h5f.root, name='superfeatures_utimes_table')
        if 'superfeatures_ids_table' in h5f.root: 
            h5f.removeNode(h5f.root, name='superfeatures_ids_table')

        utimestable = h5f.createArray(h5f.root, 'superfeatures_utimes_table', np.array(utimes))
        idstable = h5f.createArray(h5f.root, 'superfeatures_ids_table', np.array(ids))

        if 'superfeatures_table' in h5f.root: 
            h5f.removeNode(h5f.root, name='superfeatures_table')
        # sftable = h5f.createCArray(h5f.root, 'superfeatures_table', tb.Float64Atom(), shape=(len(ids), len(utimes), 8))
        feature_data = np.zeros((len(ids), len(utimes), 8), dtype=np.float32);
        for utime in utimes: 
            col = utimes_map[utime]
            query = h5f.root.superfeatures.where("""utime == %d""" % utime)
            for ro in query: 
                row = ids_map[ro['feature_id']]
                feature_data[row,col,0:2] = ro['x']
                feature_data[row,col,2:5] = ro['X']
                feature_data[row,col,5:8] = ro['normal']
            if col % 100 == 0: 
                print 'Processed utimes: %i' % col
        sftable = h5f.createArray(h5f.root, 'superfeatures_table', feature_data)    

        # Close file        
        print 'Closing h5f file'
        h5f.close()        
    
    cv2.destroyAllWindows()

    
