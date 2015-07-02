'''
============================================================
Offline KLT
============================================================
'''

# Main Idea: Track features over time, use april tags as ground truth
# python gpft_tracking.py -l ~/data/2013-09-03-artags-articulation/lcmlog-2013-09-03.01

import time, argparse
import numpy as np
import sys, cv2, collections, os.path, time

from utils.db_utils import AttrDict, DictDB
from utils.trackers.offline_klt import StandardKLT, FwdBwdKLT, GatedFwdBwdKLT

from fs_utils import LCMLogReader

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(
        usage='python offline_klt.py -l <log_file> -s <scale> -i <interval>'
    )
    parser.add_argument('--logfile', '-l', 
                        help="Input log Filename")
    parser.add_argument('--interval', '-i', 
                        help="Eval every i frames", default=1, type=int)
    parser.add_argument('--scale', '-s', 
                        help="Scale",default=1.0,type=float,
                        metavar="SCALE")
    (options, args) = parser.parse_known_args()
    
    # Required options =================================================
    if not options.logfile: 
        parser.error('Filename not given')

    # Set up LCMReader =================================================
    if options.logfile is None or not os.path.exists(options.logfile):
        print 'Invalid Filename: %s' % options.logfile
        sys.exit(0)
    reader = LCMLogReader(filename=options.logfile, scale=options.scale, sequential_read=False)
    frames = np.arange(0, reader.getNumFrames()-1, options.interval).astype(int)

    init_time = time.time();

    # Main process loop ================================================
    imgs, Xs = [], []
    for fno in frames: 
        # Get Frame
        frame = reader.getFrame(fno);
        if frame is None: break

        rgb = frame.getRGB()
        X = frame.getCloud()

        if fno % 200 == 0: 
            print 'Size: ', len(imgs)

        imgs.append(rgb)
        Xs.append(X)

        # if fno > 20 : break


    # # Standard KLT ==============================================
    # sklt = StandardKLT()
    # for im in imgs: 
    #     sklt.process_im(im)
    # sklt.tracks = sklt.get_2d_tracks()

    # OfflineKLT ================================================
    klt = FwdBwdKLT(images=imgs)
    klt.viz_pts(Xs)
    klt.tracks = klt.get_feature_data(Xs)
    # klt = GatedFwdBwdKLT(images=imgs).viz_pts(Xs)

