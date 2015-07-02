'''
============================================================
LCM Log Player sample
============================================================
'''
# python lcmlog_app.py -l ~/data/2013-09-03-artags-articulation/lcmlog-2013-09-03.01

import numpy as np
import sys, cv2, os.path, argparse, time
from fs_utils import LCMLogReader

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(
        usage='python lcmlog_app.py -l <log_file> -s <scale> -i <interval>'
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
    reader = LCMLogReader(filename=options.logfile, scale=options.scale, sequential_read=True)
    start_time = time.time()

    # Main process loop ================================================
    while True: 
        # Get Frame
        frame = reader.getNextFrame();
        if frame is None: break

        # Get rgb
        rgb = frame.getRGB()

        # Show image
        cv2.imshow('image', rgb)
        cv2.waitKey(10)

        # Only flush after the whole process
        if reader.frame_num % 100 == 0: 
            print 'Processed %i frames in %4.2f' % (reader.frame_num, time.time() - start_time)
    print 'Done!' 
