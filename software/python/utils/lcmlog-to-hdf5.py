#!/usr/bin/env python
import time, lcm, cv2, sys, os.path
import tables as tb
import numpy as np
import lcmutils as lcmutils
from ioutils import KinectFrame
from optparse import OptionParser

if __name__ == "__main__": 
    parses = OptionParser()
    parses.add_option("-f", "--filename", dest="filename", 
                      help="Input LCM log Filename", 
                      metavar="FILENAME")
    parses.add_option("-x", "--rewrite", dest="rewrite", 
                      help="Re-write DB", 
                      metavar="X", default=False)
    parses.add_option("-q", "--flush-queue", dest="flush_queue", 
                      help="Flush every N frames", 
                      metavar="Q", default=25)
    parses.add_option("-c", "--channel", dest="channel", 
                      help="Extract Channel", 
                      metavar="C", default='KINECT_FRAME')
    (options, args) = parses.parse_args()

    if options.filename is None or not os.path.exists(options.filename):
        print 'Invalid Filename: %s' % options.filename
        sys.exit(0)

    # Read lcm log file        
    reader = lcmutils.KinectLogReader(options.filename, channel=options.channel, fps=0, loop=False)    

    # Open DB
    h5f = tb.openFile('%s.h5' % options.filename, mode='a', title='%s' % options.filename)

    # Create table if it doesn't exist
    if 'log' not in h5f.root:
        logtb = h5f.createTable(h5f.root, name='log', description=KinectFrame, title='Kinect Log Data')
        print 'Creating new Log table: KinectFrame'
    else:
        logtb = h5f.root.log
        print 'Using existing Log table: KinectFrame'    

    # Find utimes set difference (lcmlog utimes - db utimes)
    log_utimes = set(reader.utime_map.keys())
    db_utimes = set(logtb.cols.utime[:])
    intersect_utimes = list(log_utimes.intersection(db_utimes));
    diff_utimes = list(log_utimes.difference(db_utimes));
        
    print 'Skipping %i frames, already in DB: ' % len(intersect_utimes)
    print 'Processing %i frames: ' % len(diff_utimes)

    # Two approaches to reading/storing from/into lcm-LOG/DB
    # First approach: Look at all diff_utimes, and seek to those that are not in DB
    
    # # Check if in DB
    # try:
    #     rows = logtb.where("""utime == %i""" % utime).next()
    # except:
    #     rows = None
    # # continue if in DB    
    # if rows is not None: continue
    
    # Second approach: Look at each utime incrementally (should be faster
    # than seeking), and update only if frame doesn't exist in DB
    
    # Implemented Below
    
    # Inits
    reader.cur_frame = 0; # Reset reader
    processed_frames = 0;
    st = time.time()
    
    # Processed 
    # First approach
    frame = logtb.row # Table pointer
    for utime in diff_utimes:
        
        # Get data from LCM-log
        data = reader.get_frame(utime)
        
        # write to DB
        frame['utime'] = data.utime
        frame['rgb'] = data.rgb
        frame['depth'] = data.depth
        frame['X'] = data.X
        frame.append()
        
        processed_frames += 1
        
        # flush every FLUSH_QUEUE frames
        if processed_frames % options.flush_queue == 0:
            now = time.time()
            logtb.flush()
            print 'Processed: %i frames | Time elapsed: %5.2f s' % (processed_frames, now - st)
            
    logtb.flush()
    print 'Processed: %i/%i frames, Skipped: %i frames' % (processed_frames, len(diff_utimes), len(intersect_utimes))

    # Indexing DB
    print 'Indexing DB ...'
    if not logtb.indexed:
        st = time.time()    
        logtb.cols.utime.createIndex()
        print 'DB indexed in %4.3f s' % (time.time() - st)
    else:
        logtb.cols.utime.reIndex()
        print 'DB already indexed, but reindexing'    
        
    print 'Closing DB'
    print h5f
    h5f.close()
