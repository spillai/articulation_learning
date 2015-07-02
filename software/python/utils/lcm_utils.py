#!/usr/bin/python
import time
import os.path
import collections
import bisect

import numpy as np
import matplotlib as mpl
import matplotlib.pylab as plt

import lcm
from lcm import EventLog

from PIL import Image
from io_utils import KinectFrame
from io_utils import KinectData

from kinect.frame_msg_t import frame_msg_t
from kinect.image_msg_t import image_msg_t
from kinect.depth_msg_t import depth_msg_t

class KinectDecoder:
    def __init__(self, scale=1., extract_rgb=True, extract_depth=True, extract_X=True):
        self.skip = int(1.0 / scale);
        assert (self.skip >= 1)
        self.extract_rgb = extract_rgb
        self.extract_depth = extract_depth    
        self.extract_X = extract_X;
        self.fx = 576.09757860;
        self.cx = 319.50;
        self.cy = 329.50;
        self.shift_offset = 1079.4753;
        self.projector_depth_baseline = 0.07214;

        h,w = 480,640
        xs,ys = np.arange(0,w), np.arange(0,h);
        fx_inv = 1.0 / self.fx;
        self.xs = (xs-self.cx)*fx_inv
        self.xs = self.xs[::self.skip] # skip pixels
        self.ys = (ys-self.cy)*fx_inv
        self.ys = self.ys[::self.skip] # skip pixels
        self.xs, self.ys = np.meshgrid(self.xs, self.ys);
    
    def decode(self, data):
        img, depth, X = [None] * 3
        if self.extract_rgb: img = self.extract_rgb_image(data)
        if self.extract_depth:
            depth = self.extract_depth_image(data)
            if self.extract_X: X = self.depth_to_cloud(depth)
        return KinectData(data.timestamp,img,depth,X)
        
    def extract_rgb_image(self, data):
        # Extract image
        img = None
        w, h = data.image.width, data.image.height;
        if data.image.image_data_format == image_msg_t.VIDEO_RGB_JPEG: 
            img = Image.frombuffer('RGB', (w,h), data.image.image_data, 
                                   'jpeg', 'RGB', None, 1)
            img = np.array(img.getdata()).reshape(img.size[0], img.size[1], 3)
        else: 

            img = np.fromstring(data.image.image_data, dtype=np.uint8)
        img = np.reshape(img, (h,w,3))
        img = img[::self.skip, ::self.skip, :] # skip pixels
        return np.array(img, order='C')
    
    def extract_depth_image(self, data):
        # Extract depth image
        w, h = data.image.width, data.image.height;
        if data.depth.compression != depth_msg_t.COMPRESSION_NONE: 
            # img = Image.frombuffer('I;16B', (w,h), data.depth.depth_data, 
            #                        'zip', 'I;16B', 1)
            raise AssertionError()
        depth = np.fromstring(data.depth.depth_data, dtype=np.uint16)
        depth = np.reshape(depth, (h,w)).astype(np.float32) * 0.001; # in m
        depth = depth[::self.skip, ::self.skip] # skip pixels
        return depth
        
    def depth_to_cloud(self, depth):
        xs = np.multiply(self.xs, depth);
        ys = np.multiply(self.ys, depth)
        X = np.dstack((xs,ys,depth))
        #X = np.hstack((np.reshape(xs, (-1, 1)), np.reshape(ys, (-1,1)), np.reshape(depth, (-1,1))))
        return X

class KinectInput: 
    def __init__(self, handler): 
        self.decoder = KinectDecoder(scale=1.0, 
                                     extract_rgb=True, extract_depth=False, extract_X=False)
        self.lc = lcm.LCM()
        self.handler = handler
        self.sub = self.lc.subscribe("KINECT_FRAME", self.lc_handler)

    def lc_handler(self, ch, data): 
        self.handler(self.decoder.decode(frame_msg_t.decode(data)))

    def __del__(self): 
        self.lc.unsubscribe(self.sub)
        
    def run(self): 
        try:
            while True:
                self.lc.handle()
        except KeyboardInterrupt:
            pass


class KinectLogReader:
    def __init__(self, filename='', channel='KINECT_FRAME', scale = 1., 
                 extract_rgb=True, extract_depth=True, extract_X=True):
        self.filename = filename
        if not os.path.exists(self.filename):         
            print 'File not found: ' % (self.filename)

        print 'Kinect Log Reader: Opening file', self.filename        

        # Args
        self.channel = channel
    
        # Log specific
        self.lc = lcm.LCM()
        self.log = EventLog(self.filename, "r")

        # Kinect Decoder
        self.decoder = KinectDecoder(scale=scale,
                                     extract_rgb=extract_rgb, extract_depth=extract_depth, extract_X=extract_X)

    
        
        # Build Index
        count = 0
        self.utime_map = collections.OrderedDict();
        st = time.time()
        for ev in self.log:
            if ev.channel == self.channel:
                data = frame_msg_t.decode(ev.data);
                self.utime_map[data.timestamp] = ev.timestamp
                count += 1
                if count % 100 == 0: print 'Indexed %s frames: sn:%ld ev:%ld' % (count, data.timestamp, ev.timestamp)
        print 'Built index: %f seconds' % (time.time() - st)

        # Keep frame index
        if len(self.utime_map):
            self.cur_frame = 0
            sensor_utime = self.utime_map.keys()[self.cur_frame]
            event_utime = self.find_closest(sensor_utime)
            self.log.c_eventlog.seek_to_timestamp(event_utime)    

    def reset(self):
        if len(self.utime_map):
            self.cur_frame = 0
            sensor_utime = self.utime_map.keys()[self.cur_frame]
            event_utime = self.find_closest(sensor_utime)
            self.log.c_eventlog.seek_to_timestamp(event_utime)    
        
    def find_closest(self, utime):
        if utime in self.utime_map.keys():
            return self.utime_map[utime]
        else:
            return None
        #print bisect.bisect_left(self.utime_map.keys(), utime)
        #inds = [utime_map.keys()[ind], utime_map.keys[ind-1]];
        #print inds
        #print 'Closest either %ld or %ld to %ld' % (self.utime_map[ind],self.utime_map[ind-1], utime)
        #arg = argmin(self.utime_map[ind],self.utime_map[ind-1]);
        #print 'Closest: %ld' % (inds[arg])
        #return self.utime_map[inds[arg]]

    def get_frame(self, sensor_utime):
        # Decode images
        event_utime = self.find_closest(sensor_utime)
        # print 'Seek to closest frame: %ld %ld' % (sensor_utime, event_utime)
        if event_utime is None: return None
        
        # Seek to the timestamp
        self.log.c_eventlog.seek_to_timestamp(event_utime)

        # Continue reading events until the right channel is read
        ev = self.log.read_next_event()
        while ev is not None:
            if ev.channel == self.channel:
                data = frame_msg_t.decode(ev.data);
                #print 'Event timestamp: ', ev.timestamp
                return self.decoder.decode(data)
            ev = self.log.read_next_event()
        return None
    
    def get_next_frame(self):
        if not self.cur_frame < len(self.utime_map): return None
        sensor_utime = self.utime_map.keys()[self.cur_frame]
        self.cur_frame += 1
        return self.get_frame(sensor_utime)
    
    def get_next_frame_wo_seek(self):
        # Continue reading events until the right channel is read
        ev = self.log.read_next_event()
        while ev is not None:
            if ev.channel == self.channel:
                data = frame_msg_t.decode(ev.data);
                sensor_utime = data.timestamp
                #print 'Event timestamp: ', ev.timestamp
                return self.decoder.decode(data)
            ev = self.log.read_next_event()
        return None        
        
    
