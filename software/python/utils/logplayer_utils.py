import os
import numpy as np
import lcm; lc = lcm.LCM()
from fs_utils import LCMLogReader

class LCMLogPlayer: 
    def __init__(self, filename): # , every_k_frames=None, k_frames=None, utimes=None): 
        if filename is None or not os.path.exists(filename):
            raise Exception('Invalid Filename: %s' % filename)
        self.reader = LCMLogReader(filename=filename)
        self.reader.reset()
        # print 'NUM FRAMES: ', self.reader.getNumFrames()

        # # Default option
        # if every_k_frames is None and k_frames is None and utimes is None: 
        #     every_k_frames = 1

        # self.every_k_frames = every_k_frames
        # self.k_frames = k_frames

        # self.fno = 0
        # self.frame_inds = self.get_frame_inds()
        # self.frame_utimes = utimes

    def iterframes(self, every_k_frames=1): 
        fnos = np.arange(0, self.reader.getNumFrames()-1, every_k_frames).astype(int)
        for fno in fnos: 
            frame = self.reader.getFrame(fno)
            assert(frame is not None)
            yield frame

    def get_frames(self, every_k_frames=1): 
        fnos = np.arange(0, self.reader.getNumFrames()-1, every_k_frames).astype(int)
        
        return [self.reader.getFrame(float(fno)) for fno in fnos]
    def get_frame_inds(self): 
        frames = None
        if self.every_k_frames is not None: 
            frames = np.arange(0, self.reader.getNumFrames()-1).astype(int)
            if self.every_k_frames != 1: frames = frames[::self.every_k_frames]
        elif self.k_frames is not None: 
            frames = np.linspace(0, self.reader.getNumFrames()-1, self.k_frames).astype(int)
        return frames
        
    def reset(self): 
        self.reader.reset()
        self.fno = 0

    def get_frame_with_percent(self, pc): 
        assert(pc >= 0.0 and pc <= 1.0)
        seek_to_index = int(pc * 1.0 * len(self.frame_inds))
        assert(seek_to_index >= 0 and seek_to_index < len(self.frame_inds))
        print 'SEEK to : ', seek_to_index
        return self.get_frame_with_index(self.frame_inds[seek_to_index])

    def get_frame_with_index(self, index): 
        return self.reader.getFrame(index)

    def get_next_frame_with_index(self): 
        if self.fno >= len(self.frame_inds): 
            return None
        frame = self.reader.getFrame(self.frame_inds[self.fno])
        self.fno += 1
        return frame

    def get_frame_with_utime(self, utime): 
        return self.reader.getFrameWithTimestamp(utime)

    def get_next_frame_with_utime(self): 
        if self.fno >= len(self.frame_utimes): 
            return None
        frame = self.reader.getFrameWithTimestamp(self.frame_utimes[self.fno])
        self.fno += 1
        return frame

    def get_next_frame(self): 
        if self.frame_utimes is not None: 
            return self.get_next_frame_with_utime()
        else: 
            return self.get_next_frame_with_index()
