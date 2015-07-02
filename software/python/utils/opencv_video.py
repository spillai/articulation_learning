import cv2, os.path
import numpy as np

class SimpleVideoPlayer(object):
    def __init__(self, filename=None): 
        if not os.path.exists(filename): 
            raise RuntimeError('Filename %s does not exist' % filename)

        self.filename = filename
        self.cap = cv2.VideoCapture(self.filename)
        
        if self.cap is None or not self.cap.isOpened():
            print 'Warning: unable to open video source: ', self.filenameCD

        self.nframes = self.get_frame_count()

    def get_frame_count(self):
        return self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        
    def reset(self):
        self.frame = 0
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def seek(self, frame):
        assert(frame >= 0 and frame < self.nframes)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        
    def read(self): 
        ret, img = self.cap.read()
        if ret: 
            self.frame = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        return ret, img

    def read_frame(self, frame): 
        self.seek(frame)
        return self.read()

if __name__ == "__main__": 
    import sys, os.path, argparse, time

    parser = argparse.ArgumentParser(
        usage='python opencv_video.py -f <video_filename>'
    )
    parser.add_argument('--filename', '-f', 
                        help="Input Filename")

    (opts, args) = parser.parse_known_args()
    p = SimpleVideoPlayer(filename=opts.filename)
    print 'Frame count: %i' % p.nframes

    while True: 

        try: 
            ret, im = p.read()
            if not ret: break

            cv2.imshow('im', im)
            cv2.waitKey(10)
            
        except KeyboardInterrupt: 
            break
        
        # except KeyboardInterrupt: 
        #     break
        
