#!/usr/bin/env python
import lcm, time
import cv2
from kinect import image_msg_t, frame_msg_t
from bot_core import image_t

from utils.lcm_utils import KinectInput
from utils.imshow_utils import imshow

feature_params = dict( maxCorners = 500,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

detector = cv2.PyramidAdaptedFeatureDetector(detector=cv2.FastFeatureDetector(16, True), 
                                                maxLevel=3) 

st = time.time()    
def my_handler(frame):
    global st

    im = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
    rgb = im.copy()
    im = cv2.GaussianBlur(im, (3, 3), 0)

    kpts = detector.detect(im)
    for kpt in kpts: 
        cv2.circle(rgb, tuple(map(int, kpt.pt)), 3, (0,255,0), -1, lineType=cv2.LINE_AA)

    cv2.imshow('rgb', rgb)
    cv2.waitKey(10)

    if time.time() - st > 1: 
        print len(kpts)
        st = time.time()

if __name__ == "__main__":     
    stream = KinectInput(my_handler).run()
