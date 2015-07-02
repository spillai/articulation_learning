#!/usr/bin/python

# Test fs_utils

import cv2
import fs_utils

# Initialize reader
reader = fs_utils.LCMLogReader();
reader.init('/home/spillai/data/2013-09-03-artags-articulation/lcmlog-2013-09-03.01', 0.5, False)

# Init MSER
mser = fs_utils.MSER3D();

for j in range(0,40): 
    # Get frame
    frame = reader.getNextFrame()
    frame.computeNormals(0.5, 0.01, 10.0)
    
    fpts = mser.update(frame)
    # cv2.imshow('img', img)
    # cv2.waitKey(10);

    for fpt in fpts: 
        print fpt.id, fpt.xyz(), fpt.normal(), fpt.tangent()#fpt.point, fpt.keypoint.size, fpt.keypoint.angle, fpt.keypoint.response
        break
