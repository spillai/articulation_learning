#!/usr/bin/env python
import lcm
import cv2
from kinect import image_msg_t, frame_msg_t
from bot_core import image_t

def my_handler(channel, data):

    # Decode
    im = frame_msg_t.decode(data).image

    # Construct bot core image
    out = image_t()

    # Populate appropriate fields
    h,w,c = im.height, im.width, 3
    out.width, out.height = w, h
    out.row_stride = w*c
    out.utime = im.timestamp

    # Propagate appropriate encoding 
    if im.image_data_format == image_msg_t.VIDEO_RGB: 
        out.pixelformat = image_t.PIXEL_FORMAT_RGB
    elif im.image_data_format == image_msg_t.VIDEO_RGB_JPEG: 
        out.pixelformat = image_t.PIXEL_FORMAT_MJPEG
    else: 
        raise AssertionError('Unknown pixel format!')
        
    # Propagate encoded/raw data, 
    out.size = im.image_data_nbytes
    out.data = im.image_data
    out.nmetadata = 0

    # Pub
    lc.publish('KINECT_IMAGE', out.encode())

if __name__ == "__main__":     
    lc = lcm.LCM()
    sub = lc.subscribe("KINECT_FRAME", my_handler)
    
    try:
        while True:
            lc.handle()
    except KeyboardInterrupt:
        pass

    lc.unsubscribe(sub)

