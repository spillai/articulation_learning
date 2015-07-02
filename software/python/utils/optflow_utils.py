import cv2
import numpy as np

def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis

def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx+fy*fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[...,0] = ang*(180/np.pi/2)
    hsv[...,1] = 255
    hsv[...,2] = np.minimum(v*4, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res



# RY = 15;
# YG = 6;
# GC = 4;
# CB = 11;
# BM = 13;
# MR = 6;
# ncols = RY + YG + GC + CB + BM + MR;
# colorwheel = np.zeros((ncols, 3)); % r g b
# color_wheel_available = False

# def make_color_wheel():
#     """
#     %   color encoding scheme
#     %   adapted from the color circle idea described at
#     %   http://members.shaw.ca/quadibloc/other/colint.htm
#     """
#     if color_wheel_available:
#         print 'Color wheel available already'
#         return
    
#     col = 0;
    
#     #RY
#     colorwheel[np.arange(0,RY), 0] = 255;
#     colorwheel[np.arange(0,RY), 1] = floor(255*np.arange(0,RY)/RY)';
#     col = col+RY;
    
#     #YG
#     colorwheel[col+np.arange(0,YG), 0] = 255 - floor(255*np.arange(0,YG)/YG)';
#     colorwheel[col+np.arange(0,YG), 1] = 255;
#     col = col+YG;
    
#     #GC
#     colorwheel[col+np.arange(0,GC), 1] = 255;
#     colorwheel[col+np.arange(0,GC), 2] = floor(255*np.arange(0,GC)/GC)';
#     col = col+GC;
    
#     #CB
#     colorwheel[col+np.arange(0,CB), 1] = 255 - floor(255*np.arange(0,CB)/CB)';
#     colorwheel[col+np.arange(0,CB), 2] = 255;
#     col = col+CB;
    
#     #BM
#     colorwheel[col+np.arange(0,BM), 2] = 255;
#     colorwheel[col+np.arange(0,BM), 0] = floor(255*np.arange(0,BM)/BM)';
#     col = col+BM;
    
#     #MR
#     colorwheel[col+np.arange(0,MR), 2] = 255 - floor(255*np.arange(0,MR)/MR)';
#     colorwheel[col+np.arange(0,MR), 0] = 255;
    

# def compute_color(u,v):
#     naninds = np.logical_or(np.isnan(u),np.isnan(v))
#     u[nanids] = 0
#     v[naninds] = 0

#     if not color_wheel_available: make_color_wheel()
#     #ncols = size(colorwheel, 1);
#     rad = np.sqrt(np.u*u+v*v);  # * is elementwise equivalent to u.*u in matlab

#     a = np.arctan2(-v, -u) / np.pi;
#     fk = (a+1)/2 * (ncols-1) + 1;  # -1~1 maped to 1~ncols
#     k0 = np.floor(fk);                 # 1, 2, ..., ncols

#     k1 = k0+1;
#     k1 = np.where(k1==ncols+1, 1, k1);

#     f = fk - k0;

    # for j in range(colorwheel.shape[1]):
    #     tmp = colorwheel[:,j]
    #     col0 = tmp(k0)/255;
    #     col1 = tmp(k1)/255;
    #     col = (1-f).*col0 + f.*col1;   
        
    #     idx = rad <= 1;   
    #     col(idx) = 1-rad(idx).*(1-col(idx));    % increase saturation with radius
        
    #     col(~idx) = col(~idx)*0.75;             % out of range
        
    #     img[:,:, j] = uint8(floor(255*col.*(1-nanIdx)));         
