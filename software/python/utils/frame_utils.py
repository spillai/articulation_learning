import cv2
import numpy as np

import utils.draw_utils as draw_utils

from utils.image.desc_utils import im_desc
from utils.camera_utils import KinectCamera

class Frame(object): 
    """
    General Frame Interface
    utime: double (index or utime)
    rgb: uint8 480x640x3
    """
    def __init__(self, utime, rgb):
        self.utime = utime
        self.rgb = rgb

class KinectFrame(Frame):
    """
    RGB-D Frame
    rgb: uint8 480x640x3
    depth: uint16 480x640x3
    """

    def __init__(self, utime, rgb, depth, compute_cloud=True, skip=1):
        Frame.__init__(self, utime, rgb)

        self.depth = depth
        if compute_cloud: 
            self.X = self.get_cloud(depth, skip=skip)

    def get_cloud(self, depth, skip=1): 
        return KinectCamera(skip=skip).get_cloud(depth)

    def viz(self): 
        draw_utils.publish_point_cloud('Test_KINECT_FRAME', self.X[::2,::2], self.rgb[::2,::2] * 1. / 255)

def frame_desc(frame, pts2d=None, mask=None): 
    # 2D description, and pts
    gray =  cv2.GaussianBlur(frame.getGray(), (3, 3), 0)        

    if pts2d is None: 
        kpts, desc = im_desc(gray, mask=mask)
        pts2d = np.vstack([kp.pt for kp in kpts])
    else: 
        _, desc = im_desc(gray, kpts=[cv2.KeyPoint(pt[0],pt[1],_size=1.0) for pt in pts2d])        

    # 3D points
    X = frame.getCloud()
    xys = pts2d.astype(int)
    pts3d = X[xys[:,1],xys[:,0],:]

    return pts2d, pts3d, desc

        
# class Frame:
#     def __init__(self, frame=None):
#         self.valid=False
        
#         if frame is None: return 
#         self.utime = frame.utime
#         self.rgb = frame.rgb
#         self.depth = frame.depth
#         self.X = frame.X
#         self.valid = True
        
#     # def valid(self):
#     #     return getattr(self, 'valid', False)
    
#     def compute_depth_mask(self):
#         assert self.valid

#         # Depth mask
#         self.depth_mask = np.bitwise_not(self.depth <= 0)

#     def get_rgb_with_depth_mask(self):
#         pass
#         # # Img with NaN mask
#         # img_with_depth_mask = np.empty_like(img)
#         # for j in range(3):
#         # 	img_with_depth_mask[:,:,j] = np.bitwise_and(img[:,:,j], depth_mask);
        
#     def compute_normals(self, smoothing_size=10, depth_change_factor=0.5):
#         # Integral normal estimation (%timeit ~52ms per loop)
#         self.normals = pcl_utils.integral_normal_estimation(self.X,
#                                                             smoothing_size=smoothing_size,
#                                                             depth_change_factor=depth_change_factor);
#         self.normals_mask = np.bitwise_not(np.any(np.isnan(self.normals), axis=2))

#     def compute_normals(self, smoothing_size=10, depth_change_factor=0.5):
#         # Integral normal estimation (%timeit ~52ms per loop)
#         self.normals = pcl_utils.integral_normal_estimation(self.X,
#                                                             smoothing_size=smoothing_size,
#                                                             depth_change_factor=depth_change_factor);
#         self.normals_mask = np.bitwise_not(np.any(np.isnan(self.normals), axis=2))
        
#     def visualize_normals(self):
#         # Normalize to range [0,1]
#         normals_img = 0.5 * (self.normals + np.ones_like(self.normals));
#         # Plot
