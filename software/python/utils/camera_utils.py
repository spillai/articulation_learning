import cv2
import numpy as np
from rigid_transform import Quaternion, RigidTransform

class Camera: 
    def __init__(self): 
        pass

class KinectCamera(Camera): 
    _calib_available = False
    def __init__(self): 
        if not KinectCamera._calib_available:
            self.setup_calibration()
    
    def setup_calibration(self): 
        # Camera matrix
        self.H,self.W = 480, 640
        self.rgbK = np.array([[528.49404721, 0, 319.5],
                              [0, 528.49404721, 239.5],
                              [0, 0, 1]], dtype=np.float64)
        
        self.depthK = np.array([[576.09757860, 0, 319.5],
                                [0, 576.09757860, 239.5],
                                [0, 0, 1]], dtype=np.float64)

        # Cached camera params 
        self.fx = self.depthK[0,0];
        self.cx = self.depthK[0,2];
        self.cy = self.depthK[1,2];
        self.fx_inv = 1.0 / self.fx;
    
        # Project shift offsets, and baseline
        self.shift_offset = 1079.4753;
        self.projector_depth_baseline = 0.07214;

        # Dist coeff
        self.D = np.array([0., 0., 0., 0.], dtype=np.float64)

        # Dense calibration map
        self.xs, self.ys = np.meshgrid((np.arange(0,self.W) - self.cx) * self.fx_inv, 
                                       (np.arange(0,self.H) - self.cy) * self.fx_inv);
    
        _, self.irgbK = cv2.invert(self.rgbK)
        self.irgbtK = self.irgbK.T
        
        # Toggle calib availability
        KinectCamera._calib_available = True
        
        
    # 16-bit depth 
    def get_cloud(self, depth, in_m=False): 
        if not in_m: depthf = depth.astype(np.float32) * 0.001; # in m
        else: depthf = depth
        xs = np.multiply(self.xs, depthf);
        ys = np.multiply(self.ys, depthf)
        X = np.dstack((xs,ys,depthf))
        return X


    # # get fundamental matrix between two poses
    # def F(self, t1, t2): 
    #     p2w1 = self.relative(t1, t2)
    #     T2w1 = p2w1.to_homogeneous_matrix()
    #     E = np.dot(np.array([[0, -T2w1[2,3], T2w1[1,3]],
    #     		      [T2w1[2,3], 0, -T2w1[0,3]], 
    #     		      [-T2w1[1,3], T2w1[0,3], 0]]), T2w1[:3,:3])
    #     return np.dot(self.irgbtK, np.dot(E, self.irgbK))

    def project(self, X, pose): 
	T = pose.to_homogeneous_matrix()
	rvec,_ = cv2.Rodrigues(T[:3,:3])
	proj,_ = cv2.projectPoints(X.reshape((1,3)), rvec,  T[:3,3], KinectCamera.depthK, KinectCamera.D)
	return proj.reshape(2)
