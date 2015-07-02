import copy
import cv2
import numpy as np

from utils.rigid_transform import RigidTransform, tf_construct, tf_construct_3pt, normalize_vec
import utils.draw_utils as draw_utils

from collections import OrderedDict
from utils.geom_utils import scale_points

# from fs_fovis_utils import FOVIS
from fs_pcl_utils import PlaneEstimationSAC
from fs_apriltags import AprilTagDetection
from fs_utils import GPFT, BirchfieldKLT, DenseTrajectories, LearDenseTrajectories#, Feature3D, IDIAPMSER, MSER3D
np.set_printoptions(precision=4, suppress=True, threshold='nan', linewidth=160)


class TrackerInfo(object): 
    def __init__(self, writer=None): 
        # self.writer = writer
        self.data = []

class AprilTagsFeatureTracker(TrackerInfo): 
    def __init__(self, writer=None, expected_id=None): 
        TrackerInfo.__init__(self)
        
        self.tracker = AprilTagDetection()
        self.expected_id = expected_id
        self.pose_init = None
        self.pose_init_inv = None

        # Keep frame pose
        self.frame_pose = draw_utils.get_frame('KINECT')

    # # TODO: Construct a rel pose graph
    # def get_pose(self, frame): 
    #     if self.expected_id is None: return None

    #     # First process
    #     self.tracker.processFrame(frame)

    #     # Get tags and save relative tf
    #     tags = self.tracker.getTags()
    #     for tag in tags: 
    #         if tag.id == self.expected_id: 
    #             pose_vec = tag.getPose()
    #             if np.isnan(pose_vec).any() or abs(np.linalg.norm(pose_vec[-4:])-1) > 1e-2: return None
    #             campose = rtf.RigidTransform.from_vec(pose_vec).inverse()
    #             if self.pose_init is None: 
    #                 self.pose_init = copy.deepcopy(campose);
    #                 self.pose_init_inv = self.pose_init.inverse()
    #                 pose_rel = rtf.RigidTransform([1,0,0,0],[0,0,0])
    #             else: 
    #                 pose_rel = self.pose_init_inv * campose
    #             return pose_rel
    #     print 'Tag %i not found!' % (self.expected_id)
    #     return None


    def get_pose_map(self, frame, ids=None): 
        self.tracker.processFrame(frame)
        tags = self.tracker.getTags()

        pose_map = dict()
        for tag in tags: 
            if ids is not None and tag.id not in ids: 
                continue

            fpts = tag.getFeatures()

            Xs = np.vstack([fpt.xyz().reshape((-1,3)) for fpt in fpts])
            Xs = Xs[~np.isnan(Xs).any(axis=1)]
            if len(Xs) != 4: continue

            p00 = np.mean(Xs, axis=0)
            p0, p1 = Xs[0], Xs[1]
            
            Ns = np.vstack([fpt.normal().reshape((-1,3)) for fpt in fpts])
            Ns = Ns[~np.isnan(Ns).any(axis=1)]
            if not len(Ns): continue            

            n0 = np.mean(Ns, axis=0)
            # print 'P0, N0, vec', p0, n0, (p1-p0) * 1.0 / np.linalg.norm(p1-p0)
            v0 = normalize_vec(p1-p0)

            R = tf_construct(n0, v0)
            # xold, yold, zold = R[:,0].copy(), R[:,1].copy(), R[:,2].copy()
            # R[:,0], R[:,1], R[:,2] = -yold, -zold, xold
            pose_map[tag.id] = RigidTransform.from_Rt(R, p00)
        return pose_map

    def get_pose_map_subpix(self, frame, ids=None): 
        # Init frame vars
        gray = frame.getGray()
        cloud = frame.getCloud()
        mask_ = np.zeros_like(gray)

        # Detect tags
        dets = self.tracker.processImage(gray)

        pose_map = dict()
        for d in dets: 
            if ids is not None and d.id not in ids: 
                continue

            det = np.vstack([f.flatten() for f in d.getFeatures()])
            # Reinit mask
            mask = mask_.copy()

            # Construct mask from detection
            sdet = scale_points(det, scale=1.75)
            hull = cv2.convexHull(sdet.astype(np.int32))
            cv2.fillConvexPoly(mask, hull, 255, cv2.LINE_AA, shift=0)

            # Retrive points within hull of detection
            pts3d = cloud[mask.astype(np.bool)]
            inds = np.isfinite(pts3d).all(axis=1)

            # SAC Plane estimation
            coeffs, _ = PlaneEstimationSAC(cloud=pts3d[inds], inlier_threshold=0.02)

            # Find intersections
            # http://www.cs.princeton.edu/courses/archive/fall00/cs426/lectures/raycast/sld017.htm
            xys = det.astype(np.int32)
            X = cloud[xys[:,1], xys[:,0]]
            dX = X / np.linalg.norm(X, axis=1)[:,np.newaxis]
            X_ = np.vstack([(-coeffs[3] / (np.inner(coeffs[:3],dx))) * dx for dx in dX])

            try: 
                pose = tf_construct_3pt(X_[0], X_[1], X_[2], origin=0.5*X_[0] + 0.5*X_[2])
                R = pose.quat.to_matrix()
                # Rnew = np.vstack([R[:,2], -R[:,1], R[:,0]]).T
                pose_map[d.id] = pose # RigidTransform.from_Rt(Rnew, pose.tvec)
            except: 
                pass

        return pose_map

    def get_pose(self, frame): 
        fpts = self.get_features(frame)

        Xs = np.vstack([fpt.xyz().reshape((-1,3)) for fpt in fpts])
        p0, p1 = Xs[0], Xs[1]
        Ns = np.vstack([fpt.normal().reshape((-1,3)) for fpt in fpts])
        XNs = np.hstack([Xs, Ns])

        # Can't retrieve pose with all nans
        if np.isnan(XNs).all(): return None

        XNs = XNs[~np.isnan(XNs).any(axis=1)]
        n0 = np.mean(XNs, axis=0)[3:]
        # print 'P0, N0, vec', p0, n0, (p1-p0) * 1.0 / np.linalg.norm(p1-p0)
        R = tf_construct(n0, (p1-p0) * 1.0 / np.linalg.norm(p1-p0))
        return RigidTransform.from_Rt(R, p0)

    def get_features(self, frame, return_mask=False): 
        self.tracker.processFrame(frame)
        tags = self.tracker.getTags()
        features = []
        for tag in tags: features.extend([ f for f in tag.getFeatures() ])
        if return_mask: 
            mask = self.tracker.getMask(scale=1.2)
            cv2.imshow('tag-mask', mask)
            return features, mask
        return features

class GPFTTracker(TrackerInfo): 
    def __init__(self, writer=None, allowed_skips=1): 
        TrackerInfo.__init__(self, writer=writer)
        self.tracker = GPFT()
        self.allowed_skips = allowed_skips;

    def get_features(self, frame, mask): 
        self.tracker.processFrame(frame, mask)
        return self.tracker.getStableFeatures()


class BirchfieldKLTTracker(TrackerInfo): 
    def __init__(self, writer=None): 
        TrackerInfo.__init__(self, writer=writer)
        self.tracker = BirchfieldKLT(num_feats=800, affine_consistency_check=1)

    def get_features(self, frame): 
        self.tracker.processFrame(frame)
        return self.tracker.getStableFeatures()

class DenseTrajectoriesTracker(TrackerInfo): 
    def __init__(self, writer=None): 
        TrackerInfo.__init__(self, writer=writer)
        self.tracker = DenseTrajectories()

    def get_features(self, frame, mask=np.array([])): 
        self.tracker.processFrame(frame, mask=mask)
        return self.tracker.getStableFeatures()

class LearDenseTrajectoriesTracker(TrackerInfo): 
    def __init__(self, writer=None): 
        TrackerInfo.__init__(self, writer=writer)
        self.tracker = LearDenseTrajectories()

    def get_features(self, frame): 
        self.tracker.processFrame(frame)
        return self.tracker.getStableFeatures()

# class IDIAPMSERTracker(TrackerInfo): 
#     def __init__(self, writer): 
#         TrackerInfo.__init__(self, writer=writer)
#         self.tracker = IDIAPMSER(delta=2, min_area=0.0005, max_area=0.5, 
#                                  max_variation=0.5, min_diversity=0.5, eight=True)

#     def get_features(self, frame): 
#         self.tracker.processFrame(frame)
#         return None; # self.tracker.getStableFeatures()

# class OpenCVMSERTracker(TrackerInfo): 
#     def __init__(self, name, writer,conv=None): 
#         TrackerInfo.__init__(self, name=name, writer=writer)
#         self.tracker = cv2.MSER(); 
#         self.color_conv = conv
#         # delta=2, min_area=0.0005, max_area=0.1, 
#         #                         max_variation=0.5, min_diversity=0.5, eight=True)

#     def get_features(self, frame): 
#         img = frame.getRGB()
#         disp = np.copy(img)
#         if self.color_conv is not None: 
#             img = cv2.cvtColor(img, self.color_conv)

#         regions = self.tracker.detect(img, None)
#         hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
#         cv2.polylines(disp, hulls, 1, (0, 255, 0))
#         cv2.imshow(''.join(['MSER-',self.name]), disp)
#         cv2.waitKey(1)

#         # self.tracker.processFrame(frame)
#         return None; # self.tracker.getStableFeatures()


# class FOVISEstimator(TrackerInfo): 
#     def __init__(self, name, table, writer): 
#         TrackerInfo.__init__(self, name=name, table=table, writer=writer)
#         self.estimator = FOVIS()

#     def get_pose(self, frame): 
#         self.estimator.processFrame(frame)
#         pose = self.estimator.getPose()
#         if np.isnan(pose).any(): return None
#         return rtf.Pose.from_vec(frame.utime, pose)
