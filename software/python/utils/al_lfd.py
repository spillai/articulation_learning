#!/usr/bin/env python
import cv2
import numpy as np
import draw_utils

import itertools, logging
from collections import defaultdict, OrderedDict

from utils.rigid_transform import RigidTransform
from utils.db_utils import DictDB, AttrDict
from utils.data_containers import Feature3DData, Pose3DData

from utils.geom_utils import scale_points, contours_from_endpoints

# from utils.dbg_utils import ColorLog
from utils.trackers.tracker_utils import AprilTagsFeatureTracker
from utils.trackers.offline_klt import StandardKLT, FwdBwdKLT, GatedFwdBwdKLT

from utils.trajectory_analysis import TrajectoryClustering
from utils.pose_estimation import PoseEstimation
from utils.articulation_analysis import ArticulationAnalysis

from utils.camera_utils import KinectCamera
from utils.correspondence_estimation import BetweenImagePoseEstimator, remove_nans
from utils.bow_utils import ObjectLocalizationBOW
from utils.frame_utils import frame_desc
np.set_printoptions(precision=2, suppress=True, threshold='nan', linewidth=160)

class TagALfD:  
    def __init__(self, params, ids=None):
        self.log = logging.getLogger(self.__class__.__name__)
        
        self.params = params
        self.ids = ids
        self.atag = AprilTagsFeatureTracker()

    def get_tags(self, frames): 
        tag_pose_map = defaultdict(lambda : OrderedDict())

        for ut_idx in range(len(frames)): 
            tag_pose_map[-1][ut_idx] = RigidTransform.identity()

        # Poses should be inserted in ascending order (ascending frame order)
        for ut_idx, frame in enumerate(frames): 
            for tid, pose in self.atag.get_pose_map(frame, self.ids).iteritems(): 
                tag_pose_map[tid][ut_idx] = pose


        # Visualize poses
        viz_poses = [pose for utime_poses in tag_pose_map.values() for (utime,pose) in utime_poses.iteritems()]
        draw_utils.publish_pose_list2('ARTAG_POSES', viz_poses, sensor_tf='KINECT')

        return tag_pose_map

    def articulation_analysis(self, poses, viz_attrib='pose_projected'): 
        """Analyze articulation give final pose estimates"""
        art_est = ArticulationAnalysis(poses, params=self.params.articulation, 
                                       viz_attrib=viz_attrib)
        return art_est

    def process_demonstration(self, frames, name): 
        # Extract images, normals, clouds
        for f in frames: 
            f.fastBilateralFilter(20.0, 0.04)
            f.computeNormals(0.5)

        # Get tag poses for each utime
        tag_pose_map = self.get_tags(frames)

        # convert to pose list
        poses = [ (tid, [(utime,pose) 
                                 for utime,pose in utime_pose_map.iteritems()] ) 
                          for tid, utime_pose_map in tag_pose_map.iteritems() ]
        # Sort by label
        poses.sort(key=lambda x: x[0])         

        # Articulation Analysis
        self.log.info('Articulation Analysis -----------------------------')
        art_est = self.articulation_analysis(poses)

        self.log.info('Saving all variables, and adding to DB -----------')
        self.model = AttrDict({'tag_pose_map': tag_pose_map, 'poses': poses, 'art_est':art_est})


class ArticulationLfD:  
    def __init__(self, params, debug_dir=None): 

        self.log = logging.getLogger(self.__class__.__name__)

        self.features_map_ = dict()
        self.params_ = params
        self.debug_dir = debug_dir

        # Initialize DB
        self.db = ArticulationDB("articulation.db")

    # =============================================================================
    # Training 
    # =============================================================================
    def setup_database(self, db): 
        # Read from DictDB and setup feature trajectories for learning
        gpft_data = Feature3DData(db.data.gpft, discretize_ms=1)
        self.log.info('--- Done building data')

        return self.setup_features(gpft_data)

        
    def setup_features(self, features): 
        # Smoothing and pruning of feature trajectories
        features.prune_by_length(
            min_track_length=self.params_.feature_selection.min_track_length
        )
        # Pick mostly top moving features
        features.pick_top_by_displacement(k=90)

        # Remove utimes/inds that are discontinuous
        features.prune_discontinuous(            
            min_feature_continuity_distance=self.params_.feature_selection.min_feature_continuity_distance,
            min_normal_continuity_angle=self.params_.feature_selection.min_normal_continuity_angle
        )
        return features

    def process_db(self, features_fn): 
        # Open DB
        self.db_ = DictDB(filename=features_fn, mode='r', ext='.h5')
        features = self.setup_database(self.db_)
        
        # Process features
        self.process_features(features)

    def process_demonstration(self, frames, name): 
        # Extract images, normals, clouds
        for f in frames: 
            f.fastBilateralFilter(20.0, 0.04)
            f.computeNormals(0.5)

        images = map(lambda x: x.getRGB(), frames)
        Xs = map(lambda x: x.getCloud(), frames)
        Ns = map(lambda x: x.getNormals(), frames)
        
        carr = images[-1][::4,::4].reshape(-1,3) * 1.0 / 255
        draw_utils.publish_point_cloud('CLOUD_KINECT', 
                                       Xs[-1][::4,::4].reshape(-1,3), c=carr[:,::-1].reshape(-1,3),
                                       sensor_tf='KINECT')


        # Extract KLT features
        self.log.info('Feature Extaction ---------------------------------')
        klt = FwdBwdKLT(images=images, oklt_params=self.params_.tracker)
        klt.features = klt.get_feature_data(Xs, Ns)
        
        # Process and extract relevant features
        self.log.info('Process Features ----------------------------------')
        features = self.setup_features(klt.features)

        # Viz feature trajectories (after outlier removal)
        features.viz_data(utimes_inds=None)  
        
        # Trajectory Clustering
        self.log.info('Feature Clustering --------------------------------')
        clusters = self.clustering(features)

        # # Write to video
        # features.write_video(frames, 'test.avi')

        # Pose Estimation
        self.log.info('Pose Estimation -----------------------------------')
        poses = self.pose_estimation(features, clusters).get_final_pose_list()
        # self.log.info(poses 

        # Articulation Analysis
        self.log.info('Articulation Analysis -----------------------------')
        art_est = self.articulation_analysis(poses)

        # Temporary! FIX! TODO
        self.log.info('Saving all variables, and adding to DB -----------')
        self.model = AttrDict({'features':features, 'clusters':clusters, 'poses': poses, 'art_est':art_est})
        # self.model = AttrDict({'poses': poses, 'art_est':art_est})

        # # Estimate motion manifold
        # print '\n\nManifold Analysis -----------------------------'
        # self.motion_analysis(frames, art_est.get_projected_poses())

        # Add visual appearance features to DB
        self.db.add_demonstration(name, frames, self.model)



    # def setup_simulated_features(self, db, attr='test1'): 
    #     # Read from DictDB and setup feature trajectories for learning
    #     gpft_data = Pose3DData(db.data[attr].samples, discretize_ms=1)
    #     print '--- Done building data'
    #     return gpft_data

    # def process_manipulation(self, manip_fn): 
    #     # Open DB
    #     self.db_ = DictDB(filename=manip_fn, mode='r', ext='.h5')
    #     self.poses_ = self.db_.data.poses

    #     # Add stationary set of poses for camera
    #     poses = [(0, [(ut, RigidTransform.identity()) for (ut,_) in self.poses_]), 
    #              (1, [(ut, pose) for (ut,pose) in self.poses_])]

        
    #     # Perform art. estimation and get projected poses 
    #     self.art_est_ = self.articulation_analysis(poses)
    #     # sampled_poses = self.art_est_.get_sampled_poses()
    #     projected_poses = self.art_est_.get_projected_poses()
    
    #     # Visualize
    #     viz_poses = []
    #     for model_id, utime_poses in projected_poses: 
    #         viz_poses.extend([pose for utime,pose in utime_poses])
    #     draw_utils.publish_pose_list('projected_poses', viz_poses, stamp=rospy.Time.now(), 
    #                                  frame_id='head_mount_kinect_rgb_optical_frame')


    # =============================================================================
    # Trajectory Clustering, PoseEstimation, Articulation Estimation
    # =============================================================================

    def clustering(self, features): 
        """
        Clustering trajectories based on rigid-point-pair-feature
        Could provide a range, and run for several range chunks
        """
        return TrajectoryClustering(features, params=self.params_.clustering, 
                                    visualize=False, save_dir=self.debug_dir)

    def pose_estimation(self, features, clusters): 
        """Pose Estimation from identified clusters"""
        pose_est = PoseEstimation(data=features, 
                                  cluster_inds=clusters.cluster_inds, 
                                  params=self.params_.pose_estimation)
        pose_est.viz_data()
        return pose_est

    def articulation_analysis(self, poses, viz_attrib='pose_projected'): 
        """Analyze articulation give final pose estimates"""
        art_est = ArticulationAnalysis(poses, params=self.params_.articulation, 
                                       viz_attrib=viz_attrib)
        return art_est


    # =============================================================================
    # Motion analysis
    # =============================================================================
    def mser_regions(self, frames, poses): 
        
        # First pose, utime
        init_utime, init_pose = poses[0]
        final_utime, final_pose = max(poses, key=lambda (utime,pose): \
                                      np.linalg.norm(pose.tvec-init_pose.tvec))
        final_pose_pts = KinectCamera().project(final_pose.tvec, RigidTransform.identity())
        
        # Estimate MSER regions @ final pose
        X_final = frames[final_utime].getCloud()
        N_final = frames[final_utime].getNormals()
        N_final8 = ((N_final + 1) * 128).astype(np.uint8)
        regions = self.mser.detect(N_final8, None)
        hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]

        # Filter regions that enclose projected pt
        # cinds = self.model.features.cluster_inds[label]
        # xys = self.model.features.xy[cinds,final_utime]
        hulls = filter(lambda hull: 
                       cv2.pointPolygonTest(hull, 
                                            tuple(final_pose_pts.astype(int)), 
                                            measureDist=False) > 0, hulls)


        # Filter regions that have a valid surface normal, and pick top
        hulls = filter(lambda hull: 
                       self.valid_region(hull, X_final, N_final), hulls)
        if not len(hulls): 
            self.log.warning('WARNING: No regions found that meet criteria!')
            return None
        hull = sorted(hulls, key=lambda hull: cv2.contourArea(hull), reverse=False)[0]
        hull3d = np.vstack(map(lambda pt: X_final[pt[1],pt[0]], np.vstack(hull)))

        mask = np.zeros_like(frames[final_utime].getGray())
            
        vis = frames[final_utime].getRGB()
        cv2.polylines(vis, [hull], 1, (0, 255, 0))
        cv2.fillConvexPoly(mask, hull, 255, cv2.LINE_AA, shift=0)
        # cv2.imshow('viz', vis)   
        # cv2.waitKey(10)

        return AttrDict({'vis':vis, 'mask':mask, 'hull3d':hull3d, 
                         'init_utime':init_utime, 'init_pose':init_pose, 
                         'final_utime':final_utime, 'final_pose':final_pose, 
                         'mask_utime':final_utime, 'all_poses':poses})

    def motion_analysis(self, frames, labeled_poses): 
        # Prediction storage {label:surface, .. }
        surface_map = dict()

        # Compute MSER features on the rgb image
        self.mser = cv2.MSER(_min_area=100, _max_area=320*240, 
                             _max_evolution=200, _edge_blur_size=15)

        # For each label
        for label, poses in labeled_poses: 
            
            # Don't do anything for ref. frame
            if label < 0: continue
            self.log.info('Motion manifold %s' % label)

            # Filter out non-sensical poses
            poses = filter(lambda (utime,pose): np.linalg.norm(pose.tvec) > 0 
                           and (not np.isnan(pose.tvec).any()), poses)

            # Checks for valid poses per label
            if not len(poses): 
                self.log.info('No valid poses for label %s' % label)


            # Store the prediction surface
            surface_info = self.mser_regions(frames, poses)
            surface_map[label] = surface_info
            
        return surface_map


    def valid_region(self, region, cloud, normals): 
        region_pts = contours_from_endpoints(region.reshape(-1,2),5)
        valid_pts = np.vstack(map(lambda pt: cloud[pt[1],pt[0]], region_pts))
        inds, = np.where(~np.isnan(valid_pts).any(axis=1))

        # Check 1: Invalid, If more than 10% are nans
        # print 'VALID PTS: ', len(inds) * 1.0 / len(valid_pts)
        if len(inds) * 1.0 / len(valid_pts) <= 0.9: 
            return False

        # # valid_pts = valid_pts[inds]

        # # Check 2: 
        # valid_normals = np.vstack(map(lambda pt: normals[pt[1],pt[0]], region_pts))
        # inds, = np.where(~np.isnan(valid_normals).any(axis=1))
        # print 'VALID_NORMALS: ', valid_normals

        return True

    # =============================================================================
    # Prediction 
    # TODO: 
    # 1. CRF normal segmentation, 
    # 2. Labeled eucl. cluster extraction
    # 3. Grabcut for surface normals
    # 4. Organized connected component segmentation
    # 5. Region growing
    # =============================================================================
    def predict(self, frame, visualize=False):
        # Query DB for top search result
        q = self.db.query(frame)
        return ;

        # Retrieve relative pose of object (in DB) w.r.t query
        # Pose prediction of query object: pose_q = pose_qo oplus pose_o
        pose_qo, obj = q['rel_pose'], q['object']
        if obj is None: return

        # print pose_qo, obj['name']

        # Learned manifold
        surface_info = obj['model'].manifold

        # Build learned hulls
        learned_hulls = []
        for label,surface in surface_info.iteritems(): 
            final_pose = surface.final_pose

            hulls3d = []
            for (utime,p) in surface.all_poses: 
                # [N x 3]
                hull3d_p = (p.oplus(final_pose.inverse())) * surface.hull3d

                # P * [N x 3]
                hulls3d.append(hull3d_p)
            learned_hulls.extend(hulls3d)

        if not visualize: 
            return

        # Plot learned manifold
        draw_utils.publish_point_cloud('LEARNED_MANIFOLD', 
                                       learned_hulls,
                                       point_type='POLYGON', c='g', 
                                       sensor_tf='KINECT')

        learned_hulls_edges = map(lambda hull: np.hstack([hull[:-1], hull[1:]]).reshape((-1,3)), 
                                  learned_hulls)
        draw_utils.publish_point_cloud('LEARNED_MANIFOLD-edges', 
                                       learned_hulls_edges, 
                                       point_type='LINES', c='#005c00', 
                                       sensor_tf='KINECT')

        # Plot predicted hulls
        predicted_hulls = [pose_qo * hull for hull in learned_hulls]
        draw_utils.publish_point_cloud('PREDICTED_MANIFOLD', 
                                       predicted_hulls,
                                       point_type='POLYGON', c='b', 
                                       sensor_tf='KINECT')

        predicted_hulls_edges = map(lambda hull: 
                                    np.hstack([hull[:-1], hull[1:]]).reshape((-1,3)), 
                                    predicted_hulls)
        draw_utils.publish_point_cloud('PREDICTED_MANIFOLD-edges', 
                                       predicted_hulls_edges, 
                                       point_type='LINES', c='#000b96', 
                                       sensor_tf='KINECT')


        # Transform poses to new reference frame
        # Debug the transform
        draw_utils.publish_point_cloud('LEARNED CLOUD', obj['model'].features.xyz, 
                                       c='r', sensor_tf='KINECT')

        q_cloud = frame.getCloud()
        q_cloud = remove_nans(q_cloud[::6,::6].reshape((-1,3)))
        draw_utils.publish_point_cloud('QUERY CLOUD', q_cloud, c='b', sensor_tf='KINECT')

        p_cloud = pose_qo.inverse() * q_cloud
        draw_utils.publish_point_cloud('CORRECTED (PREDICTED) CLOUD', p_cloud,
                                       c='g', sensor_tf='KINECT')

        # PLot predicted poses
        # aposes = # map( lambda (ut,pose): pose, 
        aposes = map(lambda (l,ut_poses): ut_poses, obj['model'].art_est.get_projected_poses())
        aposes = [pose for ut_poses in aposes for (ut,pose) in ut_poses]
        predicted_poses = [pose_qo.oplus(pose) for pose in aposes]
        draw_utils.publish_pose_list2('PREDICTED_POSES', 
                                      predicted_poses, sensor_tf='KINECT')

    def extrude_hulls(self, hulls): 
        pts = []
        for hull in hulls: 
            ehull = list(hull)
            ehull.append(hull[0])
            pts.append(ehull)
        return pts


            
class ArticulationDB: 
    """ 
    Build DB with object features, and corresponding kinematic relationships
    """
    def __init__(self, fn): 
        self.log = logging.getLogger(self.__class__.__name__)

        self.obj_loc = ObjectLocalizationBOW()
        self.db_built = False

    def describe_label(self, frame, pts2d): 
        mask = np.zeros_like(frame.getGray())

        # Construct mask from detection
        spts2d = scale_points(pts2d, scale=1.0)
        hull = cv2.convexHull(spts2d.astype(np.int32))
        cv2.fillConvexPoly(mask, hull, 255, cv2.LINE_AA, shift=0)

        # Describe convex hull mask (replace with MSER mask)
        return frame_desc(frame, pts2d=None, mask=mask)       

    def add_demonstration(self, name, frames, model, init_tidx=0): 
        """Add a demonstration by providing RGB-D frames, and lfvd variables"""
        if not (hasattr(model, 'features') and 
                hasattr(model, 'clusters') and 
                hasattr(model, 'art_est')): 
            raise RuntimeError('Cannot add demonstration: missing lfvd variables')

        # Add features per label in demonstration
        self.log.info('Adding demonstration %s' % name)

        img, gray = frames[init_tidx].getRGB(), frames[init_tidx].getGray()
        for l,cinds in model.clusters.cluster_inds.iteritems(): 
            pts2d, pts3d, desc = self.describe_label(frames[init_tidx], 
                                                     pts2d=model.features.xy[cinds,init_tidx])
            data = {
                'desc':desc, 'pts2d':pts2d, 'pts3d':pts3d, 
                'img':frames[init_tidx].getRGB(),
                'name':name, 'model': model
            }
            self.obj_loc.add_object(data)

    def query(self, frame): 
        if not self.db_built: 
            # Build db temporarily
            self.log.info('Building DB ... ')
            self.obj_loc.build_db()
            self.db_built = True

        pts2d, pts3d, desc = frame_desc(frame)
        data = { 'desc':desc, 'pts2d':pts2d, 'pts3d':pts3d, 'img':frame.getRGB()}
        detected_object = self.obj_loc.query_scene(data)
        return detected_object


class ArticulationPredictionfD: 
    """
    Estimate the surfaces involved in change detection 
    for a kinect point cloud
    
    IMPLEMENTATION TODOs: 
       - Make independent to marshalling framework
       - Learn frame, Predict frame, Learned pose list
    """
    def __init__(self, learn_frame, learned_pose_list, visualize=False): 
        assert(learned_pose_list is not None)
        self.learn_frame_ = learn_frame
        self.learned_pose_list_ = learned_pose_list
        rospy.loginfo('Initializing prediction engine. Waiting for RGB-D data')

    def predict(self, predict_frame, visualize=True):
        """
        Main process call for prediction
        """

        # Sets up initial frames
        self.frames_map = {'learn': self.learn_frame_, 'predict': predict_frame }
        self.masks_map = {'learn': None, 'predict': np.array([])};

        # Sets up viewpoints, and pretty names
        self.viewpoint_matcher = BetweenImagePoseEstimator(self.frames_map, 
                                                           masks_map=self.masks_map, 
                                                           visualize=visualize)
        self.viewpoints_map = self.viewpoint_matcher.get_viewpoints()

        # # Validate viewpoints
        # draw_utils.draw_cameras('ARTICULTION_PREDICTION_CAMS', 
        #                         [pose.inverse() for pose in self.viewpoints_map.values()], 
        #                         texts=self.viewpoints_map.keys(), 
        #                         c=['g' if k == 'predict' else 'y'
        #                            for k in self.viewpoints_map.keys()],
        #                         sensor_tf='KINECT')

        # Predict from the learned model
        rt12 = self.predict_motion('learn', 'predict')
        if np.linalg.norm(rt12.tvec) > 0.0: 
            predicted_pose_list = [rt12 * pose for (ut,pose) in self.learned_pose_list_]
            draw_utils.publish_pose_list('predicted_poses', predicted_pose_list, stamp=rospy.Time.now(), 
                                     frame_id='head_mount_kinect_rgb_optical_frame')
            return predicted_pose_list
        else: 
            return []

    def predict_motion(self, learned_name, predict_name): 
        # print '===================================='

        # Get the relative pose between prediction, 
        # and the learned manifold

        # print '==> Predicting model for ', predict_name, 'from', learned_name
        rt12 = self.viewpoint_matcher.get_relative_cloud_pose(learned_name, predict_name)
        # print '==> Relative tf between', predict_name, 'and', learned_name, rt12

        # # Debug visually
        # l_cloud, p_cloud = self.frames_map[learned_name].getCloud(), \
        #                    self.frames_map[predict_name].getCloud()

        # # Transform cloud to local frame
        # l_cloud = remove_nans(l_cloud[::6,::6].reshape((-1,3)))
        # p_cloud = remove_nans(p_cloud[::6,::6].reshape((-1,3)))

        # # Debug the transform
        # draw_utils.publish_point_cloud('LEARNED CLOUD', l_cloud, 
        #                                c='r', sensor_tf='KINECT')
        # draw_utils.publish_point_cloud('PREDICTED CLOUD', p_cloud, 
        #                                c='b', sensor_tf='KINECT')
        # draw_utils.publish_point_cloud('CORRECTED (PREDICTED) CLOUD', rt12 * l_cloud, 
        #                                c='g', sensor_tf='KINECT')

        # # Publish camera view for overlay
        # # publish_image_t('KINECT_IMAGE', self.frames_map[predict_name].getRGB())
        # cloud, bgr = self.frames_map[predict_name].getCloud(), \
        #              self.frames_map[predict_name].getRGB()
        # rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        # draw_utils.publish_point_cloud('PREDICTED_KINECT_CLOUD', 
        #                               self.frames_map[predict_name].getCloud(), 
        #                                c=rgb * 1.0 / 255)


        return rt12

        # # PLot predicted poses
        # predicted_poses = []
        # for manifolds in demo_manifold.manifold_map.values(): 
        #     predicted_poses.extend([rt12 * pose for pose in manifolds.all_poses.values()])

        # draw_utils.publish_pose_list2('PREDICTED_POSES', 
        #                               predicted_poses, sensor_tf='KINECT')




        # # Predict for predict_fn, learn from learn_fns
        # # First construct name mapping
        # self.name_to_fn_map = dict([(name, fn) 
        #                             for name, fn in name_to_fn_map.iteritems()
        #                             if ((name in predict_name) or (name in learn_name))])

        # print self.name_to_fn_map
        # self.fn_to_name_map = dict([(v,k) for k,v in name_to_fn_map.iteritems()])

        # # Check if logs are available
        # assert(learn_name in self.name_to_fn_map)
        # assert(predict_name in self.name_to_fn_map)
                
        # # Store player
        # self.player = None

        # # Compute MSER features on the rgb image
        # self.mser = cv2.MSER(_min_area=100, _max_area=320*240, 
        #                      _max_evolution=200, _edge_blur_size=15)

        # # First learn the model
        # # Per label, get the motion manifold
        # learn_fn = self.name_to_fn_map[learn_name]
        # manifold_map = self.learn_manifold(learn_fn, learned_pose_list)

        # # Store the demonstration
        # DemonstrationManifold = namedtuple('DemonstrationManifold', 
        #                                    ['name', 'manifold_map'])
        # demo_manifold = DemonstrationManifold(name=learn_name, manifold_map=manifold_map)


        # # Predict from the learned model
        # # Use the demonstrations (with names), and their manifolds learned
        # predict_manifold_map = self.predict_manifold(predict_name, demo_manifold)

        # # Get hulls for all labels
        # all_hulls3d = [surf_pred.hulls3d 
        #                for label, surf_pred in surface_predictions.iteritems()]
        # print all_hulls3d.flatten()

        # # Plot 
        # self.plot_hulls('REGION_HULLS', all_hulls3d)


        # plot_utils.imshow(np.vstack(viz), pattern='bgr')

    # def setup_frames_map(self, name_to_fn): 
    #     # Set up frames for each log
    #     frames_map = OrderedDict()
    #     for name, fn in name_to_fn.iteritems(): 
    #         player = LCMLogPlayer(fn=fn, k_frames=10)
    #         frame = player.get_frame_with_percent(0.0)
    #         frame.computeNormals(0.5)
    #         frames_map[name] = frame
    #     return frames_map

    # def setup_viewpoint_matcher(self, frames_map, masks_map, visualize): 
    #     print '===================================='
    #     print 'Setup viewpoint matcher'
       
    #     # Construct matcher
    #     matcher = BetweenImagePoseEstimator(frames_map, masks_map, visualize)
    #     return matcher

    # def predict_manifold(self, predict_name, demo_manifold): 
    #     print '===================================='

    #     # Get the relative pose between predict_name, 
    #     # and the learned demo manifold

    #     learned_name = demo_manifold.name
    #     print '==> Predicting model for ', predict_name, 'from', learned_name
    #     rt12 = self.viewpoint_matcher.get_relative_cloud_pose(learned_name, predict_name)
    #     print '==> Relative tf between', predict_name, 'and', learned_name, rt12

    #     # Debug visually
    #     l_cloud, p_cloud = self.frames_map[learned_name].getCloud(), \
    #                        self.frames_map[predict_name].getCloud()

    #     # Transform cloud to local frame
    #     l_cloud = remove_nans(l_cloud[::6,::6].reshape((-1,3)))
    #     p_cloud = remove_nans(p_cloud[::6,::6].reshape((-1,3)))

    #     # Debug the transform
    #     draw_utils.publish_point_cloud('LEARNED CLOUD', l_cloud, 
    #                                    c='r', sensor_tf='KINECT')
    #     draw_utils.publish_point_cloud('PREDICTED CLOUD', p_cloud, 
    #                                    c='b', sensor_tf='KINECT')
    #     draw_utils.publish_point_cloud('CORRECTED (PREDICTED) CLOUD', rt12 * l_cloud, 
    #                                    c='g', sensor_tf='KINECT')

    #     # Plot learned hulls
    #     learned_hulls = []
    #     for manifolds in demo_manifold.manifold_map.values(): 
    #         learned_hulls.extend(self.predict_region_trajectory(manifolds))

    #     learned_hulls = map(lambda hull: np.vstack(hull), learned_hulls)
    #     draw_utils.publish_point_cloud('LEARNED_MANIFOLD', 
    #                                    [learned_hulls[0], learned_hulls[1]], 
    #                                    point_type='POLYGON', c='g', 
    #                                    sensor_tf='KINECT')

    #     learned_hulls_edges = map(lambda hull: 
    #                               np.hstack([hull[:-1], hull[1:]]).reshape((-1,3)), 
    #                               learned_hulls)
    #     draw_utils.publish_point_cloud('LEARNED_MANIFOLD-edges', 
    #                                    learned_hulls_edges, 
    #                                    point_type='LINES', c='#005c00', 
    #                                    sensor_tf='KINECT')


    #     # Plot predicted hulls
    #     predicted_hulls = [rt12 * hull for hull in learned_hulls]
    #     draw_utils.publish_point_cloud('PREDICTED_MANIFOLD', 
    #                                    [predicted_hulls[0], predicted_hulls[1]], 
    #                                    point_type='POLYGON', c='b', 
    #                                    sensor_tf='KINECT')

    #     predicted_hulls_edges = map(lambda hull: 
    #                                 np.hstack([hull[:-1], hull[1:]]).reshape((-1,3)), 
    #                                 predicted_hulls)
    #     draw_utils.publish_point_cloud('PREDICTED_MANIFOLD-edges', 
    #                                    predicted_hulls_edges, 
    #                                    point_type='LINES', c='#000b96', 
    #                                    sensor_tf='KINECT')

    #     # Publish camera view for overlay
    #     # publish_image_t('KINECT_IMAGE', self.frames_map[predict_name].getRGB())
    #     cloud, bgr = self.frames_map[predict_name].getCloud(), \
    #                  self.frames_map[predict_name].getRGB()
    #     rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    #     draw_utils.publish_point_cloud('PREDICTED_KINECT_CLOUD', 
    #                                   self.frames_map[predict_name].getCloud(), 
    #                                    c=rgb * 1.0 / 255)

    #     # PLot predicted poses
    #     predicted_poses = []
    #     for manifolds in demo_manifold.manifold_map.values(): 
    #         predicted_poses.extend([rt12 * pose for pose in manifolds.all_poses.values()])

    #     draw_utils.publish_pose_list2('PREDICTED_POSES', 
    #                                   predicted_poses, sensor_tf='KINECT')

    # def learn_manifold(self, fn, pose_list): 
    #     print '===================================='
    #     print 'Learning manifolds for ', fn

    #     # Prediction storage
    #     manifold_map = dict()

    #     # Setup Player
    #     self.player = LCMLogPlayer(fn)

    #     # For each label
    #     viz, viz_hulls = [], []
    #     for label, poses in pose_list: 
    #         # Don't do anything for ref. frame
    #         if label < 0: continue

    #         # Filter out non-sensical poses
    #         poses = filter(lambda (utime,pose): np.linalg.norm(pose.tvec) > 0 
    #                        and (not np.isnan(pose.tvec).any()), poses)

    #         # Checks for valid poses per label
    #         if not len(poses): 
    #             print 'No valid poses for label', label

    #         # Estimate the surface, for a corresponding utime/pose
    #         vis, manifold = self.estimate_region_trajectory(poses)
            
    #         # Propagate surface predictions for each label
    #         manifold_map[label] = manifold

    #         # print 'LABEL: %i, MANIFOLD: %s' % (label, manifold)
    #     return manifold_map

        

    # def valid_region(self, region, cloud, normals): 
    #     region_pts = contours_from_endpoints(region.reshape(-1,2),5)
    #     valid_pts = np.vstack(map(lambda pt: cloud[pt[1],pt[0]], region_pts))
    #     inds, = np.where(~np.isnan(valid_pts).any(axis=1))

    #     # Check 1: Invalid, If more than 10% are nans
    #     # print 'VALID PTS: ', len(inds) * 1.0 / len(valid_pts)
    #     if len(inds) * 1.0 / len(valid_pts) <= 0.9: 
    #         return False

    #     # # valid_pts = valid_pts[inds]

    #     # # Check 2: 
    #     # valid_normals = np.vstack(map(lambda pt: normals[pt[1],pt[0]], region_pts))
    #     # inds, = np.where(~np.isnan(valid_normals).any(axis=1))
    #     # print 'VALID_NORMALS: ', valid_normals

    #     return True

    # def mser_regions(self, pose_map, init_utime, final_utime): 
    #     # Get MSER features for farthest pose
    #     # Only return valid regions that encapsulate pose's projection

    #     # Project pose into the camera image 
    #     final_pose = pose_map[final_utime]
    #     pose_pts = KinectCamera().project(final_pose.tvec, RigidTransform.identity())
    #     # print 'POSE_PTS', pose_pts

    #     frame = self.player.get_frame_with_utime(final_utime)
    #     frame.computeNormals(1.0)
            
    #     # Get images
    #     gray = frame.getGray()
    #     vis = frame.getRGB()

    #     # Convert to 3 channel 8-bit image
    #     normals = ((frame.getNormals() + 1) * 128).astype(np.uint8)

    #     # Gaussian blur, and color conversion
    #     # lab = cv2.cvtColor(vis, cv2.COLOR_BGR2LAB)

    #     regions = self.mser.detect(normals, None)
    #     # regions_lab = self.mser.detect(lab, None)
    #     # regions = itertools.chain(regions_normals, regions_lab)

    #     # Convert to hulls
    #     hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]

    #     # Filter regions that enclose projected pt
    #     hulls = filter(lambda hull: 
    #                      cv2.pointPolygonTest(hull, 
    #                                           tuple(pose_pts.astype(int)), 
    #                                           measureDist=False) > 0, hulls)

    #     # Filter regions that have a valid surface normal
    #     hulls = filter(lambda hull: 
    #                      self.valid_region(hull, frame.getCloud(), frame.getNormals()), hulls)

    #     hulls = sorted(hulls, key=lambda hull: cv2.contourArea(hull), reverse=True)
    #     # print [cv2.contourArea(hull) for hull in hulls]

    #     if not len(hulls): 
    #         print 'WARNING: No regions found that meet criteria!'

    #     print 'MSER HULLS:', len(hulls)

    #     # Get hull contour
    #     cloud = frame.getCloud()
    #     hulls3d = [ np.vstack(map(lambda pt: cloud[pt[1],pt[0]], np.vstack(hull))) 
    #                 for hull in hulls ]


    #     mask = np.zeros(shape=gray.shape)
    #     print 'MASK: ', mask.shape

    #     cv2.polylines(vis, hulls, 1, (0, 255, 0))
    #     for hull in hulls: 
    #         # bug: multiply by 4
    #         cv2.fillConvexPoly(mask, hull, 255, cv2.CV_AA, shift=0)
    #     cv2.imshow('viz', vis)
    #     return vis, mask, hulls3d

    # def check_normals(self, pts, mu): 
    #     # spts = pts[np.linspace(0, len(pts), 3).astype(int), :]
    #     if (np.dot(np.cross(pts[1] - pts[0], pts[1] - pts[2]), mu - pts[1]) > 0): 
    #         return pts
    #     else: 
    #         return pts[::-1]
        
    # def extrude_hulls(self, hulls): 
    #     pts = []
    #     for hull in hulls: 
    #         ehull = list(hull)
    #         ehull.append(hull[0])
    #         pts.append(ehull)
    #     return pts

    # def extrude_hull_volume(self, hulls): 
    #     cap, base = np.vstack(hulls[::2]), np.vstack(hulls[1::2])
    #     last_idx, caplen = None, len(cap)
    #     mu = np.mean(np.vstack([cap,base]), axis=0)

    #     for cap, base in zip(hulls[:-1], hulls[1:]): 
    #         caplen = len(cap)
    #         # Face (base)
    #         # for _idx in range(0, caplen-3): 
    #         #     idx, idx1, idx2 = _idx % caplen, (_idx+1) % caplen, (_idx+2) % caplen
    #         #     pts.extend([base[0], base[idx], base[idx1]])

    #         bpts = []
    #         for _idx in range(0, caplen+1): 
    #             idx = _idx % caplen
    #             bpts.extend([base[idx]])
    #         # bpts = self.check_normals(bpts, mu)
    #         pts.append(np.vstack(bpts))

    #         # Walls
    #         # for _idx in range(caplen): 
    #         #     idx,idx1 = _idx, (_idx+1) % caplen
    #         #     pts.extend([cap[idx], base[idx], base[idx1]])
    #         #     pts.extend([base[idx1], cap[idx1], cap[idx]])
    #         #     last_idx = idx

    #         wpts = []
    #         for _idx in range(caplen): 
    #             idx,idx1 = _idx, (_idx+1) % caplen
    #             p = [cap[idx], base[idx], base[idx1], cap[idx1]]
    #             pts.append(np.vstack(p)) # self.check_normals(p, mu)))
            
    #         # Face (cap)
    #         # for _idx in range(last_idx, last_idx + caplen-3): 
    #         #     idx, idx1, idx2 = _idx % caplen, (_idx+1) % caplen, (_idx+2) % caplen
    #         #     pts.extend([cap[last_idx], cap[idx1], cap[idx2]])
                
    #         cpts = []
    #         for _idx in range(0, caplen+1): 
    #             idx = _idx % caplen
    #             cpts.extend([cap[idx]])
    #         # cpts = self.check_normals(cpts, mu)
    #         pts.append(cpts)

    #     return pts

    # def predict_region_trajectory(self, surface): 

    #     # Create pose map
    #     pose_map = surface.all_poses
    #     final_utime, hulls3d = surface.final_utime, surface.hulls3d

    #     assert(final_utime in pose_map)

    #     # Setup the log
    #     frame = self.player.get_frame_with_utime(final_utime)
    #     h, w, ch = frame.getRGB().shape
    #     cloud_obs = frame.getCloud()[::self.ds,::self.ds]

    #     # First pose, utime
    #     init_utime = min(pose_map.keys())

    #     # Get pose observation to predict
    #     pose_obs = pose_map[final_utime]

    #     # For each of the utimes, compute the lsq. err between the
    #     # predicted cloud and actual cloud
    #     viz_all_hulls, viz_end_hulls = defaultdict(list), defaultdict(list)
    #     for utime_pred, pose_pred in pose_map.iteritems(): 
    #         pose_pTo = pose_pred * pose_obs.inverse()

    #         for hidx, hull3d in enumerate(hulls3d): 
    #             if hidx > 0: continue
    #             # Pred using pose tf in 3d
    #             hull3d_pred = pose_pTo * hull3d
    #             viz_all_hulls[hidx].append(pose_pTo * hull3d)

    #             if utime_pred == init_utime or utime_pred == final_utime: 
    #                 viz_end_hulls[hidx].append(pose_pTo * hull3d)
    #                 viz_all_hulls[hidx].insert(0, pose_pTo * hull3d)

    #     # Return the extruded hulls
    #     return list(itertools.chain(*[self.extrude_hulls(hull_pair) 
    #                                   for hull_pair in viz_all_hulls.values()]))
                
    # def estimate_region_trajectory(self, poses): 

    #     # First pose, utime
    #     init_utime, init_pose = poses[0]

    #     # Find the farthest/final pose
    #     final_utime, final_pose = max(poses, key=lambda (utime,pose): \
    #                                   np.linalg.norm(pose.tvec-init_pose.tvec))
    #     print 'INIT: ', init_utime, init_pose
    #     print 'FINAL: ', final_utime, final_pose

    #     # Get regions for utime, and only return those that enclose pose
    #     # Create pose map
    #     pose_map = dict(poses)
    #     vis, mask, hulls3d = self.mser_regions(pose_map, init_utime, final_utime)

    #     # Store the prediction surface
    #     PredictionSurface = namedtuple('PredictSurface', ['init_utime', 'init_pose',
    #                                                       'final_utime', 'final_pose',
    #                                                       'all_poses', 
    #                                                       'mask', 'mask_utime',
    #                                                       'hulls3d'])

    #     return vis, PredictionSurface(init_utime=init_utime, 
    #                                   init_pose=init_pose, 
    #                                   final_utime=final_utime, 
    #                                   final_pose=final_pose, 
    #                                   all_poses=pose_map,
    #                                   mask_utime=final_utime, 
    #                                   mask=mask, 
    #                                   hulls3d=hulls3d) 



