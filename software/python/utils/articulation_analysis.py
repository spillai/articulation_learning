#!/usr/bin/python 
# TODO: Compute relative pose mean instead of absolute mean pose
# 
# 1. Pick pose that is farthest from original pose
# 2. Compute MSER features, and extract regions that 
# may correspond to appropriate region


import cv2, copy
import numpy as np
np.set_printoptions(precision=3, suppress=True)

import itertools, logging
from collections import defaultdict, namedtuple, OrderedDict

from articulation import pose_msg_t, track_msg_t, \
    track_list_msg_t, articulated_object_msg_t

from utils.db_utils import AttrDict

import utils.plot_utils as plot_utils
import utils.draw_utils as draw_utils
import utils.imshow_utils as im_utils

from utils.camera_utils import KinectCamera
from utils.logplayer_utils import LCMLogPlayer
from utils.correspondence_estimation import BetweenImagePoseEstimator, remove_nans

from rigid_transform import Quaternion, RigidTransform, \
    tf_compose, tf_construct

from fs_articulation import ArticulationLearner
from fs_utils import LCMLogReader, publish_image_t
np.set_printoptions(precision=2, suppress=True, threshold='nan', linewidth=160)


class ArticulationAnalysis: 
    def __init__(self, pose_list, params, viz_attrib='pose_projected'): 
        self.log = logging.getLogger(self.__class__.__name__)

        self.aobj = None
        self.params = params
        self.params.num_samples = getattr(self.params, 'num_samples', None)

        # Keep frame pose
        self.viz_attrib = viz_attrib

        # Get request message for articulation learner
        req = self.publish_poses(pose_list)

        # Get label mapping from articulation learner
        self.label_map = dict(enumerate(map(lambda (l,uposes): l, pose_list)))
        self.log.debug('ALEARNER LABEL_MAP: %s\n' % self.label_map)

        # Articulation fitting given observations
        alearner = ArticulationLearner(msg=req, filters='prismatic rotational')

        self.log.debug('Articulation Learning Fit: MLESAC [%s], return_tree[%s]' % (self.params.mlesac, self.params.return_tree))
        ret = alearner.fit(optimize=self.params.mlesac, return_tree=self.params.return_tree)
        self.log.debug('Done fitting!')
        self.aobj = articulated_object_msg_t.decode(ret)

        # Save articulation eval info
        self.aobj_info = self.evaluation_info(self.aobj)
        # self.log.debug('==========================================')
        # self.log.debug('%s' % self.aobj_info.model)

        # Print stats for saving
        self.eval_info = self.eval_stats(self.aobj_info)

        # # Viz articulation
        # self.viz_articulation()

    def eval_stats(self, info): 
        
        pp = None
        viz_rots = []
        req = ['avg_error_position', 'avg_error_orientation', 'outlier_ratio', 'complexity', 'dofs']
        self.log.debug('================================================')
        for model in info.model.values(): 
            self.log.debug('Model: %i->%i Type: %s' % (model.from_id, model.to_id, model.name))

            pp = AttrDict()
            for name, value in model.iteritems(): 
                if name in req: 
                    pp[str(name)] = value
            self.log.debug(' | '.join(['%s: %4.3f' % (k,v) for k,v in pp.iteritems()]))

            if model['name'] == 'rotational': 
                rt = RigidTransform(Quaternion.from_xyzw(np.array([model.rot_axis.x, model.rot_axis.y, \
                                                                   model.rot_axis.z, model.rot_axis.w])), 
                                    np.array([model.rot_center.x, model.rot_center.y, model.rot_center.z])
                )
                viz_rots.append(rt)
                pp.rot_axis = rt.to_homogeneous_matrix()[:3,2] # z-axis
                pp.rot_center = rt.to_homogeneous_matrix()[:3,3]

            elif model['name'] == 'prismatic': 
                pp.prismatic_ori = Quaternion.from_xyzw(
                    np.array([model.rigid_orientation.x, model.rigid_orientation.y, model.rigid_orientation.z, 
                              model.rigid_orientation.w])).to_homogeneous_matrix()[:3,2]
                pp.prismatic_dir = np.array([model.prismatic_dir.x, model.prismatic_dir.y, model.prismatic_dir.z])

            elif model['name'] == 'rigid': 
                pass

            # self.log.debug('Pos err: %4.3f | Ori. err: %4.3f' % (pp.avg_error_position, pp.avg_error_orientation))
            if model.name == 'prismatic': 
                self.log.debug('Prismatic params: Ori: %s Dir: %s' % (pp.prismatic_ori, pp.prismatic_dir))
            elif model.name == 'rotational': 
                self.log.debug('Rotational params: Axis: %s Center: %s' % (pp.rot_axis, pp.rot_center))

            # HACK! At most 1 model for evaluation purposes (otherwise average out errors)
            break
        self.log.debug('================================================')
        draw_utils.publish_pose_list2('ROTATIONAL_AXIS', viz_rots, sensor_tf='KINECT')

        return pp

    def viz_articulation(self): 
        viz_poses = []

        if self.params.num_samples is None: 
            self.log.debug('VIZ PROJECTED POSES')
            projected_poses = self.get_poses(self.viz_attrib)
            for model_id, utime_poses in projected_poses: 
                viz_poses.extend([pose for utime,pose in utime_poses])            
        else: 
            self.log.debug('VIZ SAMPLED POSES')
            sampled_poses = self.get_sampled_poses(self.params.num_samples)
            for model_id, utime_poses in sampled_poses: 
                viz_poses.extend([pose for utime,pose in utime_poses])            
                
        draw_utils.publish_pose_list2('ARTICULATED_POSE_PROJECTED', 
                                              viz_poses, sensor_tf='KINECT')       

    def get_original_poses(self): 
        return self.get_poses(attrib='pose')

    def get_projected_poses(self): 
        return self.get_poses(attrib='pose_projected')

    def get_sampled_poses(self, num_samples=100): 
        return self.get_poses(attrib='pose_resampled')

    def get_poses(self, attrib='pose'): # pose_resampled, pose_projected
        assert(self.aobj is not None)
        pose_list = []
        for model in self.aobj.models: 

            # Find corr. stamp values
            chan_map = dict([(chan.name, chan.values) for chan in model.track.channels])

            # Construct projected poses with appropriate utimes
            utime_poses = [ (int(utime), 
                             RigidTransform(Quaternion.from_wxyz(pose.orientation), pose.pos)) 
                            for utime, pose in zip(chan_map['stamp'], 
                                                   getattr(model.track, attrib)) ]

            # Construct projected poses b/w from_id, and to_id with appropriate utimes
            from_id, to_id = self.label_map[model.id / self.aobj.num_parts], \
                             self.label_map[model.id % self.aobj.num_parts]
            pose_list.append(((from_id, to_id), utime_poses))

            # print 'POSE LIST', pose_list
        return pose_list

    def project_poses(self, poses): 
        for track_id, params in self.aobj_info.model.iteritems(): 
            rt = RigidTransform(Quaternion.from_xyzw(params.rot_axis.x, params.rot_axis.y, 
                                                     params.rot_axis.z, params.rot_axis.w), 
                                np.array(params.rot_center.x, params.rot_center.y, params.rot_center.z))
        self.aobj_info = self.evaluation_info(self.aobj)


    def parse_params(self, params): 
        d = AttrDict()
        for param in params: 
            k, v = param.name, param.value
            keys = k.split('.')
            if len(keys) == 2: 
                if keys[0] not in d: 
                    d[keys[0]] = AttrDict()
                d[keys[0]][keys[1]] = v
            elif len(keys) == 1: 
                d[keys[0]] = v
            else: 
                raise RuntimeError('undefined')
        return d

    def sequential_check(self, info, cand_model): 
        if cand_model.model_id in info.model: 
            return False

        if cand_model.outlier_ratio <= self.params.outlier_ratio: 
            return True
        else: 
            return False

    def evaluation_info(self, obj): 
        info = AttrDict();
        info.full_name = ''
        info.model = AttrDict()
        for model in obj.models: 
            info.full_name = ''.join([info.full_name, '-', model.name])
            self.log.debug('-------------------------------')
            from_id, to_id = self.label_map[model.id / obj.num_parts], \
                             self.label_map[model.id % obj.num_parts]
            self.log.debug('MODEL %i->%i %s ID: %i' % (from_id, to_id, model.name, model.track.id))

            # Assume correct model inference
            if self.params.return_tree: 
                info.model[model.id] = self.parse_params(model.params)
                info.model[model.id].name = model.name
                info.model[model.id].model_id = model.id
                info.model[model.id].from_id = from_id
                info.model[model.id].to_id = to_id
            else: 
                cand_model = self.parse_params(model.params)
                cand_model.name = model.name
                cand_model.model_id = model.id
                cand_model.from_id = from_id
                cand_model.to_id = to_id

                # Make sure model hasn't been added, and outlier ratio is <= 0.2 (inlier ratio >= 0.8)
                if cand_model.model_id not in info.model and \
                   cand_model.outlier_ratio <= 0.5: 
                    info.model[model.id] = cand_model
                    # self.log.debug(info.model[model.id])

        info.params = AttrDict()
        info.num_parts = obj.num_parts
        for param in obj.params: 
            info.params[param.name] = param.value

        return info

    def publish_poses(self, pose_list): 
        # Dict of poses, each given unique ID
        track_list_msg = track_list_msg_t()

        for pose_tuple in pose_list:
            pid,poses = pose_tuple
            track_msg = track_msg_t(); 
            track_msg.id = pid;
            track_msg.pose = []
            
            for utime,pose in poses: 
                pose_msg = pose_msg_t()
                pose_msg.utime = utime;
                pose_msg.id = pid;
                pose_msg.pos = pose.tvec.tolist();
                pose_msg.orientation = pose.quat.to_wxyz()  # [x,y,z,w]
                track_msg.pose.append(pose_msg)

            track_msg.pose_flags = [track_msg_t.POSE_VISIBLE for p in track_msg.pose]
            track_msg.num_poses = len(track_msg.pose)

            track_msg.pose_projected = copy.deepcopy(track_msg.pose)
            track_msg.num_poses_projected = len(track_msg.pose);

            track_msg.pose_resampled = copy.deepcopy(track_msg.pose)
            track_msg.num_poses_resampled = len(track_msg.pose);

            track_msg.channels = None
            track_msg.num_channels = 0;

            track_list_msg.tracks.append(track_msg)
            self.log.debug('Published track for %i, with %i poses' % \
                (pid, track_msg.num_poses))
       
        track_list_msg.num_tracks = len(track_list_msg.tracks)
        # lc.publish("ARTICULATION_OBJECT_TRACKS", track_list_msg.encode())
        return track_list_msg.encode()

def contours_from_endpoints(endpts, quantize=4): # quantize in pixels
    v = np.array(np.roll(endpts, 1, axis=0) - endpts, np.float32);
    vnorm = np.sqrt(np.sum(np.square(v), axis=1))
    v[:,0] = np.multiply(v[:,0],1.0/vnorm)
    v[:,1] = np.multiply(v[:,1],1.0/vnorm)

    # print vnorm, np.arange(0, vnorm[0], 4)
    
    out = []
    for j in range(len(v)):
        out.append(np.vstack([endpts[j] + mag * v[j] 
                              for mag in np.arange(0,vnorm[j],quantize)]))
    return np.vstack(out).astype(int)


class ArticulationPrediction: 
    """
    Estimate the surfaces involved in change detection 
    for a kinect point cloud
    
    IMPLEMENTATION TODOs: 
       - Make independent to marshalling framework
       - Learn frame, Predict frame, Learned pose list
    """
    def __init__(self, learn_frame, predict_frame, learned_pose_list, 
                 downsample=4, k_frames=10, visualize=False): 
        assert(learned_pose_list is not None and k_frames is not None)
        self.ds = downsample

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

        # Compute MSER features on the rgb image
        self.mser = cv2.MSER(_min_area=100, _max_area=320*240, 
                             _max_evolution=200, _edge_blur_size=15)

        # Sets up initial frames
        self.frames_map = {'learn': learn_frame, 'predict': predict_frame }
        # self.setup_frames_map(self.name_to_fn_map)

        # First learn the model
        # Per label, get the motion manifold
        learn_fn = self.name_to_fn_map[learn_name]
        manifold_map = self.learn_manifold(learn_fn, learned_pose_list)

        # Store the demonstration
        DemonstrationManifold = namedtuple('DemonstrationManifold', 
                                           ['name', 'manifold_map'])
        demo_manifold = DemonstrationManifold(name=learn_name, manifold_map=manifold_map)

        # Sets up viewpoints, and pretty names
        # Determine relative poses between logs / viewpoints
        masks_map = {learn_name: None, predict_name: np.array([])};
        for manifold in manifold_map.values(): 
            if masks_map[learn_name] is not None: 
                masks_map[learn_name] = np.bitwise_or(masks_map[learn_name], manifold.mask)
            else: 
                masks_map[learn_name] = manifold.mask
        self.viewpoint_matcher = self.setup_viewpoint_matcher(self.frames_map, 
                                                              masks_map=masks_map, 
                                                              visualize=visualize)
        viewpoints_map = self.viewpoint_matcher.get_viewpoints()

        # Validate viewpoints
        draw_utils.draw_cameras('ARTICULTION_PREDICTION_CAMS', 
                                [pose.inverse() for pose in viewpoints_map.values()], 
                                texts=viewpoints_map.keys(), 
                                c=['g' if k == predict_name else 'y'
                                   for k in viewpoints_map.keys()],
                                sensor_tf='KINECT')

        # Predict from the learned model
        # Use the demonstrations (with names), and their manifolds learned
        predict_manifold_map = self.predict_manifold(predict_name, demo_manifold)

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

    def setup_viewpoint_matcher(self, frames_map, masks_map, visualize): 
        print '===================================='
        print 'Setup viewpoint matcher'
       
        # Construct matcher
        matcher = BetweenImagePoseEstimator(frames_map, masks_map, visualize)
        return matcher

    def predict_manifold(self, predict_name, demo_manifold): 
        print '===================================='

        # Get the relative pose between predict_name, 
        # and the learned demo manifold

        learned_name = demo_manifold.name
        print '==> Predicting model for ', predict_name, 'from', learned_name
        rt12 = self.viewpoint_matcher.get_relative_cloud_pose(learned_name, predict_name)
        print '==> Relative tf between', predict_name, 'and', learned_name, rt12

        # Debug visually
        l_cloud, p_cloud = self.frames_map[learned_name].getCloud(), \
                           self.frames_map[predict_name].getCloud()

        # Transform cloud to local frame
        l_cloud = remove_nans(l_cloud[::6,::6].reshape((-1,3)))
        p_cloud = remove_nans(p_cloud[::6,::6].reshape((-1,3)))

        # Debug the transform
        draw_utils.publish_point_cloud('LEARNED CLOUD', l_cloud, 
                                       c='r', sensor_tf='KINECT')
        draw_utils.publish_point_cloud('PREDICTED CLOUD', p_cloud, 
                                       c='b', sensor_tf='KINECT')
        draw_utils.publish_point_cloud('CORRECTED (PREDICTED) CLOUD', rt12 * l_cloud, 
                                       c='g', sensor_tf='KINECT')

        # Plot learned hulls
        learned_hulls = []
        for manifolds in demo_manifold.manifold_map.values(): 
            learned_hulls.extend(self.predict_region_trajectory(manifolds))

        learned_hulls = map(lambda hull: np.vstack(hull), learned_hulls)
        draw_utils.publish_point_cloud('LEARNED_MANIFOLD', 
                                       [learned_hulls[0], learned_hulls[1]], 
                                       point_type='POLYGON', c='g', 
                                       sensor_tf='KINECT')

        learned_hulls_edges = map(lambda hull: 
                                  np.hstack([hull[:-1], hull[1:]]).reshape((-1,3)), 
                                  learned_hulls)
        draw_utils.publish_point_cloud('LEARNED_MANIFOLD-edges', 
                                       learned_hulls_edges, 
                                       point_type='LINES', c='#005c00', 
                                       sensor_tf='KINECT')


        # Plot predicted hulls
        predicted_hulls = [rt12 * hull for hull in learned_hulls]
        draw_utils.publish_point_cloud('PREDICTED_MANIFOLD', 
                                       [predicted_hulls[0], predicted_hulls[1]], 
                                       point_type='POLYGON', c='b', 
                                       sensor_tf='KINECT')

        predicted_hulls_edges = map(lambda hull: 
                                    np.hstack([hull[:-1], hull[1:]]).reshape((-1,3)), 
                                    predicted_hulls)
        draw_utils.publish_point_cloud('PREDICTED_MANIFOLD-edges', 
                                       predicted_hulls_edges, 
                                       point_type='LINES', c='#000b96', 
                                       sensor_tf='KINECT')

        # Publish camera view for overlay
        # publish_image_t('KINECT_IMAGE', self.frames_map[predict_name].getRGB())
        cloud, bgr = self.frames_map[predict_name].getCloud(), \
                     self.frames_map[predict_name].getRGB()
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        draw_utils.publish_point_cloud('PREDICTED_KINECT_CLOUD', 
                                      self.frames_map[predict_name].getCloud(), 
                                       c=rgb * 1.0 / 255)

        # PLot predicted poses
        predicted_poses = []
        for manifolds in demo_manifold.manifold_map.values(): 
            predicted_poses.extend([rt12 * pose for pose in manifolds.all_poses.values()])

        draw_utils.publish_pose_list2('PREDICTED_POSES', 
                                      predicted_poses, sensor_tf='KINECT')

    def learn_manifold(self, fn, pose_list): 
        print '===================================='
        print 'Learning manifolds for ', fn

        # Prediction storage
        manifold_map = dict()

        # Setup Player
        self.player = LCMLogPlayer(fn)

        # For each label
        for label, poses in pose_list: 
            # Don't do anything for ref. frame
            if label < 0: continue

            # Filter out non-sensical poses
            poses = filter(lambda (utime,pose): np.linalg.norm(pose.tvec) > 0 
                           and (not np.isnan(pose.tvec).any()), poses)

            # Checks for valid poses per label
            if not len(poses): 
                print 'No valid poses for label', label

            # Estimate the surface, for a corresponding utime/pose
            vis, manifold = self.estimate_region_trajectory(poses)
            
            # Propagate surface predictions for each label
            manifold_map[label] = manifold

            # print 'LABEL: %i, MANIFOLD: %s' % (label, manifold)
        return manifold_map

        

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

    def mser_regions(self, pose_map, init_utime, final_utime): 
        # Get MSER features for farthest pose
        # Only return valid regions that encapsulate pose's projection

        # Project pose into the camera image 
        final_pose = pose_map[final_utime]
        pose_pts = KinectCamera().project(final_pose.tvec, RigidTransform.identity())
        # print 'POSE_PTS', pose_pts

        frame = self.player.get_frame_with_utime(final_utime)
        frame.computeNormals(1.0)
            
        # Get images
        gray = frame.getGray()
        vis = frame.getRGB()

        # Convert to 3 channel 8-bit image
        normals = ((frame.getNormals() + 1) * 128).astype(np.uint8)

        # Gaussian blur, and color conversion
        # lab = cv2.cvtColor(vis, cv2.COLOR_BGR2LAB)

        regions = self.mser.detect(normals, None)
        # regions_lab = self.mser.detect(lab, None)
        # regions = itertools.chain(regions_normals, regions_lab)

        # Convert to hulls
        hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]

        # Filter regions that enclose projected pt
        hulls = filter(lambda hull: 
                         cv2.pointPolygonTest(hull, 
                                              tuple(pose_pts.astype(int)), 
                                              measureDist=False) > 0, hulls)

        # Filter regions that have a valid surface normal
        hulls = filter(lambda hull: 
                         self.valid_region(hull, frame.getCloud(), frame.getNormals()), hulls)

        hulls = sorted(hulls, key=lambda hull: cv2.contourArea(hull), reverse=True)
        # print [cv2.contourArea(hull) for hull in hulls]

        if not len(hulls): 
            print 'WARNING: No regions found that meet criteria!'

        print 'MSER HULLS:', len(hulls)

        # Get hull contour
        cloud = frame.getCloud()
        hulls3d = [ np.vstack(map(lambda pt: cloud[pt[1],pt[0]], np.vstack(hull))) 
                    for hull in hulls ]


        mask = np.zeros(shape=gray.shape)
        print 'MASK: ', mask.shape

        cv2.polylines(vis, hulls, 1, (0, 255, 0))
        for hull in hulls: 
            cv2.fillConvexPoly(mask, hull, 255, cv2.CV_AA, shift=0)
        cv2.imshow('viz', vis)
        return vis, mask, hulls3d

    def check_normals(self, pts, mu): 
        # spts = pts[np.linspace(0, len(pts), 3).astype(int), :]
        if (np.dot(np.cross(pts[1] - pts[0], pts[1] - pts[2]), mu - pts[1]) > 0): 
            return pts
        else: 
            return pts[::-1]
        
    def extrude_hulls(self, hulls): 
        pts = []
        for hull in hulls: 
            ehull = list(hull)
            ehull.append(hull[0])
            pts.append(ehull)
        return pts

    def extrude_hull_volume(self, hulls): 
        cap, base = np.vstack(hulls[::2]), np.vstack(hulls[1::2])
        last_idx, caplen = None, len(cap)
        mu = np.mean(np.vstack([cap,base]), axis=0)

        for cap, base in zip(hulls[:-1], hulls[1:]): 
            caplen = len(cap)
            # Face (base)
            # for _idx in range(0, caplen-3): 
            #     idx, idx1, idx2 = _idx % caplen, (_idx+1) % caplen, (_idx+2) % caplen
            #     pts.extend([base[0], base[idx], base[idx1]])

            bpts = []
            for _idx in range(0, caplen+1): 
                idx = _idx % caplen
                bpts.extend([base[idx]])
            # bpts = self.check_normals(bpts, mu)
            pts.append(np.vstack(bpts))

            # Walls
            # for _idx in range(caplen): 
            #     idx,idx1 = _idx, (_idx+1) % caplen
            #     pts.extend([cap[idx], base[idx], base[idx1]])
            #     pts.extend([base[idx1], cap[idx1], cap[idx]])
            #     last_idx = idx

            wpts = []
            for _idx in range(caplen): 
                idx,idx1 = _idx, (_idx+1) % caplen
                p = [cap[idx], base[idx], base[idx1], cap[idx1]]
                pts.append(np.vstack(p)) # self.check_normals(p, mu)))
            
            # Face (cap)
            # for _idx in range(last_idx, last_idx + caplen-3): 
            #     idx, idx1, idx2 = _idx % caplen, (_idx+1) % caplen, (_idx+2) % caplen
            #     pts.extend([cap[last_idx], cap[idx1], cap[idx2]])
                
            cpts = []
            for _idx in range(0, caplen+1): 
                idx = _idx % caplen
                cpts.extend([cap[idx]])
            # cpts = self.check_normals(cpts, mu)
            pts.append(cpts)

        return pts

    def predict_region_trajectory(self, surface): 

        # Create pose map
        pose_map = surface.all_poses
        final_utime, hulls3d = surface.final_utime, surface.hulls3d

        assert(final_utime in pose_map)

        # Setup the log
        frame = self.player.get_frame_with_utime(final_utime)
        h, w, ch = frame.getRGB().shape
        cloud_obs = frame.getCloud()[::self.ds,::self.ds]

        # First pose, utime
        init_utime = min(pose_map.keys())

        # Get pose observation to predict
        pose_obs = pose_map[final_utime]

        # For each of the utimes, compute the lsq. err between the
        # predicted cloud and actual cloud
        viz_all_hulls, viz_end_hulls = defaultdict(list), defaultdict(list)
        for utime_pred, pose_pred in pose_map.iteritems(): 
            pose_pTo = pose_pred * pose_obs.inverse()

            for hidx, hull3d in enumerate(hulls3d): 
                if hidx > 0: continue
                # Pred using pose tf in 3d
                hull3d_pred = pose_pTo * hull3d
                viz_all_hulls[hidx].append(pose_pTo * hull3d)

                if utime_pred == init_utime or utime_pred == final_utime: 
                    viz_end_hulls[hidx].append(pose_pTo * hull3d)
                    viz_all_hulls[hidx].insert(0, pose_pTo * hull3d)

        # Return the extruded hulls
        return list(itertools.chain(*[self.extrude_hulls(hull_pair) 
                                      for hull_pair in viz_all_hulls.values()]))
                
    def estimate_region_trajectory(self, poses): 

        # First pose, utime
        init_utime, init_pose = poses[0]

        # Find the farthest/final pose
        final_utime, final_pose = max(poses, key=lambda (utime,pose): \
                                      np.linalg.norm(pose.tvec-init_pose.tvec))
        print 'INIT: ', init_utime, init_pose
        print 'FINAL: ', final_utime, final_pose

        # Get regions for utime, and only return those that enclose pose
        # Create pose map
        pose_map = dict(poses)
        vis, mask, hulls3d = self.mser_regions(pose_map, init_utime, final_utime)

        # Store the prediction surface
        PredictionSurface = namedtuple('PredictSurface', ['init_utime', 'init_pose',
                                                          'final_utime', 'final_pose',
                                                          'all_poses', 
                                                          'mask', 'mask_utime',
                                                          'hulls3d'])

        return vis, PredictionSurface(init_utime=init_utime, 
                                      init_pose=init_pose, 
                                      final_utime=final_utime, 
                                      final_pose=final_pose, 
                                      all_poses=pose_map,
                                      mask_utime=final_utime, 
                                      mask=mask, 
                                      hulls3d=hulls3d) 


    # def compute_change(self, clouds): 
    #     inliers = []

    #     frame_inds = np.arange(len(clouds))
    #     idx0, rest_inds = frame_inds[0], frame_inds[1:]
        
    #     for idx1 in rest_inds: 
    #         print idx0, idx1
    #         clouds0, clouds1 = clouds[idx0].reshape((-1,3)), clouds[idx1].reshape((-1,3))
    #         draw_utils.publish_point_cloud('CHANGE_CLOUDS0', clouds0, c='g')
    #         draw_utils.publish_point_cloud('CHANGE_CLOUDS1', clouds1, c='r')
    #         inds1 = change_detection(source=clouds0, target=clouds1, 
    #                                  resolution=0.15, return_inverse=False)
    #         inds0 = change_detection(source=clouds1, target=clouds0, 
    #                                  resolution=0.15, return_inverse=False)
    #         print 'INLIERS: ', len(inds1)
            
    #         inliers.append(clouds0[inds0])
    #         inliers.append(clouds1[inds1])

    #         draw_utils.publish_point_cloud('CHANGE_DETECTION', clouds1[inds1], c='b')

    #     draw_utils.publish_point_cloud('ALL_CHANGES', np.vstack(inliers), c='b')
