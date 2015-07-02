#!/usr/bin/python 
# TODO: Compute relative pose mean instead of absolute mean pose

from __future__ import division

import numpy as np
np.set_printoptions(precision=3, suppress=True)

from collections import defaultdict, namedtuple
from rigid_transform import Quaternion, RigidTransform, tf_compose, tf_construct

import utils.draw_utils as draw_utils
import utils.plot_utils as plot_utils

from utils.db_utils import AttrDict, save_dict, load_dict
from utils.trackers.tracker_utils import AprilTagsFeatureTracker
from utils.pose_utils import mean_pose, mean_rpy, wrap_angle, wrap_angle2

import matplotlib as mpl
import matplotlib.pyplot as plt

class PoseEstimationEvalulation: 
    def __init__(self, tag_pose_map, eval_poses_map, names): 
        # Save evaluation info (tvec err, orient. err)
        self.eval_info = AttrDict()

        # Plot data
        data_map = dict()

        # Convert eval_poses to dict
        est_poses_map = {est_id: {est_label: dict(est_utime_poses) } 
                         for est_id, labeled_est_utime_poses in eval_poses_map.iteritems() 
                         for est_label, est_utime_poses in labeled_est_utime_poses }

        # # For each eval dataset, determine the overlapping utimes
        # all_utimes = set()
        # for tag_label, tag_utime_poses in tag_pose_map.iteritems(): 
        #     all_utimes = all_utimes.union(tag_utime_poses.keys())
        # # for est_id, labeled_est_utime_poses in eval_poses_map.iteritems(): 
        # #     for est_label, est_utime_poses in labeled_est_utime_poses: 
        # #         utimes = set(map(lambda (utime,pose): utime, est_utime_poses))
        # #         all_utimes = all_utimes.union(utimes)
        # all_utimes_idx = dict(zip(sorted(list(all_utimes)), np.arange(len(all_utimes))))

        # Visualize poses
        viz_poses = [pose for utime_poses in tag_pose_map.values() for (utime,pose) in utime_poses.iteritems()]
        draw_utils.publish_pose_list2('ARTAG_POSES', viz_poses, sensor_tf='KINECT')

        # For each tag label, find rel tvec err
        print '================================================='
        rem_poses = set([(est_id,label) for est_id,v in est_poses_map.iteritems() for label in v.keys()])
        rem_poses = filter(lambda (est_id,label): label >= 0, rem_poses)


        # Save tvec/ori errors
        tvec_errs, ori_errs = defaultdict(list), defaultdict(list)
        for tag_label, tag_utime_poses in tag_pose_map.iteritems(): 
            if tag_label == -1: continue
            if not len(rem_poses): break
            print '-------------------------------------------------'
            print 'Tag. Label: ', tag_label

            # Plot for each tag utime
            tag_utimes = tag_utime_poses.keys()
            tag_utimes_idx = dict(zip(tag_utimes, np.arange(len(tag_utimes))))

            # Plot relative to first
            init_tagp = None
            tag_utime_poses_rel = dict()
            for (utime,tagp) in tag_utime_poses.iteritems(): 
                if init_tagp is None: 
                    init_tagp = tagp
                tag_utime_poses_rel[utime] = init_tagp.inverse().oplus(tagp)
            self._add_aligned_data(data_map, 'Tag', tag_utime_poses_rel, index=tag_utimes_idx)

            # For each algorithm, find closest tag segment 
            for est_id, est_utime_poses_map in est_poses_map.iteritems(): 
                # print 'Est id: ', est_id
                # Eval label, and remove from list
                best = self.find_closest_segment(est_utime_poses_map, tag_utime_poses)
                print 'MATCH ID', best.label, 'ERR', best.tvec_err, best.ori_err
                tvec_errs[est_id].append(best.tvec_err), ori_errs[est_id].append(best.ori_err)

                # Add to plot # est_utime_poses_map[best.label]
                self._add_aligned_data(data_map, est_id, best.est_poses, index=tag_utimes_idx)
                
                # Remove from set
                rem_poses.remove((est_id,best.label))

        # Save tvecs, ori_err
        self.eval_info.tvec_errs = dict()
        for k,v in tvec_errs.iteritems(): 
            self.eval_info.tvec_errs[k] = np.mean(v)
        self.eval_info.ori_errs = dict()
        for k,v in ori_errs.iteritems(): 
            self.eval_info.ori_errs[k] = np.mean(v)

        # print self.eval_info

        # # Save and plot
        # fn = '/home/spillai/tmp/test_pose_eval.mat'
        # save_dict(fn, data_map)
        plot_keys = ['Tag']; plot_keys.extend(eval_poses_map.keys())
        plot_names = names; names['Tag'] = 'Tag';
        # PoseEstimationEvalulation.plot(fn, plot_keys, names)

        # Save plotting data
        self.eval_info.data_map = data_map
        self.eval_info.plot_keys = plot_keys
        self.eval_info.names = names        

    def _add_aligned_data(self, data_map, name, pose_map, index): 
        T = len(index)
        inds = map(lambda ut: index[ut] if ut in index else -1, pose_map.keys())
        inds = filter(lambda x: x >= 0, inds)
        poses = pose_map.values()

        tvecs = np.empty((T,3)) * np.nan
        tvecs[inds] = np.vstack(map(lambda x: x.tvec, poses))

        rpys = np.empty((T,3)) * np.nan
        rpys[inds] = np.vstack(map(lambda x: [x.quat.to_roll_pitch_yaw()], poses))

        data_map['Abs_%s_Roll' % name] = map(np.rad2deg, np.unwrap(rpys[:,0]))
        data_map['Abs_%s_Pitch' % name] = map(np.rad2deg, np.unwrap(rpys[:,1]))
        data_map['Abs_%s_Yaw' % name] = map(np.rad2deg, np.unwrap(rpys[:,2]))
        
        data_map['Abs_%s_X' % name] = tvecs[:,0]
        data_map['Abs_%s_Y' % name] = tvecs[:,1]
        data_map['Abs_%s_Z' % name] = tvecs[:,2]

        return

    def find_closest_segment(self, est_utime_poses_map, tag_utime_poses): 

        errs = []
        tag_utimes = tag_utime_poses.keys()
        for label, est_utime_poses in est_utime_poses_map.iteritems(): 
            est_utimes = est_utime_poses.keys()
            overlap = sorted(set(tag_utimes).intersection(set(est_utimes)))

            init_tagp, init_estp = None, None
            tag_poses, est_poses = dict(), dict()
            for utime in overlap: 
                tagp, estp = tag_utime_poses[utime], est_utime_poses[utime]

                if init_tagp is None: 
                    # Initialize tag, and estimated poses
                    init_tagp, init_estp = tagp, estp

                    # Estimate only rotation, fix translation
                    init_tagp0, init_estp0 = RigidTransform(init_tagp.quat, np.array([0,0,0])), \
                                             RigidTransform(init_estp.quat, np.array([0,0,0]))

                    # Relative rotation between tags detected, and estimated pose
                    init_rel = init_estp0.inverse() * init_tagp0

                    # Rotation of tag, and init est position
                    init_estp_t = RigidTransform(init_tagp.quat, init_estp.tvec); 

                tagp1 = init_tagp.inverse().oplus(tagp)
                estp1 = ((init_estp_t.inverse()).oplus(estp)).oplus(init_rel)
                tag_poses[utime] = tagp1
                est_poses[utime] = estp1

            # Tvec error metric for matching
            # tvec_err = np.array([np.linalg.norm(tag_utime_poses[utime].tvec - est_utime_poses[utime].tvec) 
            #                      for utime in overlap])
            # tvec_err -= tvec_err[0]
            tvec_err = np.array([np.linalg.norm(tag_poses[utime].tvec - est_poses[utime].tvec) for utime in overlap])
            ori_err = np.array([np.mean(np.rad2deg(np.fabs(
                (tag_poses[utime].inverse()).oplus(est_poses[utime]).quat.to_roll_pitch_yaw() ))) 
                                for utime in overlap])
            # print ori_err
            errs.append(AttrDict(label=label, utimes=overlap, 
                                 tvec_err=np.mean(tvec_err), ori_err=np.mean(ori_err), tag_poses=tag_poses, est_poses=est_poses))

        best = min(errs, key=lambda x: x.tvec_err)
        return best

    @staticmethod
    def plot(data_map, labels, names):
        # data_map = load_dict(filename)

        pparams = { 'linewidth':1, }
        font = { 'family': 'normal', 'weight': 'normal', 'size': 10 }
        mpl.rc('font', **font)

        # Figure 1 ==================================================
        xs = np.arange(len(data_map['Abs_Tag_X']))

        f = plt.figure(1)
        f.clf()
        f.subplots_adjust(hspace=0.55)
        ax = f.add_subplot(3, 2, 1, **{'xlabel':'Observations', 'ylabel': 'Position X(m)', 'xlim': [min(xs), max(xs)]})
        # d_err = np.fabs(np.array(data_map['Abs_Tag_X']) - 
        #                np.array(data_map['Abs_Est_X']))
        # max_err = np.max(d_err)
        for l in labels: 
            ax.plot(data_map['Abs_%s_X' % l], **pparams)
        # ax.fill_between(x=xs, 
        #                 y1=data_map['Abs_Est_X']-max_err, 
        #                 y2=data_map['Abs_Est_X']+max_err, 
        #                 alpha=0.5,
        #                 interpolate=True, facecolor='gray')
        mu = data_map['Abs_Tag_X'][0]
        off = max(max(data_map['Abs_Tag_X']) - min(data_map['Abs_Tag_X']) + 0.05, 0.15)
        ax.set_ylim([mu-off, mu+off])
        # ax.text(0.05, 0.05, 
        #         '(Avg./Max.) Error: %4.3f/%4.3f m'\
        #         % (np.mean(d_err), np.max(d_err)), transform=ax.transAxes)

        # ----------------------------------------------------------------
        ax = f.add_subplot(3, 2, 3)
        # d_err = np.fabs(np.array(data_map['Abs_Tag_Y']) - 
        #                 np.array(data_map['Abs_Est_Y']))
        # max_err = np.max(d_err)
        for l in labels: 
            ax.plot(data_map['Abs_%s_Y' % l], **pparams)
        # ax.fill_between(x=xs, 
        #                 y1=data_map['Abs_Est_Y']-max_err, 
        #                 y2=data_map['Abs_Est_Y']+max_err, alpha=0.25, interpolate=True, facecolor='gray')
        ax.set_xlim([min(xs), max(xs)])
        mu = data_map['Abs_Tag_Y'][0]
        off = max(max(data_map['Abs_Tag_Y']) - min(data_map['Abs_Tag_Y']) + 0.05, 0.15)
        ax.set_ylim([mu-off, mu+off])
        ax.set_xlabel('Observations')
        ax.set_ylabel('Position Y (m)')
        # ax.text(0.05, 0.05, 
        #         'Avg. error: %4.3f m\n'\
        #         'Max. error: %4.3f m'\
        #         % (np.mean(d_err), np.max(d_err)), transform=ax.transAxes)

        # ----------------------------------------------------------------
        ax = f.add_subplot(3, 2, 5)
        # d_err = np.max(np.fabs(np.array(data_map['Abs_Tag_Z']) - 
        #                      np.array(data_map['Abs_Est_Z'])))
        # max_err = np.max(d_err)
        for l in labels: 
            ax.plot(data_map['Abs_%s_Z' % l ], **pparams)
        # ax.fill_between(x=xs, 
        #                 y1=data_map['Abs_Est_Z']-max_err, 
        #                 y2=data_map['Abs_Est_Z']+max_err, alpha=0.25, interpolate=True, facecolor='gray')
        ax.set_xlim([min(xs), max(xs)])
        mu = data_map['Abs_Tag_Z'][0]
        off = max(max(data_map['Abs_Tag_Z']) - min(data_map['Abs_Tag_Z']) + 0.05, 0.15)
        ax.set_ylim([mu-off, mu+off])
        ax.set_xlabel('Observations')
        ax.set_ylabel('Position Z (m)')
        # ax.text(0.05, 0.05, 
        #         'Avg. error: %4.3f m\n'\
        #         'Max. error: %4.3f m'\
        #         % (np.mean(d_err), np.max(d_err)), transform=ax.transAxes)

        # ----------------------------------------------------------------
        ax = f.add_subplot(3, 2, 2)
        # d_err = np.fabs(np.array(data_map['Abs_Tag_Roll']) - 
        #                 np.array(data_map['Abs_Est_Roll']))
        # max_err = np.max(d_err)
        for l in labels: 
            ax.plot(data_map['Abs_%s_Roll' % l ], label=names[l], **pparams)
        # ax.fill_between(x=xs, 
        #                 y1=data_map['Abs_Est_Roll']-max_err, 
        #                 y2=data_map['Abs_Est_Roll']+max_err, alpha=0.25, interpolate=True, facecolor='gray')
        ax.legend(loc='upper right', fancybox=False, ncol=3)
        ax.set_xlim([min(xs), max(xs)])
        mu = data_map['Abs_Tag_Roll'][0]
        off = max(max(data_map['Abs_Tag_Roll']) - min(data_map['Abs_Tag_Roll']) + 10, 40)
        ax.set_ylim([mu-off, mu+off])
        ax.set_xlabel('Observations')
        ax.set_ylabel('Roll (deg)')
        # ax.text(0.05, 0.05, 
        #         'Avg. error: %4.3f deg\n'\
        #         'Max. error: %4.3f deg'\
        #         % (np.mean(d_err), np.max(d_err)), transform=ax.transAxes)

        # ----------------------------------------------------------------
        ax = f.add_subplot(3, 2, 4)
        # d_err = np.fabs(np.array(data_map['Abs_Tag_Pitch']) - 
        #                 np.array(data_map['Abs_Est_Pitch']))
        # max_err = np.max(d_err)
        for l in labels: 
            ax.plot(data_map['Abs_%s_Pitch' % l], **pparams)
        # ax.fill_between(x=xs, 
        #                 y1=data_map['Abs_Est_Pitch']-max_err, 
        #                 y2=data_map['Abs_Est_Pitch']+max_err, alpha=0.25, interpolate=True, facecolor='gray')
        ax.set_xlim([min(xs), max(xs)])
        mu = data_map['Abs_Tag_Pitch'][0]
        off = max(max(data_map['Abs_Tag_Pitch']) - min(data_map['Abs_Tag_Pitch']) + 10, 40)
        ax.set_ylim([mu-off, mu+off])
        ax.set_xlabel('Observations')
        ax.set_ylabel('Pitch (deg)')
        # ax.text(0.05, 0.05, 
        #         'Avg. error: %4.3f deg\n'\
        #         'Max. error: %4.3f deg'\
        #         % (np.mean(d_err), np.max(d_err)), transform=ax.transAxes)

        # ----------------------------------------------------------------
        ax = f.add_subplot(3, 2, 6)
        # d_err = np.fabs(np.array(data_map['Abs_Tag_Yaw']) - 
        #                 np.array(data_map['Abs_Est_Yaw']))
        # max_err = np.max(d_err)
        for l in labels: 
            ax.plot(data_map['Abs_%s_Yaw' % l], **pparams)
        # ax.fill_between(x=xs, 
        #                 y1=data_map['Abs_Est_Yaw']-max_err, 
        #                 y2=data_map['Abs_Est_Yaw']+max_err, alpha=0.25, interpolate=True, facecolor='gray')
        ax.set_xlim([min(xs), max(xs)])
        mu = data_map['Abs_Tag_Yaw'][0]
        off = max(max(data_map['Abs_Tag_Yaw']) - min(data_map['Abs_Tag_Yaw']) + 10, 40)
        ax.set_ylim([mu-off, mu+off])
        ax.set_xlabel('Observations')
        ax.set_ylabel('Yaw (deg)')
        # ax.text(0.05, 0.05, 
        #         'Avg. error: %4.3f deg\n'\
        #         'Max. error: %4.3f deg'\
        #         % (np.mean(d_err), np.max(d_err)), transform=ax.transAxes)

        # ----------------------------------------------------------------
        plt.tight_layout()
        plt.show()
        # ----------------------------------------------------------------
        # f.savefig('absolute-pose-comparison.png')
        # ax.set_title('Position (m) compared against April Tags')
        

if __name__ == "__main__": 
    PoseEstimationEvalulation.plot('/home/spillai/tmp/test_pose_eval.mat')

