# # Articulation estimation notebook ============================================================
# 
# 1. Run: 
# gpft_tracking.py -l ~/data/2013-08-03-feature-artags-bench/lcmlog-2013-08-03.00
# 
# TODO: 
# a. Video?
# 
# Analysis: 
# Pose estimtion errors b/w apriltags and framework
# Model param estimation b/w ""
# Prediction estimation comparison
# Voxel estimation for manifold instantiation

import logging; 
logging.basicConfig(format='%(name)s :: %(message)s',level=logging.DEBUG)

import numpy as np
import cv2; # cv2.startWindowThread()
from collections import defaultdict

import utils.plot_utils as plot_utils
import utils.draw_utils as draw_utils
import utils.imshow_utils as im_utils
import utils.io_utils as io_utils

from utils.pose_estimation_eval import PoseEstimationEvalulation
from utils.db_utils import DictDB, AttrDict

from utils.logplayer_utils import LCMLogPlayer

if __name__ == "__main__": 

    #logdir1 = '/home/spillai/data/2013_09_03_artags_articulation/lcmlog-2013-09-03.'
    logdir2 = '/home/ragtz/data/2014_01_12_artags_articulation_more/lcmlog-2014-01-12.'
    logdir3 = '/home/ragtz/data/2014_01_26_rgb_rgbd_articulation/lcmlog-2014-01-26.'
    #logdir4 = '/home/spillai/data/2014_06_14_articulation_multibody/lcmlog-2014-06-14.'
    fn_map = {

        #'L1-DRAWER-00': logdir1+'00', 'L1-DRAWER-01': logdir1+'01', 'L1-DRAWER-02': logdir1+'02', 
        #'L1-CHAIR-03': logdir1+'03', 'L1-CHAIR-04': logdir1+'04', 
        #'L1-DOOR-05': logdir1+'05', 'L1-DOOR-06': logdir1+'06', 'L1-DOOR-07': logdir1+'07', 
        #'L1-FRIDGE-08': logdir1+'08', 'L1-FRIDGE-09': logdir1+'09', 
        #'L1-PROJECTOR-10': logdir1+'10', 'L1-PROJECTOR-11': logdir1+'11', 
        # ----------------------------------------------------------------------

        'L2-DOOR-00': logdir2+'00', 'L2-DOOR-01': logdir2+'01', # w/o tags
        # 'L2-DOOR-02': logdir2+'02', # snap
        'L2-DOOR-03': logdir2+'03', # w/o tags
        'L2-DOOR-04': logdir2+'04', 'L2-DOOR-05': logdir2+'05',

        'L2-CHAIR-06': logdir2+'06', 'L2-CHAIR-07': logdir2+'07', 'L2-CHAIR-08': logdir2+'08', 
        'L2-CHAIR-09': logdir2+'09', # w/o tags
        # 'L2-CHAIR-10': logdir2+'10', # snap

        'L2-DRAWER-11': logdir2+'11', 'L2-DRAWER-12': logdir2+'12', 'L2-DRAWER-13': logdir2+'13', # w/o tags
        # 'L2-DRAWER-14': logdir2+'14', # snap
        'L2-DRAWER-15': logdir2+'15', 'L2-DRAWER-16': logdir2+'16', 

        # 'L2-FRIDGE-22': logdir2+'22', # snap
        'L2-FRIDGE-21': logdir2+'21', 'L2-FRIDGE-23': logdir2+'23',  'L2-FRIDGE-24': logdir2+'24', 'L2-FRIDGE-25': logdir2+'25', 

        'L2-MICRO-28': logdir2+'28',  'L2-MICRO-29': logdir2+'29', 
        'L2-MICRO-30': logdir2+'30',  'L2-MICRO-31': logdir2+'31', # w/o tags

        'L2-DRAWER-38': logdir2+'38', 'L2-DRAWER-39': logdir2+'39', 

        'L2-PRINTER-33': logdir2+'33', 'L2-PRINTER-34': logdir2+'34', 'L2-FRIDGE-32': logdir2+'32', 'L2-FRIDGE-26': logdir2+'26', 

        'L3-DRAWER-01': logdir3+'01',  'L3-DRAWER-02': logdir3+'02', 
        # 'L3-DRAWER-03': logdir3+'03',  'L3-DRAWER-04': logdir3+'04', # snap
        # ----------------------------------------------------------------------

        'L3-LAPTOP-09': logdir3+'09', 'L3-LAPTOP-10': logdir3+'10', 'L3-LAPTOP-11': logdir3+'11', 
        'L3-LAPTOP-21': logdir3+'21', 'L3-LAPTOP-22': logdir3+'22', 'L3-LAPTOP-24': logdir3+'24', 

        # 'L3-DOORHANDLE-17': logdir3+'17', 'L3-DOORHANDLE-18': logdir3+'18', 
        # 'L3-MONITOR-07': logdir3+'07', 'L3-MONITOR-08': logdir3+'08', 

        # 'L3-CHAIR-15': logdir3+'15', 'L3-CHAIR-16': logdir3+'16',   
        # 'L3-BOOK-27': logdir3+'27', 'L3-BOOK-28': logdir3+'28', 
        # ----------------------------------------------------------------------

        #'L4-MONITOR-00': logdir4+'00', 'L4-MONITOR-01': logdir4+'01', 'L4-MONITOR-02': logdir4+'02', 
        #'L4-MONITOR-03': logdir4+'03', 'L4-MONITOR-04': logdir4+'04', 'L4-MONITOR-05': logdir4+'05', 

        #'L4-BICYCLE-06': logdir4+'06', 'L4-BICYCLE-07': logdir4+'07', 'L4-BICYCLE-08': logdir4+'08', 
        #'L4-BICYCLE-09': logdir4+'09', 'L4-BICYCLE-10': logdir4+'10', 'L4-BICYCLE-11': logdir4+'11', 
        #'L4-BICYCLE-12': logdir4+'12', 

        #'L4-CHAIR-13': logdir4+'13', 'L4-CHAIR-14': logdir4+'14', 'L4-CHAIR-15': logdir4+'15', 
    }

    from utils.al_lfd import ArticulationLfD, ArticulationPredictionfD, TagALfD
    from utils.al_katz import KatzALfD

    # Articulation learning object
    # ============================================================================
    # ----------------------------------------------------------------------------
    lfvd_params = AttrDict(
        tracker = AttrDict( FB_check=True, OF_method='lk', scales=4, err_th=1.0 ), 
        pose_estimation = AttrDict( with_normals=False, inlier_threshold=0.04, method='ransac' ), 
        feature_selection = AttrDict( min_feature_continuity_distance=0.2, min_normal_continuity_angle = 15 * np.pi / 180, 
                                      min_track_length = 5 ), 
        clustering = AttrDict( posepair_distance_sigma = 0.01, # in m ; sig = 0.015 m
                               posepair_theta_sigma = 20 * np.pi / 180, cluster_with_normals = False, 
                               label_top_k = 2, min_track_intersection = 10 ), 
        articulation = AttrDict( mlesac=True, return_tree=True )
    )
    # ============================================================================
    # ============================================================================

    # Read log file, and process frames
    logs = [
        'L2-FRIDGE-24', 
    ]

    alg_eval = {}

    for log in logs: # fn_map.keys(): 
        frames = LCMLogPlayer(filename=fn_map[log]).get_frames(every_k_frames=5)	

        lname = log.replace('-','_')
        alg_eval[lname] = {}

        try: 
            print 'Log %s ---------------------------------' % lname

            logging.info('Ours =================================')
            al_lfvd = ArticulationLfD(lfvd_params)
            al_lfvd.process_demonstration(frames, name=lname)
            alg_eval[lname]['ours'] = al_lfvd.model.art_est.eval_info

        except Exception as e: 
            print e # 'Fail'

    logging.info('Done')

    db = DictDB('articulation_learning_eval.h5', data=alg_eval, mode='w')
    db.flush()
    db.close()


    import pprint
    pp = pprint.PrettyPrinter(indent=4)
    print 'Katz ===================================='
    # pp.pprint(al_katz.model.art_est.aobj_info.model.values())
    print 'Ours ===================================='
    pp.pprint(al_lfvd.model.art_est.aobj_info.model.values())


    # # pp.pprint(al_tag.art_est.aobj_info.model.art_est.model.values())

    # # <codecell>

    # # Print pose parameters: eval_info['ori_errs/tvec_errs']['ours/katz']
    # np.set_printoptions(precision=2, suppress=True, threshold='nan', linewidth=160)
    # alg_eval = DictDB('articulation_learning_eval.h5', mode='r').data

    # for log,data in alg_eval.iteritems(): 
    #     print 'LOG: %s ==================================' % log

    #     # Print model parameters
    #     print '\nKatz ------------------'
    #     try: 
    #         print 'tvec_err: ', data['pose_eval'].tvec_errs['katz']
    #         print 'ori_err: ', data['pose_eval'].ori_errs['katz']
    #     except: 
    #         pass
    #     try: 
    #         print data['katz']
    #     except: 
    #         pass
    #     print '\nOurs ------------------'
    #     try: 
    #         print 'tvec_err: ', data['pose_eval'].tvec_errs['ours']
    #         print 'ori_err: ', data['pose_eval'].ori_errs['ours']
    #     except: 
    #         pass
    #     try: 
    #         print data['ours']
    #     except: 
    #         pass
    #     print '\nTag ------------------'
    #     try: 
    #         print data['tag']
    #     except: 
    #         pass
    #     print '\nDiff ------------------'
    #     if 'tag' in data and 'ours' in data and 'katz' in data: 
    #         err_ours, err_katz = np.nan, np.nan
    #         for attr_str in ['prismatic_dir', 'rot_axis']: 
    #             try: 
    #                 if attr_str in data['tag'] and attr_str in data['ours']: 
    #                     err_ours = np.rad2deg(np.arccos(np.fabs(np.dot(data['tag'][attr_str], data['ours'][attr_str]))))
    #             except: 
    #                 pass

    #             try: 
    #                 if attr_str in data['tag'] and attr_str in data['katz']: 
    #                     err_katz = np.rad2deg(np.arccos(np.fabs(np.dot(data['tag'][attr_str], data['katz'][attr_str]))))
    #             except: 
    #                 pass

    #         print 'Error in %s (Ours: %4.3f), (Katz: %4.3f)' % (attr_str, err_ours, err_katz)
    #     print '================\n'

    # # # print 'Katz ----'
    # # # print al_katz.model.art_est.eval_info
    # # # print 'Ours ----'
    # # # print al_lfvd.model.art_est.eval_info
    # # # print 'Tag ----'
    # # # print al_tag.model.art_est.eval_info

    # # <codecell>

    # # frames = LCMLogPlayer(filename=fn_map[log]).iterframes(every_k_frames=10)
    # # pose_est_eval = PoseEstimationEvalulation(frames, al_katz.model.poses)
    # plot_info = alg_eval[lname]['pose_eval']
    # PoseEstimationEvalulation.plot(plot_info.data_map, plot_info.plot_keys, plot_info.names)
    # # al_lfvd.model.art_est.get_original_poses())

    # # <codecell>


    # al_lfvd.model.manifold = al_lfvd.motion_analysis(frames, al_lfvd.model.art_est.get_projected_poses())
    # al_lfvd.db.add_demonstration('test1', frames, al_lfvd.model, init_tidx=0)
    # al_lfvd.db.add_demonstration('test1', frames, al_lfvd.model, init_tidx=3)

    # # <codecell>

    # # Prediction 
    # for log in fn_map.keys(): 
    #     print 'INPUT Log: %s' % log
    #     pframe = LCMLogPlayer(filename=fn_map[log]).get_frames(every_k_frames=100)[0]
    #     al_lfvd.predict(pframe)

    #     print '==========================================================='
    # # import time
    # # st = time.time()

    # # print 'Time taken: ', time.time() - st

    # # <codecell>

    # db.query(frames[1])

