# 1. articulation-learner -k 1
# 2. python articulation_eval.py -f test.h5 OR ====> only evals single h5 db
# 2. python articulation_eval.py ====> this looks at current directory and evals all h5 files in the cwd

import numpy as np
import tables as tb
import pandas as pd
import networkx as nx
import os, sys
import lcm; lc = lcm.LCM()

import random
import matplotlib as mpl
import matplotlib.pylab as plt

# import utils.plot_utils as plot_utils
# import utils.draw_utils as draw_utils
# import utils.imshow_utils as im_utils
# import utils.io_utils as io_utils

# import utils.rigid_transform as rtf
# import utils.transformations as tf

from utils.db_utils import DictDB, AttrDict
from utils.data_containers import GroundTruthWithDepthData, Feature3DData, Pose3DData
from utils.trajectory_analysis import TrajectoryClustering, ArticulationAnalysis
from utils.pose_pair_estimation import PosePairEstimation
# from fs_utils import LCMLogReader

# # Articulation related
# import vs
# from articulation.pose_msg_t import pose_msg_t
# from articulation import track_msg_t, track_list_msg_t

from optparse import OptionParser
from collections import OrderedDict

# ===== Parameters ==== 
SAMPLE_T_TIMEFRAMES = 200; 
MIN_TRACK_INTERSECTION = SAMPLE_T_TIMEFRAMES * 0.20;

MIN_FEATURE_CONTINUITY_DISTANCE = 0.02
MIN_TRACK_LENGTH = 10; 
MIN_TRACK_INTERSECTION = 5;
MIN_TRAJECTORY_VAR = 0.05 * 0.05;

POSEPAIR_DISTANCE_SIGMA = 0.02; # in m ; sig = 0.015 m
POSEPAIR_THETA_SIGMA = 10 * np.pi / 180; # np.cos((90-5) * np.pi / 180); # sig = 5 deg


def build_info_text(info): 
    info_text = 'No. of Components: %i\n' % info.num_parts
    info_text += 'DOF: %i\n' % info.params.dof
    info_text += 'Avg. Pos. Error: %4.3f\n' % info.params.avg_error_position
    info_text += 'Avg. Orientation Error: %4.3f\n' % info.params.avg_error_orientation
    return info_text

def vizualize_articulation(msg, ax=None):
    print 'NUM PARTS: %i' % msg.num_parts
    
    # Build graph
    G = nx.Graph()
    node_labels = {}
    for j in range(msg.num_parts): 
        G.add_node(j)
        node_labels.update({j:'%i' % j})
        
    print 'NUM PARAMS: %i' % msg.num_params
    aparams = [("%s: %s" % (param.name, param.value)) 
               for param in msg.params]

    print 'NUM MODELS: %i' % msg.num_models
    edge_labels = {}
    for model in msg.models: 
        print '-------------------------------'
        from_id, to_id = model.id / msg.num_parts, \
                         model.id % msg.num_parts
        print 'MODEL %i->%i %s' % (from_id, to_id, model.name)

        aparams = dict([(param.name, param.value) 
                        for param in model.params])
        print aparams

        edge_label = model.name.title()
        if model.name == 'rotational': 
            edge_label = ''.join([edge_label, 
                                  '\n', 'Rot. radius: %3.2f m' 
                                  % (aparams['rot_radius'])])

        G.add_edge(from_id, to_id)
        edge_labels.update({(from_id,to_id): '%s' % (edge_label)})

    random.seed(1)
    colors = [random.random() for i in range(G.number_of_nodes())]
    pos=nx.shell_layout(G)
    nx.draw(G, pos, with_labels=False, arrows=False,
            nodelist=node_labels.keys(), node_color=colors, node_size=600,
            vmin=0.0, vmax=1.0)

    nx.draw_networkx_labels(G, pos, labels=node_labels,
                            font_weight='bold', font_color='white', font_size=12)

    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                                 font_weight='bold', font_size=12)
    return


def write_df(df, fn): 
    with open(fn, "w") as f:
        f.write("\\begin{tabular}{" + " | ".join(["c"] * len(df.columns)) + "}\n")
        for i, row in df.iterrows():
            f.write(" & ".join([str(x) for x in row.values]) + " \\\\\n")
        f.write("\\end{tabular}")

def eval_articulation(filename): 
    # Open DB ==============================================================
    db = DictDB(filename=filename, mode='r')

    # Overall Stats
    overall = AttrDict()
    overall.model = AttrDict()
    
    # Read from DictDB and setup feature trajectories for learning =========
    for k,v in db.data.iteritems(): 
        # if k != 'test3': continue

        gt_data = Pose3DData(v.ground_truth, discretize_ms=100)
        # gt_data.viz_data(utimes_inds=None) 

        sample_data = Feature3DData(v.samples, discretize_ms=100)
        # sampled_data.viz_data(utimes_inds=None)
        print '--- Done building data'


        # Clustering trajectories based on rigid-point-pair-feature ============
        # Could provide a range, and run for several range chunks
        tcluster = TrajectoryClustering(
            data=sample_data, 
            posepair_distance_sigma=POSEPAIR_DISTANCE_SIGMA,
            posepair_theta_sigma=POSEPAIR_THETA_SIGMA,
            min_track_intersection=1, # MIN_TRACK_INTERSECTION, 
            time_chunks=2, visualize=True, verbose=False)

        # Pose-Pair Estimation from identified clusters ========================
        ppe = PosePairEstimation(data=sample_data)
        ppe.viz_data()

        # Estimate articulation ================================================
        pose_list = AttrDict()
        pose_list.ground_truth = []
        pose_list.sample_data = ppe.get_pose_list()

        # Manually construct ground truth poses
        for ind in gt_data.valid_feat_inds: 
            ut_inds, = np.where(gt_data.idx[ind,:] != -1)
            pinds = gt_data.idx[ind,ut_inds]
            pose_list.ground_truth.append((gt_data.feature_ids[ind], 
                                           [gt_data.poses[pind] for pind in pinds]))

        # Evaluate articulation graph ==========================================
        articulation = AttrDict()

        articulation.ground_truth = ArticulationAnalysis(pose_list.ground_truth)
        articulation.ground_truth_info = articulation.ground_truth.evaluation_info();
        articulation.ground_truth_info.pretty_name = v.config_pretty_name
        articulation.ground_truth_info_text = build_info_text(articulation.ground_truth_info)

        articulation.sample_data = ArticulationAnalysis(pose_list.sample_data)
        articulation.sample_data_info = articulation.sample_data.evaluation_info();
        articulation.sample_data_info.pretty_name = v.config_pretty_name
        articulation.sample_data_info_text = build_info_text(articulation.sample_data_info)        

        # Overal statistics
        overall.model[v.config_pretty_name] = OrderedDict()
        overall.model[v.config_pretty_name]['No. of Components'] = articulation.ground_truth_info.num_parts
        overall.model[v.config_pretty_name]['DOF'] = articulation.ground_truth_info.params.dof
        overall.model[v.config_pretty_name]['Ground Truth Avg. Pos. Error'] = articulation.ground_truth_info.params.avg_error_position
        overall.model[v.config_pretty_name]['Ground Truth Avg. Orientation. Error'] = articulation.ground_truth_info.params.avg_error_orientation

        overall.model[v.config_pretty_name]['Data Avg. Pos. Error'] = articulation.sample_data_info.params.avg_error_position
        overall.model[v.config_pretty_name]['Data Avg. Orientation. Error'] = articulation.sample_data_info.params.avg_error_orientation


        # Viz articulation graph ===============================================
        f = plt.figure(1)
        f.clear()

        ax1 = f.add_subplot(121)
        ax1.set_title('Ground Truth')
        vizualize_articulation(articulation.ground_truth.msg, ax1)
        ax1.text(0.02, 0.02, articulation.ground_truth_info_text, 
                 transform=ax1.transAxes, fontsize=12, fontweight='bold')

        ax2 = f.add_subplot(122)
        ax2.set_title('Simulated Data')
        vizualize_articulation(articulation.sample_data.msg, ax2)
        ax2.text(0.02, 0.02, articulation.sample_data_info_text, 
                 transform=ax2.transAxes, fontsize=12, fontweight='bold')

        f.savefig('%s.pdf' % ('-'.join(v.config_description)))

    # Write to latex file
    latex_fn = '%s-results.tex' % (filename.replace('.h5',''))
    with open(latex_fn, 'w') as f: 
        f.write(pd.DataFrame(overall.model).T.to_latex())
    db.close()


if __name__ == "__main__": 
    parser = OptionParser()
    parser.add_option("-f", "--filename", dest="filename", 
                      type="string",
                      help="Input DB", 
                      metavar="Input DB")
    (options, args) = parser.parse_args()

    # Required options =================================================
    if not options.filename: 
        files = []
        for filename in os.listdir("."):
            if filename.endswith(".h5"):
                files.append(filename)
        if not len(files): 
            parser.error('Input DB not given')
        else: 
            for filename in files: 
                eval_articulation(filename)
    else: 
        eval_articulation(options.filename)
        
    # First create simulation data ========================================
    # os.system("python kdl_fkin_generator.py -f %s" % (options.filename))

# # gt_data = GroundTruthWithDepthData(db.data[gt_db])

# # Smoothing and pruning of feature trajectories ========================
# # gpft_data.prune_by_length(min_track_length=MIN_TRACK_LENGTH)
# # gpft_data.prune_discontinuous(min_feature_continuity_distance=MIN_FEATURE_CONTINUITY_DISTANCE)
# # gpft_data.estimate_normals(fn)
# # gpft_data.refine_xyz_data(fn)
# # gpft_data.smooth_xyz_data()

# # Viz feature trajectories =============================================
# gt_data.viz_data(utimes_inds=None) # np.arange(40,110)) 
# gpft_data.viz_data(utimes_inds=None) # np.arange(40,110)) 

# # <codecell>

# # GPFT_data should have labels =========================================
# # gpft_data.write_video(filename=fn)

# # <codecell>


# # <codecell>



# table_info = AttrDict()

# print aparams


