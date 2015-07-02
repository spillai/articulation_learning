import numpy as np
import itertools, logging

import matplotlib.pylab as plt

from sklearn import cluster as skcluster

import utils.draw_utils as draw_utils
from utils.distributions import PosePairDistribution
from utils.distributions import plot_pose_pair_density, plot_pose_pair_variation

class TrajectoryClustering: 
    def __init__(self, data, params, visualize=False, save_dir=None): 

        self.log = logging.getLogger(self.__class__.__name__)

        # Save params
        self.params = params
        self.visualize = visualize
        self.save_dir = save_dir

        # Check valid_feat_inds
        if not len(data.valid_feat_inds): 
            raise RuntimeError('No features available for clustering')
        
        # Cluster index map
        # [label] -> [inds]
        self.cluster_inds = dict()

	# 1. Init labels
	data.labels = np.zeros(data.num_feats, dtype=np.int32);
	data.labels.fill(-1);

        # 2. Single chunk clustering
        # Build similarity matrix for the full time range
        utimes_inds = np.arange(data.num_utimes)
        posepair_dict, W_dist, W_normal = self.build_similarity_matrix(data, utimes_inds)  

        # 3. Build weight matrix with normals, and displacements
        if self.params.cluster_with_normals: 
            W = np.minimum(W_dist,W_normal) 
        else: 
            W = W_dist

        # 4. Find the clustering between trajectories
        self.log.debug('===> Cluster the trajectories ...')
        valid_feat_labels = self.cluster(data, W )
        data.labels[data.valid_feat_inds] = valid_feat_labels
        self.log.debug('===> Done clustering')

        # 5. Build cluster index map
        # Only map top k clusters (based on size of each cluster)
        self.label_top_clusters(data, top_k=self.params.label_top_k)

        # Visualize clustering
        self.viz_clustering(data, data.valid_feat_inds, utimes_inds)
        self.log.debug('Number of labels: %s' % np.unique(data.labels))

    # Build cluster index map
    # Output: cluster_inds [label] -> [inds corr. to label]
    def label_top_clusters(self, data, top_k): 
        
        # Count number of instances of labels
        # Avoid -1 label
        unique_labels, unique_inv = np.unique(data.labels[np.where(data.labels != -1)], 
                                              return_inverse=True)
        num_labels = len(unique_labels)
        label_count = np.bincount(unique_inv)
        label_votes = zip(unique_labels, label_count)
        self.log.debug('Total: %i labels' % num_labels)

        # Remove labels that have count < 3 
        label_votes = filter(lambda (l,c): c >= 3, label_votes)

        # Sort (label,votes) by votes, and pick the labels with top_k votes
        label_votes.sort(key=lambda (l,c): c, reverse=True)
        labels = map(lambda (l,c): l, label_votes[:top_k])
        self.log.debug('Picking top %i labels: %s' % (len(labels), label_votes[:top_k]))

        # Unlabel rest of the labeled trajectories
        for label in unique_labels: 
            if label in labels: continue
            inds, = np.where(data.labels == label)
            data.labels[inds] = -1

        # Find exemplars for each cluster 
	for label in labels: 
            assert(label >= 0)

            # Indices that are assigned the "label"
	    inds, = np.where(data.labels == label)
            if len(inds) < 3: 
                # Complain about insufficient constraints
                raise RuntimeError('Label %i cluster has only (%i) data constraints!' % (label,len(inds)))

            # Map cluster indices
	    self.cluster_inds[label] = np.array(inds, dtype=np.int32)

    def build_similarity_matrix(self, data, utimes_inds): 
        # Find the overlapping set of features between trajectories
        self.log.debug('===> Extracting overlapping features ...')
        overlap_dict = self.extract_overlapping_features(data, 
                                                         data.valid_feat_inds, 
                                                         utimes_inds)
        self.log.debug('--- Done extracting overlapping features')
        

        # Construct the similarity matrix between trajectories
        # Utime slicing is encapsulated within the overlap_dict
        self.log.debug('===> Compute trajectory similarity ...')
        posepair_dict, W_dist, W_normal = self.compute_trajectory_similarity(data, 
                                                                             data.valid_feat_inds, 
                                                                             overlap_dict)
        self.log.debug('--- Done computing pose-pairs')

        # Viz similarity in viewer
        if self.visualize: self.viz_similarity(data, posepair_dict)

        # Viz similarity
        # if self.visualize: self.viz_similarity_matrix(W_normal) 

        # # Viz a pose pair distribution for debug purposes
        # if self.save_dir: 
        #     f = plt.figure(1002)	
        #     plt.subplots_adjust( hspace=0.4 )

        #     ax1 = f.add_subplot(411)
        #     # plt.title('Histogram of variation in distance for a rigidly connected pose-pair');
        #     self.viz_posepair_distribution(posepair_dict[(0,1)], ax1, face='green')
        
        #     ax2 = f.add_subplot(412)
        #     # plt.title('Histogram of variation in distance for a non-rigidly connected pose-pair');
        #     self.viz_posepair_distribution(posepair_dict[(0,10)], ax2, face='blue')

        #     ax3 = f.add_subplot(423)
        #     # plt.title('Histogram of variation in distance for a non-rigidly connected pose-pair');
        #     self.viz_posepair_distribution(posepair_dict[(0,1)], ax3, distribution='density', face='green')

        #     ax4 = f.add_subplot(224)
        #     # plt.title('Histogram of variation in distance for a non-rigidly connected pose-pair');
        #     self.viz_posepair_distribution(posepair_dict[(0,10)], ax4, distribution='density', face='blue')

        #     plt.savefig(''.join([self.save_dir,'pose-pair-histogram-ground-truth.pdf']))
            

        return posepair_dict, W_dist, W_normal;

    # Find overlapping tracks ==============================================
    def extract_overlapping_features(self, data, valid_inds, utimes_inds):
        overlap_dict = {}

        num_inds = len(valid_inds);         
        niter, total_iters = 0, num_inds*(num_inds-1)/2;
        for jind,kind in itertools.product(valid_inds, valid_inds):
            if kind <= jind: continue;
            # if niter % 200 == 0: self.log.debug('Processed: %i out of %i' % (niter, total_iters))
            niter += 1 

            # find nz utime indices
            tj, = np.where(data.idx[jind,utimes_inds] != -1) # <- space of utimes_inds
            tj = utimes_inds[tj] # <- original space
            tk, = np.where(data.idx[kind,utimes_inds] != -1)
            tk = utimes_inds[tk]

            # set intersection
            intersection = np.intersect1d(tj,tk);
            if intersection.size < self.params.min_track_intersection: continue;

            # total_length = max(max(tk), max(tj)) - min(min(tk), min(tj))
            # if intersection.size * 1.0 / total_length < self.min_track_intersection: continue
            # # self.log.debug('tk, tj: ', total_length, intersection.size, \
            # #     intersection.size * 1.0 / total_length)

            # # Test if the features are within 100 L1 ball radius
            # nn_inds = np.sum(np.fabs(data.xy[jind,intersection] - data.xy[kind,intersection]), 
            # 		     axis=1) < 100;
            # if not np.all(np.all(nn_inds, axis=0),axis=0): continue;
            
            # Build overlap dictionary
            overlap_dict[tuple(sorted((jind,kind)))] = intersection;
        self.log.debug('Processed: %i out of %i' % (niter, total_iters))
        return overlap_dict;

    # Compute Pose-Pairs for overlapping tracks ============================
    def compute_trajectory_similarity(self, data, valid_inds, overlap_dict): 

        # Valid inds map:  valid_inds2inds => [4,5,9,...] -> [0,1,2,...] = {4:0,5:1,9:2}
        num_valid_feats = len(valid_inds);
        valid_inds2inds = dict( zip( valid_inds, np.arange(num_valid_feats) ) )

        # Build Weight/Affinity matrix
        W_normal = np.eye(num_valid_feats);
        W_dist = np.eye(num_valid_feats);
        # W_overlapk = np.zeros_like(W_dist)

        # Compute Pose Pair Similarity
        posepair_dict = dict();
        for id_pair,overlap_utimes_inds in overlap_dict.iteritems():
            # Evaluate Pose Pair Distribution
            ppd = PosePairDistribution(data, id_pair, 
                                       overlap_utimes_inds, 
                                       distance_sigma=self.params.posepair_distance_sigma, 
                                       theta_sigma=self.params.posepair_theta_sigma);
            # self.log.debug('%s %f' % (ppd.distance_match, ppd.distance_hist))
            # self.log.debug('%s %f' % (id_pair, ppd.distance_match))
            # self.log.debug('%i %i %f' % (id_pair[0], id_pair[1], ppd.distance_match))

            if np.isnan(ppd.distance_match) or np.isnan(ppd.normal_match): continue
            posepair_dict[id_pair] = ppd;

            # Key stored is valid_feat_id -> valid_feat_id_idx via valid_inds2inds
            r, c = valid_inds2inds[id_pair[0]], valid_inds2inds[id_pair[1]]

            W_normal[r,c] = ppd.normal_match;
            W_normal[c,r] = ppd.normal_match;

            W_dist[r,c] = ppd.distance_match;
            W_dist[c,r] = ppd.distance_match;

        return posepair_dict, W_dist, W_normal
        
    # Clustering ===========================================================
    def cluster(self, data, W): 

        # Cluster labels on the valid features
        exemplars,valid_labels = skcluster.dbscan(1-W, eps=.2, min_samples=3, metric='precomputed')
        # self.log.debug('%s %s' % (exemplars, valid_labels))

        if self.visualize: 
            # Shuffle cluster inds such that the cluster labels are together
            num_valid_feats = len(data.valid_feat_inds);
            assert(num_valid_feats == len(valid_labels));
            valid_indlabels = zip(np.arange(num_valid_feats), valid_labels)
            valid_indlabels.sort(key=lambda x: x[1]) # sort to bring cluster labels together
            valid_inds_shuffled, valid_labels_shuffled = zip(*valid_indlabels)

            # Count number of instances of labels
            unique_labels, unique_inv = np.unique(valid_labels, return_inverse=True)
            num_labels = len(unique_labels)
            label_count = np.bincount(unique_inv)
            label_votes = zip(unique_labels, label_count)
            label_votes.sort(key=lambda x: x[1], reverse=True)
        
            # Viz
            self.viz_clustering_matrix(W, valid_inds_shuffled)

        # self.log.debug('%s' % valid_labels)
        return valid_labels
        

    # Draw segmented features ==============================================
    def viz_clustering(self, data, valid_inds, utimes_inds): 
        # Visualize clusters in descending order of size
        viz_pts, viz_normals, viz_colors = [], [], []
        viz_traj1, viz_traj2 = [], []

        # Indices of label in the full dataset (could also look at valid dataset)
        labels = np.unique(data.labels)
        num_labels = len(labels)
        for label in labels:
            # Only among valid_inds
            inds, = np.where(data.labels[valid_inds] == label)
            inds = valid_inds[inds]
            self.log.debug('LABEL: %s %i' % (label, len(inds)))

            # Publish each feature ind
            for idx in inds: 
                ut_inds, = np.where(data.idx[idx,utimes_inds] != -1);
                ut_inds = utimes_inds[ut_inds]

                viz_pts.append(data.xyz[idx,ut_inds,:])

                viz_traj1.append(data.xyz[idx,ut_inds[:-1],:])
                viz_traj2.append(data.xyz[idx,ut_inds[1:],:])

                viz_normals.append(data.xyz[idx,ut_inds,:] + data.normal[idx,ut_inds,:]*0.04)

                # Color by label
                carr = draw_utils.get_color_arr(label, len(ut_inds), palette_size=num_labels,
                                             color_by='label')
                viz_colors.append(carr)

        viz_pts = np.vstack(viz_pts)
        viz_colors = np.vstack(viz_colors)
        viz_normals = np.vstack(viz_normals)
        viz_traj1, viz_traj2 = np.vstack(viz_traj1), np.vstack(viz_traj2)

        draw_utils.publish_point_cloud('CLUSTERED_PTS', viz_pts, 
                                       c=viz_colors, sensor_tf='KINECT')
        draw_utils.publish_line_segments('CLUSTERED_NORMAL', viz_pts, viz_normals, 
                                         c=viz_colors, sensor_tf='KINECT')
        draw_utils.publish_line_segments('CLUSTERED_TRAJ', viz_traj1, viz_traj2, 
                                         c=viz_colors, sensor_tf='KINECT')

    # Display shuffled matrix with clusters together
    def viz_clustering_matrix(self, W, valid_inds_shuffled): 
        nrows,ncols = W.shape

        fig = plt.figure(1001)

        ax = fig.add_subplot(111)
        cax = ax.matshow(W[np.ix_(valid_inds_shuffled,valid_inds_shuffled)], origin='lower')
        fig.colorbar(cax)

        labels = [str(idx) for idx in valid_inds_shuffled]
        ax.set_xticks(np.arange(nrows))
        ax.set_yticks(np.arange(nrows))

        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)

        plt.axis('off')
        plt.title('Clustering of trajectories via DBSCAN propagation indicating # of clusters')
        if self.save_dir: plt.savefig('affinity-prop-block-wise-clusters.pdf')

        # self.log.debug('Top 10 Cluster labels')
        # self.log.debug('%s' % unique(valid_labels)[:10])

    # Viz feature similarity ===============================================
    def viz_similarity(self, data, posepair_dict): 
        # debug_ids = [8,9,10,11]

        viz_pts, viz_normals, viz_colors = [], [], []
        for id_pair,ppd in posepair_dict.iteritems():

            # if not (id_pair[0] in debug_ids or id_pair[1] in debug_ids): continue
            # self.log.debug('%s %f' % (id_pair, ppd.distance_match))

            # Get Pair and 
            idx1, idx2 = id_pair
            c = plt.cm.jet(ppd.distance_match)
            ut_inds1, = np.where(data.idx[idx1,ppd.overlap_utimes_inds] != -1)
            ut_inds1 = ppd.overlap_utimes_inds[ut_inds1]
            ut_inds2, = np.where(data.idx[idx2,ppd.overlap_utimes_inds] != -1)
            ut_inds2 = ppd.overlap_utimes_inds[ut_inds2]

            viz_pts.append(data.xyz[idx1,ut_inds1[-1],:])
            #viz_pts.append(data.xyz[idx2,ut_inds2[-1],:])
            viz_normals.append(data.xyz[idx2,ut_inds2[-1],:])
            viz_colors.append(c);

        if not len(viz_pts): return

        viz_pts = np.vstack(viz_pts)
        viz_colors = np.vstack(viz_colors)
        viz_normals = np.vstack(viz_normals)

        draw_utils.publish_point_cloud('EDGE_DATA', viz_pts, c=viz_colors, sensor_tf='KINECT')
        draw_utils.publish_line_segments('EDGE_WEIGHTS', viz_pts, viz_normals, 
                                         c=viz_colors, sensor_tf='KINECT')

    def viz_similarity_matrix_chunks(self, Ws, f): 
        for cidx, W_dist in enumerate(Ws): 
            ax = f.add_subplot(3,3,cidx+1)
            ax.set_title('Time chunk: %s' % (str(cidx+1)))
            ax.matshow(W_dist, vmin=0, vmax=1)
            ax.set_axis_off()

    def viz_similarity_matrix(self, W, save_name='similarity-matrix'):
        nrows,ncols = W.shape

        fig = plt.figure(1000)
        fig.clf()

        ax = fig.add_subplot(111)
        cax = ax.matshow(W, vmin=0, vmax=1)
        # fig.colorbar(cax)

        labels = [str(idx) for idx in np.arange(nrows)]
        ax.set_xticks(np.arange(nrows))
        ax.set_yticks(np.arange(nrows))

        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)

        if self.save_dir: plt.savefig(''.join([self.save_dir,save_name,'.pdf']), format='pdf')

    # Plot the distribution of pair of points ==============================
    def viz_posepair_distribution(self, ppd, ax, distribution='histogram', face='green'): 
        if distribution == 'density': 
            plot_pose_pair_density(ppd, face=face)
            plt.text(.8,.85,'$\mu=%4.3f~(m)$' % (np.mean(ppd.distance_hist)), 
                     ha='center',va='center',transform=ax.transAxes)

        else: 
            plot_pose_pair_variation(ppd, face=face)
            plt.text(.8,.85,'$\mu=%4.3f, ~\sigma=%4.3f~(m)$' % (np.mean(ppd.distance_hist), 
                                                              np.std(ppd.distance_hist)), 
                     ha='center',va='center',transform=ax.transAxes)
        # plt.show()
