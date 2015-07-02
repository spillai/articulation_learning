# Import libraries
import cv2
import numpy as np
# import matplotlib as mpl
import matplotlib.pylab as plt

from scipy.spatial.distance import euclidean

# Import utils
# import utils.lcmutils as lcm_utils
import utils.plot_utils as plot_utils
# import utils.tag_utils as tag_utils
import utils.draw_utils as draw_utils
import utils.point_desc as desc_utils

from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection

from fs_utils import attrs_from_labels, draw_contours_from_labels, \
    unique_label_img, build_node_graph

def compute_mean(img, labels, mask=None, compute_mean_img=False):
    """Find label-mean of the image:
    Input:     
    img - input image whose mean is taken
    labels - labels on the same image
    mask - AND mask operator on the img
    compute_mean_img - boolean
    return
    segment - averaged img by the label
    label_mean - dict of labelid -> mean of labelid    
    """
    mean_img = None
    if compute_mean_img: 
        mean_img = np.zeros_like(img)

    label_mean = attrs_from_labels(img, labels=labels, mask=mask);

    # print img.shape, label_mean        
    for label,mean in label_mean.iteritems():
        label_mean[label] = np.array(mean);

    for label,mean in label_mean.iteritems():
        # Only check elements where mask is valid
        if mask is not None: 
            inds = np.bitwise_and(labels == label, mask)
        else:
            inds = labels == label
        if np.isnan(mean).any(): continue
        if compute_mean_img: 
            mean_img[inds] = mean;

    return mean_img, label_mean


def slic_draw(labels, label, label_unaries):
    img = np.zeros_like(labels, dtype=float)
    print len(label),label_unaries.shape[0]
    assert len(label) == label_unaries.shape[0]
    for j in range(len(label)):
        inds = (labels == label[j])
        img[inds] = label_unaries[j];
    return img    

def compute_centroid(labels, mask=None):
    # Compute region centers with mesh grid
    gyx = np.dstack(np.mgrid[:labels.shape[0],:labels.shape[1]])
    _,centers = compute_mean(gyx.astype(np.float32), labels, mask=mask);

    # Flip coordinates (y,x) -> (x,y)
    for label,mean in centers.iteritems():
        centers[label] = mean[::-1]
    return centers; # (x,y)

def draw_centers(img, labels, mask=None):
    centers = compute_centroid(labels, mask=mask);

    out = img.copy()
    for label,c in centers.iteritems():
        cv2.circle(out, tuple(c.astype(int)), 2, (255,0,0), 1)
    
    return out

# # Build graph from labels
# plt.subplot(1,3,3)
# grid = dslic_labels
# vertices, edges = grid_crf.make_graph(grid)

# # compute region centers:
# gridx, gridy = np.mgrid[:grid.shape[0], :grid.shape[1]]
# centers = dict()
# for v in vertices:
#     centers[v] = [gridy[grid == v].mean(), gridx[grid == v].mean()]
    
# # plot labels
# plt.imshow(img)
    
# # overlay graph:
# for edge in edges:
#     plt.plot([centers[edge[0]][0],centers[edge[1]][0]],
#     [centers[edge[0]][1],centers[edge[1]][1]])    

def prune_graph(graph, value_map, funcs):
    # Remove nodes from the key that don't exist in the value map
    l1set = set(graph.keys())
    for l1 in graph.keys():
        if l1 not in value_map:
            l1set.remove(l1)

    # Remove nodes from values that don't satisfy func. constraints
    set_graph = dict()
    for l1 in l1set:
        pt1 = value_map[l1]

        l2set = set(graph[l1])
        for l2 in graph[l1]:
            if not (l2 in value_map and all( [func(pt1,value_map[l2]) for func in funcs] )):
                l2set.remove(l2)

        if len(l2set): set_graph[l1] = l2set;
    return set_graph

grp = GaussianRandomProjection(n_components=3);
pca = PCA(n_components=3);

class SLICDescription:
    def __init__(self, frame):
        
        # Build image (LAB) superpixels (var superpixel size / num of superpixels)
        # %timeit (~150 ms per loop)
        labels = slic.slic(frame.rgb, var=100, compactness=10,
                            op_type=slic.FOR_GIVEN_SUPERPIXEL_SIZE, color_conv=cv2.COLOR_RGB2LAB)

        # Main description
        self.description = dict()

        # Compute centers of superpixels
        try: 
            centers = compute_centroid(labels, mask=frame.depth_mask);
        except AttributeError:
            print 'Depth Mask unavailable'
            return
        
        # Compute mean_img per label
        try :
            mean_rgb, self.rgb_map = compute_mean(frame.rgb, labels,
                                                       mask=frame.depth_mask, compute_mean_img=False)
        except AttributeError:
            print 'Depth Mask unavailable'
            return
    
        # Compute mean_normals_img per label
        try :
            mean_normals, self.normals_map = compute_mean(frame.normals, labels,
                                                          mask=frame.normals_mask, compute_mean_img=False)
        except AttributeError:
            print 'Normals/Normals Mask unavailable'
            return

                    
        # Convert to R2 - Keypoints
        self.R2_map = dict([ (l,cv2.KeyPoint(c[0],c[1],_size=1,_class_id=l,_response=1.))
                             for l,c in centers.iteritems() ]) # node_id : [keypoint - R2]

        # Convert R3 - 3D points
        self.R3_map = dict([ (l,frame.X[int(c[1]),int(c[0]),:])
                             for l,c in centers.iteritems() ]) # node_id : [3D point - R3]

        # R3 x SO(2) map = R3_map + normals_map
        self.R3SO2_map = dict()
        for l,pt in self.R3_map.iteritems():
            if l in self.normals_map:
                self.R3SO2_map[l] = np.hstack((pt,self.normals_map[l]))
                                         
        # Compute connectivity graph on labels
        self.neighbors_map = slic.build_node_graph(labels); # node_id : [neighbors_node_list]

        # Prune graph connectivity by R3 (ball) HEURISTICS
        prune_funcs = []
        prune_funcs.append(lambda pt1,pt2 : bool(pt1[2] > 0.5 and pt2[2] > 0.5))
        prune_funcs.append(lambda pt1,pt2 : (not np.any(np.isnan(pt1))) and (not np.any(np.isnan(pt2))))
        prune_funcs.append(lambda pt1,pt2 : bool(euclidean(pt1,pt2) < 0.20))
        self.neighbors_map_with_depth = prune_graph(self.neighbors_map, self.R3_map, prune_funcs)
        
        # Compute Descriptors from R2_map
        # (ensure keypoints have a class_id associated with them)
        self.feat_desc = desc_utils.PointDescMap(frame.rgb, self.R2_map,
                                          graph=self.neighbors_map_with_depth)
                
        # Visualize mean_normals_img per label
        try :
            pass
        except AttributeError:
            print 'Normals unavailable; Failed to visualize'

        # Visualize mean_img per label
        try :
            pass
        except AttributeError:
            print 'Normals unavailable; Failed to visualize'


        # Visualize delaunay triangulation
        self.visualize_delaunay_triangulation();

        # Visualize surface normals
        self.visualize_surface_normals();

    # Match with a previous description
    def match(self, pdesc):
        # Check spatial constraint, then match descriptors

        # For all valid keypoints described previously,
        # look at spatial graph, and find pts in the new
        # frame that satisfy constraint
        # Then, find the closest descriptor match
        cptdesc = []
        for l in self.feat_desc.get_description().keys():
            if l in self.R2_map.keys():
                c = self.R2_map[l]
                cptdesc.append([l,([c.pt[0],c.pt[1]],self.feat_desc.get_description()[l])]) #
        print pdesc.feat_desc.get_closest_match_description(dict(cptdesc))

    def visualize_feature_desc(self,img):
        descs = np.vstack(self.feat_desc.get_description().values())
        descs_red = pca.fit_transform(descs);

        # Normalize [0,1]
        descs_red = (descs_red - descs_red.min()) / (descs_red.max() - descs_red.min())

        # Draw image
        out = np.copy(img)
        for j,l in enumerate(self.feat_desc.get_description().keys()):
            kpt = self.R2_map[l]
            col = (int(descs_red[j,0]*255),int(descs_red[j,1]*255),int(descs_red[j,2]*255))
            cv2.circle(out, (int(kpt.pt[0]),int(kpt.pt[1])), 1,
                       col, thickness=-1, lineType=cv2.LINE_AA)
        plot_utils.imshow(out)
                    
    def visualize_delaunay_triangulation(self):
        line_list = []            
        for l1,llist in self.neighbors_map_with_depth.iteritems():
            if l1 not in self.R3_map: continue
            pt1 = self.R3_map[l1]
            if np.any(np.isnan(pt1)): continue

            for l2 in llist:
                if l2 not in self.R3_map: continue
                pt2 = self.R3_map[l2]
                if np.any(np.isnan(pt2)): continue
                line_list.append(np.hstack((pt1, pt2)))
        lines = np.vstack(line_list)

        draw_utils.publish_line_segments('DELAUNAY', lines[:,:3], lines[:,-3:],
                                         c='r', size=0.02, downsample=1, sensor_tf='KINECT')

    def visualize_surface_normals(self): 
        line_list = []
        pt_list = []
        for l,n in self.normals_map.iteritems():
            if l not in self.R3_map: continue
            pt1 = self.R3_map[l]
            if np.any(np.isnan(pt1)): continue
            if np.any(np.isnan(n)): continue
            pt_list.append(pt1)
            line_list.append(np.hstack((pt1, pt1+0.04*n)))
        lines = np.vstack(line_list)
        pts = np.vstack(pt_list)

        draw_utils.publish_line_segments('NORMALS', lines[:,:3], lines[:,-3:],
                                         c='b', size=0.02, sensor_tf='KINECT')

        draw_utils.publish_point_cloud('POINTS', pts[:,:3],
                                         c='r', size=0.02, sensor_tf='KINECT')

            
        # # Build 
        #                      # Build point desc map
        # point_desc_map[frame.utime] = desc_utils.PointDescList(img, slic_keypts)
        
# def compute_mean(img, labels, func=np.mean, mask=None):
#     """Find label-mean of the image:
#     Input:     
#     img - input image whose mean is taken
#     labels - labels on the same image
#     mask - AND mask operator on the img

#     return
#     segment - averaged img by the label
#     label_mean - dict of labelid -> mean of labelid    
#     """
#     if img.ndim == 3:
#         ch = img.shape[2]
#     else:
#         ch = 1
    
#     mean_img = np.zeros_like(img)
#     unique_labels = np.unique(labels)
#     label_mean = dict()
#     for label in unique_labels:
#         # Only check elements where mask is valid
#         if mask is not None: 
#             inds = np.bitwise_and(labels == label, mask)
#         else:
#             inds = labels == label
#         mean = func(img[inds].reshape(-1,ch), axis=0);
#         if np.isnan(mean).any(): continue
        
#         label_mean[label] = mean
#         mean_img[inds] = mean;
        
#     return mean_img, label_mean
