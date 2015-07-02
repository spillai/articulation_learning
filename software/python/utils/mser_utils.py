# Import libraries
import cv2
import numpy as np
# import matplotlib as mpl
import matplotlib.pylab as plt

from scipy.spatial.distance import euclidean
from scipy.spatial import KDTree

# Import utils
# import utils.lcmutils as lcm_utils
import utils.plot_utils as plot_utils
# import utils.tag_utils as tag_utils
import utils.draw_utils as draw_utils
import utils.point_desc as desc_utils

# Import pywrappers
import fs_slic as slic
#import fs_apriltags as apriltags
#import fs_pcl_utils as pcl_utils

from shape_context import SC

# Defaultdict
from collections import defaultdict

import _vlfeat as vl
import math

from skimage.measure import approximate_polygon
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.cluster import dbscan

from itertools import product
from plot_utils import np2tuple
from geom_utils import contours_from_endpoints

class MSERDescription:
    def __init__(self, frame):
        
        # Compute MSER features on the rgb image
        # %timeit (~ ms per loop)
        mser_delta = 5
        mser_min_area = 60
        mser_max_area = 320*240
        mser_max_variation = 0.25
        mser_min_diversity = 0.2
        mser_max_evolution = 200 # 0.001 # 0.003
        mser_area_threshold = 1.01
        mser_min_margin = 0.003
        mser_edge_blur_size = 11

        self.mser = cv2.MSER(_min_area=400, _max_area=mser_max_area, _max_evolution=mser_max_evolution,
                        _edge_blur_size=mser_edge_blur_size)

        # Detect MSERs ===========================================
        self.rgb_regions, self.normal_regions = {}, {}
        ims, normals, masks = {}, {}, {}

        self.noctaves = 3
        self.octaves = np.power(2, np.arange(0,self.noctaves))

        for j,octave in enumerate(self.octaves):
            # Compute img,normals,mask at different scales
            s = 1.0 / octave
            ims[j], normals[j], masks[j] = frame.rgb, frame.normals, frame.normals_mask.astype(np.uint8)

            # Scale image appropriately
            if octave != 1: 
                ims[j] = cv2.resize(ims[j], (0,0), fx=s, fy=s,
                                    interpolation=cv2.INTER_NEAREST)
                normals[j] = cv2.resize(normals[j], (0,0), fx=s, fy=s,
                                        interpolation=cv2.INTER_NEAREST)
                masks[j] = cv2.resize(masks[j], (0,0), fx=s, fy=s,
                                      interpolation=cv2.INTER_NEAREST)

            # Gaussian blur, and color conversion
            # im = cv2.GaussianBlur(frame.rgb, (3, 3), 0)
            lab = cv2.cvtColor(ims[j], cv2.COLOR_RGB2LAB)

            # MSER Detection on RGB image
            self.rgb_regions[j] = self.mser.detect(lab, masks[j])
            
            # Convert to 3 channel 8-bit image
            normals[j] = ((normals[j] + 1) * 128).astype(np.uint8)

            # MSER Detection on surface normals
            self.normal_regions[j] = self.mser.detect(normals[j], masks[j])
        
        # Describe MSERs at each scale ==================================
        # self.mser_description = defaultdict(lambda: defaultdict(int))

        self.descriptor = cv2.DescriptorExtractor_create("OpponentSIFT");
        
        h, w, ch = frame.rgb.shape
        all_pts = []
        for j,octave in enumerate(self.octaves):

            # Extract ellipses from regions detected
            # Note: the ellipses are scaled to 0th octave
            print 'Octave: %i' % (octave)
            hulls = [cv2.convexHull(p.reshape(-1, 1, 2) * octave)
                     for p in self.normal_regions[j]]

            ellipses = [cv2.fitEllipse(contours_from_endpoints(hull.reshape(-1,2),10))
                        for hull in hulls]

            # Compute descriptor for each ellipse center
            kpts, pdpts = [], []
            for k,e in enumerate(ellipses):
                center, axes, angle = e
                
                # Detect each ellipse at each of their octaves
                kpt = cv2.KeyPoint(x=center[0], y=center[1], _size=axes[0],
                                _angle = angle, _response=axes[1]/axes[0],
                                _octave=j, _class_id=k)
                kpts.append(kpt)

            # Compute descriptor, and aggregate
            _,desc = self.descriptor.compute(ims[j], kpts); # Add mask?
            pdpts = [(k,desc_utils.PointDesc(keypt=kpt, desc=desc[k]))
                     for k,kpt in enumerate(kpts) ]
            
            # MSER description mapped to octave
            # self.mser_description[j] = dict(pdpts)
            all_pts.extend(kpts);
                
        # Collapse MSER detections across scales ========================
        # print self.mser_description[j]
        # desc_utils.PointDescMatcher(pdpts)
        out = np.array(frame.rgb, copy=True)

        # Convert to pts
        pts = np.array([[kp.pt[0], kp.pt[1]] for kp in all_pts])
        pts = np.vstack(pts)

        # DBScan
        kpinds,labels = dbscan(pts, eps=10, min_samples=1, metric='euclidean')

        # Clusters
        cluster_pts = []
        for lab in range(np.max(labels).astype(int)):
            inds, = np.where(labels == lab)
            cluster_attrs = defaultdict(list)
            for ind in inds:
                cluster_attrs['pt'].append(list(all_pts[ind].pt))
                cluster_attrs['angle'].append(all_pts[ind].angle)
                cluster_attrs['axes'].append(np.hstack((all_pts[ind].size,
                                             np.multiply(all_pts[ind].size,
                                                         all_pts[ind].response))))

            mu_center = np.mean(np.vstack(cluster_attrs['pt']), axis=0);
            mu_axes = np.mean(np.vstack(cluster_attrs['axes']), axis=0);
            mu_angle = np.mean(cluster_attrs['angle'])
            # print mu_center, mu_axes, mu_angle
            cluster_pts.append({'pt': mu_center, 'axes': mu_axes, 'angle': mu_angle})
            
        for p in cluster_pts: 
            e = np2tuple(p['pt']), np2tuple(p['axes']), p['angle']
            cv2.circle(out, np2tuple(p['pt']), 10, (0,255,0), 1)
            cv2.ellipse(out, e, (0,255,0),1)

        plot_utils.imshow(out)
        
        # print kpinds, labels
        # print 'INDS: ', len(kpinds), len(labels), len(all_pts)

        # # Add keypoints to kdtree
        # self.spatial_tree = KDTree(pts)
        # print 'Spatially described pts: ', pts.shape
        
        # # Query
        # pair_votes = defaultdict(0)
        # qpts = self.spatial_tree.query_ball_point(pts, r=10) 
        # for j,inds in enumerate(qpts):
        #     assert (j in inds)
        #     pairs = [sorted(p1,p2) for p1,p2 in product([j],inds)]
        #     for pair in pairs: pair_votes[pair] += 1
        # print pair_votes
                
    def normalize_regions(self, frame, regions):
        out = np.array(frame.rgb, copy=True)
        ellipses = [cv2.fitEllipse(p) for p in regions]
        for e in ellipses:
            cv2.ellipse(out, e, (0,255,0),1)

        normals = ((frame.normals + 1) * 128).astype(np.uint8)
        plot_utils.imshow(normals)
    
        plot_utils.imshow(np.hstack((out,normals)))
                                  
    def visualize_normals(self, frame):
        th = np.sin(np.deg2rad(75))
        mask = np.abs(frame.normals) >= np.array([th,th,0])
        
        normals = ((frame.normals + 1) * 128).astype(np.uint8)
        plot_utils.imshow(normals)

    def visualize_scales(self, frame, mser_regions):
        ims = [np.array(frame.rgb, copy=True) for j in self.octaves]
        ims.append(np.array(frame.rgb, copy=True))
        cols = [(255,0,0),(0,255,0),(0,0,255)]
        
        for j,octave in enumerate(self.octaves):

            # Work with convex hulls
            hulls = [cv2.convexHull(p.reshape(-1, 1, 2) * octave)
                     for p in mser_regions[j]]

            # Draw in last image
            cv2.polylines(ims[-1], hulls, 1, (255, 255, 0), thickness=1)

            # Draw text, and circles in last image
            for k,hull in enumerate(hulls):
                chull = tuple(np.mean(hull.reshape(-1,2),axis=0).astype(int).tolist())
                # cv2.circle(ims[-1], chull, 2, cols[j], -1)
                cv2.putText(ims[j], '%i'%k, chull, 0, 0.55, cols[j], thickness=2)

            # Draw polylines at each scale
            cv2.polylines(ims[j], hulls, 1, (255, 255, 0), thickness=1)
            cv2.putText(ims[j], 'octave: %i'%octave, (20,20), 0, 0.55, (255,0,0), thickness=3)

            # Work with ellipse fits on the convex hull
            ellipses = [cv2.fitEllipse(contours_from_endpoints(hull.reshape(-1,2),10))
                        for hull in hulls]
            # ellipses = [cv2.fitEllipse(hull.reshape(-1,2)) for hull in hulls]
            for k,e in enumerate(ellipses):
                ecenter, eaxes, eangle = e
                red_eaxes = (eaxes[0]/4,eaxes[1]/4)

                ecenter = tuple(np.array(ecenter).astype(int).tolist())
                eaxes = tuple(np.array(eaxes).astype(int).tolist())
                red_eaxes = tuple(np.array(red_eaxes).astype(int).tolist())

                e = ecenter, eaxes, eangle
                rede = ecenter,red_eaxes, eangle
                
                cv2.circle(ims[-1], ecenter, 2, cols[j], -1)
                cv2.putText(ims[j], '%i'%k, chull, 0, 0.55, cols[j], thickness=2)

                # Draw ellipses at last image
                # cv2.ellipse(ims[-1], rede, cols[j], 1)
                cv2.circle(ims[-1], ecenter, 10, cols[j], 1)
                
                # Draw ellipses at each scale
                cv2.ellipse(ims[j], e, cols[j], 1)
                
            if octave == 4:
                for k,hull in enumerate(hulls):
                    pass
                    # pts = contours_from_endpoints(hull.reshape(-1,2),10)
                    # cv2.polylines(ims[j], hulls, 1, (0, 255, 0), thickness=1)
                    # cv2.putText(ims[j], '%i'%k, tuple(np.mean(hull.reshape(-1,2),axis=0).astype(int).tolist()), 0, 0.55, (255,0,0), thickness=2)

                    # for pt in pts:
                    #     cv2.circle(ims[j], tuple(pt.astype(int).tolist()), 2, (255,255,0),-1)

                # # Match contours
                # sc1 = SC()
                # sc2 = SC()
                
                # match = np.eye(len(hulls))
                # for h1, hulls1 in enumerate(hulls):
                #     pts1 = contours_from_endpoints(hulls1.reshape(-1,2), 10)
                #     bh1,_ = sc1.compute(np.vstack((pts1,pts1[-1])))
                #     for h2, hulls2 in enumerate(hulls):
                #         if h1 == h2: continue
                #         pts2 = contours_from_endpoints(hulls2.reshape(-1,2),10)
                #         bh2,_ = sc2.compute(np.vstack((pts2,pts2[-1])))
                        
                #         match[h1,h2] = euclidean(bh1[-1],bh2[-1]) #cv2.matchShapes(pts1, pts2, 3, parameter=0 )
                        
                # plt.matshow(match)
                # print match
                # # areas = [ (cv2.contourArea(hull), np.mean(hull.reshape(-1,2), axis=0).tolist()) for hull in hulls ]
                 
                # print 'Hull: ', sorted(areas, key=lambda x: x[0])

        plt.figure(5)
        plot_utils.imshow(np.hstack(ims))
        
    def visualize_regions(self, frame, regions):
        out = np.array(frame.rgb, copy=True)
        # for p in regions:
        #     col = np.random.randint(0,256,3)
        #     for pt in p:
        #         cv2.circle(out, tuple((pt[0],pt[1])), 1, tuple(col.tolist())) # , thickness=1, lineType=cv2.LINE_AA)

        normalize = lambda x : x / \
        np.tile(np.sqrt(np.sum(np.square(x),axis=1)).reshape(-1,1), (1,2))

        for p in regions:
            col = np.random.randint(0,256,3)
            hull = [cv2.convexHull(p.reshape(-1,1,2))]

            for pt in p:
                if pt[0] > out.shape[1] or pt[1] > out.shape[0]: continue
                out[pt[1],pt[0],:] = col
                # cv2.circle(out, tuple((pt[0],pt[1])), 1, tuple(col.tolist()))
                # cv2.circle(im, tuple((pt[0],pt[1])), 1, tuple(col.tolist()))

            # v1 = np.roll(pall, 1, axis=0) - pall;
            # v2 = np.roll(pall, -1, axis=0) - pall;
            # v1,v2 = normalize(v1), normalize(v2)
            # pin = []
            # for j in range(v1.shape[0]):
            #     th = math.acos(np.dot(v1[j],v2[j]))
            #     v = (v1[j] + v2[j]) * 0.5;
            #     if th > np.pi - th: v = -v
            #     pin.append(pall[j] - 100 * v)
            # pin = np.vstack(pin)

            # print v1, v2, pall
            # cv2.polylines(out, [pall.astype(np.int)], 1, tuple(col.tolist()), thickness=2)
            # cv2.fillPoly(out, [pall.astype(np.int)], tuple(col.tolist()),
            #              lineType=cv2.LINE_AA)
            # cv2.polylines(out, [pall.astype(np.int)], 1, tuple(col.tolist()), thickness=2, shift=10)
            # break
        # hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
        # cv2.polylines(out, hulls, 1, (0, 255, 0), thickness=2)

                        
        normals = ((frame.normals + 1) * 128).astype(np.uint8)
        plot_utils.imshow(normals)

        
        plot_utils.imshow(out)
        
        # labels = slic.slic(frame.rgb, var=100, compactness=10,
        #                     op_type=slic.FOR_GIVEN_SUPERPIXEL_SIZE, color_conv=cv2.COLOR_RGB2LAB)

        # # Main description
        # self.description = dict()

        # # Compute centers of superpixels
        # try: 
        #     centers = compute_centroid(labels, mask=frame.depth_mask);
        # except AttributeError:
        #     print 'Depth Mask unavailable'
        #     return
        
        # # Compute mean_img per label
        # try :
        #     mean_rgb, self.rgb_map = compute_mean(frame.rgb, labels,
        #                                                mask=frame.depth_mask, compute_mean_img=False)
        # except AttributeError:
        #     print 'Depth Mask unavailable'
        #     return
    
        # # Compute mean_normals_img per label
        # try :
        #     mean_normals, self.normals_map = compute_mean(frame.normals, labels,
        #                                                   mask=frame.normals_mask, compute_mean_img=False)
        # except AttributeError:
        #     print 'Normals/Normals Mask unavailable'
        #     return

                    
        # # Convert to R2 - Keypoints
        # self.R2_map = dict([ (l,cv2.KeyPoint(c[0],c[1],_size=1,_class_id=l,_response=1.))
        #                      for l,c in centers.iteritems() ]) # node_id : [keypoint - R2]

        # # Convert R3 - 3D points
        # self.R3_map = dict([ (l,frame.X[int(c[1]),int(c[0]),:])
        #                      for l,c in centers.iteritems() ]) # node_id : [3D point - R3]

        # # R3 x SO(2) map = R3_map + normals_map
        # self.R3SO2_map = dict()
        # for l,pt in self.R3_map.iteritems():
        #     if l in self.normals_map:
        #         self.R3SO2_map[l] = np.hstack((pt,self.normals_map[l]))
                                         
        # # Compute connectivity graph on labels
        # self.neighbors_map = slic.build_node_graph(labels); # node_id : [neighbors_node_list]

        # # Prune graph connectivity by R3 (ball) HEURISTICS
        # prune_funcs = []
        # prune_funcs.append(lambda pt1,pt2 : bool(pt1[2] > 0.5 and pt2[2] > 0.5))
        # prune_funcs.append(lambda pt1,pt2 : (not np.any(np.isnan(pt1))) and (not np.any(np.isnan(pt2))))
        # prune_funcs.append(lambda pt1,pt2 : bool(euclidean(pt1,pt2) < 0.20))
        # self.neighbors_map_with_depth = prune_graph(self.neighbors_map, self.R3_map, prune_funcs)
        
        # # Compute Descriptors from R2_map
        # # (ensure keypoints have a class_id associated with them)
        # self.feat_desc = desc_utils.PointDescMap(frame.rgb, self.R2_map,
        #                                   graph=self.neighbors_map_with_depth)
                
        # # Visualize mean_normals_img per label
        # try :
        #     pass
        # except AttributeError:
        #     print 'Normals unavailable; Failed to visualize'

        # # Visualize mean_img per label
        # try :
        #     pass
        # except AttributeError:
        #     print 'Normals unavailable; Failed to visualize'


        # # Visualize delaunay triangulation
        # self.visualize_delaunay_triangulation();

        # # Visualize surface normals
        # self.visualize_surface_normals();






        
            
            
                # # Ellipse params
                # center, axes, angle = e
                # r = np.sqrt(np.sum(np.square(np.array(axes, np.float32) * 0.5)));
                # corners = [ [np.maximum(0, center[0] - r), np.maximum(0, center[1] - r)],
                #             [np.maximum(0, center[0] - r), np.minimum(h, center[1] + r)],
                #             [np.minimum(w, center[0] + r), np.minimum(h, center[1] + r)],
                #             [np.minimum(w, center[0] + r), np.maximum(0, center[1] - r)] ]
                # corners = np.vstack(corners)

                # x1,x2 = np.min(corners[:,0]).astype(int), np.max(corners[:,0]).astype(int)
                # y1,y2 = np.min(corners[:,1]).astype(int), np.max(corners[:,1]).astype(int)
                # print x1,x2, y1,y2

                # # Original Image
                # plt.figure(1)
                # print 'Center: ', center
                # im = np.array(frame.rgb[y1:y2,x1:x2], copy=True);
                # cref = (center[0] - x1, center[1] - y1)
                # e = cref, axes, angle
                # cv2.ellipse(im, e, (0,255,0), 2);
                # plot_utils.imshow(im)

                # # Image warp
                # scale = 1; # 128.0 / np.maximum(x2-x1, y2-y1)
                # rw, rh = int((x2-x1)*scale), int((y2-y1)*scale)
                # print rw, rh
                # T = cv2.getRotationMatrix2D((rw/2,rh/2), angle, scale)
                
                # plt.figure(2)
                # im_warped = cv2.warpAffine(im, T, dsize=(rw,rh));
                # plot_utils.imshow(im_warped)
                # print 'imw_warped: ', im_warped.shape
                
                # return
                
                # ims.append(im_warped)
                # mask = cv2.resize(im_warped, (0,0), fx=1.0, fy=s, interpolation=cv2.INTER_NEAREST)
        # ims = ims[:16]
        # w = np.ceil(np.sqrt(len(ims)))
        # h = np.ceil(len(ims)/w)
        # mosaic = np.hstack(ims)
        # plt.figure(10)
        # plot_utils.imshow(mosaic)
        # print self.mser_description
