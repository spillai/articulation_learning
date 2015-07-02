import cv2
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import Delaunay
from matplotlib import delaunay

class PointDesc(object):
    """Describe feature spatially and via image descriptor"""
    def __init__(self, keypt=None, desc=None, axes=None):
        self.keypt = keypt
        self.axes = axes
        self.set_description(desc)
        
    def set_description(self, desc):
        if getattr(desc, 'ndim', 0) == 2: 
            self.desc = desc[0,:]
            self.valid = np.sum(self.desc) != 0
        else:
            self.desc = desc
            self.valid = False
        
    def __repr__(self):
        return 'PointDesc=> Valid: %s | KeyPt: %s | Desc: %s' % (self.valid, self.keypt.pt, self.desc)


class PointDescMatcher:
    def __init__(self, pd_map, graph=None):
        """
        Init desc. list with labels, and keypoints from feature detection.
        On init, we compute the descriptor and a kd-tree for future matching
        needs. Ideally, this class is added to a dictionary with utime as a key
        in order to perform queries of nearest neighbor (both spatially and in
        descriptor feature space)
        Optional: graph that maps id to neighboring ids
        """

        # Appropriately convert desc map
        self._map = {}
        if (isinstance(pd_map, list)):
            self._map = dict(pd_map)
        else:
            self._map = pd_map
        
        # Given point description map
        # 1. Build kd-tree for spatial locations
        # 2. Build LSH hash for description
        pts = np.array([[pd.keypt.pt[0], pd.keypt.pt[1]] for pd in self._map.values()],
                       dtype=np.float32);

        # Add keypoints to kdtree
        self.spatial_tree = NearestNeighbors(n_neighbors=20, algorithm='kd_tree',
                                             leaf_size=10, p=2, warn_on_equidistant=False)
        self.spatial_tree.fit(pts)
        print 'Spatially described pts: ', pts.shape

        
                
    # def get_description(self):
    #     return self.desc_map

    # def get_match(self, pts):
    #     nn_dists,nn_inds = self.spatial_tree.kneighbors(pts);
    #     return np.vstack([self.ids[inds] for inds in nn_inds])

    # def get_closest_match_description(self, ptdesc):
    #     pts,descs = zip(*ptdesc.values())
    #     nn_dists,nn_inds = self.spatial_tree.kneighbors(pts);
    #     print nn_dists
    #     desc_matcher = cv2.BFMatcher(cv2.NORM_L2)
        
    #     all_dmatches = []
    #     for j,inds in enumerate(nn_inds):
    #         pids = self.ids[inds]
    #         dmatches = []
    #         for pid in pids:
    #             if pid not in self.desc_map: continue
    #             d = desc_matcher.match(descs[j], self.desc_map[pid])[-1].distance
    #             dmatches.append(d)
    #         all_dmatches.append(dmatches)
    #     print 'Matches: ', all_dmatches;
    #     return np.vstack([self.ids[inds] for inds in nn_inds])


class PointDescMap:
    def __init__(self, img, keypts_map, graph=None):
        """
        Init desc. list with labels, and keypoints from feature detection.
        On init, we compute the descriptor and a kd-tree for future matching
        needs. Ideally, this class is added to a dictionary with utime as a key
        in order to perform queries of nearest neighbor (both spatially and in
        descriptor feature space)
        Optional: graph that maps id to neighboring ids
        """

        # Initialize Feature Descriptor
        desc_extractor = cv2.DescriptorExtractor_create("OpponentSIFT") # cv2.SIFT()
        
        # Compute description
        bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR);
        _, desc = desc_extractor.compute(bgr_img, keypts_map.values());

        # Propagate labels 
        self.ids = np.array([k for k in keypts_map.keys()])
        keypts = np.array([[kpt.pt[0], kpt.pt[1]] for kpt in keypts_map.values()],
                                dtype=np.float32)
        
        # Only describe with valid description
        inds, = np.where(np.sum(desc,axis=1)>0)
        self.desc_map = dict(zip(self.ids[inds], list(desc[inds])))

        # Build spatial graph for valid keypoints
        print 'Described pts: ', len(self.desc_map.keys())
        self.spatial_graph = graph;
        for l in self.desc_map.keys():
            if not l in self.spatial_graph.keys():
                del self.desc_map[l]
        print 'Described pts with spatial support: ', len(self.desc_map.keys())

        # Add keypoints to kdtree
        self.spatial_tree = NearestNeighbors(n_neighbors=20, algorithm='kd_tree',
                                             leaf_size=10, p=2, warn_on_equidistant=False)
        self.spatial_tree.fit(keypts)

    def get_description(self):
        return self.desc_map

    def get_match(self, pts):
        nn_dists,nn_inds = self.spatial_tree.kneighbors(pts);
        return np.vstack([self.ids[inds] for inds in nn_inds])

    def get_closest_match_description(self, ptdesc):
        pts,descs = zip(*ptdesc.values())
        nn_dists,nn_inds = self.spatial_tree.kneighbors(pts);
        print nn_dists
        desc_matcher = cv2.BFMatcher(cv2.NORM_L2)
        
        all_dmatches = []
        for j,inds in enumerate(nn_inds):
            pids = self.ids[inds]
            dmatches = []
            for pid in pids:
                if pid not in self.desc_map: continue
                d = desc_matcher.match(descs[j], self.desc_map[pid])[-1].distance
                dmatches.append(d)
            all_dmatches.append(dmatches)
        print 'Matches: ', all_dmatches;
        return np.vstack([self.ids[inds] for inds in nn_inds])
    
        # # Build pointdesc
        # for keypt,d in zip(keypts,np.vsplit(desc,desc.shape[0])):
        #     pd = PointDesc(keypt,d)
        #     if pd.valid: self.dlist.append(pd)
        # print 'Valid keypts: %i out of %i' % (len(self.dlist), len(keypts))
        # self.keypts_feats = np.array([[pd.keypt.pt[0], pd.keypt.pt[1]] for pd in self.dlist],
        #                              dtype=np.float32)
        # self.desc_feats = np.array([pd.desc for pd in self.dlist], dtype=np.float32)

        # self.spatial_del = Delaunay(self.keypts_feats)
        # self.spatial_graph = None
        # for 
                
        # self.spatial_tree = NearestNeighbors(n_neighbors=, algorithm='kd_tree',
        #                                      leaf_size=10, p=2, warn_on_equidistant=False)
        # self.spatial_tree.fit(self.keypts_feats)

        # print 'Desc shape: ', self.desc_feats.shape
        
        # self.desc_tree = NearestNeighbors(algorithm='ball_tree',
        #                                     leaf_size=10, p=2, warn_on_equidistant=False)
        # self.desc_tree.fit(self.desc_feats)

        # # print self.spatial_tree.radius_neighbors(self.keypts_feats, 50, return_distance=True)
        # print self.desc_tree.kneighbors(self.desc_feats, 5, return_distance=True)
        

        
        
