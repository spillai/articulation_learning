import cv2

import numpy as np
import matplotlib as mpl
import matplotlib.pylab as plt

from scipy.cluster.vq import kmeans, vq
from scipy.spatial import KDTree

from sklearn.decomposition import PCA

class Vocabulary(object):
    """
    From PCV jan erik solem's book
    """
    def __init__(self, query='vq'):
        self.voc = []
        self.K, self.D = 0, 0
        self.query = query
        self.kdtree = None
        self.pca = None        

    def train(self,descriptors, k=100, subsampling=10, d=None, return_words=False):
        """ Train a vocabulary from NxD descriptors using k-means with k number of words. 
            Subsampling of training data can be used for speedup. """

        print 'Training vocab'
        N, D = descriptors.shape

        if d is None or d == D: 
            d = D
        else: 
            # PCA: reduce dim. 
            self.pca = PCA(n_components=d, whiten=True)

            descriptors = self.pca.fit_transform(descriptors)
            print 'Dimensionality reduction: (%i,%i) -> %s' % (N,D, descriptors.shape)
            
        # k-means: last number determines number of runs
        # N x D => K x D (not necessarily k (around k))
        self.voc, distortion = kmeans(descriptors[::subsampling,:], k_or_guess=k, iter=1)
        self.K, self.D = self.voc.shape

        # For kd-tree query
        if self.query == 'kdtree': 
            print 'Constructing kD-tree for scalable search'
            self.kdtree = KDTree(data=self.voc)
        print 'Vocab trained!'

        # Optionally return the words
        if return_words: 
            return self.get_words(descriptors)
 
    def reduce(self, descriptors): 
        # describe by words instead of full data
        w = self.get_words(descriptors)
        return self.voc[w]
    
    def get_words(self,descriptors, reduce_dim=True):
        """ Convert descriptors to words. """

        # Project to lower dim before quantizing
        if reduce: 
            descriptors = self.pca.fit_transform(descriptors) if self.pca is not None else descriptors
        if self.query == 'kdtree': 
            assert(self.kdtree is not None)
            # kd-tree query for nearest center
            d, w = self.kdtree.query(descriptors, k=1)
            return w
        elif self.query == 'vq': 
            # vq for nearest center
            w, d = vq(descriptors,self.voc)
            return w
        else: 
            raise RuntimeError('Unknown query type %s' % query)

def get_color_by_label(labels, default='b'): 
    return plt.cm.gist_rainbow((labels - np.min(labels)) * 1.0 / (np.max(labels) - np.min(labels))) if labels is not None else default    

def viz_words(desc, labels=None): 
    c = get_color_by_label(labels)
    data = PCA(n_components=2, whiten=True).fit_transform(desc)
    plt.scatter(data[:,0], data[:,1], c=c)

def im_desc(im, kpts=None, mask=None): 
    # desc = cv2.SIFT(nfeatures=500, nOctaveLayers=6, 
    #                 contrastThreshold=0.08, edgeThreshold=10, sigma=1.6)
    desc = cv2.SIFT()
    if kpts is not None: 
        return desc.compute(im, kpts)
    return desc.detectAndCompute(im, mask=mask)

def im_mser(im): 
    mser = cv2.MSER()
    regions = mser.detect(im, None)
    hulls = [np.vstack(cv2.convexHull(p.reshape(-1, 1, 2))) for p in regions]
    return hulls



def desc_viz(self, im, kp): 
    viz = cv2.drawKeypoints(im, kp, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow('viz', viz)
    cv2.waitKey(10)

def desc_viz_ids(im, kp, ids): 
    viz = im.copy()
    ids = (np.array(plt.cm.jet(ids * 1.0 / np.max(ids))) * 255.0).astype(int)
    for pt, pid in zip(kp, ids): 
        cv2.circle(viz, tuple(np.array(pt.pt).astype(int).tolist()), 
                   int(pt.size)/2, tuple(pid.tolist()), 
                   1, lineType=cv2.LINE_AA);

    cv2.imshow('viz', viz)
    cv2.waitKey(10)



