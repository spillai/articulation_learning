import time, os

import lcm; lc = lcm.LCM();
import vs
import bot_lcmgl as lcmgl
import cv2

import numpy as np
import matplotlib as mpl
import matplotlib.pylab as plt

from sklearn import manifold as skmanifold
from sklearn import cluster as skcluster
from sklearn import metrics as skmetrics

from scipy import cluster as spcluster
from scipy import spatial as spspatial

from bot_core import image_t

def visualize_linkage(Y, linkage): 
    # ===== Visualize hierarchical clustering tree  ====
    fig = plt.figure()
    axdendrob = fig.add_axes([0.09,0.1,0.2,0.8])
    dendrogram = spcluster.hierarchy.dendrogram(linkage, orientation='right', count_sort='ascending', show_leaf_counts=True);
    axdendrob.set_xticks([])
    axdendrob.set_yticks([])
    
    axdendrot = fig.add_axes([0.3,0.92,0.6,0.2])
    dendrogram = spcluster.hierarchy.dendrogram(linkage);
    axdendrot.set_xticks([])
    axdendrot.set_yticks([])

    # ===== Visualize distance matrix  ====
    axmatrix = fig.add_axes([0.3,0.1,0.6,0.8])
    inds = dendrogram['leaves']
    D = Y[inds,:]
    D = D[:,inds]
    im = axmatrix.matshow(D, aspect='auto', origin='lower')
    axmatrix.set_xticks([])
    axmatrix.set_yticks([])
    
    # Colorbar 
    axcolor = fig.add_axes([0.91,0.1,0.02,0.8])
    cbar = plt.colorbar(im, cax=axcolor)

def imshow(img, pattern='rgb'):
    plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0, wspace=0.01, hspace=0.01)
    if pattern=='bgr': 
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else: 
        plt.imshow(img)
    plt.axis('off')    

# def matshow(img):
#     #plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0, wspace=0.01, hspace=0.01)
#     nr, nc = img.shape
#     extent = [-0.5, nc-0.5, nr-0.5, -0.5]
#     plt.matshow(img, extent=extent, origin='upper') 
    
def cvshow(img, wait=0):
    out = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow('Frame', out);
    cv2.waitKey(wait)
    
# def matshow(img):
#     plt.matshow(img, origin='lower')
#     plt.axis('off')    


def matshow_by_labels(A, labels): 
    label_votes = get_label_votes(labels)

    idx = 0;
    D = np.zeros_like(A)
    for label,votes in label_votes: 
        inds, = np.where(labels == label)
        sz = len(inds)
        ninds = np.arange(idx,idx+sz);
        idx += sz
        print ninds, inds
        D[np.ix_(ninds,ninds)] = A[np.ix_(inds,inds)]

    # ===== Visualize distance matrix  ====
    plt.matshow(D, aspect='auto', origin='lower')
    plt.axis('off')    

def matshow_by_top_labels(A, labels, only_labels=[]): 
    label_votes = get_label_votes(labels)

    count = 0; 
    for label,votes in label_votes: 
        if label in only_labels: 
            count += votes

    if not count: 
        print 'Reverting to std. form'
        matshow_by_labels(A, labels)
        return


    print 'Creating %ix%i mat' % (count,count)
    idx = 0;
    D = np.zeros((count,count))
    for label,votes in label_votes: 
        if label not in only_labels: continue
        inds, = np.where(labels == label)
        sz = len(inds)
        ninds = np.arange(idx,idx+sz);
        idx += sz
        D[np.ix_(ninds,ninds)] = A[np.ix_(inds,inds)]

    # ===== Visualize distance matrix  ====
    plt.matshow(D, aspect='auto', origin='lower')
    plt.axis('off')    

    return D

def np2tuple(arr):
    return tuple(arr.astype(int).tolist())

def get_label_votes(labels): 
    # Count number of instances of labels
    un_labels,un_inv = np.unique(labels, return_inverse=True)
    label_count = np.bincount(un_inv)
    label_votes = zip(un_labels, label_count)
    label_votes.sort(key=lambda x: x[1], reverse=True)

    return label_votes

def send_image(ch, im):
    global lc
    out = image_t()
    h,w,c = im.shape
    out.width, out.height = w, h
    out.row_stride = w*c
    out.pixelformat = image_t.PIXEL_FORMAT_BGR
    out.size = w*h*c
    out.data = im.tostring()
    out.utime = long(time.time() * 1e6)
    out.nmetadata = 0
    # out.metadata = None
    print 'shape: ', im.shape
    lc.publish(ch, out.encode())
