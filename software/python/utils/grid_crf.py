#!/usr/bin/python
import time

import numpy as np
import matplotlib as mpl
import matplotlib.pylab as plt

def make_graph(grid):
    st = time.time()    
    # get unique labels
    vertices = np.unique(grid)
 
    # map unique labels to [1,...,num_labels]
    reverse_dict = dict(zip(vertices,np.arange(len(vertices))))
    grid = np.array([reverse_dict[x] for x in grid.flat]).reshape(grid.shape)
   
    # create edges
    down = np.c_[grid[:-1, :].ravel(), grid[1:, :].ravel()]
    right = np.c_[grid[:, :-1].ravel(), grid[:, 1:].ravel()]
    all_edges = np.vstack([right, down])
    all_edges = all_edges[all_edges[:, 0] != all_edges[:, 1], :]
    all_edges = np.sort(all_edges,axis=1)
    num_vertices = len(vertices)
    edge_hash = all_edges[:,0] + num_vertices * all_edges[:, 1]
    # find unique connections
    edges = np.unique(edge_hash)
    # undo hashing
    edges = [[vertices[x%num_vertices],
              vertices[x/num_vertices]] for x in edges]

    # compute region centers:
    gridx, gridy = np.mgrid[:grid.shape[0], :grid.shape[1]]
    centers = dict()
    for v in vertices:
        centers[v] = [gridy[grid == v].mean(), gridx[grid == v].mean()]


    print 'graph constrution took %f s' % (time.time() - st)
    return vertices, edges, centers

