#!/usr/bin/python
import numpy as np

import scipy as sp

import matplotlib as mpl
import matplotlib.pylab as plt
import matplotlib.mlab as mlab
import matplotlib.animation as animation

# from sklearn.neighbors import KernelDensity

# def plot_pose_pair_variation(ppd):
#     # ===== Viz histogram of pose-pair-normal distributions =====
#     # plt.subplot(221);
#     # n, bins, patches = plt.hist(np.hstack([v.normal for k,v in ppd.iteritems()]),
#     #                             20, normed=1, facecolor='green', alpha=0.75, range=(0,1))
#     # bincenters = 0.5*(bins[1:]+bins[:-1])
#     # y = mlab.normpdf(bincenters, 1, POSEPAIR_THETA_SIGMA)
#     # l = plt.plot(bincenters, y, 'r--', linewidth=1)
#     # plt.xlim((0,1))
#     # plt.grid(True);
#     # plt.title('Normal variation');
#     # plt.xlabel('Cosine distance (m)')
    
#     # plt.subplot(222);
#     # n, bins, patches = plt.hist(np.hstack([v.distance for k,v in ppd.iteritems()]),
#     #                             20, normed=1, facecolor='blue', alpha=0.75, range=(0,1))
#     # bincenters = 0.5*(bins[1:]+bins[:-1])
#     # y = mlab.normpdf(bincenters, 1, POSEPAIR_DISTANCE_SIGMA)
#     # l = plt.plot(bincenters, y, 'r--', linewidth=1)
#     # plt.xlim((0,0.5))
#     # plt.grid(True);
#     # plt.title('Distance variation');
#     # plt.xlabel('Euclidean distance (m)');

#     # plt.subplot(211);
#     # n, bins, patches = plt.hist(np.hstack([v.normal_pdf for k,v in ppd.iteritems()]),
#     #                             40, normed=1, facecolor='green', alpha=0.75, range=(0,1))
#     # bincenters = 0.5*(bins[1:]+bins[:-1])
#     # bincenters_ = np.linspace(bins[0],bins[-1],100)
#     # y = mlab.normpdf(bincenters_, 1, POSEPAIR_THETA_SIGMA)
#     # l = plt.plot(bincenters_, y, 'r--', linewidth=1)
#     # plt.xlim((0,1))
#     # plt.grid(True);
#     # plt.title('Normal variation');
#     # plt.xlabel('PDF of Cosine distance')
    
#     #plt.subplot(212);
#     n, bins, patches = plt.hist(np.hstack([v.distance_pdf for k,v in ppd.iteritems()]),
#                                 40, normed=1, facecolor='blue', alpha=0.75, range=(0,1))
#     bincenters = 0.5*(bins[1:]+bins[:-1])
#     bincenters_ = np.linspace(bins[0],bins[-1],100)    
#     y = mlab.normpdf(bincenters_, 1, POSEPAIR_DISTANCE_SIGMA)
#     l = plt.plot(bincenters_, y, 'r--', linewidth=1)
#     plt.xlim((0,1))
#     plt.grid(True);
#     plt.title('Distance variation');
#     plt.xlabel('PDF of Euclidean distance');
#     plt.ylabel('Num. of features');    

# ===== Viz kernel density of pose-pair =====================
def plot_pose_pair_density(ppd, kernel='gaussian', face='green'):
    mu, stddev = np.mean(ppd.distance_hist), np.std(ppd.distance_hist)
    stddev = max(0.05, stddev)
    # mind, maxd = np.amin(ppd.distance_hist), np.amax(ppd.distance_hist)
    mind, maxd = mu - 6*stddev, mu + 6*stddev
    kde = KernelDensity(kernel=kernel, bandwidth=0.01).fit(ppd.distance_hist[:,np.newaxis])

    xs = np.linspace(mind, maxd, 10000)[:,np.newaxis]
    ys = kde.score_samples(xs)

    plt.fill(xs, np.exp(ys), fc=face)
    plt.xlim((mu-0.1, mu+0.1))
    plt.grid(False);
    plt.xlabel('Gaussian KDE of relative displacement observations');
    plt.ylabel('Num. of features');    


# ===== Viz histogram of pose-pair-normal distributions =====
def plot_pose_pair_variation(ppd, face='green'):
    mu, stddev = np.mean(ppd.distance_hist), np.std(ppd.distance_hist)
    stddev = max(0.05, stddev)
    mind, maxd = mu - 6*stddev, mu + 6*stddev
    print mu, stddev
    n, bins, patches = plt.hist(ppd.distance_hist,
                                bins=np.arange(mind,maxd,stddev/8), facecolor=face, alpha=0.75) #, range=(0,1))
    bincenters = 0.5*(bins[1:]+bins[:-1])
    bincenters_ = np.linspace(bins[0],bins[-1],1000)    
    #y = mlab.normpdf(bincenters_, 1, POSEPAIR_DISTANCE_SIGMA)
    #l = plt.plot(bincenters_, y, 'r--', linewidth=1)

    plt.xlim((mu-4*stddev,mu+4*stddev))
    plt.grid(False);
    plt.xlabel('Histogram of relative displacement observations');
    plt.ylabel('Num. of features');    
    print ppd.distance_match


class PosePairDistribution:
    def __init__(self, data, id_pair, overlap_utimes_inds, 
                 distance_sigma, theta_sigma, verbose=False):

        # Evaluate for id_pair, and overlap_utimes
        jind,kind = id_pair
        self.overlap_utimes_inds = overlap_utimes_inds
       
        # Mean distance change
        self.distance = data.xyz[jind,overlap_utimes_inds,:] - \
                        data.xyz[kind,overlap_utimes_inds,:]

        self.distance = self.distance[~np.isnan(self.distance).any(axis=1)] 
        self.distance_hist = np.linalg.norm(self.distance, axis=1)

        self.zm_distance_hist = self.distance_hist - np.mean(self.distance_hist)
        self.distance_pdf = np.exp(-np.square(self.zm_distance_hist) / (2* (distance_sigma**2) ));

        self.distance_match = np.mean(np.sort(self.distance_pdf)[:len(self.distance_pdf)/4]) # np.mean(self.distance_pdf);
        # print self.distance_match, max(self.distance_hist)-min(self.distance_hist)

        # Mean angle change
        self.normal_hist = np.arccos(np.minimum(
            np.sum(np.multiply(data.normal[jind,overlap_utimes_inds,:],
                               data.normal[kind,overlap_utimes_inds,:]), axis=1), 1.)
        );
        self.normal_hist = self.normal_hist[~np.isnan(self.normal_hist)]
        self.zm_normal_hist = self.normal_hist - np.median(self.normal_hist);
        self.normal_pdf = np.exp(-np.square(self.zm_normal_hist) / (2* (theta_sigma**2) ));
        self.normal_match = np.mean(np.sort(self.normal_pdf)[:len(self.normal_pdf)/4]) # np.mean(self.normal_pdf); 
