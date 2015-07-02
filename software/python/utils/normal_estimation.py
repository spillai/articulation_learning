import numpy as np
from sklearn.cluster import KMeans
# from numba.decorators import jit, autojit

# par_integral_normal_estimation = autojit(integral_normal_estimation)

def estimate_normal(X):
    r,c = X.shape;    
    if r < 3: return np.array([np.nan]*3)
    Xs = X - np.mean(X, axis=0);
    U, s, V = np.linalg.svd(Xs, full_matrices=True)
    normal = V[:,2]
    return normal     

def normalize(x):
    norm = np.sqrt(np.sum(x**2,axis=2))
    norm = np.repeat(norm[:,:,np.newaxis], 3, axis=2)
    return np.divide(x, norm);
  
def estimate_normals(X, pts): 

    # Build randomized triangles with zero mean
    np.random.seed(seed=1)
    tri_off = np.random.normal([0,0], scale=5, size=(12,2)).astype(int)

    dconst = 1. / 576.09757860            
    ns, ds = [], []

    for xy in pts:
        # For each pt, sample 5 triangles, 15 points
        # pts = np.hstack([np.random.randint(xy[0]-5,xy[0]+5,size=(21,1)),
        #                  np.random.randint(xy[1]-5,xy[1]+5,size=(21,1))])
        pts = xy.astype(int) + tri_off;
        pts3d = X[tuple([pts[:,1], pts[:,0]])]

        # Split to 3 sets of lines
        tri3d = np.array_split(pts3d, 3)

        # Compute surface tangents
        v1, v2 = np.vstack(tri3d[0] - tri3d[1]), np.vstack(tri3d[0] - tri3d[2]);
        v1norm = np.sqrt(np.sum(np.square(v1), axis=1))
        v2norm = np.sqrt(np.sum(np.square(v2), axis=1))
        v1 *= 1. / np.tile(v1norm[:,np.newaxis], [1,3])
        v2 *= 1. / np.tile(v2norm[:,np.newaxis], [1,3])

        # Compute surface normals
        ns_ = [np.cross(v1_,v2_) for v1_,v2_ in zip(v1,v2)]
        nsnorm_ = np.vstack([n / np.linalg.norm(n) for n in ns_])

        # Remove any nans
        inds,_ = np.where(~np.isnan(nsnorm_).any(axis=1, keepdims=True))
        nsnorm_ = nsnorm_[inds]

        # Continue if not enough size
        if len(nsnorm_) < 2: 
            ns.append([np.nan, np.nan, np.nan])
            ds.append(np.nan)
            continue

        # Flip normals that point away from camera 
        flipinds, = np.where(nsnorm_[:,2] > 0)
        nsnorm_[flipinds] = nsnorm_[flipinds] * -1.0

        ns.append(np.mean(nsnorm_, axis=0))
        ds.append(np.mean(pts3d[:,2], axis=0))
        # print pts3d
        # # K-means (with k=2), pick most dominant cluster
        # kmeans = KMeans(n_clusters=2, init='k-means++').fit(nsnorm_)
        # # print 'NSNORM: \n', nsnorm_
        # label = np.argmax(np.bincount(kmeans.labels_.astype(int)))
        # inds, = np.where(kmeans.labels_.astype(int) == label)
        # # print 'LABELS: \n', kmeans.labels_.astype(int)
        # # print 'TOP LABEL: %i \n%s' % (label,nsnorm_[inds])
        # # print 'MEAN: \n', np.mean(nsnorm_[inds], axis=0)
        # ns.append(np.mean(nsnorm_[inds], axis=0))

    return ns, ds
        

def integral_normal_estimation(depth, sz, downsample=1):
    """
    Estimates the normals block wise from an organized point cloud
    """
    # (xnan,ynan,znan) = np.where(depth==0)
    # print znan    
    # nan = np.zeros_like(depth, dtype='bool');
    #nan[np.ix_(xnan,ynan)] = True;

    h, w, c = depth.shape
    depth = depth[np.ix_(range(0,h,downsample), range(0,w,downsample))];    
    h, w, c = depth.shape
    normals = np.zeros_like(depth)
    mean = np.zeros_like(depth)

    assert c == 3
    assert sz % 2 == 0

    dintegral = np.cumsum(np.cumsum(depth.astype('double'), axis=1), axis=0);
    dintegral = np.hstack((np.zeros((h,1,3)), dintegral));
    dintegral = np.vstack((np.zeros((1,dintegral.shape[1],3)), dintegral))    

    xs = np.arange(sz/2,w-sz/2);
    ys = np.arange(sz/2,h-sz/2);
    
    mean[np.ix_(ys,xs)] = (dintegral[np.ix_(ys+(sz/2),xs+(sz/2))] +
                            dintegral[np.ix_(ys-(sz/2),xs-(sz/2))] -
                            dintegral[np.ix_(ys+(sz/2),xs-(sz/2))] -
                            dintegral[np.ix_(ys-(sz/2),xs+(sz/2))]) / (sz*sz)

    normals[np.ix_(ys,xs)] = normalize(np.cross(mean[np.ix_(ys,xs-sz/2)] - mean[np.ix_(ys,xs+sz/2)],
                                mean[np.ix_(ys+sz/2,xs)] - mean[np.ix_(ys-sz/2,xs)]))

    
    return normals

    # for y in range(size/2,h-size/2):
    #     for x in range(size/2,w-size/2):
    #         patch = depth[y-size/2:y+size/2, x-size/2:y+size/2,:]
    #         X = np.hstack((np.reshape(patch[:,:,0], (-1, 1)),
    #                         np.reshape(patch[:,:,1], (-1,1)),
    #                         np.reshape(patch[:,:,2], (-1,1))))
    #         mean[y,x,:] = np.mean(X, axis=0)
            #normals[y,x,:] = estimate_normal(X)
