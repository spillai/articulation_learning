import numpy as np

def scale_points(X, scale): 
    mu_X = np.mean(X, axis=0)
    dX = X - mu_X
    dX_mag = np.linalg.norm(dX, axis=1)[:,np.newaxis]
    dX = dX / dX_mag
    return mu_X + np.multiply(dX, dX_mag * scale)

def contours_from_endpoints(endpts, quantize=4): # quantize in pixels
    v = np.array(np.roll(endpts, 1, axis=0) - endpts, np.float32);
    vnorm = np.sqrt(np.sum(np.square(v), axis=1))
    v[:,0] = np.multiply(v[:,0],1.0/vnorm)
    v[:,1] = np.multiply(v[:,1],1.0/vnorm)

    # print vnorm, np.arange(0, vnorm[0], 4)
    
    out = []
    for j in range(len(v)):
        out.append(np.vstack([endpts[j] + mag * v[j] for mag in np.arange(0,vnorm[j],quantize)]))
    return np.vstack(out).astype(int)
