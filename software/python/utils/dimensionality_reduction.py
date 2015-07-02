import numpy as np
import matplotlib as mpl
from pose_utils import wrap_angle
from collections import namedtuple
from sklearn import decomposition as skdecomposition
from rigid_transform import Pose, Quaternion
fit_stats = namedtuple('fit_stats', ['residual', 'n_comps'])
def randomized_pca_compression(X, explained_variance=0.95): 
    
    # Randomized PCA on position 
    n,m = X.shape
    rpca_pos = skdecomposition.RandomizedPCA(copy=True).fit(X+1e-5) # add eps to prevent nan

    # cumsum of Explained variance at most 0.95
    n_comps = np.argmax(np.cumsum(rpca_pos.explained_variance_ratio_)>explained_variance)+1
    print 'POS: %s, NCOMPS: %i' % (rpca_pos.explained_variance_ratio_, n_comps)

    # Only retain top n_comps (as explained variance ratio)
    Y = rpca_pos.transform(X); Y[:,n_comps:] = 0;
    # Transform back to original space
    Xred = rpca_pos.inverse_transform(Y)
    return fit_stats(np.linalg.norm(Xred-X)/(n-1), n_comps), Xred

def reduce_poses(poses): 
    # Convert to wrapped angle 
    X = [ np.hstack([ p.tvec, 
                      wrap_angle(
                          np.array( p.quat.to_roll_pitch_yaw() ) 
                      ) 
                  ]) for p in poses 
    ]
    X = np.vstack(X)

    # Randomize PCA on position
    xyz_red_stats, poses_xyz_red = randomized_pca_compression(X[:,:3], 0.95)
    rpy_red_stats, poses_rpy_red = randomized_pca_compression(X[:,-3:])


    poses_red = [Pose(-1, Quaternion.from_rpy(rpy), xyz)
                      for xyz,rpy in zip(poses_xyz_red, poses_rpy_red)]
    return poses_red


