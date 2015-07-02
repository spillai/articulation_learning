import numpy as np
import transformations as tf
from utils.rigid_transform import RigidTransform

from math import radians
from cmath import rect, phase

def wrap_angle(theta): 
    return np.fmod(np.array(theta)+2*np.pi,2*np.pi)

def wrap_angle_pi(theta): 
    inds = (theta > np.pi); theta[inds] = theta[inds] - 2*np.pi;
    return theta
    
# def wrap_angle2(rad): 
#     return phase(rect(1, rad))

def wrap_angle2(rads): 
    return [phase(rect(1, rad)) for rad in rads]

def median_pose(poses): 
    """HACK: Compute the mean pose of set of tfs"""
    mu_pos = np.median(poses[:,:3], axis=0)
    
    rpy_poses = [wrap_angle(tf.euler_from_quaternion(pose)) for pose in poses[:,-4:]]
    mu_rpy = np.median(np.vstack(rpy_poses), axis=0);
    mu_quat = tf.quaternion_from_euler(mu_rpy[0], mu_rpy[1], mu_rpy[2])
    return np.hstack([mu_pos,mu_quat])

def mean_angle(rads):
    return phase(sum( [rect(1, r) for r in rads])/len(rads))

def mean_rpy(rpy): 
    """Compute the mean pose of set of tfs"""
    mu_rpy = [mean_angle(rpy[:,0]), mean_angle(rpy[:,1]), mean_angle(rpy[:,2])]
    return mu_rpy[0], mu_rpy[1], mu_rpy[2]

def mean_pose(poses): 
    """Compute the mean pose of set of tfs"""
    mu_pos = np.mean(np.vstack([pose.tvec for pose in poses]), axis=0)
    rpy = np.vstack([ np.array(pose.quat.to_roll_pitch_yaw()) for pose in poses])
    mu_rpy = mean_rpy(rpy)
    # print 'RPY: ', rpy, mu_rpy
    return RigidTransform.from_roll_pitch_yaw_x_y_z(mu_rpy[0], mu_rpy[1], mu_rpy[2], 
                                                    mu_pos[0], mu_pos[1], mu_pos[2])

if __name__ == "__main__": 
    for angles in [[350, 10], [90, 180, 270, 360], [10, 20, 30]]:
        print('The mean angle of', map(radians, angles), 'is:', round(mean_angle(map(radians, angles)), 12), 'radians')

# def mean_pose(poses): 
#     """HACK: Compute the mean pose of set of tfs"""
#     mu_pos = np.mean(poses[:,:3], axis=0)
    
#     rpy_poses = [tf.wrap_angle(tf.euler_from_quaternion(pose) for pose in poses[:,-4:]]
#     mu_rpy = np.mean(np.vstack(rpy_poses), axis=0);
#     mu_quat = tf.quaternion_from_euler(mu_rpy[0], mu_rpy[1], mu_rpy[2])
#     return np.hstack([mu_pos,mu_quat])
