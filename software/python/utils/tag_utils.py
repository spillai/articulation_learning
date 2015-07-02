import time

import cv2
import lcm
import vs
# import bot_lcmgl as lcmgl
import sysconfig
import numpy as np
import matplotlib as mpl
import matplotlib.pylab as plt

from rigid_transform import RigidTransform

# ===== Globals ====
lc = lcm.LCM()

# Kinect sensor    
kinect_pose = vs.obj_t();
kinect_pose.id = 1;
kinect_pose.x, kinect_pose.y, kinect_pose.z = 0.15, 0.2, 1.48;
kinect_pose.roll, kinect_pose.pitch, kinect_pose.yaw = -90.5 * np.pi / 180, 2.0 * np.pi / 180, -96 * np.pi / 180;

def draw(img, tags): 
    # ===== Visualize  ====
    for tag in tags: 
        tid,pts = tag.id, tag.p;
        cv2.line(img, tuple(pts[2,:]), tuple(pts[3,:]), (0,0,255), thickness=1)
        cv2.line(img, tuple(pts[2,:]), tuple(pts[1,:]), color=(0,255,0))
        c = tuple(np.mean(np.vstack((pts[2,:],pts[0,:])), axis=0))
        cv2.circle(img, c, 4, (255,0,0), 2)
        cv2.putText(img, ' %i'%tid, c, 0, 0.75, (255,0,0), thickness=2)        
    plt.imshow(img)

def construct_tf(utime, p1, p2, p3): 
    x1 = p1-p2;
    x1 /= np.linalg.norm(x1)

    y1 = p3-p2;
    y1 /= np.linalg.norm(y1)

    z1 = np.cross(x1,y1);
    z1 /= np.linalg.norm(z1)

    y1 = np.cross(z1,x1);
    y1 /= np.linalg.norm(y1)

    R = np.vstack((x1,y1,z1))
    T = np.eye(4)

    T[:3,:3] = R.T
    T[:3,3] = p2.T
    return RigidTransform.from_homogenous_matrix(T)

def compute_tag_pose(utime, img, cloud, tags): 
    tag_poses = []
    for tag in tags: 
        tid,pts = tag.id, tag.p;
        pose = construct_tf(utime, 
                            cloud[int(pts[3,1]),int(pts[3,0])],
                            cloud[int(pts[2,1]),int(pts[2,0])],
                            cloud[int(pts[1,1]),int(pts[1,0])]); 
        tag_poses.append((tid,pose))
    return tag_poses

id_generator = dict()
def get_unique_id(ch):
    if ch in id_generator:
        return id_generator[ch]
    else:
        uid = 10000 + len(id_generator.keys())
        id_generator[ch] = uid
        return uid
        
def publish_poses(pub_channel, poses, sensor_tf='KINECT'):
    # pose list collection msg
    pose_list_msg = vs.obj_collection_t();
    pose_list_msg.id = get_unique_id(pub_channel);
    pose_list_msg.name = pub_channel;
    pose_list_msg.type = vs.obj_collection_t.AXIS3D;
    pose_list_msg.reset = True;

    # fill out points
    pose_list_msg.objs = [];
    pose_list_msg.nobjs = len(poses);             

    # Get kinect pose
    kinect_pose_tf = RigidTransform.from_roll_pitch_yaw_x_y_z(
        kinect_pose.roll, kinect_pose.pitch, kinect_pose.yaw,
        kinect_pose.x, kinect_pose.y, kinect_pose.z)

    for pid,pose_tf in poses:
        if sensor_tf: 
            pose_tf = kinect_pose_tf * pose_tf
        pose_list_msg.objs.append(pose_tf.to_vs_obj_t(id=len(pose_list_msg.objs)));

    lc.publish("OBJ_COLLECTION", pose_list_msg.encode())
    # print 'Published %i poses' % (len(poses))

    # carr = plt.cm.Spectral(np.arange(len(arr)))
    # publish_point_cloud(pub_channel+'_POINTS', arr[:,:3], c=carr[:,:3])
