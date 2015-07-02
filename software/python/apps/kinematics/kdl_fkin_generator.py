'''
============================================================
KDL Forward Kinematics Generator
============================================================
'''
# Generate different configurations (rigid, pris, rotational)
# python kdl_fkin_generator.py -c <config> -o <output db>
# python kdl_fkin_generator.py -o tests-without-noise.h5 -c ~/envoy/software/perception/spillai-sandbox/config/articulation-simulation.yaml -n 0.000
# python kdl_fkin_generator.py -o tests-with-0.0005-noise.h5 -c ~/envoy/software/perception/spillai-sandbox/config/articulation-simulation.yaml -n 0.0005
# python kdl_fkin_generator.py -o tests-with-0.001-noise.h5 -c ~/envoy/software/perception/spillai-sandbox/config/articulation-simulation.yaml -n 0.001
# python kdl_fkin_generator.py -o tests-with-0.002-noise.h5 -c ~/envoy/software/perception/spillai-sandbox/config/articulation-simulation.yaml -n 0.002

import lcm
import numpy as np
import PyKDL as kdl
import time

from optparse import OptionParser
import yaml

import utils.rigid_transform as rtf
import utils.draw_utils as draw_utils

from fs_utils import Feature3D
from utils.log_utils import *
from utils.db_utils import AttrDict, DictDB
from utils.io_utils import Feature3DTable, Pose3DTable, Feature3DWriter, Pose3DWriter

import xml.etree.ElementTree as ET

def kdl_T(f):
    return np.array([[f.M[0,0], f.M[0,1], f.M[0,2], f.p[0]],
                         [f.M[1,0], f.M[1,1], f.M[1,2], f.p[1]],
                         [f.M[2,0], f.M[2,1], f.M[2,2], f.p[2]],
                         [0,0,0,1]])
def kdl_R(f):
    return np.array([[f.M[0,0], f.M[0,1], f.M[0,2]],
                         [f.M[1,0], f.M[1,1], f.M[1,2]],
                         [f.M[2,0], f.M[2,1], f.M[2,2]]])

def kdl_t(f):
    return np.array([f.p[0],f.p[1],f.p[2]])

def get_type(t): 
    tvec = [float(v) for v in t['tvec'].strip('[]').split(' ')]
    rvec = [float(v) for v in t['rvec'].strip('[]').split(' ')]
    if t['type'] == 'rigid': 
        return (kdl.Joint.TransY, kdl.Vector(tvec[0], tvec[1], tvec[2]))
    elif t['type'] == 'prismatic': 
        return (kdl.Joint.TransX, kdl.Vector(tvec[0], tvec[1], tvec[2]))
    elif t['type'] == 'rotational': 
        return (kdl.Joint.RotZ, kdl.Vector(tvec[0], tvec[1], tvec[2]))

def build_fpt(utime, fid, pose): 
    f = AttrDict()
    f.utime = utime
    f.id = fid
    f.point = np.array([0,0])
    f.keypoint = AttrDict()
    f.keypoint.size, f.keypoint.angle, f.keypoint.response = 0, 0, 0
    f.xyz_ = pose.tvec
    f.normal_ = pose.to_homogeneous_matrix()[:3,0]
    f.tangent_ = np.array([np.nan,np.nan,np.nan])
    f.xyz = lambda : f.xyz_
    f.normal = lambda : f.normal_
    f.tangent = lambda : f.tangent_
    return f

def get_joint_config(configuration): 
    joint_dict = {}
    for idx,config in enumerate([c.split(',') for c in configuration]): 
        joint_dict.update({idx: dict([c.split('=') for c in config])})
    return joint_dict

def default_config(): 
    return """type=rigid,tvec=[0.0 0 0],rvec=[0 0 0],min=0,max=0.0;"""\
        """type=prismatic,tvec=[0.2 0 0],rvec=[0 0 0],min=0,max=0.1;"""\
        """type=rotational,tvec=[0.2 0 0],rvec=[0 0 0],min=0,max=%3.2f""" % (90 * np.pi / 180)

def default_rotrotrot_config(): 
    return """type=rigid,tvec=[0.0 0 0],rvec=[0 0 0],min=0,max=0.0;"""\
        """type=rotational,tvec=[0.2 0 0],rvec=[0 0 0],min=0,max=%3.2f;"""\
        """type=rotational,tvec=[0.3 0 0],rvec=[0 0 0],min=0,max=%3.2f;"""\
        """type=rotational,tvec=[0.4 0 0],rvec=[0 0 0],min=0,max=%3.2f""" % \
        (90 * np.pi / 180, 90 * np.pi / 180, 90 * np.pi / 180)


def kinematic_tree_sampler(configuration, simulate=False, sigma_noise=0.):
    ground_truth_poses, sampled_poses = [], []

    # Build Chain ======================================================
    chain = kdl.Chain()
    
    # Setup joint dict
    joint_dict = get_joint_config(configuration)
    njoints = len(joint_dict)

    print 'Num Joints: %i' % njoints
    print 'Input Joint Configuration: %s' % joint_dict
    print 'Num Features per joint: %i' % options.nfeats

    # Setup utimes, d
    utimes = np.linspace(0,10*1e6,100)
    jntVals = np.zeros((njoints,len(utimes)))
    
    # Add segments based on configuration
    config_description = []
    for idx,v in joint_dict.iteritems(): 
        if 'type' not in v or 'tvec' not in v or 'rvec' not in v: continue;
        val = get_type(v)
        chain.addSegment(kdl.Segment(kdl.Joint(val[0]), kdl.Frame(val[1])))

        if 'min' not in v or 'max' not in v: continue;
        jntVals[idx,:] = np.linspace(float(v['min']),float(v['max']), len(utimes))
        config_description.append(v['type'])
    config_description.append('noise-%5.4f' % sigma_noise)

    # FK ===============================================================
    fk = kdl.ChainFkSolverPos_recursive(chain)
    end_jntArray = kdl.JntArray(njoints)    

    # Keep relevant link info for sampling
    joint_feats = dict();

    # Viz
    viz_poses, viz_points1 = [], []
    # draw_utils.publish_sensor_frame('KINECT_FRAME')

    # Sample ===========================================================
    # For each timestamp, and each joint
    for ut_idx,utime in enumerate(utimes):
        # Build frame
        endeff = kdl.Frame()    
        for jnt_idx in range(njoints):
            end_jntArray[jnt_idx] = jntVals[jnt_idx][ut_idx];

        tf_feats, rpy_feats = [], []
        for jnt_idx in range(njoints):       
            fk.JntToCart(end_jntArray, endeff, jnt_idx+1)
            pose = rtf.RigidTransform.from_homogenous_matrix(kdl_T(endeff))
            ground_truth_poses.extend([(utime,jnt_idx*options.nfeats,pose)])
            viz_poses.append(pose)

            # First time step, init points
            if ut_idx == 0: 
                samples = np.random.normal([0,0,0], scale=0.2, size=(options.nfeats,3))
                if sigma_noise != 0.: 
                    samples += np.random.normal([0,0,0], scale=sigma_noise, 
                                                size=(options.nfeats,3))
                joint_feats[jnt_idx] = {}
                joint_feats[jnt_idx]['init'] = pose
                joint_feats[jnt_idx]['samples'] = [rtf.RigidTransform([0,0,0,1], sample) 
                                                   for sample in samples]

            # Poses are transformed based on the sampling
            pose_init = joint_feats[jnt_idx]['init']
            poses_samples = joint_feats[jnt_idx]['samples']
            poses = [pose * posei for posei in poses_samples]
            if sigma_noise != 0.: 
                noise = np.random.normal([0,0,0], scale=sigma_noise, 
                                         size=(options.nfeats,3))
                poses = [rtf.RigidTransform([0,0,0,1], 
                                            np.random.normal([0,0,0], 
                                                             scale=sigma_noise)) * pose
                         for pose in poses]
            viz_points1.extend([pose.tvec for pose in poses])

            # Build Feature3D pts and Pose3D ground truth
            fpts = [build_fpt(utime,jnt_idx*options.nfeats+sidx,p) 
                    for sidx,p in enumerate(poses)]
            sampled_poses.extend(fpts)


        if simulate: time.sleep(0.1)
        draw_utils.publish_pose_list2('SIMULATED_POSES', viz_poses)
        draw_utils.publish_point_cloud('SIMULATED_POSES_SAMPLED_POINTS', 
                                       np.vstack(viz_points1), c='b')

    return ground_truth_poses, sampled_poses, config_description

if __name__ == "__main__": 
    parser = OptionParser()
    parser.add_option("-o", "--output-file", dest="filename", 
                      help="Output db", 
                      metavar="FILENAME")
    parser.add_option("-c", "--config", dest="configuration", 
                      help="Configurations ((r)igid|(p)ris|ro(t))",
                      default=default_rotrotrot_config(),
                      type="string",
                      metavar="Configuration")
    parser.add_option("-f", "--feats", dest="nfeats", 
                      help="Num feats per link",default=5,type="string",
                      metavar="Features")
    parser.add_option("-n", "--noise-sigma", dest="noise", 
                      help="Observation noise",default=0.0,type="float",
                      metavar="Observation noise")
    parser.add_option("-s", "--simulate", dest="simulate", 
                      help="Simulate (add sleep)",default=False,type=int,
                      metavar="Simulate")
    (options, args) = parser.parse_args()
    
    # Required options =================================================
    if not options.filename: parser.error('Filename not given')
    if not options.configuration: parser.error('Configuration not given')

    # Open the configuration script
    with open(options.configuration) as f: 
        config = yaml.load(f)

    # Check tests
    if 'tests' not in config: 
        print 'No tests listed in %s' % options.configuration
        exit(1)

    # Setup DB =========================================================
    db = DictDB(filename=options.filename, mode='w')

    # Parse through tests
    for tname, t in config['tests'].iteritems(): 
        if 'description' not in t: continue;

        print('Running test %s with \n%s' % (tname,t['description']))
        print t['description']

        # Generate ground truth and samples from configuration 
        ground_truth_poses, sampled_poses, \
            config_description = kinematic_tree_sampler(t['description'], 
                                                        simulate=options.simulate, 
                                                        sigma_noise=options.noise)

        db.data[tname] = AttrDict()
        db.data[tname].config = t
        db.data[tname].config_description = config_description
        db.data[tname].config_pretty_name = t['name']

        db.data[tname].samples = Feature3DWriter(data=sampled_poses); 
        db.data[tname].ground_truth = Pose3DWriter(data=ground_truth_poses); 

        # db.data[tname].samples = (Feature3DWriter(), sampled_poses)

        # break
    db.flush()
    db.close()
    print 'Done saving to %s' % options.filename
    # # Add Segments
    # for c in tree.iter(tag='joint'): 
    #     parent, child, tvec, rvec = [None] * 4
    #     d = dict([(ch.tag, ch.attrib) for ch in c.getchildren()])
    #     parent,child = link_dict[d['parent']['link']], link_dict[d['child']['link']]

    #     # Get joint params
    #     jtype = get_type(c.attrib['type'])
    #     tvec, rvec = get_origin(d)

    #     seg = kdl.Segment(kdl.Joint(jtype), kdl.Frame(kdl.Vector(tvec[0], tvec[1], tvec[2])))
    #     # print c.tag, c.attrib, parent, child


    # # Setup utimes, d
    # utimes = np.linspace(0,10*1e6,100)
    # jntVals = np.zeros((njoints,len(utimes)))

    # # Forward kinematics
    # jntArray = kdl.JntArray(njoints)

        
    # exit(1)


    # # Parse tree
    # tree = ET.parse(options.urdf)
    # root = tree.getroot()
    # # link_dict = dict([(c.attrib['name'],idx) for idx,c in enumerate(tree.iter(tag='link'))])


        
