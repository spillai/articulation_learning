#!/usr/bin/python
import time
import os.path, collections
from optparse import OptionParser

import tables as tb
import numpy as np
# from dbg_utils import set_trace

# ===== Feature3DWriter ====
class TableWriter: 
    def __init__(self): 
        pass

class Feature3DWriter(TableWriter): 
    def __init__(self, data): 
        self.description = {
            'utime': tb.Int64Col(), 
            'id': tb.Int64Col(), # feature ID
            'point': tb.Float32Col(shape=(2,)), # image coordinates
            'size': tb.Float32Col(), 
            'angle': tb.Float32Col(), 
            'response': tb.Float32Col(), 
            'xyz': tb.Float32Col(shape=(3,)), # 3-D 
            'normal': tb.Float32Col(shape=(3,)), # 3-D 
            'tangent': tb.Float32Col(shape=(3,)) # 3-D 
        }
        self.data = data;

    def write(self, table): 
        if not isinstance(self.data, list): 
            raise TypeError, 'data must be a list'

        # print 'Writing data to table'
        ro = table.row
        for idx,d in enumerate(self.data):
            ro['utime'] = d.utime
            ro['id'] = d.id
            ro['point'] = d.point.reshape(-1)
            ro['size'] = d.keypoint.size
            ro['angle'] = d.keypoint.angle
            ro['response'] = d.keypoint.response
            ro['xyz'] = d.xyz().reshape(-1)
            # print 'xyz: ', d.xyz().reshape(-1)
            ro['normal'] = d.normal().reshape(-1)
            ro['tangent'] = d.tangent().reshape(-1)
            ro.append()

            if idx % int(len(self.data)/10) == 0: 
                print 'Processed %3.2f %%' % (idx * 100.0 / len(self.data))

# ===== Pose3DWriter ====
class Pose3DWriter(TableWriter):
    def __init__(self, data):
        self.description = { 
            'utime': tb.Int64Col(),       # sensor utime index
            'id': tb.Int64Col(),       # feature ID
            'xyz': tb.Float32Col(shape=(3,)),    # 3-D 
            'quaternion': tb.Float32Col(shape=(4,)),    # Quaternion
        }
        self.data = data

    def write(self, table): 
        if not isinstance(self.data, list): 
            raise TypeError, 'data must be a list'

        ro = table.row
        # set_trace()
        for idx,d in enumerate(self.data): 
            utime, pid, pose = d
            ro['utime'] = utime
            ro['id'] = pid
            ro['xyz'] = pose.tvec.reshape(-1)
            ro['quaternion'] = pose.quat.q.reshape(-1)
            ro.append()

            if idx % int(len(self.data)/10) == 0: 
                print 'Processed %3.2f %%' % (idx * 100.0 / len(self.data))

KinectData = collections.namedtuple('KinectData', ['utime','rgb','depth','X'])
# SuperFeature = collections.namedtuple('SuperFeature', ['utime','feature_id','x','X','normal'])

# ===== KinectFrame ====
class KinectFrame(tb.IsDescription):
    """
    KinectFrame Table:
    -    utime: sensor utime from log
    -    rgb: rgb in uint8 (c-contiguous)
    -    depth: depth map in float32
    -    X: organized pcl in float32 (h x w x 3)    
    """
    utime = tb.Int64Col()       # sensor utime index
    rgb = tb.UInt8Col(shape=(480,640,3))        # img (np.uint8)
    depth = tb.Float32Col(shape=(480,640))    # depth img (np.float32)
    X = tb.Float32Col(shape=(480,640,3))     # organized pcl (np.float32)

# ===== SuperFeatureTable ====
class SuperFeatureTable(tb.IsDescription):
    """
    SuperFeature Table:
    -    utime: sensor utime from log
    -    feature_id: ID of the feature
    -    x: image coordinate of feature
    -    X: 3-D coordinate of feature
    -    normal: surface normal of feature
    """
    utime = tb.Int64Col()       # sensor utime index
    feature_id = tb.Int64Col()       # feature ID
    x = tb.Float32Col(shape=(2,))    # image coordinates
    X = tb.Float32Col(shape=(3,))    # 3-D 
    normal = tb.Float32Col(shape=(3,))  # 3-D surface normal

# ===== Feature3DTable ====
class Feature3DTable(tb.IsDescription):
    """
    SuperFeature Table:
    """
    utime = tb.Int64Col()       # sensor utime index
    id = tb.Int64Col()       # feature ID
    point = tb.Float32Col(shape=(2,))    # image coordinates
    size = tb.Float32Col()    
    angle = tb.Float32Col()    
    response = tb.Float32Col()    
    xyz = tb.Float32Col(shape=(3,))    # 3-D 
    normal = tb.Float32Col(shape=(3,))    # 3-D 
    tangent = tb.Float32Col(shape=(3,))    # 3-D 

# ===== Pose3DTable ====
class Pose3DTable(tb.IsDescription):
    """
    Pose3D Table:
    """
    utime = tb.Int64Col()       # sensor utime index
    id = tb.Int64Col()       # feature ID
    xyz = tb.Float32Col(shape=(3,))    # 3-D 
    quaternion = tb.Float32Col(shape=(4,))    # Quaternion

def get_output_folder(fn, ext='.h5'): 
    log_path, log_file = os.path.split(fn)
    if not len(log_path): log_path = os.getcwd()
    processed_dir = '%s/processed' % (log_path)
    if not os.path.exists(processed_dir): os.makedirs(processed_dir)
    processed_filename = '%s/%s%s' % (processed_dir, log_file, ext)
    return processed_filename

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise

def get_processed_filename(fn, dir_name='processed'): 
    log_path, log_file = os.path.split(fn)
    if not len(log_path): log_path = os.getcwd()
    processed_dir = '%s/%s' % (log_path, dir_name)
    if not os.path.exists(processed_dir): os.makedirs(processed_dir)
    processed_filename = '%s/%s' % (processed_dir, log_file)
    return processed_filename

def make_sandbox_with_date(): 
    today = time.strftime("%Y-%m-%d", time.gmtime()), 
    processed_dir = '%s/processed/%s' % (os.getcwd(),today)
    if not os.path.exists(processed_dir): os.makedirs(processed_dir)
    return processed_dir

