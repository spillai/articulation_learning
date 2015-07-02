'''
============================================================
Testing basic ISAM Slam3d
============================================================
'''

import time
import numpy as np
import sys, cv2, collections, os.path, time
from optparse import OptionParser
from collections import defaultdict, namedtuple

from fs_isam import Pose3d, Slam3D

if __name__ == "__main__": 
    parser = OptionParser()
    (options, args) = parser.parse_args()
    
    slam = Slam3D()
    
    noise = np.eye(6) * 0.01

    slam.add_node_with_factor(0, Pose3d(0, 0, 0, 0.1, 0.2, 0.3), noise);
    slam.add_node(1);
    # slam.add_node();
    slam.add_node(3);
    slam.add_node(4);
    slam.add_node(7);
    # slam.add_node_with_factor(2, Pose3d(1, 0, 0, 0.1, 0.2, 0.3), noise);
    # slam.add_node_with_factor(3, Pose3d(2, 0, 0, 0.1, 0.2, 0.3), noise);

    slam.add_edge_factor(0, 1, Pose3d(1.0, 0, 0, 0, 0, 0), noise)
    slam.add_edge_factor(1, 3, Pose3d(1.0, 0, 0, 0, 0, 0), noise)
    # slam.add_edge_factor(3, 4, Pose3d(1.0, 0, 0, 0, 0, 0), noise)
    slam.add_edge_factor(3, 4, Pose3d(1.0, 0, 0, 0, 0, 0), noise)
    slam.add_edge_factor(4, 7, Pose3d(1.0, 0, 0, 0, 0, 0), noise)

    t1 = time.time()
    slam.batch_optimization();
    print 'Graph Optimization took', time.time() - t1, 's'
    
    print '===================================='
    print 'Printing SLAM graph: '
    slam.print_graph()
    print '===================================='
    print 'Printing SLAM graph stats: '
    slam.print_stats()
    print '===================================='
    print 'Printing all: '
    slam.print_all2()
    print '===================================='
    for node in slam.get_nodes(): 
        print node.x, node.y, node.z, node.roll, node.pitch, node.yaw
