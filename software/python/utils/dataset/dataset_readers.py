import cv2
import numpy as np
import os, fnmatch, time

from collections import defaultdict, namedtuple

def read_dir_recursive(directory, pattern): 
    # Get directory, and filename pattern
    directory = os.path.expanduser(directory)

    if not os.path.exists(directory): 
        raise Exception("""Path %s doesn't exist""" % directory)

    # Build dictionary [full_root_path -> pattern_matched_files]
    fn_map = {}
    for root, dirs, files in os.walk(directory): 

        # Filter only filename matches 
        matches = [os.path.join(root, fn) 
                   for fn in fnmatch.filter(files, pattern)]
        if not len(matches): continue
        fn_map[root] = matches
    return fn_map

class ImageDatasetReader: 
    """
    Simple Image Dataset Reader
    Refer to this class and ImageDatasetWriter for input/output
    """
    def __init__(self, directory, template='img_%i.png', pattern='*.png'):
        # Get directory, and filename pattern
        self.directory = os.path.expanduser(directory)
        self.template = template
        self.pattern = pattern
        self.files = []
        self.cidx = 0

        # Index starts at 1
        frame_info = namedtuple('frame_info', ['index', 'rgb_fn'])
        idx = 1;
        while True: 
            rgb_path = os.path.join(self.directory, self.template % idx)
            print rgb_path
            if not os.path.exists(rgb_path): 
                break

            self.files.append(
                frame_info(index=idx, rgb_fn=rgb_path)
            )
            idx += 1

    def reset(self): 
        self.cidx = 0

    def get_next(self): 
        if self.cidx >= len(self.files):
            return None
        im = cv2.imread(self.files[self.cidx].rgb_fn)
        self.cidx += 1
        return im


    # Retrieve frame (would be nice to use generator instead)
    def get_frame(self, rgb_fn, depth_fn): 
        rgb, depth = cv2.imread(rgb_fn, 1), cv2.imread(depth_fn, -1)
        return rgb, depth

class RGBDDatasetReaderUW:
    """
    RGB-D Dataset reader 
    http://rgbd-dataset.cs.washington.edu/dataset.html
    """

    def __init__(self, directory): 
        import scipy.io
        from utils.frame_utils import Frame, KinectFrame

        # Initialize dataset reader
        fn_map = read_dir_recursive(directory, '*.png')

        # Build rgbd filename map [object_type -> [(rgb_fn,depth_fn), .... ]
        self.rgbd_fn_map = defaultdict(list)
        self.mat_fn_map = {}

        # Index starts at 1
        frame_info = namedtuple('frame_info', ['index', 'rgb_fn', 'depth_fn'])
        for root,v in fn_map.iteritems(): 
            idx = 1;
            while True: 
                rgb_path = os.path.join(root, '%s_%i.png' % 
                                        (os.path.basename(root), idx))
                depth_path = os.path.join(root, '%s_%i_depth.png' % 
                                          (os.path.basename(root), idx))
                if not os.path.exists(rgb_path) or not os.path.exists(depth_path): 
                    break

                self.rgbd_fn_map[os.path.basename(root)].append(
                    frame_info(index=idx, rgb_fn=rgb_path, depth_fn=depth_path)
                )
                idx += 1

        # Convert mat files
        mat_map = read_dir_recursive(directory, '*.mat')
        for files in mat_map.values(): 
            for fn in files: 
                self.mat_fn_map[os.path.splitext(os.path.basename(fn))[0]] = fn
        print self.mat_fn_map

        # Store relevant metadata about dataset
        self.total_frames = idx
        self.rgb = True

    # Retrieve metadata
    def load_metadata(self, mat_fn): 
        return scipy.io.loadmat(mat_fn, squeeze_me=True, struct_as_record=False)

    # Get metadata for a particular category
    def get_metadata(self, category): 
        matfile = self.mat_fn_map[category]
        return self.load_metadata(matfile)

    # Get bounding boxes for a particular category
    def get_bboxes(self, category): 
        return self.get_metadata(category)['bboxes']

    # Get files for a particular category
    def get_files(self, category): 
        return self.rgbd_fn_map[category]

    # Retrieve frame (would be nice to use generator instead)
    def get_frame(self, rgb_fn, depth_fn): 
        rgb, depth = cv2.imread(rgb_fn, 1), cv2.imread(depth_fn, -1)
        return rgb, depth

def test_rgbd_uw(): 
    # Read dataset
    rgbd_data_uw = RGBDDatasetReaderUW('~/data/rgbd-datasets/udub/rgbd-scenes')

    # Get metadata for object (bounding boxe)s
    # Note: length of metadata and frames should be the same
    matfile = rgbd_data_uw.get_metadata('table_1')
    bboxes = rgbd_data_uw.get_bboxes('table_1')

    # Get files for object
    files = rgbd_data_uw.get_files('table_1')

    # Construct frames with (rgb, d) filenames
    runtime = []
    for fidx, (bbox, f) in enumerate(zip(bboxes, files)): 
        rgb, depth = rgbd_data_uw.get_frame(f.rgb_fn, f.depth_fn)
        t1 = time.time()
        KinectFrame(f.index, rgb, depth, skip=1)
        print [(bb.category, bb.instance, bb.top, bb.bottom, bb.left, bb.right) 
               for bb in np.array([bbox]).flatten()]
        runtime.append((time.time() - t1) * 1e3)
        if fidx % 10 == 0: print 'Processed ', fidx
    print 'Done! Processed %i items. %i bboxes' % (len(files), len(bboxes))
    print 'Average runtime: ', np.mean(runtime), 'ms'


def test_imagedataset_reader(): 
    # Read dataset
    rgb_data = ImageDatasetReader(directory='~/data/rgb-dataset-test', 
                                  pattern='*.png', template='img_%i.png')
    
    while True: 
        im = rgb_data.get_next()
        if im is None: 
            break

        print im.shape

if __name__ == "__main__": 
    # RGBD UW dataset
    # test_rgbd_uw()

    # ImageDatasetReader
    test_imagedataset_reader()
    
