import cv2, os

class ImageDatasetWriter(object):
    """
    Simple Image Writer 
    """

    def __init__(self, directory, template='img_%i.png'): 
        # Initialize directory
        if not os.path.exists(directory): 
            os.makedirs(directory)
        self.root = directory
        self.template = template
        self.idx = 1

    def write(self, im, idx=None):
        idx_ = self.idx if idx is None else idx
        rgb_path = os.path.join(self.root, self.template % (idx_))
        cv2.imwrite(rgb_path, im)
        if idx is None: 
            self.idx = self.idx + 1

        print 'Written: %s' % rgb_path

class RGBDImageDatasetWriter(object): 
    def __init__(self, directory): 
        self.rgb_writer = ImageDatasetWriter(directory=directory, template='img_%i.png')
        self.depth_writer = ImageDatasetWriter(directory=directory, template='img_%i_depth.png')

    def write(self, frame): 
        self.rgb_writer.write(frame.getRGB())
        self.depth_writer.write(frame.getDepth())
        
if __name__ == "__main__": 
    # Write dataset
    writer = RGBDImageDatasetWriter(directory='/home/spillai/data/rgb-dataset-test/')

    # Read from lcm log 
    import numpy as np
    from fs_utils import LCMLogReader

    player = LCMLogReader(filename='/home/spillai/data/2014_01_12_artags_articulation_more/lcmlog-2014-01-12.24')

    while True: 
        f = player.getNextFrame()
        if f is None: break
        writer.write(f)
