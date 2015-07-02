from __future__ import division
import cv2
import math
import numpy as np
import matplotlib as mpl
import matplotlib.pylab as plt
import collections
import itertools as it

class ImageViewer:
    def __init__(self):
        self.w, self.h = 1280, 960
        self.frame = np.empty((self.h,self.w,3), dtype=np.uint8) #(h,w)
        self.image_map = collections.OrderedDict();
        self.grid_sz = np.array([1,1])
        self.tile_sz = np.array([self.w,self.h]) # (w,h)
        self.max_grid_width = 4;
    
    def reconfigure_grid(self):
        cv2.namedWindow('cv viewer', cv2.WINDOW_NORMAL)
        print 'image map size: ', len(self.image_map)
        self.grid_sz[0] = min(math.ceil(np.sqrt(len(self.image_map))), self.max_grid_width)
        self.grid_sz[1] = math.ceil(len(self.image_map) / self.grid_sz[0])
        print 'grid size: ', self.grid_sz
    
        self.tile_sz = np.divide(np.array([self.w,self.h]), self.grid_sz);
        print 'tile size: ', self.tile_sz

        #keys = self.image_map.keys()    
        #for k in keys:
        #    self.image_map[k] = cv2.resize(self.image_map[k], (self.tile_sz[0], self.tile_sz[1]))        

    def overlay_attributes(self, frame, name, r):
        rect = [int(i) for i in r]
        tlx,tly,brx,bry = rect    
        cv2.rectangle(frame, (tlx,bry-15), (brx,bry), (20,20,20), -1)
        cv2.putText(frame, '%s'%name, (tlx+5,bry-5), 0, 0.35, (200,200,200), thickness=1)        
        cv2.rectangle(frame, (tlx,tly), (brx,bry), (55,55,55), 1)
    
    def draw_image(self, name, img, rx, ry, rw, rh):
        tlx, tly = rx, ry;
        brx, bry = rx+rw, ry+rh;
        h,w,c = img.shape
        s = max(h/rh, w/rw);
        #print 'scale: ', (h/rh, w/rw), img.shape
        sw, sh = min(int(w*s),rw), min(int(h*s),rh)
        #print 'sw, sh', sw, sh
        #img = cv2.resize(img, (rw,rh))        
        img = cv2.resize(img, (sw,sh))
        xoff, yoff = math.floor(abs(rw-sw)/2), math.floor(abs(rh-sh)/2)
        #print 'xoff, yoff', xoff, yoff

        slx,sly = tlx+xoff, tly+yoff
        srx,sry = tlx+xoff+sw, tly+yoff+sh
        self.image_map[name] = img;
        self.frame[sly:sry,:,:][:,slx:srx,:] = img
    
        self.overlay_attributes(self.frame, name, [tlx,tly,brx,bry])
    
    def draw(self, name, img): 
        keys = self.image_map.keys()
        for k in keys:
                j = keys.index(k)
                x = j % self.tile_sz[0]
                y = math.floor(j / self.tile_sz[0])
                print 'draw: ', k, x, y
                self.draw_image(name, img, x * self.tile_sz[0], y * self.tile_sz[1], self.tile_sz[0], self.tile_sz[1])        
        

    def imshow(self, name, img):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        reconfigure = not self.image_map.has_key(name)
        self.image_map[name] = None
        if reconfigure: self.reconfigure_grid()    
        self.draw(name, img)

        if self.frame.size:
            cv2.imshow('cv viewer', self.frame)
            cv2.waitKey(10)
        
    
viewer = ImageViewer();
def imshow(name, img):
    # print 'static imshow', viewer
    if viewer:
        viewer.imshow(name, img)                

def grouper(n, iterable, fillvalue=None):
    '''grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx'''
    args = [iter(iterable)] * n
    return it.izip_longest(fillvalue=fillvalue, *args)

def mosaic(w, imgs):
    '''Make a grid from images.

    w    -- number of grid columns
    imgs -- images (must have same size and format)
    '''
    imgs = iter(imgs)
    img0 = imgs.next()
    pad = np.zeros_like(img0)
    imgs = it.chain([img0], imgs)
    rows = grouper(w, imgs, pad)
    return np.vstack(map(np.hstack, rows))
    
