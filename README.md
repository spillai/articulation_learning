## Build

Check ```software/tobuild.txt``` and ```software/externals/tobuild.txt```

**Build system**
```shell
cd ~/articulation_learning/software
make -j 8
```

**Note:**
- Look at ```externals/pcl-pod/pcl_common_include.patch``` - you might have to copy over some files from
  ```software/externals/pcl-pod/pcl-1.7.0/common/include/pcl/``` to
  ```software/build/include/pcl/ (conversions.h, register_point_struct.h, point_types_conversion.h)```

## General setup

**Terminal 1**
```shell
bot-param-server config/articulation_learning.cfg
```
**Terminal 2**
```shell
cd ~/articulation_learning/software/python/apps
python articulation_learning_demo.py
```

## Code Organization: 

  **config/**
 - config scripts such as wheelchair.cfg for bot-param-server
  **software/**
 - compilation of pods, and modules for articulation learning
  **scripts/**
 - scripts for running various utilities

## Directory layout
------------------------------------------------
  - software: 
    - Within the software folder, code is organized as follows
    - Comment/Uncomment pods in the ```tobuild.txt``` file to add/remove unnecessary
   dependencies
    - apps: 
     - mostly test apps that are used to check/run various implementations

------------------------------------------------
  - drivers: 
    - kinect driver for reading/publishing kinect data over lcm

------------------------------------------------
  - externals: 
    - external dependencies include opencv, pcl, apriltags, eigen, isam, lcm,
     libbot, and a few other visualization tools

------------------------------------------------
  - modules: 
    - articulation: main articulation learning module
    - birchfield's klt
    - dbow2: bag of words implementation
    - more modules

------------------------------------------------
  - python: 
    - notebooks: 
    - ipython notebooks for articulation learning 
    - run articulation_learning.ipynb
  - utils: 
    - all the different components in the articulation learning implmented in
      python with certain modules wrapped in C++ (see pywrappers)
    - look at ```al_lfd.py``` and ```articulation_learning_demo.py``` initially for articulation learning
    
------------------------------------------------
  - pywrappers: 
    - apriltags, articulation, pcl, and fs_utils wrapped utilities for python access
    - utils/: provides containers, and registered conversions from/to opencv<=>numpy 

------------------------------------------------
  - utilities: 
    - general purpose utilities for opencv, pcl, lcm, and opencv-based rgbd containers. Also a few visualization utilities

------------------------------------------------
  - viewer: 
    - al-viewer for visualizing results/data


## BASHRC setup

```shell
  export ALEARN=$HOME/articulation_learning
  export PATH=$ALEARN/bin:$PATH
  export LD_LIBRARY_PATH=$ALEARN/software/build/lib/:$ALEARN/software/build/lib64:$LD_LIBRARY_PATH
  export PYTHONPATH=$ALEARN/software/python:$ALEARN/software/build/lib:$PYTHONPATH
  export PYTHONPATH=$ALEARN/software/build/lib/python2.7/dist-packages:$ALEARN/software/build/lib/python2.7/site-packages:$PYTHONPATH
  export DYLD_LIBRARY_PATH=$ALEARN/software/build/lib:$DYLD_LIBRARY_PATH
  export PKG_CONFIG_PATH=/usr/lib/pkgconfig:/usr/share/pkgconfig/:/usr/share/cmake/Modules:$PKG_CONFIG_PATH
  export PKG_CONFIG_PATH=$ALEARN/software/build/lib64/pkgconfig/:$ALEARN/software/build/lib64/pkgconfig:$PKG_CONFIG_PATH
```

## DATA

```shell
ssh spillai@virgo2
```

