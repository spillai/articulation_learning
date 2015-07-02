#!/usr/bin/env python

from distutils.core import setup

setup(name='flann',
      version='1.6.9',
      description='Fast Library for Approximate Nearest Neighbors',
      author='Marius Muja',
      author_email='mariusm@cs.ubc.ca',
      license='BSD',
      url='http://www.cs.ubc.ca/~mariusm/flann/',
      packages=['pyflann', 'pyflann.io', 'pyflann.bindings', 'pyflann.util', 'pyflann.lib'],
      package_dir={'pyflann.lib':'/home/marius/ubc/flann/build/lib'},
      package_data={'pyflann.lib': ['libflann.so','flann.dll', 'libflann.dylib']}, 
)
