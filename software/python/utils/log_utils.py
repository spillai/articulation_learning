#!/usr/bin/python

import logging
# logging.basicConfig(level=logging.DEBUG,
#                     format='(%(threadName)-10s) %(message)s',
# )
def green(s):
    return '\033[1;32m%s\033[m' % s
def red(s):
    return '\033[1;31m%s\033[m' % s
def yellow(s):
    return '\033[1;33m%s\033[m' % s
def blue(s):
    return '\033[1;34m%s\033[m' % s
def magenta(s):
    return '\033[1;35m%s\033[m' % s

# glog = lambda x: logging.debug(green(x))
# rlog = lambda x: logging.debug(red(x))
# ylog = lambda x: logging.debug(yellow(x))
# blog = lambda x: logging.debug(blue(x))
# mlog = lambda x: logging.debug(magenta(x))

# gprint = lambda x: print green(x)
# rprint = lambda x: print red(x)
# yprint = lambda x: print yellow(x)
# bprint = lambda x: print blue(x)
# mprint = lambda x: print magenta(x) 
