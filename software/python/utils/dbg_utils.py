# import sys
# from IPython.core.debugger import Pdb
#from IPython.core import interactiveshell as IPShell
# from IPython.core import ipapi

#shell = IPShell(argv=[''])

# def set_trace():
#     ip = ipapi.get()
#     def_colors = ip.colors
#     Pdb(def_colors).set_trace(sys._getframe().f_back)

import logging
from termcolor import colored

class ColorLog(object):
    colormap = dict(
        debug=dict(color='grey', attrs=['bold']),
        info=dict(color='white'),
        warn=dict(color='yellow', attrs=['bold']),
        warning=dict(color='yellow', attrs=['bold']),
        error=dict(color='red'),
        critical=dict(color='red', attrs=['bold']),
    )
 
    def __init__(self, logger):
        self._log = logger
 
    def __getattr__(self, name):
        if name in ['debug', 'info', 'warn', 'warning', 'error', 'critical']:
            return lambda s, *args: getattr(self._log, name)(
                                colored(s, **self.colormap[name]), *args)
 
        return getattr(self._log, name)
 
# log = ColorLog(logging.getLogger(__name__))
